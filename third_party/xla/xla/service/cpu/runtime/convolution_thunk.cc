/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "xla/service/cpu/runtime/convolution_thunk.h"

#define EIGEN_USE_THREADS

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "Eigen/Core"  // from @eigen_archive
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "xla/executable_run_options.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/service/cpu/runtime_conv2d_acl.h"
#include "xla/service/cpu/runtime_conv2d_mkl.h"
#include "xla/service/cpu/runtime_conv_impl.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::cpu {
namespace {

bool IsSupportedType(PrimitiveType primitive_type) {
  return primitive_type == PrimitiveType::F16 ||
         primitive_type == PrimitiveType::F32;
}

auto GetConvolutionRank(const Shape& input_shape) {
  // Convolution rank is the number of spatial dimensions. Besides spatial
  // dimensions, input shape contains two other dimensions (batch size and the
  // number of channels).
  return input_shape.dimensions_size() - 2;
}

bool CanUseMklDnn(const DebugOptions& debug_options,
                  int64_t feature_group_count, PrimitiveType primitive_type) {
  if (!debug_options.xla_cpu_use_mkl_dnn() || feature_group_count != 1 ||
      primitive_type != PrimitiveType::F32) {
    return false;
  }

  if (!debug_options.xla_cpu_multi_thread_eigen()) {
    LOG(WARNING) << "Using Eigen instead of MKL-DNN for single-threaded "
                    "convolution.";
    return false;
  }

  return true;
}

bool CanUseACL(const DebugOptions& debug_options,
               PrimitiveType primitive_type) {
  return debug_options.xla_cpu_use_acl() &&
         primitive_type == PrimitiveType::F32;
}

auto MakeRunOptions(const Eigen::ThreadPoolDevice* threadpool) {
  ExecutableRunOptions run_options;
  run_options.set_intra_op_thread_pool(threadpool);
  return run_options;
}

}  // namespace

absl::StatusOr<std::unique_ptr<ConvolutionThunk>> ConvolutionThunk::Create(
    Info info, const DebugOptions& debug_options,
    BufferAllocation::Slice input_buffer, const Shape& input_shape,
    BufferAllocation::Slice kernel_buffer, const Shape& kernel_shape,
    BufferAllocation::Slice output_buffer, const Shape& output_shape,
    const ConvolutionDimensionNumbers& dnums, const Window& window,
    int64_t feature_group_count) {
  // TODO(abanas): Add shape verification (batch size, feature count, etc.)
  auto primitive_type = input_shape.element_type();
  if (!IsSupportedType(primitive_type)) {
    VLOG(2) << "Unsupported element type: " << primitive_type;
    return InvalidArgument("ConvolutionThunk: Unsupported element type");
  }

  int64_t convolution_rank = GetConvolutionRank(input_shape);
  if (convolution_rank > 3) {
    VLOG(2) << "Incorrect convolution rank: " << convolution_rank;
    return InvalidArgument("ConvolutionThunk: Incorrect convolution rank");
  }

  absl::InlinedVector<int64_t, 2> input_dims;
  absl::InlinedVector<int64_t, 2> kernel_dims;
  absl::InlinedVector<int64_t, 2> output_dims;

  // We lower 1D convolutions into calls to the same Eigen function as 2D
  // convolutions, except that we pretend that the 1D convolution is really
  // a 2D convolution with the missing dimension set to 1.  We also adjust
  // the padding, dilation parameters as needed.
  if (convolution_rank == 1) {
    input_dims.push_back(1);
    kernel_dims.push_back(1);
    output_dims.push_back(1);
  }

  // Configuration.
  bool multi_threaded = debug_options.xla_cpu_multi_thread_eigen();
  bool use_mkl_dnn =
      CanUseMklDnn(debug_options, feature_group_count, primitive_type);
  bool use_acl = CanUseACL(debug_options, primitive_type);

  // Input tensor.
  int64_t input_batch = input_shape.dimensions(dnums.input_batch_dimension());
  for (int d : dnums.input_spatial_dimensions()) {
    input_dims.push_back(input_shape.dimensions(d));
  }
  int64_t input_channels =
      input_shape.dimensions(dnums.input_feature_dimension());

  // Kernel tensor.
  for (int d : dnums.kernel_spatial_dimensions()) {
    kernel_dims.push_back(kernel_shape.dimensions(d));
  }
  int64_t kernel_channels =
      kernel_shape.dimensions(dnums.kernel_input_feature_dimension());
  int64_t kernel_filters =
      kernel_shape.dimensions(dnums.kernel_output_feature_dimension());

  // Output tensor.
  for (int d : dnums.output_spatial_dimensions()) {
    output_dims.push_back(output_shape.dimensions(d));
  }

  // Extract the window stride for the convolution.
  absl::InlinedVector<int64_t, 2> strides;
  absl::InlinedVector<std::pair<int64_t, int64_t>, 2> padding;
  absl::InlinedVector<int64_t, 2> base_dilation;
  absl::InlinedVector<int64_t, 2> window_dilation;
  if (convolution_rank == 1) {
    strides.push_back(1);
    padding.push_back({0, 0});
    base_dilation.push_back(1);
    window_dilation.push_back(1);
  }
  for (const auto& d : window.dimensions()) {
    strides.push_back(d.stride());
    padding.push_back({d.padding_low(), d.padding_high()});
    base_dilation.push_back(d.base_dilation());
    window_dilation.push_back(d.window_dilation());
  }

  auto valid_num_dims = [](absl::Span<const int64_t> xs) {
    return xs.size() >= 2 && xs.size() <= 3;
  };
  TF_RET_CHECK(valid_num_dims(input_dims)) << input_dims.size();
  TF_RET_CHECK(valid_num_dims(kernel_dims));
  TF_RET_CHECK(valid_num_dims(output_dims));
  TF_RET_CHECK(valid_num_dims(strides));
  TF_RET_CHECK(padding.size() >= 2 && padding.size() <= 3);
  TF_RET_CHECK(valid_num_dims(base_dilation));
  TF_RET_CHECK(valid_num_dims(window_dilation));

  return absl::WrapUnique(new ConvolutionThunk(
      std::move(info), std::move(input_buffer), input_shape,
      std::move(kernel_buffer), kernel_shape, std::move(output_buffer),
      output_shape, input_batch, input_dims, input_channels, kernel_dims,
      kernel_channels, kernel_filters, output_dims, strides, padding,
      base_dilation, window_dilation, feature_group_count, multi_threaded,
      use_mkl_dnn, use_acl));
}

ConvolutionThunk::ConvolutionThunk(
    Info info, BufferAllocation::Slice input_buffer, const Shape& input_shape,
    BufferAllocation::Slice kernel_buffer, const Shape& kernel_shape,
    BufferAllocation::Slice output_buffer, const Shape& output_shape,
    int64_t input_batch, absl::InlinedVector<int64_t, 2> input_dims,
    int64_t input_channels, absl::InlinedVector<int64_t, 2> kernel_dims,
    int64_t kernel_channels, int64_t kernel_filters,
    absl::InlinedVector<int64_t, 2> output_dims,
    absl::InlinedVector<int64_t, 2> strides,
    absl::InlinedVector<std::pair<int64_t, int64_t>, 2> padding,
    absl::InlinedVector<int64_t, 2> base_dilation,
    absl::InlinedVector<int64_t, 2> window_dilation,
    int64_t feature_group_count, bool multi_threaded, bool use_mkl_dnn,
    bool use_acl)
    : Thunk(Kind::kConvolution, std::move(info)),
      input_buffer_(input_buffer),
      input_shape_(input_shape),
      kernel_buffer_(kernel_buffer),
      kernel_shape_(kernel_shape),
      output_buffer_(output_buffer),
      output_shape_(output_shape),
      input_batch_(input_batch),
      input_dims_(std::move(input_dims)),
      input_channels_(input_channels),
      kernel_dims_(std::move(kernel_dims)),
      kernel_channels_(kernel_channels),
      kernel_filters_(kernel_filters),
      output_dims_(std::move(output_dims)),
      strides_(std::move(strides)),
      padding_(std::move(padding)),
      base_dilation_(std::move(base_dilation)),
      window_dilation_(std::move(window_dilation)),
      feature_group_count_(feature_group_count),
      multi_threaded_(multi_threaded),
      use_mkl_dnn_(use_mkl_dnn),
      use_acl_(use_acl) {}

tsl::AsyncValueRef<Thunk::ExecuteEvent> ConvolutionThunk::Execute(
    const ExecuteParams& params) {
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });

  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase input_data,
      params.buffer_allocations->GetDeviceAddress(input_buffer_));
  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase kernel_data,
      params.buffer_allocations->GetDeviceAddress(kernel_buffer_));
  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase output_data,
      params.buffer_allocations->GetDeviceAddress(output_buffer_));

  VLOG(3) << absl::StreamFormat("ConvolutionThunk::Execute");
  VLOG(3) << absl::StreamFormat("input: %s in slice %s (%p)",
                                input_shape_.ToString(true),
                                input_buffer_.ToString(), input_data.opaque());
  VLOG(3) << absl::StreamFormat(
      "kernel: %s in slice %s (%p)", kernel_shape_.ToString(true),
      kernel_buffer_.ToString(), kernel_data.opaque());
  VLOG(3) << absl::StreamFormat(
      "output: %s in slice %s (%p)", output_shape_.ToString(true),
      output_buffer_.ToString(), output_data.opaque());
  auto primitive_type = input_shape_.element_type();

  // TODO(abanas): Actually, this dispatch should probably be done at emit time,
  // not at runtime. Benchmark it, and create separate class for every
  // convolution if needed.
  if (input_dims_.size() == 2) {
    auto input_rows = input_dims_[0];
    auto input_cols = input_dims_[1];
    auto kernel_rows = kernel_dims_[0];
    auto kernel_cols = kernel_dims_[1];
    auto output_rows = output_dims_[0];
    auto output_cols = output_dims_[1];
    auto row_stride = strides_[0];
    auto col_stride = strides_[1];
    auto padding_top = padding_[0].first;
    auto padding_bottom = padding_[0].second;
    auto padding_left = padding_[1].first;
    auto padding_right = padding_[1].second;
    auto input_row_dilation = base_dilation_[0];
    auto input_col_dilation = base_dilation_[1];
    auto kernel_row_dilation = window_dilation_[0];
    auto kernel_col_dilation = window_dilation_[1];

    if (use_mkl_dnn_) {
      // NOTE: This is the basic support for MKL-DNN. Performance was not
      // benchmarked and is likely not good, the design could be improved
      // (e.g. creating run_options is a hack).
      auto run_options = MakeRunOptions(params.intra_op_threadpool);
      __xla_cpu_runtime_MKLConv2DF32(
          &run_options, static_cast<float*>(output_data.opaque()),
          static_cast<float*>(input_data.opaque()),
          static_cast<float*>(kernel_data.opaque()), input_batch_, input_rows,
          input_cols, input_channels_, kernel_rows, kernel_cols,
          kernel_channels_, kernel_filters_, output_rows, output_cols,
          row_stride, col_stride, padding_top, padding_bottom, padding_left,
          padding_right, input_row_dilation, input_col_dilation,
          kernel_row_dilation, kernel_col_dilation);
    } else if (use_acl_) {
      // NOTE: This is the basic support for ACL. Performance was not
      // benchmarked and is likely not good, the design could be improved
      // (e.g. creating run_options is a hack).
      auto run_options = MakeRunOptions(params.intra_op_threadpool);
      __xla_cpu_runtime_ACLConv2DF32(
          &run_options, static_cast<float*>(output_data.opaque()),
          static_cast<float*>(input_data.opaque()),
          static_cast<float*>(kernel_data.opaque()), input_batch_, input_rows,
          input_cols, input_channels_, kernel_rows, kernel_cols,
          kernel_channels_, kernel_filters_, output_rows, output_cols,
          row_stride, col_stride, padding_top, padding_bottom, padding_left,
          padding_right, input_row_dilation, input_col_dilation,
          kernel_row_dilation, kernel_col_dilation, feature_group_count_);
    } else {
      // Eigen 2D convolution.
      auto dispatch_2d = [&](auto type_tag, const auto& eigen_device) {
        using scalar_type = decltype(type_tag);
        tensorflow::xla::EigenConv2DImpl(
            eigen_device, static_cast<scalar_type*>(output_data.opaque()),
            static_cast<scalar_type*>(input_data.opaque()),
            static_cast<scalar_type*>(kernel_data.opaque()), input_batch_,
            input_rows, input_cols, input_channels_, kernel_rows, kernel_cols,
            kernel_channels_, kernel_filters_, output_rows, output_cols,
            row_stride, col_stride, padding_top, padding_bottom, padding_left,
            padding_right, input_row_dilation, input_col_dilation,
            kernel_row_dilation, kernel_col_dilation, feature_group_count_);
      };

      if (primitive_type == PrimitiveType::F16) {
        if (multi_threaded_) {
          dispatch_2d(Eigen::half(), *params.intra_op_threadpool);
        } else {
          dispatch_2d(Eigen::half(), Eigen::DefaultDevice());
        }
      } else {
        if (multi_threaded_) {
          dispatch_2d(float(), *params.intra_op_threadpool);
        } else {
          dispatch_2d(float(), Eigen::DefaultDevice());
        }
      }
    }
  } else if (input_dims_.size() == 3) {
    // Eigen 3D convolution.
    auto input_x = input_dims_[0];
    auto input_y = input_dims_[1];
    auto input_z = input_dims_[2];
    auto kernel_x = kernel_dims_[0];
    auto kernel_y = kernel_dims_[1];
    auto kernel_z = kernel_dims_[2];
    auto output_x = output_dims_[0];
    auto output_y = output_dims_[1];
    auto output_z = output_dims_[2];
    auto x_stride = strides_[0];
    auto y_stride = strides_[1];
    auto z_stride = strides_[2];
    auto padding_x_before = padding_[0].first;
    auto padding_x_after = padding_[0].second;
    auto padding_y_before = padding_[1].first;
    auto padding_y_after = padding_[1].second;
    auto padding_z_before = padding_[2].first;
    auto padding_z_after = padding_[2].second;
    auto input_x_dilation = base_dilation_[0];
    auto input_y_dilation = base_dilation_[1];
    auto input_z_dilation = base_dilation_[2];
    auto kernel_x_dilation = window_dilation_[0];
    auto kernel_y_dilation = window_dilation_[1];
    auto kernel_z_dilation = window_dilation_[2];

    auto dispatch_3d = [&](auto type_tag, const auto& eigen_device) {
      using scalar_type = decltype(type_tag);
      tensorflow::xla::EigenConv3DImpl(
          eigen_device, static_cast<scalar_type*>(output_data.opaque()),
          static_cast<scalar_type*>(input_data.opaque()),
          static_cast<scalar_type*>(kernel_data.opaque()), input_batch_,
          input_x, input_y, input_z, input_channels_, kernel_x, kernel_y,
          kernel_z, kernel_channels_, kernel_filters_, output_x, output_y,
          output_z, x_stride, y_stride, z_stride, padding_x_before,
          padding_x_after, padding_y_before, padding_y_after, padding_z_before,
          padding_z_after, input_x_dilation, input_y_dilation, input_z_dilation,
          kernel_x_dilation, kernel_y_dilation, kernel_z_dilation,
          feature_group_count_);
    };

    if (primitive_type == PrimitiveType::F16) {
      if (multi_threaded_) {
        dispatch_3d(Eigen::half(), *params.intra_op_threadpool);
      } else {
        dispatch_3d(Eigen::half(), Eigen::DefaultDevice());
      }
    } else {
      if (multi_threaded_) {
        dispatch_3d(float(), *params.intra_op_threadpool);
      } else {
        dispatch_3d(float(), Eigen::DefaultDevice());
      }
    }
  }

  // TODO(abanas): Execute asynchronously in multi-thread mode
  // usingEigen::ThreadPoolDevice.
  return OkExecuteEvent();
}

}  // namespace xla::cpu
