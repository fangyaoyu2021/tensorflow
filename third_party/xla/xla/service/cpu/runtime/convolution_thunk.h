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

#ifndef XLA_SERVICE_CPU_RUNTIME_CONVOLUTION_THUNK_H_
#define XLA_SERVICE_CPU_RUNTIME_CONVOLUTION_THUNK_H_

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

// Performs 1D, 2D or 3D convolution.
class ConvolutionThunk final : public Thunk {
 public:
  static absl::StatusOr<std::unique_ptr<ConvolutionThunk>> Create(
      Info info, const DebugOptions& debug_options,
      BufferAllocation::Slice input_buffer, const Shape& input_shape,
      BufferAllocation::Slice kernel_buffer, const Shape& kernel_shape,
      BufferAllocation::Slice output_buffer, const Shape& output_shape,
      const ConvolutionDimensionNumbers& dnums, const Window& window,
      int64_t feature_group_count);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  Thunk::BufferUses buffer_uses() const final {
    return {{input_buffer_, BufferUse::kRead},
            {kernel_buffer_, BufferUse::kRead},
            {output_buffer_, BufferUse::kWrite}};
  }

 private:
  ConvolutionThunk(
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
      bool use_acl);

  BufferAllocation::Slice input_buffer_;
  Shape input_shape_;

  BufferAllocation::Slice kernel_buffer_;
  Shape kernel_shape_;

  BufferAllocation::Slice output_buffer_;
  Shape output_shape_;

  int64_t input_batch_;
  absl::InlinedVector<int64_t, 2> input_dims_;
  int64_t input_channels_;
  absl::InlinedVector<int64_t, 2> kernel_dims_;
  int64_t kernel_channels_;
  int64_t kernel_filters_;
  absl::InlinedVector<int64_t, 2> output_dims_;
  absl::InlinedVector<int64_t, 2> strides_;
  absl::InlinedVector<std::pair<int64_t, int64_t>, 2> padding_;
  absl::InlinedVector<int64_t, 2> base_dilation_;
  absl::InlinedVector<int64_t, 2> window_dilation_;
  int64_t feature_group_count_;
  bool multi_threaded_;
  bool use_mkl_dnn_;
  bool use_acl_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_RUNTIME_CONVOLUTION_THUNK_H_
