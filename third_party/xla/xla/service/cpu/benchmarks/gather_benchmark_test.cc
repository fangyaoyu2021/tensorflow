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

#include <cstdint>
#include <random>
#include <string_view>
#include <vector>

#include "absl/strings/str_cat.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/shape_util.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/test_benchmark.h"

namespace xla::cpu {

static void BM_GatherS32(benchmark::State& state) {
  int64_t d0 = state.range(0);
  int64_t d1 = state.range(1);

  std::string_view hlo = R"(
    HloModule gather_s32_d$d0_d$d1

    ENTRY e {
      operand = s32[$d0,$d1] parameter(0)
      indices = s32[2] parameter(1)
      ROOT gather = s32[2,$d1] gather(operand, indices),
          offset_dims={1},
          collapsed_slice_dims={0},
          start_index_map={0},
          index_vector_dim=1,
          slice_sizes={1, $d1}
    }
  )";

  std::minstd_rand0 engine;

  auto operand_shape = ShapeUtil::MakeShape(S32, {d0, d1});
  auto indices_shape = ShapeUtil::MakeShape(S32, {2});
  auto operand =
      *LiteralUtil::CreateRandomLiteral<S32>(operand_shape, &engine, 0, 100);
  auto indices = *LiteralUtil::CreateRandomLiteral<S32>(
      indices_shape, &engine, 0, static_cast<int32_t>(d0 - 1));

  std::vector<const Literal*> args = {&operand, &indices};
  CHECK_OK(
      RunHloBenchmark(state, hlo, args,
                      {{"$d0", absl::StrCat(d0)}, {"$d1", absl::StrCat(d1)}}));
}

BENCHMARK(BM_GatherS32)
    ->MeasureProcessCPUTime()
    ->ArgPair(3, 3)
    ->ArgPair(3, 32)
    ->ArgPair(3, 64)
    ->ArgPair(3, 128)
    ->ArgPair(3, 256)
    ->ArgPair(3, 512)
    ->ArgPair(10, 3)
    ->ArgPair(10, 32)
    ->ArgPair(10, 64)
    ->ArgPair(10, 128)
    ->ArgPair(10, 256)
    ->ArgPair(10, 512)
    ->ArgPair(100, 3)
    ->ArgPair(100, 32)
    ->ArgPair(100, 64)
    ->ArgPair(100, 128)
    ->ArgPair(100, 256)
    ->ArgPair(100, 512);

}  // namespace xla::cpu
