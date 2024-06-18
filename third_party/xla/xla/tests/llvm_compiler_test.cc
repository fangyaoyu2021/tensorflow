/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/llvm_compiler.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "llvm/IR/Module.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/literal_util.h"
#include "xla/service/backend.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/test_helpers.h"
#include "xla/tests/verified_hlo_module.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace {

class LLVMCompilerTest : public ::testing::Test {
 public:
  void SetUp() override {
    absl::StatusOr<std::unique_ptr<Backend>> backend_or_status =
        Backend::CreateDefaultBackend();
    ASSERT_IS_OK(backend_or_status.status());
    backend_ = std::move(backend_or_status).value();
  }

 protected:
  std::unique_ptr<Backend> backend_;

  static std::string TestName() {
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
  }

  std::unique_ptr<HloModule> CreateNewVerifiedModule() {
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsFromFlags());
    return std::make_unique<VerifiedHloModule>(
        TestName(), config, /*verifier_layout_sensitive=*/false,
        /*allow_mixed_precision_in_hlo_verifier=*/true,
        backend_->compiler()->ShapeSizeBytesFunction());
  }
};

TEST_F(LLVMCompilerTest, HooksTest) {
  int pre_opt_hook_call_count = 0;
  int post_opt_hook_call_count = 0;

  auto pre_opt_hook = [&pre_opt_hook_call_count](const llvm::Module&) {
    ++pre_opt_hook_call_count;
    return absl::OkStatus();
  };
  auto post_opt_hook = [&post_opt_hook_call_count](const llvm::Module&) {
    ++post_opt_hook_call_count;
    return absl::OkStatus();
  };

  // Create HLO module, and run the compiler.
  auto builder = HloComputation::Builder(TestName());
  builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0)));

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(builder.Build());

  LLVMCompiler* compiler =
      tensorflow::down_cast<xla::LLVMCompiler*>(backend_->compiler());
  compiler->SetPreOptimizationHook(pre_opt_hook);
  compiler->SetPostOptimizationHook(post_opt_hook);

  ASSERT_TRUE(compiler
                  ->RunBackend(std::move(hlo_module),
                               backend_->default_stream_executor(),
                               /*device_allocator=*/nullptr)
                  .ok());

  // Test that hooks were called.
  EXPECT_EQ(1, pre_opt_hook_call_count);
  EXPECT_EQ(1, post_opt_hook_call_count);
}

TEST_F(LLVMCompilerTest, MultiModuleCompilation) {
  HloComputation::Builder builder(TestName());
  builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0)));

  std::unique_ptr<HloModule> hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(builder.Build());

  auto module_group = std::make_unique<HloModuleGroup>("test_module_group");
  module_group->push_back(hlo_module->Clone());
  module_group->push_back(std::move(hlo_module));

  std::vector<std::vector<se::StreamExecutor*>> executors;
  executors.push_back({backend_->default_stream_executor()});
  executors.push_back({backend_->default_stream_executor()});

  EXPECT_IS_OK(backend_->compiler()->Compile(std::move(module_group),
                                             std::move(executors),
                                             backend_->memory_allocator()));
}

}  // namespace
}  // namespace xla
