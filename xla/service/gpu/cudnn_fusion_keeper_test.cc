#include "xla/tests/hlo_test_base.h"
#include <gtest/gtest.h>

#include "xla/service/gpu/cudnn_fusion_keeper.h"

namespace xla {
namespace gpu {
namespace {

using FusionKeeperTest = HloTestBase;

TEST_F(FusionKeeperTest, FusionKeeperWorks) {
  RunAndFilecheckHloRewrite(R"(
fusion_computation1.4 {
  Arg_0.5 = s32[] parameter(0)
  Arg_1.6 = s32[] parameter(1)
  Arg_2.7 = s32[] parameter(2)
  multiply.8 = s32[] multiply(Arg_1.6, Arg_2.7)
  ROOT subtract.9 = s32[] subtract(Arg_0.5, multiply.8)
}

ENTRY main.11 {
  Arg_0.1 = s32[] parameter(0)
  Arg_1.2 = s32[] parameter(1)
  Arg_2.3 = s32[] parameter(2)
  ROOT custom-call.10 = s32[] custom-call(Arg_0.1, Arg_1.2, Arg_2.3),
    custom_call_target="__cudnn$fusion", called_computations={fusion_computation1.4}
}
                          )",
                            CuDnnFusionKeeper(), R"(
; CHECK: kCustom
                          )");
}

} // namespace
} // namespace gpu
} // namespace xla
