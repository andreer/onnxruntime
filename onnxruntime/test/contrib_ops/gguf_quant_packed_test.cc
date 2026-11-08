// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "contrib_ops/cpu/quantization/gguf_quant_tables.h"
#include "contrib_ops/cpu/quantization/gguf_vec_dot_cpu.h"
#include "core/common/float16.h"
#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "core/session/inference_session.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

using namespace onnxruntime::contrib::gguf_quant;

uint8_t NextPortableByte(std::mt19937& rng) {
  return static_cast<uint8_t>(rng() & 0xFFu);
}

float NextPortableSignedFloat(std::mt19937& rng, float scale) {
  // std::mt19937::operator()() is strictly specified by the C++ standard to produce identical
  // uint32_t sequences across libstdc++, libc++, and MSVC STL.
  const int32_t v = static_cast<int32_t>(rng() & 0xFFFFu) - 32768;
  return static_cast<float>(v) * (scale / 32768.0f);
}

std::vector<uint8_t> MakeSyntheticGgufTensor(
    std::mt19937& rng, int64_t num_rows, int64_t K, int ggml_type) {
  const int64_t rb = GgmlRowBytes(ggml_type, K);
  ORT_ENFORCE(rb > 0);
  int64_t bel = 0, bbytes = 0;
  ORT_ENFORCE(GgmlTypeBlockInfo(ggml_type, bel, bbytes));

  std::vector<uint8_t> raw(static_cast<size_t>(num_rows * rb));
  for (size_t i = 0; i < raw.size(); ++i) {
    raw[i] = NextPortableByte(rng);
  }

  for (int64_t r = 0; r < num_rows; ++r) {
    uint8_t* row_ptr = raw.data() + static_cast<size_t>(r * rb);
    for (int64_t b = 0; b < K / bel; ++b) {
      uint8_t* blk = row_ptr + static_cast<size_t>(b * bbytes);
      if (ggml_type == kGgmlTypeF32) {
        for (int64_t j = 0; j < bel; ++j) {
          float v = static_cast<float>(static_cast<int>(NextPortableByte(rng)) - 128) * 0.002f;
          std::memcpy(blk + j * 4, &v, sizeof(float));
        }
      } else if (ggml_type == kGgmlTypeF16) {
        const uint16_t h = MLFloat16(static_cast<float>(static_cast<int>(NextPortableByte(rng)) - 128) * 0.002f).val;
        std::memcpy(blk, &h, sizeof(uint16_t));
      } else if (ggml_type == kGgmlTypeBF16) {
        const uint16_t bf = BFloat16(static_cast<float>(static_cast<int>(NextPortableByte(rng)) - 128) * 0.002f).val;
        std::memcpy(blk, &bf, sizeof(uint16_t));
      } else if (ggml_type == kGgmlTypeQ6_K) {
        blk[208] = 0x00;
        blk[209] = 0x18;  // fp16 0.001953125
      } else if (ggml_type == kGgmlTypeIQ4_XS || ggml_type == kGgmlTypeIQ4_NL) {
        blk[0] = 0x00;
        blk[1] = 0x10;  // fp16 0.000061035
      } else {
        blk[0] = 0x00;
        blk[1] = 0x20;  // fp16 0.0078125
        if (ggml_type == kGgmlTypeQ5_K) {
          blk[2] = 0x00;
          blk[3] = 0x30;  // fp16 0.125 (= 16 * d) to balance unsigned 5-bit quants (0..31)
        }
      }
    }
  }
  return raw;
}

TEST(GgufQuantPackedTest, VecDotGgufRowAndGateUpMatchDequantizeAcrossAllFormats) {
  std::mt19937 rng(20261107);
  constexpr int64_t K = 2560;
  std::vector<float> x(static_cast<size_t>(K));
  for (int64_t j = 0; j < K; ++j) {
    x[static_cast<size_t>(j)] = NextPortableSignedFloat(rng, 0.25f);
  }

  const int types[] = {
      kGgmlTypeIQ2_XXS,
      kGgmlTypeIQ2_XS,
      kGgmlTypeIQ2_S,
      kGgmlTypeIQ3_XXS,
      kGgmlTypeIQ4_NL,
      kGgmlTypeIQ4_XS,
      kGgmlTypeQ2_0,
      kGgmlTypeQ5_0,
      kGgmlTypeQ4_0,
      kGgmlTypeQ8_0,
      kGgmlTypeQ5_K,
      kGgmlTypeQ6_K,
      kGgmlTypeF32,
      kGgmlTypeF16,
      kGgmlTypeBF16,
  };

  std::vector<float> deq_g(static_cast<size_t>(K));
  std::vector<float> deq_u(static_cast<size_t>(K));

  for (int gt : types) {
    const int64_t rb = GgmlRowBytes(gt, K);
    ASSERT_GT(rb, 0);
    auto w = MakeSyntheticGgufTensor(rng, 8, K, gt);
    const uint8_t* g_row = w.data();
    const uint8_t* u_row = w.data() + static_cast<size_t>(rb);
    const uint8_t* r2_row = w.data() + static_cast<size_t>(2 * rb);
    const uint8_t* r3_row = w.data() + static_cast<size_t>(3 * rb);

    ASSERT_TRUE(DequantizeGgufRowCpu(g_row, gt, K, deq_g.data()));
    ASSERT_TRUE(DequantizeGgufRowCpu(u_row, gt, K, deq_u.data()));

    double ref_g = 0.0;
    double ref_u = 0.0;
    for (int64_t j = 0; j < K; ++j) {
      ref_g += static_cast<double>(deq_g[static_cast<size_t>(j)]) * x[static_cast<size_t>(j)];
      ref_u += static_cast<double>(deq_u[static_cast<size_t>(j)]) * x[static_cast<size_t>(j)];
    }

    const float dot_g = VecDotGgufRowCpu(gt, g_row, x.data(), K);
    const float dot_u = VecDotGgufRowCpu(gt, u_row, x.data(), K);
    const float dot_r2 = VecDotGgufRowCpu(gt, r2_row, x.data(), K);
    const float dot_r3 = VecDotGgufRowCpu(gt, r3_row, x.data(), K);
    const float scalar_g = VecDotGgufRowScalar(gt, g_row, x.data(), K);

    float fused_g = 0.0f;
    float fused_u = 0.0f;
    VecDotGgufGateUpRowCpu(gt, g_row, gt, u_row, x.data(), K, &fused_g, &fused_u);

    float dot4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    VecDotGgufRow4xCpu(gt, g_row, u_row, r2_row, r3_row, x.data(), K, dot4);

    std::vector<GgufQ8Block> x_q8(static_cast<size_t>(K / 32));
    QuantizeRowQ8Cpu(x.data(), K, x_q8.data());
    const float q8_g = VecDotGgufRowQ8Cpu(gt, g_row, x_q8.data(), K);
    const float q8_u = VecDotGgufRowQ8Cpu(gt, u_row, x_q8.data(), K);
    const float q8_r2 = VecDotGgufRowQ8Cpu(gt, r2_row, x_q8.data(), K);
    const float q8_r3 = VecDotGgufRowQ8Cpu(gt, r3_row, x_q8.data(), K);
    const float q8_scalar_g = VecDotGgufRowQ8Scalar(gt, g_row, x_q8.data(), K);

    float q8_gate2[2] = {0.0f, 0.0f};
    float q8_up2[2] = {0.0f, 0.0f};
    VecDotGgufGateUpRow2xQ8Cpu(gt, g_row, r2_row, gt, u_row, r3_row, x_q8.data(), K, q8_gate2, q8_up2);

    float q8_dot4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    VecDotGgufRow4xQ8Cpu(gt, g_row, u_row, r2_row, r3_row, x_q8.data(), K, q8_dot4);

    const float tol = static_cast<float>(std::max(1e-4, std::abs(ref_g) * 1e-5));
    EXPECT_NEAR(dot_g, static_cast<float>(ref_g), tol) << "ggml_type=" << gt;
    EXPECT_NEAR(dot_u, static_cast<float>(ref_u), tol) << "ggml_type=" << gt;
    EXPECT_NEAR(scalar_g, static_cast<float>(ref_g), tol) << "ggml_type=" << gt;
    EXPECT_NEAR(fused_g, dot_g, 1e-5f) << "ggml_type=" << gt;
    EXPECT_NEAR(fused_u, dot_u, 1e-5f) << "ggml_type=" << gt;
    EXPECT_NEAR(dot4[0], dot_g, 1e-5f) << "ggml_type=" << gt;
    EXPECT_NEAR(dot4[1], dot_u, 1e-5f) << "ggml_type=" << gt;
    EXPECT_NEAR(dot4[2], dot_r2, 1e-5f) << "ggml_type=" << gt;
    EXPECT_NEAR(dot4[3], dot_r3, 1e-5f) << "ggml_type=" << gt;

    const float q8_exact_tol = static_cast<float>(std::max(1e-4, std::abs(q8_scalar_g) * 1e-5));
    EXPECT_NEAR(q8_g, q8_scalar_g, q8_exact_tol) << "ggml_type=" << gt;
    EXPECT_NEAR(q8_dot4[0], q8_g, q8_exact_tol) << "ggml_type=" << gt;
    EXPECT_NEAR(q8_dot4[1], q8_u, q8_exact_tol) << "ggml_type=" << gt;
    EXPECT_NEAR(q8_dot4[2], q8_r2, q8_exact_tol) << "ggml_type=" << gt;
    EXPECT_NEAR(q8_dot4[3], q8_r3, q8_exact_tol) << "ggml_type=" << gt;
    EXPECT_NEAR(q8_gate2[0], q8_g, q8_exact_tol) << "ggml_type=" << gt;
    EXPECT_NEAR(q8_gate2[1], q8_r2, q8_exact_tol) << "ggml_type=" << gt;
    EXPECT_NEAR(q8_up2[0], q8_u, q8_exact_tol) << "ggml_type=" << gt;
    EXPECT_NEAR(q8_up2[1], q8_r3, q8_exact_tol) << "ggml_type=" << gt;

    const double fp_vec[8] = {
        dot_g,
        dot_u,
        dot_r2,
        dot_r3,
        VecDotGgufRowCpu(gt, w.data() + static_cast<size_t>(4 * rb), x.data(), K),
        VecDotGgufRowCpu(gt, w.data() + static_cast<size_t>(5 * rb), x.data(), K),
        VecDotGgufRowCpu(gt, w.data() + static_cast<size_t>(6 * rb), x.data(), K),
        VecDotGgufRowCpu(gt, w.data() + static_cast<size_t>(7 * rb), x.data(), K),
    };
    float q8_dot4_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    VecDotGgufRow4xQ8Cpu(
        gt,
        w.data() + static_cast<size_t>(4 * rb),
        w.data() + static_cast<size_t>(5 * rb),
        w.data() + static_cast<size_t>(6 * rb),
        w.data() + static_cast<size_t>(7 * rb),
        x_q8.data(), K, q8_dot4_hi);
    const double q8_vec[8] = {
        q8_dot4[0],
        q8_dot4[1],
        q8_dot4[2],
        q8_dot4[3],
        q8_dot4_hi[0],
        q8_dot4_hi[1],
        q8_dot4_hi[2],
        q8_dot4_hi[3],
    };
    double dot_pq = 0.0, norm_p = 0.0, norm_q = 0.0;
    for (int m = 0; m < 8; ++m) {
      dot_pq += fp_vec[m] * q8_vec[m];
      norm_p += fp_vec[m] * fp_vec[m];
      norm_q += q8_vec[m] * q8_vec[m];
    }
    const double cos_sim = dot_pq / (std::sqrt(norm_p) * std::sqrt(norm_q));
    EXPECT_GT(cos_sim, 0.9999) << "ggml_type=" << gt;
  }
}

TEST(GgufQuantPackedTest, MoEPackedCpuAndCudaAndHybridOffloadParity) {
  std::mt19937 rng(33953);
  constexpr int64_t num_experts = 8;
  constexpr int64_t hidden_size = 256;
  constexpr int64_t inter_size = 64;
  constexpr int64_t top_k = 4;
  constexpr int64_t num_rows = 8;

  struct TypeCombo {
    int gate_type;
    int up_type;
    int down_type;
  };
  const TypeCombo combos[] = {
      {kGgmlTypeIQ2_XXS, kGgmlTypeIQ2_XXS, kGgmlTypeQ2_0},
      {kGgmlTypeIQ2_XS, kGgmlTypeIQ2_XS, kGgmlTypeQ2_0},
      {kGgmlTypeIQ2_S, kGgmlTypeIQ2_S, kGgmlTypeIQ4_NL},
      {kGgmlTypeIQ3_XXS, kGgmlTypeIQ3_XXS, kGgmlTypeQ2_0},
      {kGgmlTypeIQ3_XXS, kGgmlTypeIQ3_XXS, kGgmlTypeIQ4_NL},
      {kGgmlTypeIQ4_XS, kGgmlTypeIQ4_XS, kGgmlTypeQ5_0},
      {kGgmlTypeIQ2_S, kGgmlTypeIQ2_XXS, kGgmlTypeQ2_0},
  };

  std::vector<BFloat16> x_bf16(static_cast<size_t>(num_rows * hidden_size));
  std::vector<BFloat16> r_bf16(static_cast<size_t>(num_rows * num_experts));
  for (auto& v : x_bf16) v = BFloat16(NextPortableSignedFloat(rng, 0.25f));
  for (auto& v : r_bf16) v = BFloat16(NextPortableSignedFloat(rng, 0.25f));

  for (const auto& c : combos) {
    const int64_t gate_rb = GgmlRowBytes(c.gate_type, hidden_size);
    const int64_t up_rb = GgmlRowBytes(c.up_type, hidden_size);
    const int64_t down_rb = GgmlRowBytes(c.down_type, inter_size);
    auto gate_w = MakeSyntheticGgufTensor(rng, num_experts * inter_size, hidden_size, c.gate_type);
    auto up_w = MakeSyntheticGgufTensor(rng, num_experts * inter_size, hidden_size, c.up_type);
    auto down_w = MakeSyntheticGgufTensor(rng, num_experts * hidden_size, inter_size, c.down_type);

    // 1. CPU reference via ExecuteMoEPackedColdRoutesCpu
    std::vector<float> topk_scales(static_cast<size_t>(num_rows * top_k));
    std::vector<int> topk_indices(static_cast<size_t>(num_rows * top_k));
    ComputeMoEPackedSoftmaxTopKCpu<BFloat16>(
        r_bf16.data(), static_cast<int>(num_rows), static_cast<int>(num_experts),
        static_cast<int>(top_k), true, topk_scales.data(), topk_indices.data());

    std::vector<int> all_routes(static_cast<size_t>(num_rows * top_k));
    for (size_t i = 0; i < all_routes.size(); ++i) all_routes[i] = static_cast<int>(i);
    std::vector<BFloat16> ref_fc2(static_cast<size_t>(num_rows * top_k * hidden_size));
    auto serial_for = [](std::ptrdiff_t count, const auto& fn) {
      for (std::ptrdiff_t i = 0; i < count; ++i) fn(i);
    };
    ExecuteMoEPackedColdRoutesCpu<BFloat16>(
        serial_for, x_bf16.data(), gate_w.data(), up_w.data(), down_w.data(),
        nullptr, nullptr, nullptr,
        static_cast<int>(hidden_size), static_cast<int>(inter_size), static_cast<int>(top_k),
        c.gate_type, c.up_type, c.down_type, gate_rb, up_rb, down_rb,
        topk_indices.data(), all_routes.data(), static_cast<int>(num_rows * top_k), ref_fc2.data());

    // Verify expert-grouped M=8 batch execution matches M=1 single-token execution bit-for-bit.
    std::vector<BFloat16> single_tok_fc2(static_cast<size_t>(num_rows * top_k * hidden_size));
    for (int64_t row = 0; row < num_rows; ++row) {
      int tok_routes[top_k];
      for (int64_t r = 0; r < top_k; ++r) {
        tok_routes[r] = static_cast<int>(row * top_k + r);
      }
      ExecuteMoEPackedColdRoutesCpu<BFloat16>(
          serial_for, x_bf16.data(), gate_w.data(), up_w.data(), down_w.data(),
          nullptr, nullptr, nullptr,
          static_cast<int>(hidden_size), static_cast<int>(inter_size), static_cast<int>(top_k),
          c.gate_type, c.up_type, c.down_type, gate_rb, up_rb, down_rb,
          topk_indices.data(), tok_routes, static_cast<int>(top_k),
          single_tok_fc2.data() + static_cast<size_t>(row * top_k * hidden_size));
    }
    for (size_t i = 0; i < ref_fc2.size(); ++i) {
      EXPECT_EQ(ref_fc2[i].val, single_tok_fc2[i].val) << "i=" << i;
    }

    std::vector<BFloat16> y_ref(static_cast<size_t>(num_rows * hidden_size));
    for (int64_t row = 0; row < num_rows; ++row) {
      for (int64_t col = 0; col < hidden_size; ++col) {
        float sum = 0.0f;
        for (int64_t r = 0; r < top_k; ++r) {
          sum += topk_scales[static_cast<size_t>(row * top_k + r)] *
                 ref_fc2[static_cast<size_t>((row * top_k + r) * hidden_size + col)].ToFloat();
        }
        y_ref[static_cast<size_t>(row * hidden_size + col)] = FloatToActTypeCpu<BFloat16>(sum);
      }
    }

    // 2. Verify OpTester on CPUExecutionProvider and CUDAExecutionProvider
    OpTester tester("MoEPacked", 1, kMSDomain);
    tester.AddAttribute<int64_t>("k", top_k);
    tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
    tester.AddAttribute<std::string>("activation_type", "swiglu");
    tester.AddAttribute<int64_t>("gate_type", c.gate_type);
    tester.AddAttribute<int64_t>("up_type", c.up_type);
    tester.AddAttribute<int64_t>("down_type", c.down_type);
    tester.AddInput<BFloat16>("input", {num_rows, hidden_size}, x_bf16);
    tester.AddInput<BFloat16>("router_probs", {num_rows, num_experts}, r_bf16);
    tester.AddInput<uint8_t>("gate_experts_weights", {num_experts, inter_size, gate_rb}, gate_w, true);
    tester.AddInput<uint8_t>("up_experts_weights", {num_experts, inter_size, up_rb}, up_w, true);
    tester.AddInput<uint8_t>("down_experts_weights", {num_experts, hidden_size, down_rb}, down_w, true);
    tester.AddOutput<BFloat16>("output", {num_rows, hidden_size}, y_ref);
    tester.SetOutputAbsErr("output", 8e-2f);
    tester.SetOutputRelErr("output", 8e-2f);

    std::vector<std::unique_ptr<IExecutionProvider>> cpu_ep;
    cpu_ep.push_back(DefaultCpuExecutionProvider());
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cpu_ep);
  }
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
