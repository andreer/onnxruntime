// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlas.h"

#if defined(MLAS_SBGEMM_AVAILABLE)

#include "bench_util.h"
#include "core/util/thread_utils.h"

#include <stdexcept>

static const std::vector<std::string> sbgemm_bench_arg_names = {"M", "N", "K", "Threads"};

// fp32 A and B, computed in bf16 (as MatMul does with mlas.enable_gemm_fastmath_arm64_bfloat16).
// pack_b: B is packed ahead of time, as for a constant initializer; otherwise B is converted on every call.
void SBGEMM(benchmark::State& state, bool pack_b) {
  if (!MlasBf16AccelerationSupported()) {
    state.SkipWithMessage("no bf16 acceleration on this CPU");
    return;
  }
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));
  const int threads = static_cast<int>(state.range(3));
  if (M == 0 || N == 0 || K == 0 || threads <= 0) throw std::invalid_argument("M, N, K and Threads must be > 0");

  auto A = RandomVectorUniform(M * K, -1.0f, 1.0f);
  auto B = RandomVectorUniform(N * K, -1.0f, 1.0f);
  std::vector<float> C(M * N);

  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp;
  if (threads > 1) {
    OrtThreadPoolParams tpo;
    tpo.thread_pool_size = threads;
    tpo.auto_set_affinity = true;
    tp = onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(), tpo,
                                                    onnxruntime::concurrency::ThreadPoolType::INTRA_OP);
  }

  std::vector<uint8_t> packed_b;
  MLAS_SBGEMM_DATA_PARAMS params;
  params.A = A.data();
  params.lda = K;
  params.C = C.data();
  params.ldc = N;
  params.AIsfp32 = true;
  if (pack_b) {
    const size_t size = MlasSBGemmPackBSize(CblasNoTrans, CblasNoTrans, true, N, K, nullptr);
    if (size == 0) {
      state.SkipWithMessage("B packing not supported");
      return;
    }
    packed_b.resize(size);
    MlasSBGemmConvertPackB(CblasNoTrans, CblasNoTrans, true, N, K, B.data(), N, packed_b.data(), nullptr);
    params.B = packed_b.data();
    params.ldb = 0;
    params.BIsfp32 = false;
    params.BIsPacked = true;
  } else {
    params.B = B.data();
    params.ldb = N;
    params.BIsfp32 = true;
  }

  MlasSBGemmBatch(CblasNoTrans, CblasNoTrans, M, N, K, 1, &params, tp.get(), nullptr);
  for (auto _ : state) {
    MlasSBGemmBatch(CblasNoTrans, CblasNoTrans, M, N, K, 1, &params, tp.get(), nullptr);
  }
  state.counters["GFLOPS"] = benchmark::Counter(2.0 * M * N * K, benchmark::Counter::kIsIterationInvariantRate,
                                                benchmark::Counter::kIs1000);
}

static void SBGemmSizes(benchmark::Benchmark* b) {
  b->ArgNames(sbgemm_bench_arg_names);
  // Square-ish shapes.
  b->ArgsProduct({{63, 255, 1023}, {63, 255, 1023}, {63, 255, 1023}, {1, 4, 8}});
  // Small BERT-style encoder (hidden 384, intermediate 1536) at several token counts.
  b->ArgsProduct({{32, 128, 512}, {384, 1536}, {384}, {1, 4, 8}});
  b->ArgsProduct({{32, 128, 512}, {384}, {1536}, {1, 4, 8}});
  // Few rows (a single short sequence, pooled outputs) and narrow N.
  b->ArgsProduct({{1, 2, 3, 4}, {384, 1536}, {384}, {1, 4, 8}});
  b->ArgsProduct({{63, 255, 1023}, {8, 12, 13, 16}, {384}, {1, 4, 8}});
  // Attention probabilities x V: M = K = sequence length, N = head dimension.
  for (int64_t L : {128, 512}) {
    for (int64_t N : {32, 64}) {
      for (int64_t T : {1, 4, 8}) {
        b->Args({L, N, L, T});
      }
    }
  }
}

BENCHMARK_CAPTURE(SBGEMM, PACKB, true)->Apply(SBGemmSizes)->UseRealTime();
BENCHMARK_CAPTURE(SBGEMM, NOPACKB, false)->Apply(SBGemmSizes)->UseRealTime();

#endif  // MLAS_SBGEMM_AVAILABLE
