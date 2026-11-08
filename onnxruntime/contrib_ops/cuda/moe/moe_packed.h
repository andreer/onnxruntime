// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "core/providers/cuda/cuda_kernel.h"
#if !defined(__CUDACC__)
#include <memory>
#include <vector>
#include "core/framework/moe_offload_plan.h"
#include "core/providers/cuda/qmoe_session_controller.h"
#include "contrib_ops/cuda/moe/qmoe_spin_pool.h"
#endif

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

inline constexpr int kMoEPackedMaxChunkRows = 2048;
// Covers 512/640-element K-loop increments and rounded 2048-element finalize tiles.
inline constexpr int64_t kMoEPackedMaxDimension = std::numeric_limits<int>::max() - 2047;

struct alignas(16) MoEPackedTile {
  int expert;
  int start_route;
  int count;
  int pad;
};

struct MoEPackedParams {
  int num_rows{0};
  int num_experts{0};
  int hidden_size{0};
  int inter_size{0};
  int k{0};
  bool normalize_routing_weights{true};
  bool bf16_tc_supported{false};
  int gate_type{0};
  int up_type{0};
  int down_type{0};
  int64_t gate_row_bytes{0};
  int64_t up_row_bytes{0};
  int64_t down_row_bytes{0};
  const int32_t* expert_to_slot{nullptr};
  const void* gate_bias{nullptr};
  const void* up_bias{nullptr};
  const void* down_bias{nullptr};
};

Status GetMoEPackedWorkspaceBytes(const MoEPackedParams& params, size_t elem_bytes, size_t& workspace_bytes);

Status GetMoEPackedBf16KernelSupport(int device_id, bool& supported);

Status LaunchMoEPackedCuda(
    cudaStream_t stream,
    bool is_bf16,
    const void* input,
    const void* router_probs,
    const uint8_t* gate_weights,
    const uint8_t* up_weights,
    const uint8_t* down_weights,
    const MoEPackedParams& params,
    void* workspace,
    void* output);

Status LaunchMoEPackedRoutingCuda(
    cudaStream_t stream,
    bool is_bf16,
    const void* router_probs,
    const MoEPackedParams& params,
    void* workspace,
    float** out_d_topk_scales,
    int** out_d_topk_indices,
    void** out_d_fc2_out,
    void** out_d_cold_fc2,
    int** out_d_cold_route_ids);

Status LaunchMoEPackedHotExpertsCuda(
    cudaStream_t stream,
    bool is_bf16,
    const void* input,
    const uint8_t* gate_weights,
    const uint8_t* up_weights,
    const uint8_t* down_weights,
    const MoEPackedParams& params,
    int resident_slots,
    void* workspace);

Status LaunchMoEPackedScatterColdAndFinalizeCuda(
    cudaStream_t stream,
    bool is_bf16,
    const float* d_topk_scales,
    const void* d_cold_fc2,
    const int* d_cold_route_ids,
    int num_cold_routes,
    const MoEPackedParams& params,
    void* d_fc2_out,
    void* output);

#if !defined(__CUDACC__)
template <typename T>
class MoEPacked final : public CudaKernel, public onnxruntime::moe_offload::IMoeOffloadNode {
 public:
  explicit MoEPacked(const OpKernelInfo& info);
  ~MoEPacked() override;
  Status ComputeInternal(OpKernelContext* context) const override;

  const onnxruntime::moe_offload::MoeExpertSourceDescriptor& GetSourceDescriptor() const noexcept override {
    return offload_source_desc_;
  }
  const onnxruntime::moe_offload::MoeBackendCapabilities& GetBackendCapabilities() const noexcept override {
    return offload_capabilities_;
  }
  Status InitializeAdmittedPlan(const onnxruntime::moe_offload::MoeOffloadSessionPolicy& policy,
                                const onnxruntime::moe_offload::MoeNodeExecutionPlan& plan) override;

 private:
  void SetUpOffload(const OpKernelInfo& info);
  Status EnsureHybridStagingCapacity(int64_t rows, int64_t hidden_size, int64_t inter_size) const;

  int64_t k_{0};
  int64_t normalize_routing_weights_{1};
  std::string activation_type_;
  int64_t gate_type_{-1};
  int64_t up_type_{-1};
  int64_t down_type_{-1};
  bool bf16_tc_supported_{false};

  onnxruntime::moe_offload::MoeOffloadSessionPolicy offload_policy_;
  onnxruntime::moe_offload::MoeExpertSourceDescriptor offload_source_desc_;
  onnxruntime::moe_offload::MoeBackendCapabilities offload_capabilities_;
  onnxruntime::moe_offload::MoeNodeExecutionPlan offload_plan_;
  int32_t offload_layer_index_{-1};
  int32_t gpu_experts_override_{-1};
  bool offload_node_registered_{false};
  bool offload_active_{false};
  int32_t resident_slots_{0};
  std::vector<int32_t> host_expert_to_slot_;

  IAllocatorUniquePtr<int32_t> d_expert_to_slot_;
  IAllocatorUniquePtr<uint8_t> d_resident_gate_w_;
  IAllocatorUniquePtr<uint8_t> d_resident_up_w_;
  IAllocatorUniquePtr<uint8_t> d_resident_down_w_;
  IAllocatorUniquePtr<void> d_gate_bias_;
  IAllocatorUniquePtr<void> d_up_bias_;
  IAllocatorUniquePtr<void> d_down_bias_;

  const uint8_t* cpu_gate_w_{nullptr};
  const uint8_t* cpu_up_w_{nullptr};
  const uint8_t* cpu_down_w_{nullptr};
  const T* cpu_gate_bias_{nullptr};
  const T* cpu_up_bias_{nullptr};
  const T* cpu_down_bias_{nullptr};

  mutable std::mutex staging_mu_;
  mutable int64_t staged_max_rows_{0};
  mutable void* pinned_d2h_{nullptr};
  mutable void* pinned_h2d_{nullptr};
  mutable size_t pinned_d2h_bytes_{0};
  mutable size_t pinned_h2d_bytes_{0};
  mutable std::vector<int> route_to_tok_buf_;
  mutable std::vector<int> unique_toks_buf_;
  mutable std::vector<float> cold_x_f32_buf_;
  mutable std::vector<float> cold_inter_buf_;
  mutable std::vector<uint64_t> cold_x_q8_buf_;
  mutable std::vector<uint64_t> cold_inter_q8_buf_;
  mutable cudaEvent_t d2h_event_{nullptr};
  mutable cudaEvent_t h2d_consumed_event_{nullptr};
  mutable bool h2d_in_flight_{false};
};
#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
