// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/moe/moe_packed.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <type_traits>
#include <vector>

#include "contrib_ops/cpu/quantization/gguf_quant_tables.h"
#include "contrib_ops/cpu/quantization/gguf_vec_dot_cpu.h"
#include "core/common/parse_string.h"
#include "core/common/safeint.h"
#include "core/framework/allocator.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {
bool PackedTensorBytesFit(int64_t rows, int64_t columns, size_t element_bytes) {
  size_t row_bytes = 0;
  size_t bytes = 0;
  return rows >= 0 && columns >= 0 &&
         IAllocator::CalcMemSizeForArray(static_cast<size_t>(columns), element_bytes, &row_bytes) &&
         IAllocator::CalcMemSizeForArray(static_cast<size_t>(rows), row_bytes, &bytes) &&
         bytes <= static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max());
}
}  // namespace

#define REGISTER_MOE_PACKED_CUDA(T)                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      MoEPacked,                                                         \
      kMSDomain, 1,                                                      \
      T,                                                                 \
      kCudaExecutionProvider,                                            \
      (*KernelDefBuilder::Create())                                      \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())         \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()), \
      MoEPacked<T>);

REGISTER_MOE_PACKED_CUDA(MLFloat16)
REGISTER_MOE_PACKED_CUDA(BFloat16)

#undef REGISTER_MOE_PACKED_CUDA

namespace {

int32_t ParseGpuExpertsOverride(const OpKernelInfo& info) {
  std::string raw = info.GetConfigOptions().GetConfigOrDefault("ep.cuda.moe_packed_gpu_experts", "");
  if (raw.empty()) {
    if (const char* env = std::getenv("ORT_MOE_PACKED_GPU_EXPERTS"); env != nullptr && env[0] != '\0') {
      raw = env;
    }
  }
  if (raw.empty()) return -1;
  int32_t val = -1;
  if (TryParseStringWithClassicLocale(raw, val) && val >= 0) {
    return val;
  }
  return -1;
}

std::string GetInputDefName(const OpKernelInfo& info, size_t slot) {
  const auto& defs = info.node().InputDefs();
  if (slot < defs.size() && defs[slot] != nullptr && defs[slot]->Exists()) {
    return defs[slot]->Name();
  }
  return {};
}

class GgufWorkerPool final {
 public:
  explicit GgufWorkerPool(int num_threads) {
    const int workers = std::max(0, num_threads - 1);
    workers_.reserve(static_cast<size_t>(workers));
    for (int i = 0; i < workers; ++i) {
      workers_.emplace_back([this]() { WorkerLoop(); });
    }
  }

  ~GgufWorkerPool() {
    stop_.store(true, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lock(sleep_mu_);
      cv_.notify_all();
    }
    for (auto& w : workers_) {
      if (w.joinable()) w.join();
    }
  }

  template <typename Fn>
  void ParallelFor(std::ptrdiff_t count, Fn& fn) {
    if (count <= 0) return;
    if (workers_.empty() || count == 1) {
      for (std::ptrdiff_t i = 0; i < count; ++i) fn(i);
      return;
    }
    std::lock_guard<std::mutex> serial(call_mu_);
    fn_ = [](void* ctx, std::ptrdiff_t idx) { (*static_cast<Fn*>(ctx))(idx); };
    ctx_ = &fn;
    done_.store(0, std::memory_order_relaxed);
    ++gen_;
    ticket_.store(Pack(gen_, static_cast<uint64_t>(count), 0), std::memory_order_seq_cst);
    if (sleeping_.load(std::memory_order_seq_cst) > 0) {
      std::lock_guard<std::mutex> lock(sleep_mu_);
      cv_.notify_all();
    }
    DrainTasks();
    while (done_.load(std::memory_order_acquire) != static_cast<size_t>(count)) {
      CpuPause();
    }
  }

  static GgufWorkerPool& Default() {
    static GgufWorkerPool pool(std::clamp(qmoe::DiscoverPhysicalCoreCount(), 16, 32));
    return pool;
  }

 private:
  static constexpr uint64_t kFieldBits = 20;
  static constexpr uint64_t kFieldMask = (uint64_t{1} << kFieldBits) - 1;

  static uint64_t Pack(uint64_t gen, uint64_t count, uint64_t next) noexcept {
    return (gen << (2 * kFieldBits)) | (count << kFieldBits) | next;
  }

  static bool HasWork(uint64_t ticket) noexcept {
    return (ticket & kFieldMask) < ((ticket >> kFieldBits) & kFieldMask);
  }

  static void CpuPause() noexcept {
#if defined(_MSC_VER) && (defined(_M_AMD64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) && defined(__GNUC__)
    __asm__ __volatile__("yield" ::: "memory");
#endif
  }

  void DrainTasks() {
    size_t local_done = 0;
    uint64_t ticket = ticket_.load(std::memory_order_acquire);
    while (HasWork(ticket)) {
      const uint64_t next = ticket & kFieldMask;
      if (ticket_.compare_exchange_weak(
              ticket, ticket + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
        fn_(ctx_, static_cast<std::ptrdiff_t>(next));
        ++local_done;
        ticket = ticket_.load(std::memory_order_acquire);
      }
    }
    if (local_done > 0) {
      done_.fetch_add(local_done, std::memory_order_release);
    }
  }

  void WorkerLoop() {
    while (!stop_.load(std::memory_order_relaxed)) {
      DrainTasks();
      auto t0 = std::chrono::steady_clock::now();
      for (uint32_t spins = 1; !stop_.load(std::memory_order_relaxed); ++spins) {
        if (HasWork(ticket_.load(std::memory_order_acquire))) break;
        CpuPause();
        if ((spins & 0x3ffu) == 0 &&
            std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(50)) {
          std::unique_lock<std::mutex> lock(sleep_mu_);
          sleeping_.fetch_add(1, std::memory_order_seq_cst);
          cv_.wait(lock, [this]() {
            return stop_.load(std::memory_order_relaxed) ||
                   HasWork(ticket_.load(std::memory_order_seq_cst));
          });
          sleeping_.fetch_sub(1, std::memory_order_seq_cst);
          t0 = std::chrono::steady_clock::now();
        }
      }
    }
  }

  std::vector<std::thread> workers_;
  std::mutex call_mu_;
  std::mutex sleep_mu_;
  std::condition_variable cv_;
  std::atomic<bool> stop_{false};
  uint64_t gen_{0};
  std::atomic<uint64_t> ticket_{0};
  std::atomic<size_t> done_{0};
  std::atomic<int> sleeping_{0};
  void (*fn_)(void*, std::ptrdiff_t) = nullptr;
  void* ctx_ = nullptr;
};

}  // namespace

template <typename T>
MoEPacked<T>::MoEPacked(const OpKernelInfo& info) : CudaKernel(info) {
  ORT_ENFORCE(info.GetAttr<int64_t>("k", &k_).IsOK() && k_ > 0 && k_ <= 16,
              "MoEPacked requires k in 1..16");
  normalize_routing_weights_ = info.GetAttrOrDefault<int64_t>("normalize_routing_weights", 1);
  activation_type_ = info.GetAttrOrDefault<std::string>("activation_type", "swiglu");
  ORT_ENFORCE(activation_type_ == "swiglu", "MoEPacked only supports activation_type='swiglu'");
  ORT_ENFORCE(info.GetAttr<int64_t>("gate_type", &gate_type_).IsOK(), "MoEPacked requires gate_type");
  ORT_ENFORCE(info.GetAttr<int64_t>("up_type", &up_type_).IsOK(), "MoEPacked requires up_type");
  ORT_ENFORCE(info.GetAttr<int64_t>("down_type", &down_type_).IsOK(), "MoEPacked requires down_type");
  ORT_ENFORCE(gguf_quant::IsGgmlTypeSupported(gate_type_), "MoEPacked: unsupported gate_type=", gate_type_);
  ORT_ENFORCE(gguf_quant::IsGgmlTypeSupported(up_type_), "MoEPacked: unsupported up_type=", up_type_);
  ORT_ENFORCE(gguf_quant::IsGgmlTypeSupported(down_type_), "MoEPacked: unsupported down_type=", down_type_);
  if constexpr (std::is_same_v<T, BFloat16>) {
    ORT_ENFORCE(GetDeviceProp().major >= 8, "MoEPacked BF16 requires compute capability 8.0 or higher");
    ORT_THROW_IF_ERROR(GetMoEPackedBf16KernelSupport(GetDeviceId(), bf16_tc_supported_));
  }
  gpu_experts_override_ = ParseGpuExpertsOverride(info);
  SetUpOffload(info);
}

template <typename T>
MoEPacked<T>::~MoEPacked() {
  if (d2h_event_ != nullptr) {
    static_cast<void>(cudaEventDestroy(d2h_event_));
    d2h_event_ = nullptr;
  }
  if (h2d_consumed_event_ != nullptr) {
    static_cast<void>(cudaEventDestroy(h2d_consumed_event_));
    h2d_consumed_event_ = nullptr;
  }
  if (pinned_d2h_ != nullptr) {
    static_cast<void>(cudaFreeHost(pinned_d2h_));
    pinned_d2h_ = nullptr;
  }
  if (pinned_h2d_ != nullptr) {
    static_cast<void>(cudaFreeHost(pinned_h2d_));
    pinned_h2d_ = nullptr;
  }
  if (offload_node_registered_) {
    if (auto controller = GetQMoESessionController()) {
      controller->UnregisterMoeOffloadNode(this);
    }
  }
}

template <typename T>
Status MoEPacked<T>::EnsureHybridStagingCapacity(int64_t rows, int64_t hidden_size, int64_t inter_size) const {
  if (d2h_event_ == nullptr) {
    ORT_RETURN_IF_ERROR(CUDA_CALL(cudaEventCreateWithFlags(&d2h_event_, cudaEventDisableTiming)));
  }
  if (h2d_consumed_event_ == nullptr) {
    ORT_RETURN_IF_ERROR(CUDA_CALL(cudaEventCreateWithFlags(&h2d_consumed_event_, cudaEventDisableTiming)));
  }
  if (rows <= staged_max_rows_) {
    return Status::OK();
  }
  const int64_t alloc_rows = std::max<int64_t>(rows, 16);
  const size_t max_routes = SafeInt<size_t>(alloc_rows) * k_;
  size_t idx_bytes = 0;
  ORT_RETURN_IF_NOT(IAllocator::CalcMemSizeForArrayWithAlignment(max_routes, sizeof(int), 64, &idx_bytes),
                    "MoEPacked staging index byte size overflow");
  const size_t d2h_req = SafeInt<size_t>(idx_bytes) + SafeInt<size_t>(alloc_rows) * hidden_size * sizeof(T);
  const size_t h2d_req = SafeInt<size_t>(idx_bytes) + SafeInt<size_t>(max_routes) * hidden_size * sizeof(T);

  if (d2h_req > pinned_d2h_bytes_) {
    if (pinned_d2h_ != nullptr) {
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaFreeHost(pinned_d2h_)));
      pinned_d2h_ = nullptr;
    }
    ORT_RETURN_IF_ERROR(CUDA_CALL(cudaHostAlloc(&pinned_d2h_, d2h_req, cudaHostAllocDefault)));
    pinned_d2h_bytes_ = d2h_req;
  }
  if (h2d_req > pinned_h2d_bytes_) {
    if (pinned_h2d_ != nullptr) {
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaFreeHost(pinned_h2d_)));
      pinned_h2d_ = nullptr;
    }
    ORT_RETURN_IF_ERROR(CUDA_CALL(cudaHostAlloc(&pinned_h2d_, h2d_req, cudaHostAllocDefault)));
    pinned_h2d_bytes_ = h2d_req;
  }

  route_to_tok_buf_.resize(max_routes);
  unique_toks_buf_.resize(max_routes);
  cold_x_f32_buf_.resize(SafeInt<size_t>(alloc_rows) * hidden_size);
  cold_inter_buf_.resize(SafeInt<size_t>(max_routes) * inter_size);
  const size_t q8_u64_per_block = (sizeof(gguf_quant::GgufQ8Block) + sizeof(uint64_t) - 1) / sizeof(uint64_t);
  cold_x_q8_buf_.resize(SafeInt<size_t>(alloc_rows) * (hidden_size / 32) * q8_u64_per_block + 8);
  cold_inter_q8_buf_.resize(SafeInt<size_t>(max_routes) * (inter_size / 32) * q8_u64_per_block + 8);
  staged_max_rows_ = alloc_rows;
  return Status::OK();
}

template <typename T>
void MoEPacked<T>::SetUpOffload(const OpKernelInfo& info) {
  using onnxruntime::moe_offload::MoeOffloadSessionPolicy;
  std::filesystem::path model_path = GetSessionModelPath();
  ORT_THROW_IF_ERROR(MoeOffloadSessionPolicy::ParseFromConfigOptions(
      info.GetConfigOptions(), model_path, offload_policy_));
  if (offload_policy_.mode == onnxruntime::moe_offload::OffloadMode::kDisabled) {
    return;
  }

  const Tensor* gate_w = nullptr;
  const Tensor* up_w = nullptr;
  const Tensor* down_w = nullptr;
  if (!info.TryGetConstantInput(2, &gate_w) || gate_w == nullptr ||
      !info.TryGetConstantInput(3, &up_w) || up_w == nullptr ||
      !info.TryGetConstantInput(4, &down_w) || down_w == nullptr ||
      gate_w->Location().device.Type() != OrtDevice::CPU ||
      up_w->Location().device.Type() != OrtDevice::CPU ||
      down_w->Location().device.Type() != OrtDevice::CPU ||
      gate_w->Shape().NumDimensions() != 3 ||
      up_w->Shape().NumDimensions() != 3 ||
      down_w->Shape().NumDimensions() != 3) {
    return;
  }

  const int32_t num_e = static_cast<int32_t>(gate_w->Shape()[0]);
  const int32_t inter = static_cast<int32_t>(gate_w->Shape()[1]);
  const int32_t hidden = static_cast<int32_t>(down_w->Shape()[1]);
  if (num_e <= 0 || num_e > 512 || inter <= 0 || hidden <= 0) {
    return;
  }

  cpu_gate_w_ = gate_w->Data<uint8_t>();
  cpu_up_w_ = up_w->Data<uint8_t>();
  cpu_down_w_ = down_w->Data<uint8_t>();
  static_cast<void>(onnxruntime::moe_offload::PromoteFileMappingToSharedReadOnly(cpu_gate_w_, gate_w->SizeInBytes()));
  static_cast<void>(onnxruntime::moe_offload::PromoteFileMappingToSharedReadOnly(cpu_up_w_, up_w->SizeInBytes()));
  static_cast<void>(onnxruntime::moe_offload::PromoteFileMappingToSharedReadOnly(cpu_down_w_, down_w->SizeInBytes()));

  const Tensor* gate_b = nullptr;
  const Tensor* up_b = nullptr;
  const Tensor* down_b = nullptr;
  if (info.TryGetConstantInput(5, &gate_b) && gate_b != nullptr && gate_b->Location().device.Type() == OrtDevice::CPU) {
    cpu_gate_bias_ = gate_b->Data<T>();
  }
  if (info.TryGetConstantInput(6, &up_b) && up_b != nullptr && up_b->Location().device.Type() == OrtDevice::CPU) {
    cpu_up_bias_ = up_b->Data<T>();
  }
  if (info.TryGetConstantInput(7, &down_b) && down_b != nullptr && down_b->Location().device.Type() == OrtDevice::CPU) {
    cpu_down_bias_ = down_b->Data<T>();
  }

  auto controller = GetQMoESessionController();
  if (controller == nullptr) {
    return;
  }
  if (model_path.empty()) {
    model_path = controller->GetModelPath();
    if (!model_path.empty()) {
      ORT_THROW_IF_ERROR(MoeOffloadSessionPolicy::ParseFromConfigOptions(
          info.GetConfigOptions(), model_path, offload_policy_));
    }
  } else if (controller->GetModelPath().empty()) {
    controller->SetModelPath(model_path);
  }

  int64_t attr_layer = -1;
  if (info.GetAttr<int64_t>("layer_index", &attr_layer).IsOK() && attr_layer >= 0) {
    offload_layer_index_ = static_cast<int32_t>(attr_layer);
  } else {
    int32_t parsed_layer = -1;
    if (MoeOffloadSessionPolicy::TryExtractLayerIndexFromName(info.node().Name(), parsed_layer)) {
      offload_layer_index_ = parsed_layer;
    } else {
      const std::string in2_name = GetInputDefName(info, 2);
      if (!in2_name.empty() && MoeOffloadSessionPolicy::TryExtractLayerIndexFromName(in2_name, parsed_layer)) {
        offload_layer_index_ = parsed_layer;
      }
    }
  }

  offload_source_desc_.op_type = "MoEPacked";
  offload_source_desc_.quant_type = "gguf";
  offload_source_desc_.stable_node_id = info.node().Name();
  offload_source_desc_.num_experts = num_e;
  offload_source_desc_.top_k = static_cast<int32_t>(k_);
  offload_source_desc_.hidden_size = hidden;
  offload_source_desc_.inter_size = inter;
  offload_source_desc_.block_size = 32;
  offload_source_desc_.weight_bits = 2;
  offload_source_desc_.weights_prepacked = true;
  offload_source_desc_.placement.device_id = controller->DeviceId();
  offload_source_desc_.shard_spec.global_num_experts = num_e;
  offload_source_desc_.shard_spec.local_num_experts = num_e;
  offload_source_desc_.shard_spec.global_hidden_size = hidden;
  offload_source_desc_.shard_spec.global_inter_size = inter;
  offload_source_desc_.shard_spec.local_hidden_size = hidden;
  offload_source_desc_.shard_spec.local_inter_size = inter;
  offload_source_desc_.shard_spec.quant_block_size = 32;

  auto record_slice = [&](onnxruntime::moe_offload::ExpertBankKind kind, int slot_idx, const Tensor* t) {
    if (!t || num_e <= 0) return;
    onnxruntime::moe_offload::MoeExpertBankSlice slice;
    slice.kind = kind;
    slice.input_index = slot_idx;
    slice.initializer_name = GetInputDefName(info, static_cast<size_t>(slot_idx));
    slice.elem_type = static_cast<ONNXTensorElementDataType>(t->GetElementType());
    slice.full_shape = t->Shape();
    slice.total_bytes = t->SizeInBytes();
    slice.bytes_per_expert = t->SizeInBytes() / static_cast<size_t>(num_e);
    slice.host_data = (t->Location().device.Type() == OrtDevice::CPU) ? t->DataRaw() : nullptr;
    slice.is_file_mapped = onnxruntime::moe_offload::IsReloadableFileMapping(slice.host_data, slice.total_bytes);
    offload_source_desc_.banks[static_cast<size_t>(kind)] = std::move(slice);
  };
  record_slice(onnxruntime::moe_offload::ExpertBankKind::kFc1Weights, 2, gate_w);
  record_slice(onnxruntime::moe_offload::ExpertBankKind::kFc1Scales, 3, up_w);
  record_slice(onnxruntime::moe_offload::ExpertBankKind::kFc2Weights, 4, down_w);

  const size_t bias_bytes =
      (gate_b ? gate_b->SizeInBytes() : 0) + (up_b ? up_b->SizeInBytes() : 0) + (down_b ? down_b->SizeInBytes() : 0);
  const size_t per_expert_bytes =
      (gate_w->SizeInBytes() + up_w->SizeInBytes() + down_w->SizeInBytes() + bias_bytes) / static_cast<size_t>(num_e);

  offload_capabilities_.supports_full_gpu_resident = true;
  offload_capabilities_.supports_hybrid_doorbell_decode = true;
  offload_capabilities_.supports_gpu_streaming_prefill = true;
  offload_capabilities_.supports_file_backed_bounded_host_cache = true;
  offload_capabilities_.supports_cuda_graph_capture = true;
  offload_capabilities_.decode_kernel_skips_negative_slots = true;
  offload_capabilities_.source_bytes_per_expert = per_expert_bytes;
  offload_capabilities_.gpu_slot_bytes = per_expert_bytes;
  offload_capabilities_.gpu_fixed_table_bytes = static_cast<size_t>(num_e) * sizeof(int32_t);
  offload_capabilities_.cpu_slot_bytes = per_expert_bytes;
  offload_capabilities_.min_host_cache_slots = 1;

  controller->RegisterMoeOffloadNode(this, offload_policy_, offload_layer_index_);
  offload_source_desc_.layer_index = offload_layer_index_;
  offload_node_registered_ = true;
}

template <typename T>
Status MoEPacked<T>::InitializeAdmittedPlan(const onnxruntime::moe_offload::MoeOffloadSessionPolicy& policy,
                                            const onnxruntime::moe_offload::MoeNodeExecutionPlan& plan) {
  offload_policy_ = policy;
  offload_plan_ = plan;
  offload_layer_index_ = plan.layer_index;

  const int32_t num_e = plan.num_experts;
  resident_slots_ = std::clamp<int32_t>(plan.resident_slots, 0, num_e);

  host_expert_to_slot_.assign(static_cast<size_t>(num_e), -1);
  if (resident_slots_ == num_e) {
    for (int32_t e = 0; e < num_e; ++e) {
      host_expert_to_slot_[static_cast<size_t>(e)] = e;
    }
  } else {
    for (int32_t s = 0; s < resident_slots_; ++s) {
      const int32_t e = (s < static_cast<int32_t>(plan.initial_resident_experts.size()))
                            ? plan.initial_resident_experts[static_cast<size_t>(s)]
                            : ((s < static_cast<int32_t>(plan.ranked_experts.size()))
                                   ? plan.ranked_experts[static_cast<size_t>(s)]
                                   : s);
      if (e >= 0 && e < num_e) {
        host_expert_to_slot_[static_cast<size_t>(e)] = s;
      }
    }
  }

  AllocatorPtr alloc = Info().GetAllocator(OrtMemTypeDefault);
  ORT_RETURN_IF_NOT(alloc != nullptr, "MoEPacked InitializeAdmittedPlan requires a valid CUDA device allocator.");

  d_expert_to_slot_ = GetScratchBuffer<int32_t>(static_cast<size_t>(num_e), nullptr);
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpy(
      d_expert_to_slot_.get(), host_expert_to_slot_.data(),
      static_cast<size_t>(num_e) * sizeof(int32_t), cudaMemcpyHostToDevice)));

  if (resident_slots_ > 0) {
    const Tensor* gate_w = nullptr;
    const Tensor* up_w = nullptr;
    const Tensor* down_w = nullptr;
    ORT_RETURN_IF_NOT(Info().TryGetConstantInput(2, &gate_w) && gate_w != nullptr &&
                          Info().TryGetConstantInput(3, &up_w) && up_w != nullptr &&
                          Info().TryGetConstantInput(4, &down_w) && down_w != nullptr,
                      "MoEPacked InitializeAdmittedPlan requires constant expert weight tensors.");

    const size_t gate_exp_bytes = gate_w->SizeInBytes() / static_cast<size_t>(num_e);
    const size_t up_exp_bytes = up_w->SizeInBytes() / static_cast<size_t>(num_e);
    const size_t down_exp_bytes = down_w->SizeInBytes() / static_cast<size_t>(num_e);

    d_resident_gate_w_ = GetScratchBuffer<uint8_t>(static_cast<size_t>(resident_slots_) * gate_exp_bytes, nullptr);
    d_resident_up_w_ = GetScratchBuffer<uint8_t>(static_cast<size_t>(resident_slots_) * up_exp_bytes, nullptr);
    d_resident_down_w_ = GetScratchBuffer<uint8_t>(static_cast<size_t>(resident_slots_) * down_exp_bytes, nullptr);

    if (resident_slots_ == num_e) {
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpy(
          d_resident_gate_w_.get(), cpu_gate_w_, gate_w->SizeInBytes(), cudaMemcpyHostToDevice)));
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpy(
          d_resident_up_w_.get(), cpu_up_w_, up_w->SizeInBytes(), cudaMemcpyHostToDevice)));
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpy(
          d_resident_down_w_.get(), cpu_down_w_, down_w->SizeInBytes(), cudaMemcpyHostToDevice)));
    } else {
      for (int32_t e = 0; e < num_e; ++e) {
        const int32_t s = host_expert_to_slot_[static_cast<size_t>(e)];
        if (s < 0) continue;
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpy(
            d_resident_gate_w_.get() + static_cast<size_t>(s) * gate_exp_bytes,
            cpu_gate_w_ + static_cast<size_t>(e) * gate_exp_bytes,
            gate_exp_bytes, cudaMemcpyHostToDevice)));
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpy(
            d_resident_up_w_.get() + static_cast<size_t>(s) * up_exp_bytes,
            cpu_up_w_ + static_cast<size_t>(e) * up_exp_bytes,
            up_exp_bytes, cudaMemcpyHostToDevice)));
        ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpy(
            d_resident_down_w_.get() + static_cast<size_t>(s) * down_exp_bytes,
            cpu_down_w_ + static_cast<size_t>(e) * down_exp_bytes,
            down_exp_bytes, cudaMemcpyHostToDevice)));
      }
    }
  }

  if (resident_slots_ < num_e) {
    ORT_RETURN_IF_ERROR(EnsureHybridStagingCapacity(
        16, offload_source_desc_.hidden_size, offload_source_desc_.inter_size));
  }

  offload_active_ = true;
  return Status::OK();
}

template <typename T>
Status MoEPacked<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router_probs = context->Input<Tensor>(1);
  const Tensor* gate_w = context->Input<Tensor>(2);
  const Tensor* up_w = context->Input<Tensor>(3);
  const Tensor* down_w = context->Input<Tensor>(4);

  const auto& in_shape = input->Shape();
  const auto& rp_shape = router_probs->Shape();
  const auto& gw_shape = gate_w->Shape();
  const auto& uw_shape = up_w->Shape();
  const auto& dw_shape = down_w->Shape();

  ORT_RETURN_IF_NOT(in_shape.NumDimensions() >= 2, "MoEPacked input must have rank >= 2");
  ORT_RETURN_IF_NOT(rp_shape.NumDimensions() == 2, "MoEPacked router_probs must be 2D [num_rows, num_experts]");
  ORT_RETURN_IF_NOT(gw_shape.NumDimensions() == 3 && uw_shape.NumDimensions() == 3 && dw_shape.NumDimensions() == 3,
                    "MoEPacked expert weights must be 3D [num_experts, out_dim, row_bytes]");

  const size_t last_axis = in_shape.NumDimensions() - 1;
  const int64_t hidden_size = in_shape[last_axis];
  const int64_t num_rows = in_shape.SizeToDimension(last_axis);
  const int64_t num_experts = rp_shape[1];
  const int64_t inter_size = gw_shape[1];

  ORT_RETURN_IF_NOT(hidden_size > 0 && hidden_size <= kMoEPackedMaxDimension &&
                        inter_size > 0 && inter_size <= kMoEPackedMaxDimension,
                    "MoEPacked requires hidden_size and inter_size in 1..", kMoEPackedMaxDimension);
  ORT_RETURN_IF_NOT(num_rows >= 0, "MoEPacked requires a nonnegative row count");
  ORT_RETURN_IF_NOT(rp_shape[0] == num_rows, "MoEPacked router_probs row count mismatch");
  ORT_RETURN_IF_NOT(num_experts > 0 && num_experts <= 512, "MoEPacked supports up to 512 experts");
  ORT_RETURN_IF_NOT(k_ <= num_experts, "MoEPacked requires k <= num_experts");
  ORT_RETURN_IF_NOT(gw_shape[0] == num_experts && uw_shape[0] == num_experts && dw_shape[0] == num_experts,
                    "MoEPacked expert count mismatch");
  ORT_RETURN_IF_NOT(uw_shape[1] == inter_size && dw_shape[1] == hidden_size,
                    "MoEPacked expert inner/outer dimension mismatch");
  ORT_RETURN_IF_NOT((hidden_size % 32) == 0 && (inter_size % 32) == 0,
                    "MoEPacked requires hidden_size and inter_size to be multiples of 32");

  const int64_t exp_gate_rb = gguf_quant::GgmlRowBytes(static_cast<int>(gate_type_), hidden_size);
  const int64_t exp_up_rb = gguf_quant::GgmlRowBytes(static_cast<int>(up_type_), hidden_size);
  const int64_t exp_down_rb = gguf_quant::GgmlRowBytes(static_cast<int>(down_type_), inter_size);
  ORT_RETURN_IF_NOT(exp_gate_rb > 0 && exp_up_rb > 0 && exp_down_rb > 0 &&
                        gw_shape[2] == exp_gate_rb && uw_shape[2] == exp_up_rb && dw_shape[2] == exp_down_rb,
                    "MoEPacked expert weight row_bytes mismatch");
  constexpr bool is_bf16 = std::is_same_v<T, BFloat16>;
  ORT_RETURN_IF_NOT(PackedTensorBytesFit(num_rows, hidden_size, sizeof(T)) &&
                        PackedTensorBytesFit(num_rows, num_experts, sizeof(T)) &&
                        PackedTensorBytesFit(num_experts * inter_size, exp_gate_rb, 1) &&
                        PackedTensorBytesFit(num_experts * inter_size, exp_up_rb, 1) &&
                        PackedTensorBytesFit(num_experts * hidden_size, exp_down_rb, 1),
                    "MoEPacked tensor byte size exceeds the supported address range");

  // Process in chunks of at most 2048 rows so workspace stays compact during large prefills.
  const int64_t max_chunk = std::min<int64_t>(num_rows, kMoEPackedMaxChunkRows);
  const int64_t max_routes = max_chunk * k_;
  const int64_t max_tiles = max_routes <= 128 ? max_routes : (max_routes + 7) / 8 + num_experts;
  ORT_RETURN_IF_NOT((hidden_size + 7) / 8 <= GetDeviceProp().maxGridSize[0] &&
                        (inter_size + 7) / 8 <= GetDeviceProp().maxGridSize[0] &&
                        max_tiles <= GetDeviceProp().maxGridSize[1] &&
                        max_chunk <= GetDeviceProp().maxGridSize[1],
                    "MoEPacked dimensions exceed the CUDA grid limits");

  MoEPackedParams max_params{};
  max_params.num_rows = static_cast<int>(max_chunk);
  max_params.num_experts = static_cast<int>(num_experts);
  max_params.hidden_size = static_cast<int>(hidden_size);
  max_params.inter_size = static_cast<int>(inter_size);
  max_params.k = static_cast<int>(k_);
  max_params.normalize_routing_weights = (normalize_routing_weights_ != 0);
  max_params.bf16_tc_supported = bf16_tc_supported_;
  max_params.gate_type = static_cast<int>(gate_type_);
  max_params.up_type = static_cast<int>(up_type_);
  max_params.down_type = static_cast<int>(down_type_);
  max_params.gate_row_bytes = exp_gate_rb;
  max_params.up_row_bytes = exp_up_rb;
  max_params.down_row_bytes = exp_down_rb;

  size_t ws_bytes = 0;
  ORT_RETURN_IF_ERROR(GetMoEPackedWorkspaceBytes(max_params, sizeof(T), ws_bytes));
  Tensor* output = context->Output(0, in_shape);
  if (num_rows == 0) return Status::OK();
  auto workspace = GetScratchBuffer<uint8_t>(ws_bytes, GetComputeStream(context));

  const uint8_t* in_bytes = static_cast<const uint8_t*>(input->DataRaw());
  const uint8_t* rp_bytes = static_cast<const uint8_t*>(router_probs->DataRaw());
  uint8_t* out_bytes = static_cast<uint8_t*>(output->MutableDataRaw());
  cudaStream_t stream = Stream(context);

  const bool weights_on_cpu = (gate_w->Location().device.Type() == OrtDevice::CPU);
  if (!offload_active_ && !weights_on_cpu) {
    for (int64_t r0 = 0; r0 < num_rows;) {
      const int64_t chunk_rows = std::min<int64_t>(kMoEPackedMaxChunkRows, num_rows - r0);
      MoEPackedParams chunk_params = max_params;
      chunk_params.num_rows = static_cast<int>(chunk_rows);
      const size_t input_offset = SafeInt<size_t>(r0) * hidden_size * sizeof(T);
      const size_t router_offset = SafeInt<size_t>(r0) * num_experts * sizeof(T);
      ORT_RETURN_IF_ERROR(LaunchMoEPackedCuda(
          stream,
          is_bf16,
          in_bytes + input_offset,
          rp_bytes + router_offset,
          gate_w->Data<uint8_t>(),
          up_w->Data<uint8_t>(),
          down_w->Data<uint8_t>(),
          chunk_params,
          workspace.get(),
          out_bytes + input_offset));
      r0 += chunk_rows;
    }
    return Status::OK();
  }

  // Full GPU resident via admitted offload plan:
  if (offload_active_ && resident_slots_ == static_cast<int32_t>(num_experts)) {
    for (int64_t r0 = 0; r0 < num_rows;) {
      const int64_t chunk_rows = std::min<int64_t>(kMoEPackedMaxChunkRows, num_rows - r0);
      MoEPackedParams chunk_params = max_params;
      chunk_params.num_rows = static_cast<int>(chunk_rows);
      chunk_params.expert_to_slot = d_expert_to_slot_.get();
      const size_t input_offset = SafeInt<size_t>(r0) * hidden_size * sizeof(T);
      const size_t router_offset = SafeInt<size_t>(r0) * num_experts * sizeof(T);
      ORT_RETURN_IF_ERROR(LaunchMoEPackedCuda(
          stream,
          is_bf16,
          in_bytes + input_offset,
          rp_bytes + router_offset,
          d_resident_gate_w_.get(),
          d_resident_up_w_.get(),
          d_resident_down_w_.get(),
          chunk_params,
          workspace.get(),
          out_bytes + input_offset));
      r0 += chunk_rows;
    }
    return Status::OK();
  }

  // Hybrid GPU-hot / CPU-cold or 100% CPU-cold execution:
  const uint8_t* host_gw = cpu_gate_w_ != nullptr ? cpu_gate_w_ : gate_w->Data<uint8_t>();
  const uint8_t* host_uw = cpu_up_w_ != nullptr ? cpu_up_w_ : up_w->Data<uint8_t>();
  const uint8_t* host_dw = cpu_down_w_ != nullptr ? cpu_down_w_ : down_w->Data<uint8_t>();
  const int eff_resident_slots = offload_active_ ? static_cast<int>(resident_slots_) : 0;

  std::lock_guard<std::mutex> staging_lock(staging_mu_);
  ORT_RETURN_IF_ERROR(EnsureHybridStagingCapacity(max_chunk, hidden_size, inter_size));

  const size_t max_routes_cap = SafeInt<size_t>(staged_max_rows_) * k_;
  size_t idx_offset = 0;
  ORT_RETURN_IF_NOT(IAllocator::CalcMemSizeForArrayWithAlignment(max_routes_cap, sizeof(int), 64, &idx_offset),
                    "MoEPacked staging index byte size overflow");
  int* h_topk_indices = static_cast<int*>(pinned_d2h_);
  T* h_input = reinterpret_cast<T*>(static_cast<uint8_t*>(pinned_d2h_) + idx_offset);
  int* h_cold_route_ids = static_cast<int*>(pinned_h2d_);
  T* h_cold_fc2 = reinterpret_cast<T*>(static_cast<uint8_t*>(pinned_h2d_) + idx_offset);

  for (int64_t r0 = 0; r0 < num_rows;) {
    const int64_t chunk_rows = std::min<int64_t>(kMoEPackedMaxChunkRows, num_rows - r0);
    const int chunk_routes = static_cast<int>(chunk_rows * k_);
    MoEPackedParams chunk_params = max_params;
    chunk_params.num_rows = static_cast<int>(chunk_rows);
    chunk_params.expert_to_slot = offload_active_ ? d_expert_to_slot_.get() : nullptr;

    const size_t input_offset = SafeInt<size_t>(r0) * hidden_size * sizeof(T);
    const size_t router_offset = SafeInt<size_t>(r0) * num_experts * sizeof(T);
    const void* chunk_in = in_bytes + input_offset;
    const void* chunk_rp = rp_bytes + router_offset;
    void* chunk_out = out_bytes + input_offset;

    if (h2d_in_flight_) {
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaEventSynchronize(h2d_consumed_event_)));
      h2d_in_flight_ = false;
    }

    float* d_topk_scales = nullptr;
    int* d_topk_indices = nullptr;
    void* d_fc2_out = nullptr;
    void* d_cold_fc2 = nullptr;
    int* d_cold_route_ids = nullptr;

    ORT_RETURN_IF_ERROR(LaunchMoEPackedRoutingCuda(
        stream,
        is_bf16,
        chunk_rp,
        chunk_params,
        workspace.get(),
        &d_topk_scales,
        &d_topk_indices,
        &d_fc2_out,
        &d_cold_fc2,
        &d_cold_route_ids));

    ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
        h_topk_indices, d_topk_indices,
        static_cast<size_t>(chunk_routes) * sizeof(int),
        cudaMemcpyDeviceToHost, stream)));
    ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
        h_input, chunk_in,
        static_cast<size_t>(chunk_rows * hidden_size) * sizeof(T),
        cudaMemcpyDeviceToHost, stream)));
    ORT_RETURN_IF_ERROR(CUDA_CALL(cudaEventRecord(d2h_event_, stream)));

    // Launch GPU hot experts on stream BEFORE waiting on d2h_event_ so GPU Tensor Cores
    // execute hot experts concurrently while the CPU processes cold experts.
    if (eff_resident_slots > 0) {
      ORT_RETURN_IF_ERROR(LaunchMoEPackedHotExpertsCuda(
          stream,
          is_bf16,
          chunk_in,
          d_resident_gate_w_.get(),
          d_resident_up_w_.get(),
          d_resident_down_w_.get(),
          chunk_params,
          eff_resident_slots,
          workspace.get()));
    }

    ORT_RETURN_IF_ERROR(CUDA_CALL(cudaEventSynchronize(d2h_event_)));

    int num_cold = 0;
    for (int r = 0; r < chunk_routes; ++r) {
      const int e = h_topk_indices[r];
      const bool is_cold = !offload_active_ ||
                           (e >= 0 && e < static_cast<int>(host_expert_to_slot_.size()) &&
                            host_expert_to_slot_[static_cast<size_t>(e)] < 0);
      if (is_cold) {
        h_cold_route_ids[num_cold++] = r;
      }
    }

    if (num_cold > 0) {
      auto parallel_for = [](std::ptrdiff_t count, auto& fn) {
        GgufWorkerPool::Default().ParallelFor(count, fn);
      };
      gguf_quant::ExecuteMoEPackedColdRoutesCpuWithScratch<T>(
          parallel_for,
          h_input,
          host_gw,
          host_uw,
          host_dw,
          cpu_gate_bias_,
          cpu_up_bias_,
          cpu_down_bias_,
          static_cast<int>(hidden_size),
          static_cast<int>(inter_size),
          static_cast<int>(k_),
          static_cast<int>(gate_type_),
          static_cast<int>(up_type_),
          static_cast<int>(down_type_),
          exp_gate_rb,
          exp_up_rb,
          exp_down_rb,
          h_topk_indices,
          h_cold_route_ids,
          num_cold,
          cold_x_f32_buf_.data(),
          cold_inter_buf_.data(),
          route_to_tok_buf_.data(),
          unique_toks_buf_.data(),
          h_cold_fc2,
          reinterpret_cast<gguf_quant::GgufQ8Block*>(
              (reinterpret_cast<uintptr_t>(cold_x_q8_buf_.data()) + 15u) & ~uintptr_t{15u}),
          reinterpret_cast<gguf_quant::GgufQ8Block*>(
              (reinterpret_cast<uintptr_t>(cold_inter_q8_buf_.data()) + 15u) & ~uintptr_t{15u}));

      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
          d_cold_fc2, h_cold_fc2,
          static_cast<size_t>(num_cold * hidden_size) * sizeof(T),
          cudaMemcpyHostToDevice, stream)));
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(
          d_cold_route_ids, h_cold_route_ids,
          static_cast<size_t>(num_cold) * sizeof(int),
          cudaMemcpyHostToDevice, stream)));
      ORT_RETURN_IF_ERROR(CUDA_CALL(cudaEventRecord(h2d_consumed_event_, stream)));
      h2d_in_flight_ = true;
    }

    ORT_RETURN_IF_ERROR(LaunchMoEPackedScatterColdAndFinalizeCuda(
        stream,
        is_bf16,
        d_topk_scales,
        d_cold_fc2,
        d_cold_route_ids,
        num_cold,
        chunk_params,
        d_fc2_out,
        chunk_out));
    r0 += chunk_rows;
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
