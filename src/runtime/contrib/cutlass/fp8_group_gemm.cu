/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <cuda_fp16.h>
#include <float.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "../../cuda/cuda_common.h"

// clang-format off
#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
// clang-format on

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

using namespace cute;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per group

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

inline size_t aligned(size_t value, size_t alignment = 16) {
  return (value + alignment - 1) / alignment * alignment;
}

template <typename ElementA, typename ElementB, typename ElementC, typename StrideA,
          typename StrideB, typename StrideC>
__global__ void prepare_group_gemm_arguments(
    const ElementA** ptr_A, const ElementB** ptr_B, ElementC** ptr_D,
    typename ProblemShape::UnderlyingProblemShape* problem_sizes, StrideA* stride_A,
    StrideB* stride_B, StrideC* stride_D, ElementA* x, ElementB* weight, ElementC* out,
    int64_t* total_rows, int64_t n, int64_t k, int64_t num_experts) {
  int expert_id = threadIdx.x;
  if (expert_id >= num_experts) return;
  int prev_rows = expert_id == 0 ? 0 : total_rows[expert_id - 1];
  ptr_A[expert_id] = x + prev_rows * k;
  ptr_B[expert_id] = weight + expert_id * k * n;
  ptr_D[expert_id] = out + prev_rows * n;
  problem_sizes[expert_id] = {static_cast<int>(total_rows[expert_id] - prev_rows),
                              static_cast<int>(n), static_cast<int>(k)};
  stride_A[expert_id] = cute::make_stride(k, Int<1>{}, int64_t{0});
  stride_B[expert_id] = cute::make_stride(k, Int<1>{}, int64_t{0});
  stride_D[expert_id] = cute::make_stride(n, Int<1>{}, int64_t{0});
}

template <typename ElementA, typename ElementB, typename ElementC>
void cutlass_fp8_moe_gemm(ElementA* x, ElementB* weight, int64_t* total_rows_before_expert,
                          uint8_t* workspace, int64_t workspace_size, int64_t total_rows, int64_t n,
                          int64_t k, int64_t num_experts, ElementC* out, cudaStream_t stream) {
  // A matrix configuration
  using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
  constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementA>::value;  // Alignment of A matrix in units of elements
                                                    // (up to 16 bytes)

  // B matrix configuration
  using LayoutB = cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
  constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementB>::value;  // Alignment of B matrix in units of elements
                                                    // (up to 16 bytes)

  // C/D matrix configuration
  using LayoutC = cutlass::layout::RowMajor;  // Layout type for C and D matrix operands
  constexpr int AlignmentC =
      128 / cutlass::sizeof_bits<ElementC>::value;  // Alignment of C matrix in units of elements
                                                    // (up to 16 bytes)

  // Core kernel configurations
  using ElementAccumulator = float;  // Element type for internal accumulation
  using ArchTag =
      cutlass::arch::Sm90;  // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag
  using TileShape = Shape<_128, _256, _64>;              // Threadblock-level tile size
  using ClusterShape = Shape<_2, _2, _1>;                // Shape of the threadblocks in a cluster
  using StageCountType =
      cutlass::gemm::collective::StageCountAuto;  // Stage count maximized based on the tile size
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum;  // Kernel to launch
  using EpilogueSchedule = cutlass::epilogue::PtrArrayNoSmemWarpSpecialized;   // Epilogue to launch

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC*, AlignmentC, ElementC, LayoutC*, AlignmentC,
      EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementA, LayoutA*, AlignmentA, ElementB, LayoutB*, AlignmentB,
      ElementAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::UnderlyingStrideA;
  using StrideB = typename Gemm::GemmKernel::UnderlyingStrideB;
  using StrideC = typename Gemm::GemmKernel::UnderlyingStrideC;
  using StrideD = typename Gemm::GemmKernel::UnderlyingStrideD;

  typename Gemm::EpilogueOutputOp::Params epilogue_params{ElementAccumulator(1.0f),
                                                          ElementAccumulator(0.0f)};
  // set up pointers
  std::ptrdiff_t offset = 0;
  const ElementA** ptr_A = reinterpret_cast<const ElementA**>(workspace + offset);
  offset += aligned(sizeof(ElementA*) * num_experts);
  const ElementB** ptr_B = reinterpret_cast<const ElementB**>(workspace + offset);
  offset += aligned(sizeof(ElementB*) * num_experts);
  ElementC** ptr_D = reinterpret_cast<ElementC**>(workspace + offset);
  offset += aligned(sizeof(ElementC*) * num_experts);
  typename ProblemShape::UnderlyingProblemShape* problem_sizes =
      reinterpret_cast<typename ProblemShape::UnderlyingProblemShape*>(workspace + offset);
  offset += sizeof(typename ProblemShape::UnderlyingProblemShape) * num_experts;
  StrideA* stride_A = reinterpret_cast<StrideA*>(workspace + offset);
  offset += sizeof(StrideA) * num_experts;
  StrideB* stride_B = reinterpret_cast<StrideB*>(workspace + offset);
  offset += sizeof(StrideB) * num_experts;
  StrideC* stride_D = reinterpret_cast<StrideC*>(workspace + offset);
  offset += sizeof(StrideC) * num_experts;
  prepare_group_gemm_arguments<<<1, num_experts, 0, stream>>>(
      ptr_A, ptr_B, ptr_D, problem_sizes, stride_A, stride_B, stride_D, x, weight, out,
      total_rows_before_expert, n, k, num_experts);
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {static_cast<int>(num_experts), problem_sizes, /*problem_sizes_host=*/nullptr},
      {ptr_A, stride_A, ptr_B, stride_B},
      {epilogue_params, const_cast<const ElementC**>(ptr_D), stride_D, ptr_D, stride_D},
      hw_info};
  Gemm gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(arguments));
  int64_t remaining_workspace = workspace_size - static_cast<int64_t>(offset);
  CHECK_GE(remaining_workspace, gemm_op.get_workspace_size(arguments));
  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace + offset, stream));
  CUTLASS_CHECK(gemm_op.run());
}

namespace tvm {
namespace runtime {

template <typename ElementA, typename ElementB, typename ElementC>
void tvm_cutlass_fp8_moe_gemm(NDArray x, NDArray weight, NDArray total_rows_before_expert,
                              NDArray workspace, int64_t total_rows, int64_t n, int64_t k,
                              int64_t num_experts, NDArray out) {
  // Workspace is used for storing device-side group gemm arguments and cutlass internal workspace.
  // Recommened size is 4MB.
  auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(func != nullptr);
  cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());
  cutlass_fp8_moe_gemm(static_cast<ElementA*>(x->data), static_cast<ElementB*>(weight->data),
                       static_cast<int64_t*>(total_rows_before_expert->data),
                       static_cast<uint8_t*>(workspace->data), workspace->shape[0], total_rows, n,
                       k, num_experts, static_cast<ElementC*>(out->data), stream);
}

TVM_REGISTER_GLOBAL("cutlass.moe_gemm_e5m2_e5m2_fp16")
    .set_body_typed(
        tvm_cutlass_fp8_moe_gemm<cutlass::float_e5m2_t, cutlass::float_e5m2_t, cutlass::half_t>);

TVM_REGISTER_GLOBAL("cutlass.moe_gemm_e5m2_e4m3_fp16")
    .set_body_typed(
        tvm_cutlass_fp8_moe_gemm<cutlass::float_e5m2_t, cutlass::float_e4m3_t, cutlass::half_t>);

TVM_REGISTER_GLOBAL("cutlass.moe_gemm_e4m3_e4m3_fp16")
    .set_body_typed(
        tvm_cutlass_fp8_moe_gemm<cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::half_t>);

}  // namespace runtime
}  // namespace tvm

#endif  // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED