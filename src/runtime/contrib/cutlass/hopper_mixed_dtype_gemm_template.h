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

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

// Unfortunately the below header requires
// some cute APIs to be in the global namespace
using namespace cute;

#include "cutlass/epilogue/collective/collective_builder.hpp"

using MmaType = cutlass::half_t;
using QuantType = cutlass::float_e4m3_t;
constexpr int TileShapeK = 128 * 8 / sizeof_bits<MmaType>::value;

// A matrix configuration
using ElementA = MmaType;                   // Element type for A matrix operand
using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
constexpr int AlignmentA =
    128 / cutlass::sizeof_bits<ElementA>::value;  // Memory access granularity/alignment of A matrix
                                                  // in units of elements (up to 16 bytes)

// B matrix configuration
using ElementB = QuantType;                    // Element type for B matrix operand
using LayoutB = cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
constexpr int AlignmentB =
    128 / cutlass::sizeof_bits<ElementB>::value;  // Memory access granularity/alignment of B matrix
                                                  // in units of elements (up to 16 bytes)

// This example manually swaps and transposes, so keep transpose of input layouts
using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

using ElementZero = cutlass::half_t;
using ElementScale = cutlass::half_t;
using LayoutScale = cutlass::layout::RowMajor;

// C/D matrix configuration
using ElementC = cutlass::half_t;           // Element type for C and D matrix operands
using LayoutC = cutlass::layout::RowMajor;  // Layout type for C and D matrix operands
constexpr int AlignmentC =
    128 / cutlass::sizeof_bits<ElementC>::value;  // Memory access granularity/alignment of C matrix
                                                  // in units of elements (up to 16 bytes)

// D matrix configuration
using ElementD = ElementC;
using LayoutD = LayoutC;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

// Core kernel configurations
// TODO(csullivan): check if we can use half here for element acc
using ElementAccumulator = float;  // Element type for internal accumulation
using ElementCompute = float;      // Element type for epilogue computation
using ArchTag =
    cutlass::arch::Sm90;  // Tag indicating the minimum SM that supports the intended feature
using OperatorClass = cutlass::arch::OpClassTensorOp;        // Operator class tag
using TileShape = Shape<_128, _256, cute::Int<TileShapeK>>;  // Threadblock-level tile size
using ClusterShape = Shape<_2, _1, _1>;  // Shape of the threadblocks in a cluster
using KernelSchedule =
    cutlass::gemm::KernelTmaWarpSpecializedCooperativeMixedInput;  // Kernel to launch based on the
                                                                   // default setting in the
                                                                   // Collective Builder
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape, EpilogueTileType,
    ElementAccumulator, ElementAccumulator,
    // Transpose layout of D here since we use explicit swap + transpose
    // the void type for C tells the builder to allocate 0 smem for the C matrix.
    // We can enable this if beta == 0 by changing ElementC to void below.
    ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type, AlignmentC, ElementD,
    typename cutlass::layout::LayoutTranspose<LayoutD>::type, AlignmentD,
    EpilogueSchedule  // This is the only epi supporting the required swap + transpose.
    >::CollectiveOp;

// ============================================================ MIXED INPUT NO SCALES
// ============================================================================ The collective will
// infer that the narrow type should be upcasted to the wide type. We swap A and B operands to the
// builder here
using CollectiveMainloopConvertOnly = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementB, LayoutB_Transpose, AlignmentB, ElementA, LayoutA_Transpose,
    AlignmentA, ElementAccumulator, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule>::CollectiveOp;

using GemmKernelConvertOnly =
    cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,  // Indicates ProblemShape
                                         CollectiveMainloopConvertOnly, CollectiveEpilogue>;

using GemmConvertOnly = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelConvertOnly>;

// =========================================================== MIXED INPUT WITH SCALES
// =========================================================================== The Scale information
// must get paired with the operand that will be scaled. In this example, B is scaled so we make a
// tuple of B's information and the scale information.
using CollectiveMainloopScaleOnly = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, cute::tuple<ElementB, ElementScale>, LayoutB_Transpose, AlignmentB,
    ElementA, LayoutA_Transpose, AlignmentA, ElementAccumulator, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule>::CollectiveOp;

using GemmKernelScaleOnly =
    cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,  // Indicates ProblemShape
                                         CollectiveMainloopScaleOnly, CollectiveEpilogue>;

using GemmScaleOnly = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelScaleOnly>;

// =========================================================== MIXED INPUT WITH SCALES AND ZEROS
// ================================================================== We specify scale + zero
// elements to indicate that we require both. Scales and biases have the same format.
using CollectiveMainloopScaleWithZeroPoint = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, cute::tuple<ElementB, ElementScale, ElementZero>, LayoutB_Transpose,
    AlignmentB, ElementA, LayoutA_Transpose, AlignmentA, ElementAccumulator, TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule>::CollectiveOp;

using GemmKernelScaleWithZeroPoint =
    cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,  // Indicates ProblemShape
                                         CollectiveMainloopScaleWithZeroPoint, CollectiveEpilogue>;

using GemmScaleWithZeroPoint =
    cutlass::gemm::device::GemmUniversalAdapter<GemmKernelScaleWithZeroPoint>;
