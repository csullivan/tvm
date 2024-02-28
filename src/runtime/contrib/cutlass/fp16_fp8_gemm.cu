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
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "hopper_mixed_dtype_gemm_template.h"

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
using StrideC = typename GemmKernelScaleWithZeroPoint::StrideC;
using StrideD = typename GemmKernelScaleWithZeroPoint::StrideD;
using StrideS = typename CollectiveMainloopScaleWithZeroPoint::StrideScale;

template <typename GemmArguments>
GemmArguments gemm_args_from_packed_args(int64_t m, int64_t n, int64_t k, int64_t l, int64_t g,
                                         float alpha, float beta, cutlass::half_t* A,
                                         cutlass::float_e4m3_t* B, cutlass::half_t* C,
                                         cutlass::half_t* D, cutlass::half_t* S) {
  // TODO(csullivan): check if batch handling is correct
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, l));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, l));
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(n, m, l));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(n, m, l));
  auto stride_S =
      cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(n, (k + g - 1) / g, l));

  // Construct GEMM arguments
  GemmArguments args = {cutlass::gemm::GemmUniversalMode::kGemm,
                        {n, m, k, l},
                        {B, stride_B, A, stride_A, S, stride_S, g},
                        {{alpha, beta}, C, stride_C, D, stride_D}};

  return args;
}

namespace tvm {
namespace runtime {

void _cutlass_mixed_dtype_gemm_fp16_fp8_scale(DLTensor* A, DLTensor* B, DLTensor* C, DLTensor* S,
                                              int64_t m, int64_t n, int64_t k, int64_t l, int64_t g,
                                              DLTensor* D) {
  auto rawA = static_cast<cutlass::half_t*>(A->data);
  auto rawB = static_cast<cutlass::float_e4m3_t*>(B->data);
  cutlass::half_t* rawC = nullptr;
  if (C) {
    rawC = static_cast<cutlass::half_t*>(C->data);
  }
  auto rawD = static_cast<cutlass::half_t*>(D->data);
  auto rawS = static_cast<cutlass::half_t*>(S->data);

  float alpha = 1.0;
  float beta = 0.0;
  if (rawC) {
    beta = 1.0;
  }
  auto args = gemm_args_from_packed_args<typename GemmScaleOnly::Arguments>(
      m, n, k, l, g, alpha, beta, rawA, rawB, rawC, rawD, rawS);

  GemmScaleOnly gemm;

  size_t workspace_size = GemmScaleOnly::get_workspace_size(args);

  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(gemm.can_implement(args));
  CUTLASS_CHECK(gemm.initialize(args, workspace.get()));
  CUTLASS_CHECK(gemm.run());
}

void _cutlass_mixed_dtype_matmul_fp16_fp8_scale(DLTensor* A, DLTensor* B, DLTensor* S, int64_t m,
                                                int64_t n, int64_t k, int64_t l, int64_t g,
                                                DLTensor* D) {
  _cutlass_mixed_dtype_gemm_fp16_fp8_scale(A, B, nullptr, S, m, n, k, l, g, D);
}

TVM_REGISTER_GLOBAL("cutlass.mixed_dtype_matmul_fp16_fp8_scale")
    .set_body_typed(_cutlass_mixed_dtype_matmul_fp16_fp8_scale);
TVM_REGISTER_GLOBAL("cutlass.mixed_dtype_gemm_fp16_fp8_scale")
    .set_body_typed(_cutlass_mixed_dtype_gemm_fp16_fp8_scale);
}  // namespace runtime
}  // namespace tvm
