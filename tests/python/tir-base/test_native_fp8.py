# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import sys
import pytest

import tvm
from tvm.script import tir as T
import numpy as np
import tvm.testing


@tvm.testing.requires_cuda_compute_version(9)
def test_e4m3_conversions():
    dtype = "e4m3_float8"

    @T.prim_func
    def add(
        A: T.Buffer((64,), dtype),
        B: T.Buffer((64,), dtype),
        C: T.Buffer((64,), dtype),
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i in range(64):
            with T.block("C"):
                v_i = T.axis.spatial(64, i)
                T.reads(A[v_i], B[v_i])
                T.writes(C[v_i])
                C[v_i] = T.Cast(dtype, T.Cast("float16", A[v_i]) + T.Cast("float16", B[v_i]))

    sch = tvm.tir.Schedule(add)
    block = sch.get_block("C")
    b = sch.get_loops(block)
    bx, tx = sch.split(b[0], factors=[None, 32])
    sch.bind(bx, "blockIdx.x")
    sch.bind(tx, "threadIdx.x")

    target = "cuda"
    fadd = tvm.build(sch.mod, target=target)

    cuda_src = fadd.imported_modules[0].get_source()
    assert "fp8_e4_t" in cuda_src, "FP8E4M3 (fp8_e4_t) datatype not found in generated CUDA"

    dev = tvm.device(target, 0)

    numpytype = "float8_e4m3fn"
    a = tvm.nd.array(np.random.uniform(low=0, high=5, size=64).astype(numpytype), dev)
    b = tvm.nd.array(np.random.uniform(low=0, high=5, size=64).astype(numpytype), dev)
    c = tvm.nd.array(np.zeros(64, dtype=numpytype), dev)
    fadd(a, b, c)

    tvm.testing.assert_allclose(
        c.numpy().astype("float16"), (a.numpy() + b.numpy()).astype("float16")
    )


native_dtype, promoted_dtype = tvm.testing.parameters(
    ("e4m3_float8", "float32"),
    ("e4m3_float8", "float16"),
    ("e4m3_float8x2", "float32x2"),
    ("e4m3_float8x2", "float16x2"),
    ("e4m3_float8x4", "float32x4"),
    # Supported via half4 vector type extension in codegen
    ("e4m3_float8x4", "float16x4"),
)


@tvm.testing.requires_cuda_compute_version(9)
def test_e4m3x2_conversions(native_dtype, promoted_dtype):
    vector_length = 64

    @T.prim_func
    def add(
        A: T.Buffer((vector_length,), native_dtype),
        B: T.Buffer((vector_length,), native_dtype),
        C: T.Buffer((vector_length,), native_dtype),
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i in range(vector_length):
            with T.block("C"):
                v_i = T.axis.spatial(vector_length, i)
                T.reads(A[v_i], B[v_i])
                T.writes(C[v_i])
                C[v_i] = T.Cast(
                    native_dtype, T.Cast(promoted_dtype, A[v_i]) + T.Cast(promoted_dtype, B[v_i])
                )

    sch = tvm.tir.Schedule(add)
    block = sch.get_block("C")
    b = sch.get_loops(block)
    bx, tx = sch.split(b[0], factors=[None, 32])
    sch.bind(bx, "blockIdx.x")
    sch.bind(tx, "threadIdx.x")

    target = "cuda"
    fadd = tvm.build(sch.mod, target=target)
    cuda_src = fadd.imported_modules[0].get_source()
    dev = tvm.device(target, 0)

    numpytype = "float8_e4m3fn"
    if "x" in native_dtype:
        lanes = int(native_dtype.split("x")[-1])
    else:
        lanes = 1

    if "x" in promoted_dtype:
        promoted_base_dtype = promoted_dtype.split("x")[0]
    else:
        promoted_base_dtype = promoted_dtype

    np_shape = (vector_length, lanes) if lanes > 1 else (vector_length,)
    a_np = np.random.uniform(low=0, high=5, size=np_shape).astype(numpytype)
    a = tvm.nd.empty(shape=(vector_length,), dtype=native_dtype, device=dev)
    a.copyfrom(a_np)
    b_np = np.random.uniform(low=0, high=5, size=np_shape).astype(numpytype)
    b = tvm.nd.empty(shape=(vector_length,), dtype=native_dtype, device=dev)
    b.copyfrom(b_np)
    c = tvm.nd.empty(shape=(vector_length,), dtype=native_dtype, device=dev)
    fadd(a, b, c)

    tvm.testing.assert_allclose(
        c.numpy().astype(promoted_base_dtype), (a_np + b_np).astype(promoted_base_dtype)
    )


if __name__ == "__main__":
    tvm.testing.main()
