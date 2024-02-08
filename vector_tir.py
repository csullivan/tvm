import tvm
import numpy as np
from utils import Timer

dtype = "e4m3_float8"
numpytype = "float8_e4m3fn"
# dtype="float16"
# numpytype="float16"
# n = pow(2,12)
n = 64


def vector_test():
    print(f"Testing vector addition with dtype={dtype}, size={n}")

    low, high = 0, 5

    A = tvm.te.placeholder((n,), name="A", dtype=dtype)
    B = tvm.te.placeholder((n,), name="B", dtype=dtype)
    C = tvm.te.compute(A.shape, lambda i: A[i] + B[i], name="C")

    prim_func = tvm.te.create_prim_func([A, B, C])
    prim_func.show()
    sch = tvm.tir.Schedule(prim_func)
    block = sch.get_block("C")
    b = sch.get_loops(block)
    bx, tx = sch.split(b[0], factors=[None, 32])
    sch.bind(bx, "blockIdx.x")
    sch.bind(tx, "threadIdx.x")
    sch.mod.show()

    target = "cuda"
    fadd = tvm.build(sch.mod, target=target)

    cuda_code = fadd.imported_modules[0].get_source()
    with open("tmp_cuda.cu", "w") as f:
        f.write(cuda_code)

    dev = tvm.device(target, 0)
    a = tvm.nd.array(np.random.uniform(low=low, high=high, size=n).astype(numpytype), dev)
    b = tvm.nd.array(np.random.uniform(low=low, high=high, size=n).astype(numpytype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=numpytype), dev)

    with Timer("fadd"):
        fadd(a, b, c)
        dev.sync()

    print(f"Runtime arrays:\na={a}\nb={b}")
    print(f"Result:\nc={c}")
    print(a.dtype, b.dtype, c.dtype)

    A = a.numpy()
    B = b.numpy()
    C = c.numpy()
    answer = A + B
    for i in range(n):
        if C[i] != answer[i]:
            print(
                f"Value mistmatch. i={i} a[i]={A[i]} b[i]={B[i]} c[i]={C[i]} (expected c[i]={answer[i]})"
            )


if __name__ == "__main__":
    vector_test()
