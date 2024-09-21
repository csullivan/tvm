import tvm
import tvm.testing
from tvm import te, tir
from tvm.script import tir as T, ir as I
import numpy as np

@tvm.testing.requires_cuda
def test_matmul():
    @I.ir_module    
    class Module:
        @T.prim_func
        def matmul(A: T.Buffer((64, 16,), "float16"),
                B: T.Buffer((16, 256), "float16"),
                C: T.Buffer((64, 256), "float16")):
            c_regs = T.alloc_buffer((128,), "float32", scope="local")
            A_shared = T.alloc_buffer((64*16), "float16", scope="shared")
            B_shared = T.alloc_buffer((16*256), "float16", scope="shared")
            for r in range(128):
                for n in range(8):
                    with T.block("load_A"):
                        v = T.axis.spatial(64*16, n*128 + r)
                        A_shared[v] = A[v//16, v%16]

                for n in range(32):
                    with T.block("load_B"):
                        v = T.axis.spatial(16*256, n*128 + r)
                        B_shared[v] = B[v//256, v%256]
        
                T.call_device_func("wgmma", 
                                c_regs.data,
                                A_shared.data,
                                B_shared.data,
                                )
    # # Create a schedule
    sch = tir.Schedule(Module)
    
    # # Work on the main function
    sch.work_on("matmul")
    

    
    # # Get the block
    block = sch.get_block("load_A")
    lo = sch.get_loops(block)
    lb, lt = sch.split(lo[0], [None, 128])
    sch.bind(lb, "blockIdx.x")
    sch.bind(lt, "threadIdx.x")

    
    sch.mod.show()
    
    # Build the function
    cuda_mod = tvm.build(sch.mod, target="cuda")

    # Prepare input data
    a_np = np.random.uniform(size=(64, 16,)).astype(np.float16)
    b_np = np.random.uniform(size=(16, 256,)).astype(np.float16)
    c_np = np.zeros((64, 256,), dtype=np.float16)

    # Get CUDA context
    cuda_ctx = tvm.cuda(0)

    # Transfer data to GPU
    a_tvm = tvm.nd.array(a_np, cuda_ctx)
    b_tvm = tvm.nd.array(b_np, cuda_ctx)
    c_tvm = tvm.nd.array(c_np, cuda_ctx)

    # Run the kernel
    cuda_mod(a_tvm, b_tvm, c_tvm)

    # Transfer results back to CPU
    c_np = c_tvm.numpy()

    # Verify results
    np.testing.assert_allclose(c_np, a_np @ b_np, rtol=1e-5)

    print("Matrix multiplication test passed!")

if __name__ == "__main__":
    test_matmul()