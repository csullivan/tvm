#include <nccl.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/ndarray.h>

#include "cuda_common.h"

namespace tvm {
namespace runtime {

class NCCLModule {
public:
    static NCCLModule& GetInstance() {
        static NCCLModule instance;
        return instance;
    }

    void Init(int world_size, int rank) {
        this->world_size_ = world_size;
        this->rank_ = rank;
        cudaSetDevice(rank);

        ncclCommInitRank(&comm_, world_size_, nuid_, rank_);
    }

    ncclComm_t GetCommunicator() const {
        return comm_;
    }

    const ncclUniqueId& InitUniqueId() {
        ncclGetUniqueId(&nuid_);
        return nuid_;
    }

    void SetUniqueId(const ncclUniqueId& id) {
        nuid_ = id;
    }

private:
    NCCLModule() : comm_(nullptr) {}
    NCCLModule(const NCCLModule&) = delete;
    void operator=(const NCCLModule&) = delete;

    ncclComm_t comm_;
    int world_size_;
    int rank_;
    ncclUniqueId nuid_;
};

TVM_REGISTER_GLOBAL("tvm.nccl.init")
.set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
    int world_size = args[0];
    int rank = args[1];
    NCCLModule::GetInstance().Init(world_size, rank);
});

TVM_REGISTER_GLOBAL("tvm.nccl.get_comm")
.set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
    *rv = reinterpret_cast<void*>(NCCLModule::GetInstance().GetCommunicator());
});

TVM_REGISTER_GLOBAL("tvm.nccl.init_unique_id")
.set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
    const ncclUniqueId& uid = NCCLModule::GetInstance().InitUniqueId();
    tvm::runtime::NDArray uid_ndarray = tvm::runtime::NDArray::Empty({NCCL_UNIQUE_ID_BYTES}, DataType::Int(8), {kDLCPU, 0});
    memcpy(uid_ndarray->data, uid.internal, NCCL_UNIQUE_ID_BYTES);
    *rv = uid_ndarray;
});

TVM_REGISTER_GLOBAL("tvm.nccl.set_unique_id")
.set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
    tvm::runtime::NDArray uid_ndarray = args[0];
    
    int64_t total_size = 1;
    for (int i = 0; i < uid_ndarray->ndim; ++i) {
        total_size *= uid_ndarray->shape[i];
    }
    
    int64_t size_in_bytes = total_size * (uid_ndarray->dtype.bits / 8);
    if (uid_ndarray->byte_offset + NCCL_UNIQUE_ID_BYTES > uid_ndarray->byte_offset + size_in_bytes) {
        LOG(FATAL) << "Invalid size for ncclUniqueId bytes";
        return;
    }
    
    ncclUniqueId id;
    memcpy(id.internal, uid_ndarray->data, NCCL_UNIQUE_ID_BYTES);
    NCCLModule::GetInstance().SetUniqueId(id);
});


TVM_REGISTER_GLOBAL("tvm.distributed.collective.allgather").set_body([](TVMArgs args, TVMRetValue* rv) {
    DLTensor* in = args[0];
    DLTensor* out = args[1];

    ncclComm_t comm = NCCLModule::GetInstance().GetCommunicator();
    cudaStream_t stream = CUDAThreadEntry::ThreadLocal()->stream;
    ncclDataType_t ncclDataType;
    if (in->dtype.code == kDLFloat) {
      switch (in->dtype.bits) {
        case 16: ncclDataType = ncclFloat16; break;
        case 32: ncclDataType = ncclFloat32; break;
        case 64: ncclDataType = ncclFloat64; break;
        default: LOG(FATAL) << "Unsupported float bit-width: " << in->dtype.bits; return;
      }
    } else if (in->dtype.code == kDLInt) {
      switch (in->dtype.bits) {
        case 8: ncclDataType = ncclInt8; break;
        case 32: ncclDataType = ncclInt32; break;
        case 64: ncclDataType = ncclInt64; break;
        default: LOG(FATAL) << "Unsupported int bit-width: " << in->dtype.bits; return;
      }
    } else {
      LOG(FATAL) << "Unsupported DLTensor data type: " << in->dtype.code; return;
    }

    size_t sendcount = 1;
    for (int i = 0; i < in->ndim; ++i) {
      sendcount *= in->shape[i];
    }

    ncclAllGather(in->data, out->data, sendcount, ncclDataType, comm, stream);
});


} // namespace runtime
} // namespace tvm
