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

if(USE_CUDA AND USE_TRTLLM)

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  
  ### Build the trt-llm common libs
  set(TRTLLM_DIR ${PROJECT_SOURCE_DIR}/3rdparty/TensorRT-LLM)
  set(TRTLLM_COMMON_DIR ${TRTLLM_DIR}/cpp/tensorrt_llm/common)
  file(GLOB TRTLLM_COMMON_SRCS "${TRTLLM_COMMON_DIR}/*.cpp")
  file(GLOB TRTLLM_COMMON_CU_SRCS "${TRTLLM_COMMON_DIR}/*.cu")
  
  # Don't bring in unnecessary dependencies from trt-llm
  list(REMOVE_ITEM TRTLLM_COMMON_SRCS "${TRTLLM_COMMON_DIR}/mpiUtils.cpp")
  list(REMOVE_ITEM TRTLLM_COMMON_SRCS "${TRTLLM_COMMON_DIR}/cudaAllocator.cpp")
  include_directories(${TRTLLM_DIR}/cpp/include/ ${TRTLLM_DIR}/cpp ${CUDA_INCLUDE_DIRS})
  
  add_library(tensorrt_llm_common SHARED ${TRTLLM_COMMON_SRCS} ${TRTLLM_COMMON_CU_SRCS})
  target_compile_definitions(tensorrt_llm_common PRIVATE -DENABLE_FP8 -DENABLE_BF16)
  
  set_property(TARGET tensorrt_llm_common PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET tensorrt_llm_common PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set(TRTLLM_COMMON_BINARY_DIR ${CMAKE_BINARY_DIR}/tensorrt_llm_common_build)
  
  # list(APPEND RUNTIME_SRCS src/runtime/contrib/trtllm/fp8_quantization.cc)

  message(STATUS "Build with TensorRT-LLM")
endif()
