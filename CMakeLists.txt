cmake_minimum_required(VERSION 3.25)
project(llm_serving_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(BUILD_CPP_RUNTIME "Build C++ runtime components" ON)
option(ENABLE_GRPC "Enable gRPC-based coordinator/worker" ON)
option(ENABLE_CUDA_WORKER "Enable CUDA worker stub" ON)

if(BUILD_CPP_RUNTIME)
  add_library(runtime_common STATIC
    runtime/cpp/common/types.h
    runtime/cpp/common/types.cpp
  )
  target_include_directories(runtime_common PUBLIC runtime/cpp)

  include(CheckLanguage)
  check_language(CUDA)
  if(ENABLE_CUDA_WORKER AND CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  elseif(ENABLE_CUDA_WORKER)
    message(WARNING "CUDA requested but no CUDA compiler found; building worker without CUDA stub.")
    set(ENABLE_CUDA_WORKER OFF)
  endif()

  if(ENABLE_GRPC)
    find_package(Protobuf QUIET)
    find_package(gRPC QUIET)

    if(Protobuf_FOUND AND gRPC_FOUND)
      message(STATUS "gRPC + Protobuf found: enabling generated stubs")
      set(PROTO_FILE ${CMAKE_SOURCE_DIR}/runtime/proto/runtime.proto)
      set(GEN_DIR ${CMAKE_BINARY_DIR}/generated)
      file(MAKE_DIRECTORY ${GEN_DIR})

      add_custom_command(
        OUTPUT ${GEN_DIR}/runtime.pb.cc ${GEN_DIR}/runtime.pb.h ${GEN_DIR}/runtime.grpc.pb.cc ${GEN_DIR}/runtime.grpc.pb.h
        COMMAND protobuf::protoc
        ARGS --cpp_out=${GEN_DIR} --grpc_out=${GEN_DIR} --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin> -I ${CMAKE_SOURCE_DIR}/runtime/proto ${PROTO_FILE}
        DEPENDS ${PROTO_FILE}
        VERBATIM
      )

      add_library(runtime_proto STATIC
        ${GEN_DIR}/runtime.pb.cc
        ${GEN_DIR}/runtime.grpc.pb.cc
      )
      target_include_directories(runtime_proto PUBLIC ${GEN_DIR})
      target_link_libraries(runtime_proto PUBLIC gRPC::grpc++ protobuf::libprotobuf)

      add_executable(runtime_coordinator_cpp runtime/cpp/coordinator/main.cpp)
      target_link_libraries(runtime_coordinator_cpp PRIVATE runtime_common runtime_proto)

      if(ENABLE_CUDA_WORKER)
        add_executable(runtime_worker_cpp
          runtime/cpp/worker/main.cpp
          runtime/cpp/worker/cuda_stub.cu
        )
        target_compile_definitions(runtime_worker_cpp PRIVATE RUNTIME_ENABLE_CUDA_STUB=1)
      else()
        add_executable(runtime_worker_cpp runtime/cpp/worker/main.cpp)
        target_compile_definitions(runtime_worker_cpp PRIVATE RUNTIME_ENABLE_CUDA_STUB=0)
      endif()
      target_link_libraries(runtime_worker_cpp PRIVATE runtime_common runtime_proto)

    else()
      message(WARNING "gRPC/Protobuf not found. Building placeholder binaries without generated RPC stubs.")
      add_executable(runtime_coordinator_cpp runtime/cpp/coordinator/main.cpp)
      target_link_libraries(runtime_coordinator_cpp PRIVATE runtime_common)

      if(ENABLE_CUDA_WORKER)
        add_executable(runtime_worker_cpp
          runtime/cpp/worker/main.cpp
          runtime/cpp/worker/cuda_stub.cu
        )
        target_compile_definitions(runtime_worker_cpp PRIVATE RUNTIME_ENABLE_CUDA_STUB=1)
      else()
        add_executable(runtime_worker_cpp runtime/cpp/worker/main.cpp)
        target_compile_definitions(runtime_worker_cpp PRIVATE RUNTIME_ENABLE_CUDA_STUB=0)
      endif()
      target_link_libraries(runtime_worker_cpp PRIVATE runtime_common)
    endif()
  endif()
endif()
