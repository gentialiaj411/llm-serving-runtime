#pragma once

#include <grpcpp/grpcpp.h>

#include <memory>
#include <string>

#include "runtime.grpc.pb.h"

namespace runtime::worker {

class GrpcWorkerService final : public runtime::v1::WorkerService::Service {
 public:
  grpc::Status Generate(
      grpc::ServerContext* context,
      const runtime::v1::GenerateRequest* request,
      runtime::v1::GenerateResponse* response) override;
};

std::unique_ptr<grpc::Server> BuildGrpcWorkerServer(
    const std::string& bind_addr,
    GrpcWorkerService* service);

}  // namespace runtime::worker
