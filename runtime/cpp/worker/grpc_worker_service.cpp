#include "worker/grpc_worker_service.h"

#include <cstdint>
#include <sstream>
#include <vector>

namespace runtime::worker {

namespace {
std::vector<std::string> SplitWords(const std::string& text) {
  std::istringstream iss(text);
  std::vector<std::string> words;
  std::string w;
  while (iss >> w) {
    words.push_back(w);
  }
  if (words.empty()) {
    words.push_back("hello");
  }
  return words;
}
}  // namespace

grpc::Status GrpcWorkerService::Generate(
    grpc::ServerContext* /*context*/,
    const runtime::v1::GenerateRequest* request,
    runtime::v1::GenerateResponse* response) {
  const auto words = SplitWords(request->prompt());
  const auto max_tokens = request->max_tokens() == 0 ? 1U : request->max_tokens();

  std::string text;
  text.reserve(max_tokens * 6);
  for (std::uint32_t i = 0; i < max_tokens; ++i) {
    const std::string& token = words[i % words.size()];
    response->add_tokens(token);
    if (!text.empty()) {
      text.append(" ");
    }
    text.append(token);
  }

  response->set_request_id(request->request_id());
  response->set_text(text);
  return grpc::Status::OK;
}

std::unique_ptr<grpc::Server> BuildGrpcWorkerServer(
    const std::string& bind_addr,
    GrpcWorkerService* service) {
  grpc::ServerBuilder builder;
  builder.AddListeningPort(bind_addr, grpc::InsecureServerCredentials());
  builder.RegisterService(service);
  return builder.BuildAndStart();
}

}  // namespace runtime::worker
