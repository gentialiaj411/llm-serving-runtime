#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace runtime::common {

struct GenerateRequest {
  std::string request_id;
  std::string prompt;
  std::uint32_t max_tokens{64};
  float temperature{0.0F};
};

struct GenerateResponse {
  std::string request_id;
  std::string text;
  std::vector<std::string> tokens;
};

}  // namespace runtime::common
