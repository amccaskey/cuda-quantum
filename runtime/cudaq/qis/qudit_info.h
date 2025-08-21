#pragma once

#include <cstddef>

namespace cudaq {

struct QuditInfo {
  std::size_t levels = 0;
  std::size_t id = 0;
  QuditInfo(std::size_t _levels, std::size_t _id) : levels(_levels), id(_id) {}
  bool operator==(const QuditInfo &other) const {
    return levels == other.levels && id == other.id;
  }
};
} // namespace cudaq
