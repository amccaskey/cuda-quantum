#pragma once

#include "channel.h"

#include <stdarg.h>

namespace cudaq::driver {

struct target_info {};

// Spec device IDs are implicit in the vector index here
static std::vector<std::unique_ptr<channel>> communication_channels;

void initialize(const target_info &info) {}

std::size_t marshal_arguments(std::size_t deviceId, const char *argFmtStr,
                              ...) {
  std::size_t numArgs = 0;
  for (const char *p = argFmtStr; (p = strchr(p, '%')) != NULL; numArgs++, p++)
    ;

  va_list args;
  va_start(args, numArgs);

  // do something with these args
  std::vector<argument> marshaledArgs;
  // Fill the args...

  auto &channel = communication_channels.at(deviceId);

  return channel->marshal(marshaledArgs);
}

} // namespace cudaq::driver