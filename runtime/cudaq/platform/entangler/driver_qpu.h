#pragma once
#include "cudaq/platform/qpu.h"

namespace cudaq {
class DriverQPU : public QPU {
public:
  DriverQPU() = default;

  cudaq::KernelThunkResultType
  launchKernel(const std::string &name, cudaq::KernelThunkType kernelFunc,
               void *args, std::uint64_t argsSize, std::uint64_t resultOffset,
               const std::vector<void *> &rawArgs) override {}

  void setExecutionContext(cudaq::ExecutionContext *context) override {}

  void resetExecutionContext() override {}
};

} // namespace cudaq