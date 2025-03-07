#pragma once

namespace cudaq::driver {

class argument {};

class channel {
public:
  virtual void connect() const = 0;
  virtual std::size_t marshal(const std::vector<argument> &arguments) const = 0;
  virtual void invoke_function(const std::string &symbolName,
                               std::size_t argumentIdentifier) const = 0;
  virtual void free_arguments(std::size_t identifier) const = 0;

};

} // namespace cudaq::driver