#pragma once

namespace cudaq {
template <typename Derived, typename Base>
Derived *crtp_cast(Base *u) {
  return static_cast<Derived *>(u);
}
} // namespace cudaq
