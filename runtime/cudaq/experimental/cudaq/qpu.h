/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

namespace cudaq {

/// Allowed configuration value types for QPU configuration
/// Supports common types needed for QPU setup: strings for endpoints/tokens,
/// numeric types for parameters, and booleans for flags
using allowed_config_t =
    std::variant<std::string, bool, int, std::size_t, double, float>;

/// Configuration map for QPU initialization
/// Maps string keys to type-erased configuration values
using qpu_configuration = std::unordered_map<std::string, allowed_config_t>;

/// @brief Extract a typed value from a QPU configuration map
/// @tparam T The expected type of the configuration value
/// @param config The configuration map to extract from
/// @param key The configuration key to look up
/// @return std::optional<T> containing the value if found and type matches,
///         std::nullopt otherwise
///
/// This function provides type-safe extraction of configuration values.
/// It returns nullopt if either:
/// - The key doesn't exist in the configuration map
/// - The stored value's type doesn't match the requested type T
///
/// Example usage:
/// ```cpp
/// qpu_configuration config;
/// config["endpoint"] = std::string("http://localhost:5000");
/// config["shots"] = 1000;
///
/// auto endpoint = extract_config<std::string>(config, "endpoint");
/// if (endpoint) {
///   // Use endpoint.value()
/// }
///
/// auto shots = extract_config<int>(config, "shots");
/// auto missing = extract_config<double>(config, "nonexistent"); // nullopt
/// ```
template <typename T>
std::optional<T> extract_config(const qpu_configuration &config,
                                const std::string &key) {
  auto it = config.find(key);
  if (it == config.end()) {
    return std::nullopt; // Key doesn't exist
  }

  if (auto *value = std::get_if<T>(&it->second)) {
    return *value; // Type matches
  }
  return std::nullopt; // Type mismatch
}

// ============================================================================
// Base QPU Class (CRTP Pattern)
// ============================================================================

/// Base QPU class using CRTP (Curiously Recurring Template Pattern)
///
/// This class provides the foundation for all QPU implementations, whether
/// built-in or external plugins. It uses CRTP for static polymorphism,
/// enabling zero-overhead abstraction.
///
/// Example:
/// ```cpp
/// class my_qpu : public qpu<my_qpu, simulator<my_qpu>, local_trait<my_qpu>> {
/// public:
///   std::string name() const { return "my_qpu"; }
///   void configure(const heterogeneous_map& config) { /* ... */ }
///   // Implement trait methods...
/// };
/// ```
template <typename Derived, typename... Traits>
class qpu : public Traits... {
protected:
  qpu_configuration m_configuration;

public:
  qpu() = default;

  /// Construct with configuration
  qpu(const qpu_configuration &config) : m_configuration(config) {}

  /// Get the name of this QPU (must be implemented by derived class)
  std::string name() const { return crtp_cast<Derived>(this)->name(); }

  /// Get the current configuration
  qpu_configuration &get_configuration() { return m_configuration; }
  /// Get the current configuration (const)
  const qpu_configuration &get_configuration() const { return m_configuration; }
};

// ============================================================================
// Helper Templates
// ============================================================================

/// Helper for always-false static_assert (used in templates)
template <typename T>
inline constexpr bool always_false_v = false;

} // namespace cudaq
