/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/utils/extension_point.h"

namespace cudaq::config {
class TargetConfig;
}

namespace cudaq::v2 {

class qpu_handle;

/// \struct platform_metadata
/// \brief Encapsulates platform and target configuration metadata for a QPU.
///
/// This struct holds references to the target configuration, target options,
/// and the initial configuration string for the quantum platform.
struct platform_metadata {
  /// \brief Reference to the target configuration.
  const cudaq::config::TargetConfig &target_config;
  /// \brief List of target-specific options.
  const std::vector<std::string> &target_options;
  /// \brief Initial configuration string.
  const std::string &initial_config_str;
};

namespace config {

/// @brief platform_config is an extension point for CUDA-Q
/// platform developers. Subtypes should implement configure_qpus
/// in order to configure the platform based on the target and
/// provided user information. Configure the platform implies
/// setting the platform_qpus reference.
class platform_config : public extension_point<platform_config> {
public:
  virtual void configure_qpus(std::vector<std::unique_ptr<qpu_handle>> &,
                              const platform_metadata &) = 0;
};

/// Decode the base64 encoded string and extract the --target-options 
/// from user command line
std::vector<std::string> get_options(const std::string &encoded);
void load_target_config(cudaq::config::TargetConfig &config,
                        const std::string &backend);
} // namespace config

} // namespace cudaq::v2
