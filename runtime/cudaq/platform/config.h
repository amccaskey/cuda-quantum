/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/Logger.h"
#include "qpu.h"

#include "llvm/Support/Base64.h"

#include <map>

namespace cudaq::config {

/// @brief platform_config is an extension point for CUDA-Q
/// platform developers. Subtypes should implement configure_qpus
/// in order to configure the platform based on the target and
/// provided user information. Configure the platform implies
/// setting the platform_qpus reference.
class platform_config : public extension_point<platform_config> {
public:
  virtual void configure_qpus(std::vector<std::unique_ptr<qpu>> &,
                              const platform_metadata &) = 0;
};

std::vector<std::string> get_options(const std::string &encoded) {

  llvm::SmallVector<llvm::StringRef> args;
  std::string targetArgsString = encoded;
  if (targetArgsString.starts_with("base64_")) {
    if (targetArgsString.size() > 7) {
      auto targetArgsStr = targetArgsString.substr(7);
      std::vector<char> decodedStr;
      if (auto err = llvm::decodeBase64(targetArgsStr, decodedStr)) {
        llvm::errs() << "DecodeBase64 error for '" << targetArgsStr
                     << "' string.";
        abort();
      }
      std::string decoded(decodedStr.data(), decodedStr.size());
      targetArgsString = decoded;
    } else {
      targetArgsString = "";
    }
  }
  llvm::StringRef(targetArgsString).split(args, ' ', -1, false);
  std::vector<std::string> targetArgv;
  for (const auto &arg : args) {
    std::string targetArgsStr = arg.str();
    if (targetArgsStr.starts_with("base64_")) {
      targetArgsStr.erase(0, 7); // erase "base64_"
      std::vector<char> decodedStr;
      if (auto err = llvm::decodeBase64(targetArgsStr, decodedStr)) {
        llvm::errs() << "DecodeBase64 error for '" << targetArgsStr
                     << "' string.";
        abort();
      }
      std::string decoded(decodedStr.data(), decodedStr.size());
      cudaq::info("Decoded '{}' from '{}'", decoded, targetArgsStr);
      targetArgsStr = decoded;
    }
    targetArgv.emplace_back(targetArgsStr);
  }
  return targetArgv;
}

} // namespace cudaq::config
