
/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "config.h"

#include "llvm/Support/Base64.h"

#include "cudaq/Support/TargetConfig.h"
#include "cudaq/utils/cudaq_utils.h"

#include "common/Logger.h"
#include <filesystem>
#include <fstream>

namespace cudaq::v2::config {

void load_target_config(cudaq::config::TargetConfig &config,
                        const std::string &backend) {
  std::filesystem::path cudaqLibPath{cudaq::getCUDAQLibraryPath()};
  auto platformPath = cudaqLibPath.parent_path().parent_path() / "targets";
  std::string fileName = backend + std::string(".yml");
  auto configFilePath = platformPath / fileName;
  cudaq::info("Config file path = {}", configFilePath.string());
  std::ifstream configFile(configFilePath.string());
  std::string configContents((std::istreambuf_iterator<char>(configFile)),
                             std::istreambuf_iterator<char>());
  llvm::yaml::Input Input(configContents.c_str());
  Input >> config;
}

/// Decode the base64 encoded string and extract the --target-options
/// from user command line
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
} // namespace cudaq::v2::config
