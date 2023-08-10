/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "Token.h"

using namespace cudaq;

bool Token::isKeyword() const {
  switch (kind) {
  default:
    return false;
#define TOK_KEYWORD(SPELLING)                                                  \
  case kw_##SPELLING:                                                          \
    return true;
#include "TokenKinds.def"
  }
}

//===----------------------------------------------------------------------===//
// Location processing.
//===----------------------------------------------------------------------===//

llvm::SMLoc Token::getLoc() const {
  return llvm::SMLoc::getFromPointer(spelling.data());
}

llvm::SMLoc Token::getEndLoc() const {
  return llvm::SMLoc::getFromPointer(spelling.data() + spelling.size());
}

llvm::SMRange Token::getLocRange() const {
  return llvm::SMRange(getLoc(), getEndLoc());
}

//===----------------------------------------------------------------------===//
// Helpers to decode specific sorts of tokens.
//===----------------------------------------------------------------------===//

std::optional<uint64_t> Token::getIntegerValue() const {
  uint64_t result = 0;
  if (spelling.getAsInteger(10, result))
    return std::nullopt;
  return result;
}

std::optional<double> Token::getFloatingPointValue() const {
  double result = 0;
  if (spelling.getAsDouble(result))
    return std::nullopt;
  return result;
}
