/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"

namespace cudaq {

class Token {
public:
  enum Kind {
#define TOK_MARKER(NAME) NAME,
#define TOK_IDENTIFIER(NAME) NAME,
#define TOK_LITERAL(NAME) NAME,
#define TOK_PUNCTUATION(NAME, SPELLING) NAME,
#define TOK_KEYWORD(SPELLING) kw_##SPELLING,
#include "TokenKinds.def"
  };

  Token(Kind kind, llvm::StringRef spelling) : kind(kind), spelling(spelling) {}

  llvm::StringRef getSpelling() const { return spelling; }

  Kind getKind() const { return kind; }

  bool is(Kind k) const { return kind == k; }

  bool isAny(Kind k1, Kind k2) const { return is(k1) || is(k2); }

  template <typename... T>
  bool isAny(Kind k1, Kind k2, Kind k3, T... others) const {
    if (is(k1))
      return true;
    return isAny(k2, k3, others...);
  }

  bool isNot(Kind k) const { return kind != k; }

  template <typename... T>
  bool isNot(Kind k1, Kind k2, T... others) const {
    return !isAny(k1, k2, others...);
  }

  bool isKeyword() const;

  //===--------------------------------------------------------------------===//
  // Location processing.
  //===--------------------------------------------------------------------===//

  llvm::SMLoc getLoc() const;
  llvm::SMLoc getEndLoc() const;
  llvm::SMRange getLocRange() const;

  //===--------------------------------------------------------------------===//
  // Helpers to decode specific sorts of tokens.
  //===--------------------------------------------------------------------===//

  llvm::Optional<uint64_t> getIntegerValue() const;
  llvm::Optional<double> getFloatingPointValue() const;

private:
  Kind kind;
  llvm::StringRef spelling;
};

} // namespace cudaq
