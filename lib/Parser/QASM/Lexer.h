/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "Token.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
class Location;
class MLIRContext;
} // namespace mlir

namespace cudaq {

class Lexer {
public:
  explicit Lexer(mlir::MLIRContext *context, const llvm::SourceMgr &sourceMgr);

  const llvm::SourceMgr &getSourceMgr() const { return sourceMgr; }

  Token nextToken();

  mlir::Location translateLocation(mlir::SMLoc loc);

private:
  // Helpers.
  Token createToken(Token::Kind kind, const char *tokStart) {
    return Token(kind, llvm::StringRef(tokStart, curPtr - tokStart));
  }

  Token emitError(const char *loc, const llvm::Twine &message);

  // Lexer implementation methods.
  Token eatIdentifier(const char *tokStart);
  Token eatNumber(const char *tokStart);
  Token eatString(const char *tokStart);
  void eatComment();
  void eatWhitespace();

  mlir::MLIRContext *const context;
  const llvm::SourceMgr &sourceMgr;
  const mlir::StringAttr bufferNameIdentifier;

  mlir::StringRef curBuffer;
  const char *curPtr;

  Lexer(const Lexer &) = delete;
  void operator=(const Lexer &) = delete;
};

} // namespace cudaq
