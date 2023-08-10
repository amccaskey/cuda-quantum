/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "Lexer.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace cudaq {

class Parser {
public:
  Parser(mlir::MLIRContext *context, const llvm::SourceMgr &sourceMgr,
         mlir::ModuleOp module)
      : lexer(context, sourceMgr), curToken(lexer.nextToken()), module(module),
        builder(module.getBodyRegion()) {}

  mlir::ParseResult parse();

private:
  //===--------------------------------------------------------------------===//
  // Location
  //===--------------------------------------------------------------------===//

  mlir::Location translateLocation(mlir::SMLoc loc) {
    return lexer.translateLocation(loc);
  }

  //===--------------------------------------------------------------------===//
  // Token Parsing
  //===--------------------------------------------------------------------===//

  bool check(Token::Kind kind) { return curToken.is(kind); }

  void consume() { curToken = lexer.nextToken(); }

  /// If the current token has the specified kind, consume it and return
  /// success. Otherwise, return failure.
  bool consumeIf(Token::Kind kind) {
    if (curToken.isNot(kind))
      return false;
    curToken = lexer.nextToken();
    return true;
  }

  /// Consume the specified token if present and return success.  On failure,
  /// output a diagnostic and return failure.
  mlir::ParseResult expect(Token::Kind expected);

  //===--------------------------------------------------------------------===//
  // QASM Parsing
  //===--------------------------------------------------------------------===//
  mlir::ParseResult parseArgument(mlir::Value &arg, int64_t *index = nullptr);
  mlir::ParseResult parseCRegDecl();
  mlir::ParseResult parseHeader();
  mlir::ParseResult parseIdentifier(mlir::StringRef &result);
  mlir::ParseResult parseInclude();
  mlir::ParseResult parseInteger(int64_t &result);
  mlir::ParseResult parseQRegDecl();
  mlir::ParseResult parseGateStmt();
  mlir::ParseResult parseMeasure();
  mlir::ParseResult parseParameter(mlir::Value &param);
  mlir::ParseResult parseReal(double &result);

  //===--------------------------------------------------------------------===//
  // Helpers
  //===--------------------------------------------------------------------===//

  std::string getTopLevelName() const;

  const llvm::SourceMgr &getSourceMgr() const { return lexer.getSourceMgr(); }

  mlir::ParseResult parseCommaSeparatedList(
      mlir::function_ref<mlir::ParseResult()> parseElementFne);

  //===--------------------------------------------------------------------===//
  // Members
  //===--------------------------------------------------------------------===//

  /// The lexer for the source file we're parsing.
  Lexer lexer;

  /// This is the next token that hasn't been consumed yet.
  Token curToken;

  /// The module we are parsing into.
  mlir::ModuleOp module;
  mlir::OpBuilder builder;
  llvm::StringMap<std::pair<mlir::Value, mlir::SmallVector<mlir::Value>>>
      symbolTable;

  Parser(const Parser &) = delete;
  void operator=(const Parser &) = delete;
};

} // namespace cudaq
