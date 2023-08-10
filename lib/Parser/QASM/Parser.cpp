/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "Parser.h"
#include "Lexer.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Parser/QASM.h"
#include "llvm/Support/Path.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace cudaq {
mlir::OwningOpRef<mlir::ModuleOp>
parseQASMFile(mlir::MLIRContext *context, const llvm::SourceMgr &sourceMgr) {
  const auto *sourceBuf = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  mlir::Location loc = mlir::FileLineColLoc::get(
      context, sourceBuf->getBufferIdentifier(), /*line=*/0, /*column=*/0);
  mlir::OwningOpRef<mlir::ModuleOp> module(mlir::ModuleOp::create(loc));
  if (Parser(context, sourceMgr, *module).parse())
    return {};
  return module;
}

mlir::OwningOpRef<mlir::ModuleOp>
parseQASMCode(mlir::MLIRContext *context, const llvm::StringRef &sourceStr) {
  auto sourceBuf = llvm::MemoryBuffer::getMemBuffer(sourceStr, "sourceName");
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(sourceBuf), llvm::SMLoc());
  return parseQASMFile(context, sourceMgr);
}

//===----------------------------------------------------------------------===//

mlir::ParseResult Parser::parse() {
  if (parseHeader())
    return mlir::failure();

  // Create the top-level circuit op in the MLIR module.
  mlir::Location loc = translateLocation(curToken.getLoc());
  auto circuit = builder.create<mlir::func::FuncOp>(
      loc, getTopLevelName(), builder.getFunctionType({}, {}));
  builder.setInsertionPointToEnd(circuit.addEntryBlock());

  while (1) {
    if (check(Token::eof))
      break;
    loc = translateLocation(curToken.getLoc());
    switch (curToken.getKind()) {
    case Token::Kind::kw_barrier:
    case Token::Kind::kw_gate:
    case Token::Kind::kw_if:
      mlir::emitError(loc, "Not implemented");
      return mlir::failure();

    case Token::Kind::kw_creg:
      if (parseCRegDecl())
        return mlir::failure();
      break;

    case Token::Kind::kw_CX:
    case Token::Kind::kw_U:
    case Token::Kind::identifier:
      if (parseGateStmt())
        return mlir::failure();
      break;

    case Token::Kind::kw_include:
      mlir::emitWarning(loc, "Currently unsupported, will ignore");
      if (parseInclude())
        return mlir::failure();
      break;

    case Token::Kind::kw_measure:
      if (parseMeasure())
        return mlir::failure();
      break;

    case Token::Kind::kw_qreg:
      if (parseQRegDecl())
        return mlir::failure();
      break;

    default:
      mlir::emitError(loc, "Unexpected token: " + curToken.getSpelling());
      return mlir::failure();
    }
  }
  loc = translateLocation(curToken.getLoc());
  builder.create<mlir::func::ReturnOp>(loc);
  return mlir::success();
}

mlir::ParseResult Parser::parseCRegDecl() {
  mlir::Location loc = translateLocation(curToken.getLoc());
  mlir::StringRef identifier;
  int64_t size;
  if (expect(Token::Kind::kw_creg) || parseIdentifier(identifier) ||
      expect(Token::Kind::l_square) || parseInteger(size) ||
      expect(Token::Kind::r_square) || expect(Token::Kind::semicolon))
    return mlir::failure();
  auto memrefType = mlir::MemRefType::get({size}, builder.getIntegerType(1));
  auto allocaOp = builder.create<mlir::memref::AllocOp>(loc, memrefType);
  // FIXME: Check if already exists
  symbolTable.try_emplace(
      identifier, std::make_pair(allocaOp.getResult(),
                                 mlir::SmallVector<mlir::Value>(size, {})));
  return mlir::success();
}

mlir::ParseResult Parser::parseHeader() {
  if (expect(Token::Kind::kw_OPENQASM))
    return mlir::failure();
  // Gambiarra: checking for version as a real number
  mlir::Location loc = translateLocation(curToken.getLoc());
  if (!check(Token::real)) {
    mlir::emitError(loc, "Expected a version number X.X");
    return mlir::failure();
  }
  double version = curToken.getFloatingPointValue().value();
  loc = translateLocation(curToken.getLoc());
  if (version != 2.0) {
    mlir::emitError(
        loc, "Unsupported version.  The parser barely supports OpenQASM 2.0");
    return mlir::failure();
  }
  // Consume the token.  We know it is a real, so no need to check;
  consume();
  return expect(Token::Kind::semicolon);
}

mlir::ParseResult Parser::parseIdentifier(mlir::StringRef &result) {
  if (!check(Token::Kind::identifier))
    return mlir::failure();
  result = curToken.getSpelling();
  consume();
  return mlir::success();
}

// Doesn't really matter if it's sucessful or not, for now we gracefully ignore
mlir::ParseResult Parser::parseInclude() {
  if (expect(Token::Kind::kw_include) || expect(Token::Kind::string) ||
      expect(Token::Kind::semicolon))
    return mlir::failure();
  return mlir::success();
}

mlir::ParseResult Parser::parseInteger(int64_t &result) {
  if (!check(Token::Kind::integer))
    return mlir::failure();
  result = curToken.getIntegerValue().value();
  consume();
  return mlir::success();
}

mlir::ParseResult Parser::parseQRegDecl() {
  mlir::Location loc = translateLocation(curToken.getLoc());
  mlir::StringRef identifier;
  int64_t size;
  if (expect(Token::Kind::kw_qreg) || parseIdentifier(identifier) ||
      expect(Token::Kind::l_square) || parseInteger(size) ||
      expect(Token::Kind::r_square) || expect(Token::Kind::semicolon))
    return mlir::failure();
  auto allocaOp = builder.create<quake::AllocaOp>(loc, size);
  // FIXME: Check if already exists
  symbolTable.try_emplace(
      identifier, std::make_pair(allocaOp.getResult(),
                                 mlir::SmallVector<mlir::Value>(size, {})));
  return mlir::success();
}

mlir::ParseResult Parser::parseGateStmt() {
  mlir::Location loc = translateLocation(curToken.getLoc());
  mlir::SmallVector<mlir::Value> params;
  mlir::SmallVector<mlir::Value> args;
  mlir::StringRef identifier;

  if (parseIdentifier(identifier))
    return mlir::failure();
  if (consumeIf(Token::Kind::l_paren)) {
    if (parseCommaSeparatedList(
            [&]() { return parseParameter(params.emplace_back()); }) ||
        expect(Token::Kind::r_paren))
      return mlir::failure();
  }
  if (parseCommaSeparatedList(
          [&]() { return parseArgument(args.emplace_back()); }) ||
      expect(Token::Kind::semicolon))
    return mlir::failure();

  mlir::ArrayRef<mlir::Value> controls;
  mlir::ArrayRef<mlir::Value> targets;
  if (identifier == "cx" || identifier == "CX") {
    identifier = "x";
    controls = mlir::ArrayRef<mlir::Value>(args.data(), 1);
    targets = mlir::ArrayRef<mlir::Value>(args.data() + 1, 1);
  } else {
    targets = args;
  }
  if (identifier == "h")
    builder.create<quake::HOp>(loc, controls, targets);
  else if (identifier == "rz")
    builder.create<quake::RzOp>(loc, params, controls, targets);
  else if (identifier == "s")
    builder.create<quake::SOp>(loc, controls, targets);
  else if (identifier == "sdg")
    builder.create<quake::SOp>(loc, /*is_adj=*/true, params, controls, targets);
  else if (identifier == "t")
    builder.create<quake::TOp>(loc, controls, targets);
  else if (identifier == "tdg")
    builder.create<quake::TOp>(loc, /*is_adj=*/true, params, controls, targets);
  else if (identifier == "x")
    builder.create<quake::XOp>(loc, controls, targets);
  else if (identifier == "y")
    builder.create<quake::YOp>(loc, controls, targets);
  else if (identifier == "z")
    builder.create<quake::ZOp>(loc, controls, targets);
  else
    mlir::emitError(loc, "Unknown op: " + identifier);

  return mlir::success();
}

mlir::ParseResult Parser::parseParameter(mlir::Value &param) {
  mlir::Location loc = translateLocation(curToken.getLoc());
  double value;
  bool is_minus = consumeIf(Token::Kind::minus);
  if (parseReal(value))
    return mlir::failure();
  if (is_minus)
    value *= -1;
  param = builder.create<mlir::arith::ConstantFloatOp>(
      loc, llvm::APFloat(value), builder.getF64Type());
  return mlir::success();
}

// Statement: `measure` qubit | qreg `->` bit | creg
mlir::ParseResult Parser::parseMeasure() {
  mlir::Location loc = translateLocation(curToken.getLoc());
  mlir::Value qArg;
  mlir::Value cArg;
  int64_t index;
  if (expect(Token::Kind::kw_measure) || parseArgument(qArg) ||
      expect(Token::Kind::arrow) || parseArgument(cArg, &index) ||
      expect(Token::Kind::semicolon))
    return mlir::failure();

  // FIXME: Handle measuring a register
  if (qArg.getType().isa<quake::VeqType>()) {
    mlir::emitError(loc, "Cannot handle measuring registers yet.");
    return mlir::failure();
  }
  auto mzOp = builder.create<quake::MzOp>(loc, builder.getIntegerType(1), qArg);
  auto constOp = builder.create<mlir::arith::ConstantIndexOp>(loc, index);
  builder.create<mlir::memref::StoreOp>(loc, mzOp, cArg, constOp.getResult());
  return mlir::success();
}

mlir::ParseResult Parser::parseArgument(mlir::Value &arg, int64_t *index) {
  mlir::Location loc = translateLocation(curToken.getLoc());
  mlir::StringRef identifier;
  int64_t idx;
  if (parseIdentifier(identifier) || expect(Token::Kind::l_square) ||
      parseInteger(idx) || expect(Token::Kind::r_square))
    return mlir::failure();
  if (index)
    *index = idx;
  auto &[qvec, qubits] = symbolTable[identifier];
  if (!qvec.getType().isa<quake::VeqType>()) {
    arg = qvec;
    return mlir::success();
  }
  if (!qubits[idx]) {
    auto constOp = builder.create<mlir::arith::ConstantIndexOp>(loc, idx);
    qubits[idx] = builder.create<quake::ExtractRefOp>(loc, qvec, constOp);
  }
  arg = qubits[idx];
  return mlir::success();
}

mlir::ParseResult Parser::parseReal(double &result) {
  if (!check(Token::Kind::real))
    return mlir::failure();
  result = curToken.getFloatingPointValue().value();
  consume();
  return mlir::success();
}

mlir::ParseResult Parser::parseCommaSeparatedList(
    llvm::function_ref<mlir::ParseResult()> parseElementFn) {
  if (parseElementFn())
    return mlir::failure();
  // Otherwise we have a list of comma separated elements.
  while (consumeIf(Token::comma)) {
    if (parseElementFn())
      return mlir::failure();
  }
  return mlir::success();
}

mlir::ParseResult Parser::expect(Token::Kind expected) {
  if (consumeIf(expected))
    return mlir::success();
  return mlir::failure();
}

std::string Parser::getTopLevelName() const {
  auto &sourceMgr = getSourceMgr();
  auto mainBuffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  mlir::StringRef bufferName = mainBuffer->getBufferIdentifier();
  if (bufferName.empty())
    bufferName = "qasm_unknown";
  else {
    bufferName = llvm::sys::path::filename(bufferName);
    bufferName.consume_back(".qasm");
  }
  return bufferName.str();
}
} // namespace cudaq