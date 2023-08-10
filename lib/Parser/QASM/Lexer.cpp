/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "Lexer.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"

using namespace cudaq;

static mlir::StringAttr
getMainBufferNameIdentifier(const llvm::SourceMgr &sourceMgr,
                            mlir::MLIRContext *context) {
  auto mainBuffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  mlir::StringRef bufferName = mainBuffer->getBufferIdentifier();
  if (bufferName.empty())
    bufferName = "<unknown>";
  return mlir::StringAttr::get(context, bufferName);
}

Lexer::Lexer(mlir::MLIRContext *context, const llvm::SourceMgr &sourceMgr)
    : context(context), sourceMgr(sourceMgr),
      bufferNameIdentifier(getMainBufferNameIdentifier(sourceMgr, context)) {
  auto bufferID = sourceMgr.getMainFileID();
  curBuffer = sourceMgr.getMemoryBuffer(bufferID)->getBuffer();
  curPtr = curBuffer.begin();
}

// Return the next token in the buffer. If this is the end of buffer, it
// return the EOF token.
Token Lexer::nextToken() {
lex_next_token:
  eatWhitespace();
  const char *tokenStart = curPtr;
  switch (*curPtr++) {
  case 0:
    return createToken(Token::eof, tokenStart);

  case '\r':
    if (*curPtr == '\n')
      ++curPtr;
    [[fallthrough]];
  case '\n':
    goto lex_next_token;

  case '/':
    if (*curPtr == '/') {
      eatComment();
      goto lex_next_token;
    }
    return createToken(Token::slash, tokenStart);

  // clang-format off
  case '0': case '1': case '2': case '3': case '4':
  case '5': case '6': case '7': case '8': case '9':
    return eatNumber(tokenStart);
    // clang-format on

  case 'C':
    if (*curPtr == 'X') {
      ++curPtr;
      return createToken(Token::kw_CX, tokenStart);
    }
    break;

  case 'U':
    return createToken(Token::kw_U, tokenStart);

  // clang-format off
  case 'O':
  case 'a': case 'b': case 'c': case 'd': case 'e': case 'f': case 'g':
  case 'h': case 'i': case 'j': case 'k': case 'l': case 'm': case 'n':
  case 'o': case 'p': case 'q': case 'r': case 's': case 't': case 'u':
  case 'v': case 'w': case 'x': case 'y': case 'z':
    return eatIdentifier(tokenStart);
    // clang-format on

  case '[':
    return createToken(Token::l_square, tokenStart);

  case ']':
    return createToken(Token::r_square, tokenStart);

  case '(':
    return createToken(Token::l_paren, tokenStart);

  case ')':
    return createToken(Token::r_paren, tokenStart);

  case '{':
    return createToken(Token::l_brace, tokenStart);

  case '}':
    return createToken(Token::r_brace, tokenStart);

  case '*':
    return createToken(Token::star, tokenStart);

  case '+':
    return createToken(Token::plus, tokenStart);

  case '-':
    if (*curPtr == '>') {
      ++curPtr;
      return createToken(Token::arrow, tokenStart);
    }
    return createToken(Token::minus, tokenStart);

  case '^':
    return createToken(Token::caret, tokenStart);

  case ';':
    return createToken(Token::semicolon, tokenStart);

  case '=':
    if (*curPtr == '=') {
      ++curPtr;
      return createToken(Token::equalequal, tokenStart);
    }
    break;

  case ',':
    return createToken(Token::comma, tokenStart);

  case '"':
    return eatString(tokenStart);

  default:
    break;
  }
  return createToken(Token::error, tokenStart);
}

/// Encode the specified source location information into a Location object
/// for attachment to the IR or error reporting.
mlir::Location Lexer::translateLocation(mlir::SMLoc loc) {
  unsigned mainFileID = sourceMgr.getMainFileID();
  auto lineAndColumn = sourceMgr.getLineAndColumn(loc, mainFileID);
  return mlir::FileLineColLoc::get(context, bufferNameIdentifier,
                                   lineAndColumn.first, lineAndColumn.second);
}

Token Lexer::emitError(const char *loc, const llvm::Twine &message) {
  llvm::SMLoc smLoc = llvm::SMLoc::getFromPointer(loc);
  auto &sourceMgr = getSourceMgr();
  sourceMgr.PrintMessage(smLoc, llvm::SourceMgr::DK_Error, message);
  return createToken(Token::error, loc);
}

// Match [_A-Za-z0-9]*, we have already matched [a-z]
Token Lexer::eatIdentifier(const char *tokenStart) {
  while (std::isalpha(*curPtr) || std::isdigit(*curPtr) || *curPtr == '_') {
    ++curPtr;
  }
  // Check to see if this identifier is a keyword.
  llvm::StringRef spelling(tokenStart, curPtr - tokenStart);
  Token::Kind kind = llvm::StringSwitch<Token::Kind>(spelling)
#define TOK_KEYWORD(SPELLING) .Case(#SPELLING, Token::kw_##SPELLING)
#include "TokenKinds.def"
                         .Default(Token::identifier);

  return Token(kind, spelling);
}

// Match [.0-9]*, we have already matched [0-9]
// TODO: handle exponents
Token Lexer::eatNumber(const char *tokenStart) {
  assert(std::isdigit(curPtr[-1]));

  while (std::isdigit(*curPtr)) {
    ++curPtr;
  }
  if (*curPtr != '.') {
    return createToken(Token::integer, tokenStart);
  }
  ++curPtr;
  while (std::isdigit(*curPtr)) {
    ++curPtr;
  }
  return createToken(Token::real, tokenStart);
}

Token Lexer::eatString(const char *tokStart) {
  assert(curPtr[-1] == '"');

  while (true) {
    switch (*curPtr++) {
    case '"':
      return createToken(Token::string, tokStart);

    case 0:
    case '\n':
    case '\v':
    case '\f':
      return emitError(curPtr - 1, "expected '\"' in string literal");

    case '\\':
      if (*curPtr == '"' || *curPtr == '\\' || *curPtr == 'n' || *curPtr == 't')
        ++curPtr;
      else
        return emitError(curPtr - 1, "unknown escape in string literal");
      continue;

    default:
      continue;
    }
  }
}

// We have just read the // characters from input.  Eat until we find the
// newline or EOF character thats terminate the comment.
void Lexer::eatComment() {
  assert(*curPtr == '/' && curPtr[-1] == '/');

  while (*curPtr != 0 && *curPtr != '\n' && *curPtr != '\r') {
    ++curPtr;
  }
}

void Lexer::eatWhitespace() {
  if ((*curPtr == ' ') || (*curPtr == '\t')) {
    ++curPtr;
    while ((*curPtr == ' ') || (*curPtr == '\t'))
      ++curPtr;
  }
}
