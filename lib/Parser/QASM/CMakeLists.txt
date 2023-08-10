add_mlir_library(OpenQASMParser
  Lexer.cpp
  Parser.cpp
  Token.cpp

  LINK_LIBS PUBLIC
  MLIRIR
)
