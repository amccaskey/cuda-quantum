
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/PDLL/AST/Context.h"
#include "mlir/Tools/PDLL/AST/Nodes.h"
#include "mlir/Tools/PDLL/CodeGen/CPPGen.h"
#include "mlir/Tools/PDLL/CodeGen/MLIRGen.h"
#include "mlir/Tools/PDLL/ODS/Context.h"
#include "mlir/Tools/PDLL/Parser/Parser.h"

#include <filesystem>
#include <set>

using namespace mlir;
using namespace mlir::pdll;

/// @brief Retrieve the path of this executable, borrowed from
/// the Clang Driver
std::string getExecutablePath(const char *argv0, bool canonicalPrefixes) {
  if (!canonicalPrefixes) {
    SmallString<128> executablePath(argv0);
    if (!llvm::sys::fs::exists(executablePath))
      if (llvm::ErrorOr<std::string> p =
              llvm::sys::findProgramByName(executablePath))
        executablePath = *p;
    return std::string(executablePath.str());
  }
  void *p = (void *)(intptr_t)getExecutablePath;
  return llvm::sys::fs::getMainExecutable(argv0, p);
}

int main(int argc, char **argv) {
  // FIXME: This is necessary because we link in TableGen, which defines its
  // options as static variables.. some of which overlap with our options.
  llvm::cl::ResetCommandLineParser();

  llvm::cl::opt<bool> install("install", llvm::cl::desc(""),
                              llvm::cl::init(false));

  llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"),
      llvm::cl::value_desc("filename"));

  llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));

  llvm::cl::list<std::string> includeDirs(
      "I", llvm::cl::desc("Directory of include files"),
      llvm::cl::value_desc("directory"), llvm::cl::Prefix);

  llvm::cl::opt<std::string> inputSplitMarker{
      "split-input-file", llvm::cl::ValueOptional,
      llvm::cl::callback([&](const std::string &str) {
        // Implicit value: use default marker if flag was used without value.
        if (str.empty())
          inputSplitMarker.setValue(kDefaultSplitMarker);
      }),
      llvm::cl::desc("Split the input file into chunks using the given or "
                     "default marker and process each chunk independently"),
      llvm::cl::init("")};
  llvm::cl::opt<std::string> outputSplitMarker(
      "output-split-marker",
      llvm::cl::desc("Split marker to use for merging the ouput"),
      llvm::cl::init(kDefaultSplitMarker));

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "CUDAQ Optimizer Compiler");

  // Set up the input file.
  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> inputFile =
      openInputFile(inputFilename, &errorMessage);
  if (!inputFile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  // We need the location of this cudaq-quake executable so that we can get the
  // install path
  std::string executablePath = getExecutablePath(argv[0], true);
  std::filesystem::path cudaqQuakePath{executablePath};
  auto installBinPath = cudaqQuakePath.parent_path();
  auto cudaqInstallPath = installBinPath.parent_path();

  auto cudaqCompilerInstallPath = cudaqInstallPath / "passes";
  if (!std::filesystem::exists(cudaqCompilerInstallPath))
    std::filesystem::create_directories(cudaqCompilerInstallPath);

  // If we are creating a dependency file, we'll also need to track what files
  // get included during processing.
  std::set<std::string> includedFilesStorage;
  std::set<std::string> *includedFiles = nullptr;

  // The split-input-file mode is a very specific mode that slices the file
  // up into small pieces and checks each independently.
  std::string outputStr;
  llvm::raw_string_ostream outputStrOS(outputStr);
  llvm::SourceMgr sourceMgr;
  sourceMgr.setIncludeDirs(includeDirs);
  sourceMgr.AddNewSourceBuffer(std::move(inputFile), SMLoc());

  ods::Context odsContext;
  ast::Context astContext(odsContext);
  FailureOr<ast::Module *> module = parsePDLLAST(astContext, sourceMgr, false);
  if (failed(module))
    return 1;

  // Add the files that were included to the set.
  if (includedFiles) {
    for (unsigned i = 1, e = sourceMgr.getNumBuffers(); i < e; ++i) {
      includedFiles->insert(
          sourceMgr.getMemoryBuffer(i + 1)->getBufferIdentifier().str());
    }
  }

  MLIRContext mlirContext;
  auto pdlModule =
      codegenPDLLToMLIR(&mlirContext, astContext, sourceMgr, **module);
  if (!pdlModule)
    return 1;

  if (outputFilename == "-") {
    pdlModule->print(outputStrOS, OpPrintingFlags().enableDebugInfo());
    llvm::outs() << outputStr;
    return 0;
  }

  if (install) {
    std::string buffer;
    llvm::raw_string_ostream ostream(buffer);
    if (failed(writeBytecodeToFile(pdlModule.get(), ostream))) {
      llvm::errs() << "Failed to write bytecode\n";
      return 1;
    }
    auto n = (cudaqCompilerInstallPath / outputFilename.getValue()).string();
    llvm::outs() << "Writing to passes file " << n << "\n";
    std::unique_ptr<llvm::ToolOutputFile> outputFile = openOutputFile(
        (cudaqCompilerInstallPath / outputFilename.getValue()).string(),
        &errorMessage);
    if (!outputFile) {
      llvm::errs() << errorMessage << "\n";
      return 1;
    }
    outputFile->os() << buffer;
    outputFile->keep();
  }
  return 0;
}