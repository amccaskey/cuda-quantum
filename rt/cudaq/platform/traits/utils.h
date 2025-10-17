#pragma once 

#include <vector> 
#include <string> 

namespace cudaq::traits {

/// \brief Metadata describing a quantum operation.
struct operation_metadata {
  static inline const std::vector<double>
      empty{}; ///< Static empty parameter vector.

  const std::string &name;               ///< Name of the operation.
  const std::vector<double> &parameters; ///< Operation parameters.
  bool isAdjoint = false; ///< Whether this operation is an adjoint.

  /// \brief Construct metadata for a parameterless operation.
  /// \param n Name of the operation.
  operation_metadata(const std::string &n) : name(n), parameters(empty) {}

  /// \brief Construct metadata for a parameterized operation.
  /// \param n Name of the operation.
  /// \param p Operation parameters.
  operation_metadata(const std::string &n, const std::vector<double> &p)
      : name(n), parameters(p) {}

  /// \brief Construct metadata for a parameterized operation with adjoint
  /// flag. \param n Name of the operation. \param p Operation parameters.
  /// \param isadj True if this is an adjoint operation.
  operation_metadata(const std::string &n, const std::vector<double> &p,
                     bool isadj)
      : name(n), parameters(p), isAdjoint(isadj) {}
};
}