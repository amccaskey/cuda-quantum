/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "fermion_handler.h"
#include "common/FmtCore.h"

namespace cudaq::experimental::fermion {

std::size_t
fermion_handler::num_qubits(const details::operator_data &thisPtr) const {
  if (thisPtr.productTerms.empty())
    return 0;
  auto [data, coeffs] = thisPtr;
  return data[0].size() / 2;
}

/// @brief Add the given spin_op to this one and return *this
void fermion_handler::addAssign(details::operator_data &thisPtr,
                                const details::operator_data &v) {}

/// @brief Multiply the given spin_op with this one and return *this
void fermion_handler::multAssign(details::operator_data &thisPtr,
                                 const details::operator_data &v) {}

/// @brief Return true if this spin_op is equal to the given one. Equality
/// here does not consider the coefficients.
bool fermion_handler::checkEquality(const details::operator_data &thisPtr,
                                    const details::operator_data &v) const {
  return false;
}

std::string fermion_handler::to_string(const details::operator_data &thisPtr,
                                       bool printCoeffs) const {
  auto &[m_data, m_coefficients] = thisPtr;
  return "";
}

} // namespace cudaq::experimental::fermion