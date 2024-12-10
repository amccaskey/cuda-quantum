/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/operator.h"

namespace cudaq::experimental::fermion {

class fermion_handler : public details::operator_handler {
public:
  std::size_t num_qubits(const details::operator_data &thisPtr) const override;

  void addAssign(details::operator_data &thisPtr,
                 const details::operator_data &v) override;
  void multAssign(details::operator_data &thisPtr,
                  const details::operator_data &v) override;

  bool checkEquality(const details::operator_data &thisPtr,
                     const details::operator_data &v) const override;
  std::string to_string(const details::operator_data &thisPtr,
                        bool printCoeffs) const override;
};

using fermion_op = quantum_operator<fermion_handler>;

fermion_op create(std::size_t idx);
fermion_op annihilate(std::size_t idx);

} // namespace cudaq::experimental::fermion