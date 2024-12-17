/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/operator.h"

namespace cudaq::experimental::particle {

// data encoding for particle ops is
// QIDX0 1/0 QIDX1 1/0 QIDX2 1/0 ...
// QIDX0 1/0 QIDX1 1/0 QIDX2 1/0 ...
//

class particle_handler : public details::operator_handler {
public:
  details::operator_data initialize() const override;
  std::vector<double>
  get_data_representation(const details::operator_data &) const override {
    return {};
  }

  operator_matrix to_matrix(const details::operator_data &,
                            const parameter_map &p) const override;

  std::size_t num_qubits(const details::operator_data &thisPtr) const override;
  std::vector<operator_matrix>
  get_support_matrices(const details::operator_data &,
                       const parameter_map &p) const override;

  void add_assign(details::operator_data &thisPtr,
                  const details::operator_data &v) override;
  void mult_assign(details::operator_data &thisPtr,
                   const details::operator_data &v) override;

  bool check_equality(const details::operator_data &thisPtr,
                      const details::operator_data &v) const override;
  std::string to_string(const details::operator_data &thisPtr,
                        bool printCoeffs) const override;
};

using particle_op = operator_<particle_handler>;

particle_op create(std::size_t idx);
particle_op annihilate(std::size_t idx);

particle_op position(std::size_t idx);
particle_op momentum(std::size_t idx);

} // namespace cudaq::experimental::particle
