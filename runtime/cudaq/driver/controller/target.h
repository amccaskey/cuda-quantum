/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/utils/extension_point.h"

namespace cudaq::config {
class TargetConfig;
}

namespace cudaq::driver {

class target : public extension_point<target> {
public:
  virtual void initialize(const config::TargetConfig &) = 0;
  
  // loading calibrated waveforms? 

  // allocating static hamiltonian terms 
  virtual void allocate(std::size_t num) = 0;
  virtual void deallocate(std::size_t num) = 0; 
  
  // allocating time dynamic hamiltonian terms
//   virtual drive_line allocate_drive(...) = 0; 
//   virtual readout_line allocate_readout(...) = 0; 

  virtual void apply_opcode(const std::string &opCode,
                            const std::vector<double> &params,
                            const std::vector<std::size_t> &qudits) = 0;
  
//   virtual void drive(...) = 0; 
//   virtual void modulate(...) = 0; 
  

//   virtual ReadoutResult readout(...) = 0; 

  virtual std::size_t measure_z(std::size_t qudit) = 0; 
};

} // namespace cudaq::driver