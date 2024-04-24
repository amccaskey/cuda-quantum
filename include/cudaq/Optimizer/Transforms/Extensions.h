/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "mlir/Dialect/Transform/IR/TransformDialect.h"

namespace cudaq {
class CudaqTransformExtensions
    : public mlir::transform::TransformDialectExtension<
          CudaqTransformExtensions> {
public:
  using Base::Base;
  void init();
};
} // namespace cudaq