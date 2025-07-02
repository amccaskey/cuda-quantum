# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ctad-maybe-unsupported")
set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

# Create the CUDA-Q Library
add_library(${LIBRARY_NAME} 
         SHARED cudaq.cpp 
                target_control.cpp
                algorithms/draw.cpp
                algorithms/evolve.cpp
                algorithms/schedule.cpp
                qis/state.cpp
                utils/cudaq_utils.cpp
                utils/matrix.cpp
                distributed/mpi_plugin.cpp
                platformv2/platform.cpp 
                platformv2/config.cpp
                platformv2/qpus/qpu_state.cpp
                platformv2/qpus/remote_state.cpp
                )

set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS ${LIBRARY_NAME})

target_include_directories(${LIBRARY_NAME}
  PUBLIC $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tpls/eigen>
  PRIVATE .)

target_link_libraries(${LIBRARY_NAME}
  PUBLIC dl cudaq-operator cudaq-common cudaq-nlopt cudaq-ensmallen
  PRIVATE fmt::fmt-header-only CUDAQTargetConfigUtil)

if (CUDA_FOUND)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_include_directories(${LIBRARY_NAME}
    PRIVATE $<BUILD_INTERFACE:${CUDAToolkit_INCLUDE_DIRS}>
  )
  target_link_libraries(${LIBRARY_NAME}
    PRIVATE CUDA::cudart_static)
  target_compile_definitions(${LIBRARY_NAME} PRIVATE CUDAQ_HAS_CUDA)
endif()

add_subdirectory(platformv2)
add_subdirectory(algorithms)
add_subdirectory(builder)
add_subdirectory(domains)
add_subdirectory(operators)

install(TARGETS ${LIBRARY_NAME} EXPORT cudaq-targets DESTINATION lib)

install(EXPORT cudaq-targets
        FILE CUDAQTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/cudaq)

# Install CUDAQ MPI interface header, reference implementation, and build script
# for manual activation on a local system if needed.
install(
  FILES
      distributed/distributed_capi.h
      distributed/builtin/mpi_comm_impl.cpp
      distributed/builtin/activate_custom_mpi.sh
  DESTINATION distributed_interfaces
  PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
)
