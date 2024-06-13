# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #


set(LIBRARY_NAME cudaq2)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ctad-maybe-unsupported")
set(INTERFACE_POSITION_INDEPENDENT_CODE ON)
add_library(target_config_empty OBJECT target_config.cpp)
set (CMAKE_ENABLE_EXPORTS TRUE)

add_link_options(-Wl,--no-as-needed)
# Create the CUDA-Q Library
add_library(${LIBRARY_NAME} 
         SHARED cudaq.cpp 
                simulators/qpp/QppCircuitSimulator.cpp
                spin/spin_op.cpp
                spin/matrix.cpp
                qis/execution_manager.cpp 
                qis/managers/default/DefaultExecutionManager.cpp
                utils/cudaq_utils.cpp 
                utils/Logger.cpp
                utils/MeasureCounts.cpp 
                utils/NoiseModel.cpp
                utils/Trace.cpp
                utils/Future.cpp
                platform/quantum_platform.cpp
                platform/default/DefaultQuantumPlatform.cpp
                platform/common/QuantumExecutionQueue.cpp
                )

set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS ${LIBRARY_NAME})

if (CUDA_FOUND)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  target_sources(${LIBRARY_NAME} PRIVATE 
     simulators/custatevec/CuStateVecCircuitSimulator.cu
     simulators/custatevec/CuStateVecCircuitSimulatorF32.cu
     )

  string(REPLACE "." ";" CUDA_VERSION_LIST ${CMAKE_CUDA_COMPILER_VERSION})
  list(GET CUDA_VERSION_LIST 0 CUDA_MAJOR_VERSION)

  find_library(CUSTATEVEC_LIB
      NAMES custatevec
      HINTS
      ${CUSTATEVEC_ROOT}/lib64
      ${CUSTATEVEC_ROOT}/lib
      ${CUSTATEVEC_ROOT}/lib64/${CUDA_MAJOR_VERSION}
      ${CUSTATEVEC_ROOT}/lib/${CUDA_MAJOR_VERSION})

  get_filename_component(CUSTATEVEC_LIB_DIR ${CUSTATEVEC_LIB} DIRECTORY)
  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${CUSTATEVEC_LIB_DIR}")

  target_include_directories(${LIBRARY_NAME}
    PUBLIC $<INSTALL_INTERFACE:include>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/qruntime>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tpls/eigen>
            $<BUILD_INTERFACE:${CUDAToolkit_INCLUDE_DIRS}>
            $<BUILD_INTERFACE:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
            $<BUILD_INTERFACE:${CUSTATEVEC_ROOT}/include>)

  target_link_libraries(${LIBRARY_NAME}
    PUBLIC dl 
    PRIVATE fmt::fmt-header-only 
            CUDA::cudart_static 
            spdlog::spdlog 
            libqpp 
            ${CUSTATEVEC_LIB}
            ${CUDA_LIBRARIES} 
            cublas
            target_config_empty)

  target_compile_definitions(${LIBRARY_NAME} PRIVATE CUDAQ_HAS_CUDA)
else()
  target_include_directories(${LIBRARY_NAME}
    PUBLIC $<INSTALL_INTERFACE:include>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/qruntime>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tpls/eigen>
    PRIVATE .)

  target_link_libraries(${LIBRARY_NAME}
    PUBLIC dl
    PRIVATE fmt::fmt-header-only 
            spdlog::spdlog 
            libqpp 
            target_config_empty)
endif()

install(TARGETS ${LIBRARY_NAME} EXPORT cudaq2-targets DESTINATION lib)

install(EXPORT cudaq2-targets
        FILE CUDAQTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/cudaq2)

add_subdirectory(simulators/targets)
# Install CUDAQ MPI interface header, reference implementation, and build script
# for manual activation on a local system if needed.
# install(
#   FILES
#       distributed/distributed_capi.h
#       distributed/builtin/mpi_comm_impl.cpp
#       distributed/builtin/activate_custom_mpi.sh
#   DESTINATION distributed_interfaces
#   PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
# )
