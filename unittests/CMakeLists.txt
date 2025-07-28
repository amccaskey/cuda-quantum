# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set (CMAKE_CXX_FLAGS
     "${CMAKE_CXX_FLAGS} -Wno-attributes -Wno-ctad-maybe-unsupported")
# clear any flags for static linking
SET(CMAKE_EXE_LINKER_FLAGS "")
SET(CMAKE_SHARED_LINKER_FLAGS "")
set (CMAKE_ENABLE_EXPORTS TRUE)
set (CUDAQ_BACKEND_CONFIG_GLUE ${CMAKE_SOURCE_DIR}/tools/nvqpp/backendConfig.cpp)

add_subdirectory(plugin)

file (GLOB 
  CUDAQ_RUNTIME_TEST_SOURCES 
    integration/*.cpp 
    qir/*.cpp 
    qis/*.cpp 
    common/*.cpp
)
create_simulator_runtime_test(test_runtime 
  EXTRA_SOURCES runtime_tester.cpp
  DISABLE_INTEGRATION TRUE
)

# Create runtime test for the QPP backend
create_simulator_runtime_test(qpp 
    EXTRA_SOURCES backends/QPPTester.cpp 
    FP_TYPE FP64
    BACKEND_CONFIG "qpp-cpu"
    EXTRA_LIBS libqpp
)

# Create runtime test for the QPP Density Matrix backend
create_simulator_runtime_test(dm 
    EXTRA_SOURCES backends/QPPDMTester.cpp 
    FP_TYPE FP64
    BACKEND_CONFIG "density-matrix-cpu"
    EXTRA_LIBS libqpp
)

# Create runtime test for the Stim backend
create_simulator_runtime_test(stim ""
    FP_TYPE FP64
    BACKEND_CONFIG "stim"
)

# Create an executable for SpinOp UnitTests
add_executable(test_spin main.cpp spin_op/SpinOpTester.cpp operators/utils.cpp)
target_link_libraries(test_spin
  PRIVATE 
  cudaq
  cudaq-operator
  gtest_main)
target_include_directories(test_spin PRIVATE operators)
gtest_discover_tests(test_spin)

# Create an executable for operators UnitTests
set(CUDAQ_OPERATOR_TEST_SOURCES
   operators/utils.cpp
   operators/scalar_op.cpp
   operators/matrix_op.cpp
   operators/spin_op.cpp
   operators/boson_op.cpp
   operators/fermion_op.cpp
   operators/conversions.cpp
   operators/product_op.cpp
   operators/sum_op.cpp
   operators/rydberg_hamiltonian.cpp
   operators/manipulation.cpp
)
add_executable(test_operators main.cpp ${CUDAQ_OPERATOR_TEST_SOURCES})
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_operators PRIVATE -Wl,--no-as-needed)
endif()
target_link_libraries(test_operators
  PRIVATE
  cudaq-operator
  cudaq
  gtest_main)
gtest_discover_tests(test_operators)

create_simulator_runtime_test(test_utils 
  EXTRA_SOURCES utils/UtilsTester.cpp utils/Matrix.cpp 
  DISABLE_INTEGRATION TRUE
)

# build the test qudit execution manager
add_subdirectory(qudit)
create_simulator_runtime_test(test_qudit 
   EXTRA_SOURCES qudit/SimpleQuditTester.cpp
   DISABLE_INTEGRATION TRUE 
   QPU_LIB cudaq-qpu-qudit 
   TARGET_OPTIONS "--target-option qpu:SimpleQuditSimulator"
)

# build the test photonics execution manager
create_simulator_runtime_test(test_photonics 
   EXTRA_SOURCES photonics/PhotonicsTester.cpp
   DISABLE_INTEGRATION TRUE 
   QPU_LIB cudaq-qpu-photonics 
   TARGET_OPTIONS "--target-option qpu:PhotonicsQPU"
)

add_executable(test_extension_point extension/test_extension_point.cpp)
target_link_libraries(test_extension_point PRIVATE GTest::gtest_main cudaq)
gtest_discover_tests(test_extension_point)

add_subdirectory(Optimizer)
# FIXME add these back
# add_subdirectory(backends)
add_subdirectory(output_record)

if (CUDA_FOUND) 
  if (CUSTATEVEC_ROOT)
    # Create runtime test for the custatevec fp32 backend
    create_simulator_runtime_test(custatevec_fp32 
        FP_TYPE FP32
        BACKEND_CONFIG "nvidia"
        GPU_REQUIRED TRUE
    )

    # Create unit test for the MQPU support
    create_simulator_runtime_test(test_mqpu 
        BACKEND_CONFIG "nvidia"
        EXTRA_SOURCES mqpu/mqpu_tester.cpp
        GPU_REQUIRED TRUE 
        DISABLE_INTEGRATION TRUE
        TARGET_OPTIONS "--target-option mqpu" 
    )
  endif()
  

  # Test CUDAQ_OBSERVE_FROM_SAMPLING=ON mode.
  # (term-by-term expectation value calculation by applying change-of-basis gates then reverting them)
  # FIXME AJM Refactoring, this does not seem to pass along the environment variable. 
  # I suspect its not doing it on main either.
  create_simulator_runtime_test(test_custatevec_observe_from_sampling 
      BACKEND_CONFIG "nvidia" 
      EXTRA_SOURCES 
        integration/builder_tester.cpp
        integration/deuteron_variational_tester.cpp
        integration/gradient_tester.cpp
        integration/nlopt_tester.cpp
      DISABLE_INTEGRATION TRUE 
      GPU_REQUIRED TRUE 
      ENV_VARS "CUDAQ_OBSERVE_FROM_SAMPLING=1"
  )

  if (MPI_CXX_FOUND)
    # Count the number of GPUs
    find_program(NVIDIA_SMI "nvidia-smi")
    if(NVIDIA_SMI)
      execute_process(COMMAND bash -c "nvidia-smi --list-gpus | wc -l" OUTPUT_VARIABLE NGPUS)
      # Only build this test if we have more than 1 GPU
      if (${NGPUS} GREATER_EQUAL 2)
        create_simulator_runtime_test(test_mpi 
          BACKEND_CONFIG "nvidia"
          EXTRA_SOURCES mqpu/mpi_mqpu_tester.cpp
          GPU_REQUIRED TRUE 
          # MGPUS REQUIRED?
          DISABLE_INTEGRATION TRUE 
          TARGET_OPTIONS "--target-option mqpu"
        )
        configure_file("mqpu/run_mpi.sh.in"
                    "${CMAKE_BINARY_DIR}/unittests/run_mpi.sh" @ONLY)
        add_test(NAME MPITest COMMAND ${MPIEXEC} --allow-run-as-root -np 2 bash ${CMAKE_BINARY_DIR}/unittests/run_mpi.sh)
        set_tests_properties(MPITest PROPERTIES LABELS "gpu_required;mgpus_required")
      endif()
    endif()
  endif()

  create_simulator_runtime_test(test_gpu_get_state 
       BACKEND_CONFIG "nvidia"
       TARGET_OPTIONS "--target-option fp64"
       GPU_REQUIRED TRUE 
       DISABLE_INTEGRATION TRUE 
       EXTRA_SOURCES gpu/get_state_tester.cu 
    )
  target_compile_options(test_gpu_get_state PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -Wno-attributes>
  )

  find_package(CUDAToolkit REQUIRED)

  find_library(CUTENSOR_LIB
    NAMES   cutensor libcutensor.so.2
    HINTS
        ${CUTENSOR_ROOT}/lib64
        ${CUTENSOR_ROOT}/lib
        ${CUTENSOR_ROOT}/lib64/${CUDAToolkit_VERSION_MAJOR}
        ${CUTENSOR_ROOT}/lib/${CUDAToolkit_VERSION_MAJOR}
  )

  find_library(CUTENSORNET_LIB
    NAMES   cutensornet libcutensornet.so.2
    HINTS   
        ${CUTENSORNET_ROOT}/lib64
        ${CUTENSORNET_ROOT}/lib
        ${CUTENSORNET_ROOT}/lib64/${CUDAToolkit_VERSION_MAJOR}
        ${CUTENSORNET_ROOT}/lib/${CUDAToolkit_VERSION_MAJOR}
        ${CUDENSITYMAT_ROOT}/lib/
)

  # Create an executable for dynamics UnitTests
  create_simulator_runtime_test(test_dynamics 
     EXTRA_SOURCES 
      dynamics/test_RungeKuttaIntegrator.cpp
      dynamics/test_CuDensityMatState.cpp
      dynamics/test_CuDensityMatTimeStepper.cpp
      dynamics/test_CuDensityMatExpectation.cpp
      dynamics/test_EvolveSingle.cpp
      dynamics/test_EvolveApi.cpp
      dynamics/test_EvolveBatchedImpl.cpp
      dynamics/test_BatchedEvolveApi.cpp
      dynamics/test_BatchingDetection.cpp
    DISABLE_INTEGRATION TRUE 
    EXTRA_OPTIONS -DCUDAQ_ANALOG_TARGET
    BACKEND_CONFIG "dynamics" 
    GPU_REQUIRED TRUE
    EXTRA_LIBS  
      ${CUDENSITYMAT_ROOT}/lib/libcudensitymat.so.0
      CUDA::cudart
      CUDA::cublas
      CUDA::cusparse
      ${CUTENSOR_LIB}
      ${CUTENSORNET_LIB}
  )

  # Multi-QPU evolve_async test
  create_simulator_runtime_test(test_evolve_async 
     EXTRA_SOURCES mqpu/dynamics_async_tester.cpp 
     GPU_REQUIRED TRUE 
     #MGPU REQUIRED?
     DISABLE_INTEGRATION TRUE 
     TARGET_OPTIONS "--target-option mqpu"
     BACKEND_CONFIG "dynamics"
     EXTRA_OPTIONS -DCUDAQ_ANALOG_TARGET
     EXTRA_LIBS 
       ${CUDENSITYMAT_ROOT}/lib/libcudensitymat.so.0
       CUDA::cudart
       CUDA::cublas
       ${CUTENSORNET_LIB}
       CUDA::cusparse
       ${CUTENSOR_LIB}
    )
endif() 

if (MPI_CXX_FOUND)
  set(NUM_PROCS 4)
  create_simulator_runtime_test(test_mpi_plugin 
    EXTRA_SOURCES mpi/mpi_tester.cpp
    DISABLE_INTEGRATION TRUE 
    EXTRA_OPTIONS "-DNUM_PROCS=${NUM_PROCS}"
    HAS_MAIN_CPP TRUE 
    SKIP_GTEST_DISCOVER TRUE
  )
  # Check if `--allow-run-as-root` is supported (OpenMPI)
  # Note: MPICH doesn't need `--allow-run-as-root`.
  execute_process(COMMAND ${MPIEXEC} --allow-run-as-root -np ${NUM_PROCS} hostname ERROR_VARIABLE CHECK_ALLOW_RUN_AS_ROOT_RESULTS ERROR_STRIP_TRAILING_WHITESPACE OUTPUT_QUIET)
  if ("${CHECK_ALLOW_RUN_AS_ROOT_RESULTS}" STREQUAL "")
    set(MPI_EXEC_CMD_ARGS "--allow-run-as-root")
  endif()

  add_test(NAME MPIApiTest COMMAND ${MPIEXEC} ${MPI_EXEC_CMD_ARGS} -np ${NUM_PROCS} ${CMAKE_BINARY_DIR}/unittests/test_mpi_plugin)
endif()


# Make it so we can get function symbols
set (CMAKE_ENABLE_EXPORTS TRUE)

# AJM Aug 23 2025 - Thien can help with this...

# if(TARGET nvqir-tensornet)
#   message(STATUS "Building cutensornet backend tests.")
#   create_tests_with_backend(tensornet "")
#   create_tests_with_backend(tensornet-mps "")
#   create_tests_with_backend(tensornet-fp32 "")
#   create_tests_with_backend(tensornet-mps-fp32 "")
#   if (MPI_CXX_FOUND)
#     # Count the number of GPUs
#     find_program(NVIDIA_SMI "nvidia-smi")
#     if(NVIDIA_SMI)
#       execute_process(COMMAND bash -c "nvidia-smi --list-gpus | wc -l" OUTPUT_VARIABLE NGPUS)
#       # Only build this test if we have more than 1 GPUs
#       if (${NGPUS} GREATER_EQUAL 2)
#         message(STATUS "Building cutensornet MPI tests.")
#         add_executable(test_tensornet_mpi mpi/tensornet_mpi_tester.cpp)
#         if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
#           target_link_options(test_tensornet_mpi PRIVATE -Wl,--no-as-needed)
#         endif()
#         target_link_libraries(test_tensornet_mpi
#             PRIVATE
#             cudaq
#             fmt::fmt-header-only
#             cudaq-platform-default
#             nvqir-tensornet
#             gtest)
#         add_test(NAME TensornetMPITest COMMAND ${MPIEXEC} --allow-run-as-root -np 2 ${CMAKE_BINARY_DIR}/unittests/test_tensornet_mpi)
#         set_tests_properties(TensornetMPITest PROPERTIES LABELS "gpu_required;mgpus_required")
#       endif() # NGPUS
#     endif() # NVIDIA_SMI
#   endif() # MPI_CXX_FOUND

#   # Test CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE=ON mode (on a few test cases that have cudaq::observe)
#   add_executable(test_tensornet_observe_path_reuse
#     integration/builder_tester.cpp
#     integration/deuteron_variational_tester.cpp
#     integration/observe_result_tester.cpp
#     integration/noise_tester.cpp # This test contains noisy observe test cases.
#   )
#   target_include_directories(test_tensornet_observe_path_reuse PRIVATE .)
#   target_compile_definitions(test_tensornet_observe_path_reuse
#                              PRIVATE -DNVQIR_BACKEND_NAME=tensornet)
#   if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
#     target_link_options(test_tensornet_observe_path_reuse PRIVATE -Wl,--no-as-needed)
#   endif()
#   target_link_libraries(test_tensornet_observe_path_reuse
#     PRIVATE
#     cudaq
#     cudaq-builder
#     cudaq-platform-default
#     nvqir-tensornet
#     gtest_main)
#   # Run this test with "CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE=TRUE"
#   gtest_discover_tests(test_tensornet_observe_path_reuse TEST_SUFFIX _PathReuse PROPERTIES ENVIRONMENT "CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE=ON" PROPERTIES LABELS "gpu_required")
# endif()

# AJM Aug 23, 2025 - These things are all part of CUDA-QX now, we should drop them 

# if (CUDAQ_ENABLE_PYTHON)
#   if (NOT Python_FOUND)
#     message(FATAL_ERROR "find_package(Python) not run?")
#   endif()
#   execute_process(COMMAND ${Python_EXECUTABLE} -c "import openfermionpyscf"
#     OUTPUT_VARIABLE PYSCF_output
#     ERROR_VARIABLE  PYSCF_error
#     RESULT_VARIABLE PYSCF_result)

#   if(NOT ${PYSCF_result} EQUAL 0)
#     message(STATUS "OpenFermion PySCF not available for chemistry tests.")
#     return()
#   endif()

#   message(STATUS "OpenFermion PySCF found, enabling chemistry tests.")
#   add_executable(test_domains main.cpp domains/ChemistryTester.cpp)
#   if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
#     target_link_options(test_domains PRIVATE -Wl,--no-as-needed)
#   endif()
#   target_compile_definitions(test_domains PRIVATE -DNVQIR_BACKEND_NAME=qpp -DCUDAQ_SIMULATION_SCALAR_FP64)
#   target_include_directories(test_domains PRIVATE .)
#   set_property(TARGET test_domains PROPERTY ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python")
#   target_link_libraries(test_domains
#     PRIVATE
#     cudaq
#     cudaq-platform-default
#     nvqir nvqir-qpp
#     cudaq-operator
#     cudaq-chemistry
#     cudaq-pyscf
#     gtest_main)
#   gtest_discover_tests(test_domains
#     TEST_SUFFIX _Sampling PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python")
# endif()
