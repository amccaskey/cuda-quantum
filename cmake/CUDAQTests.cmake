# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#------------------------------------------------------------------------------
# create_simulator_runtime_test(<backend> [OPTIONS ...])
#
# Create a GoogleTest-based runtime test executable for a given quantum
# simulator backend. This function streamlines the setup of integration and
# unit tests for CUDA-Q backends by configuring sources, compile definitions,
# libraries, and test discovery.
#
# Arguments:
#   <backend>
#     The name of the simulator backend (e.g., qpp, stim, custatevec_fp32).
#     Used to name the test target and set backend-specific definitions.
#
# Options (all optional, specified as KEY VALUE pairs):
#   EXTRA_SOURCES <files...>
#     Additional source files to include in the test executable.
#
#   FP_TYPE <FP32|FP64>
#     Floating-point precision type for simulation (sets compile definitions).
#
#   BACKEND_CONFIG <string>
#     String to pass as a backend configuration (sets compile definitions).
#
#   EXTRA_LIBS <libraries...>
#     Additional libraries to link with the test executable.
#
#   TARGET_OPTIONS <string>
#     Backend-specific target options, passed as a base64-encoded string in a
#     compile definition.
#
#   EXTRA_OPTIONS <options...>
#     Additional compile options for the test target.
#
#   ENV_VARS <env_vars>
#     Environment variables to set when running the tests (passed to
#     gtest_discover_tests).
#
#   QPU_LIB <library>
#     Override the default QPU simulator library to link.
#
#   HAS_MAIN_CPP <bool>
#     If TRUE, do not add main.cpp as a source (assume EXTRA_SOURCES includes a main).
#
#   DISABLE_INTEGRATION <bool>
#     If TRUE, do not add the default integration test sources.
#
#   GPU_REQUIRED <bool>
#     If TRUE, label the test as requiring a GPU.
#
# Usage Example:
#   create_simulator_runtime_test(qpp
#     EXTRA_SOURCES backends/QPPTester.cpp
#     FP_TYPE FP64
#     BACKEND_CONFIG "qpp-cpu"
#     EXTRA_LIBS libqpp
#   )
#
# This creates a test executable named "test_runtime_qpp" with the specified
# sources, floating-point type, backend configuration, and linked libraries.
#
#------------------------------------------------------------------------------
function(create_simulator_runtime_test backend)
    # Parse optional arguments for FP type and backend config
    cmake_parse_arguments(ARG
        "DISABLE_INTEGRATION;HAS_MAIN_CPP;GPU_REQUIRED;SKIP_GTEST_DISCOVER"   # Options (flags)
        "FP_TYPE;BACKEND_CONFIG;QPU_LIB;TARGET_OPTIONS" # One-value keywords
        "EXTRA_SOURCES;EXTRA_LIBS;EXTRA_OPTIONS;ENV_VARS"          # Multi-value keywords
        ${ARGN}
    )
    if (NOT ARG_DISABLE_INTEGRATION)
       set(test_name "test_runtime_${backend}")
    else() 
       set(test_name "${backend}")
    endif() 

    # Convert to uppercase
    string(TOUPPER "${backend}" backend_upper)

    # Create the definition name
    set(backend_define "CUDAQ_BACKEND_${backend_upper}")
    set (CUDAQ_BACKEND_CONFIG_GLUE ${CMAKE_SOURCE_DIR}/tools/nvqpp/backendConfig.cpp)
    
    add_executable(${test_name} 
        ${CUDAQ_BACKEND_CONFIG_GLUE} 
        ${ARG_EXTRA_SOURCES}
    )
    
    if (NOT ARG_HAS_MAIN_CPP) 
      target_sources(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/unittests/main.cpp)
    endif()

    if (NOT ARG_DISABLE_INTEGRATION)
      target_sources(${test_name} PRIVATE ${CUDAQ_RUNTIME_TEST_SOURCES})
    endif() 

    target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/unittests .)
    target_compile_definitions(${test_name} PRIVATE 
        -DNVQIR_BACKEND_NAME=${backend} -D${backend_define}
    )
    
    # Add FP precision flag if specified
    if(ARG_FP_TYPE)
        target_compile_definitions(${test_name} PRIVATE
            -DCUDAQ_SIMULATION_SCALAR_${ARG_FP_TYPE}
        )
    endif()
    
    # Add backend config if specified
    if(ARG_BACKEND_CONFIG)
        string(REPLACE "," ";" backend_config_semicolon "${ARG_BACKEND_CONFIG}")
        string(REPLACE ";" "\\;" backend_config_escaped "${backend_config_semicolon}")
        target_compile_definitions(${test_name} PRIVATE
            "-DNVQPP_TARGET_BACKEND_CONFIG=\"${backend_config_escaped}\""
        )
    endif()
    
    if (ARG_TARGET_OPTIONS) 
      # Run base64 on the string, capturing output with no newline
      execute_process(
         COMMAND bash -c "echo -n '${ARG_TARGET_OPTIONS}' | base64 --wrap=0"
         OUTPUT_VARIABLE BASE64_ENCODED
         OUTPUT_STRIP_TRAILING_WHITESPACE
      )

      set(BASE64_ARG "base64_${BASE64_ENCODED}")
      target_compile_definitions(${test_name} PRIVATE -DNVQPP_TARGET_OPTIONS="${BASE64_ARG}")
    endif() 

    if (ARG_EXTRA_OPTIONS)
      foreach(item ${ARG_EXTRA_OPTIONS})
        target_compile_options(${test_name} PRIVATE ${item})
      endforeach()
    endif()
    
    # Default to our circuit simulators
    set(QPU_LIB cudaq-qpu-simulator)
    # Allow the client to change this though
    if (ARG_QPU_LIB)
      set(QPU_LIB ${ARG_QPU_LIB})
    endif()
    target_link_libraries(${test_name} PRIVATE 
        nvqir # For NVQIRTester.cpp
        cudaq 
        ${QPU_LIB} 
        fmt::fmt-header-only 
        cudaq-builder 
        gtest_main
        ${ARG_EXTRA_LIBS}
    )
    
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
      target_link_options(${test_name} PRIVATE -Wl,--no-as-needed)
    endif()

    if (NOT ARG_SKIP_GTEST_DISCOVER)
      if (NOT ARG_GPU_REQUIRED)
        gtest_discover_tests(${test_name} PROPERTIES ENVIRONMENT ${ARG_ENV_VARS})
      else()
        gtest_discover_tests(${test_name} PROPERTIES LABELS "gpu_required" PROPERTIES ENVIRONMENT "${ARG_ENV_VARS}")
      endif()
    endif()
    
endfunction()