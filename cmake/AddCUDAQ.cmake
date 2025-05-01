# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This file is derived from                                                    #
# https://github.com/llvm/circt/blob/main/cmake/modules/AddCIRCT.cmake         #
# CIRCT is an LLVM incubator project under Apache License 2.0 with LLVM        #
# Exceptions.                                                                  #
# ============================================================================ #

include_guard()

function(add_cudaq_dialect dialect dialect_namespace)
  set(LLVM_TARGET_DEFINITIONS ${dialect}Dialect.td)
  mlir_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls -dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect=${dialect_namespace})
  add_public_tablegen_target(${dialect}DialectIncGen)
  set(LLVM_TARGET_DEFINITIONS ${dialect}Ops.td)
  mlir_tablegen(${dialect}Ops.h.inc -gen-op-decls)
  mlir_tablegen(${dialect}Ops.cpp.inc -gen-op-defs)
  add_public_tablegen_target(${dialect}OpsIncGen)
  set(LLVM_TARGET_DEFINITIONS ${dialect}Types.td)
  mlir_tablegen(${dialect}Types.h.inc -gen-typedef-decls -typedefs-dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs -typedefs-dialect=${dialect_namespace})
  add_public_tablegen_target(${dialect}TypesIncGen)
  add_dependencies(cudaq-headers
    ${dialect}DialectIncGen ${dialect}OpsIncGen ${dialect}TypesIncGen)
endfunction()

function(add_cudaq_interface interface)
  set(LLVM_TARGET_DEFINITIONS ${interface}.td)
  mlir_tablegen(${interface}.h.inc -gen-op-interface-decls)
  mlir_tablegen(${interface}.cpp.inc -gen-op-interface-defs)
  add_public_tablegen_target(${interface}IncGen)
  add_dependencies(cudaq-headers ${interface}IncGen)
endfunction()

function(add_cudaq_doc tablegen_file output_path command)
  set(LLVM_TARGET_DEFINITIONS ${tablegen_file}.td)
  string(MAKE_C_IDENTIFIER ${output_path} output_id)
  tablegen(MLIR ${output_id}.md ${command} ${ARGN})
  set(GEN_DOC_FILE ${CUDAQ_BINARY_DIR}/docs/${output_path}.md)
  add_custom_command(
    OUTPUT ${GEN_DOC_FILE}
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/${output_id}.md
    ${GEN_DOC_FILE}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_id}.md)
  add_custom_target(${output_id}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(cudaq-doc ${output_id}DocGen)
endfunction()

function(add_cudaq_dialect_doc dialect dialect_namespace)
  add_cudaq_doc(${dialect} Dialects/${dialect} -gen-dialect-doc -dialect ${dialect_namespace})
endfunction()

function(add_cudaq_library name)
  add_mlir_library(${ARGV} DISABLE_INSTALL)
  add_cudaq_library_install(${name})
endfunction()

# Adds a CUDA Quantum dialect library target for installation. This should normally
# only be called from add_cudaq_dialect_library().
function(add_cudaq_library_install name)
  install(TARGETS ${name} COMPONENT ${name} EXPORT CUDAQTargets)
  set_property(GLOBAL APPEND PROPERTY CUDAQ_ALL_LIBS ${name})
  set_property(GLOBAL APPEND PROPERTY CUDAQ_EXPORTS ${name})
endfunction()

function(add_cudaq_dialect_library name)
  set_property(GLOBAL APPEND PROPERTY CUDAQ_DIALECT_LIBS ${name})
  add_cudaq_library(${ARGV} DEPENDS cudaq-headers)
endfunction()

function(add_cudaq_translation_library name)
  set_property(GLOBAL APPEND PROPERTY CUDAQ_TRANSLATION_LIBS ${name})
  add_cudaq_library(${ARGV} DEPENDS cudaq-headers)
endfunction()

function(add_target_config name)
  install(FILES ${name}.yml DESTINATION targets)
  configure_file(${name}.yml ${CMAKE_BINARY_DIR}/targets/${name}.yml COPYONLY)
endfunction()

function(add_target_mapping_arch providerName name)
  install(FILES ${name} DESTINATION targets/mapping/${providerName})
  configure_file(${name} ${CMAKE_BINARY_DIR}/targets/mapping/${providerName}/${name} COPYONLY)
endfunction()

#[=======================================================================[.rst:
cudaq_sanitize_plugin_for_python
---------------------------------
Creates a Python-compatible version of a shared library by removing problematic 
dependencies that may conflict with Python runtime environments.
Synopsis
^^^^^^^^
.. code-block:: cmake
  cudaq_sanitize_plugin_for_python(<target_name>
                                   [REMOVE_DEPENDENCIES <dep1> <dep2> ...])
Description
^^^^^^^^^^^
This function creates a sanitized copy of a shared library target specifically 
for use in Python environments. It uses ``patchelf`` to remove specified dynamic 
library dependencies that might cause conflicts or issues when the library is 
loaded in Python.
The function creates:
* A copy of the original shared library with a ``libpy-`` prefix
* An imported CMake target named ``py-<target_name>``
* A build target ``py-<target_name>-build`` to ensure the sanitized library is built
Parameters
^^^^^^^^^^
``<target_name>``
  The name of the existing shared library target to sanitize. Must be a valid
  CMake target of type ``SHARED_LIBRARY``.
``REMOVE_DEPENDENCIES <dep1> <dep2> ...``
  Optional list of dynamic library dependencies to remove from the target.
  If not specified, defaults to removing ``libcudaq-mlir-runtime.so``.
Requirements
^^^^^^^^^^^^
* ``patchelf`` must be installed and available in the system PATH
* The target must be a shared library (``SHARED_LIBRARY`` type)
* Linux environment (uses ``.so`` library extension)
Behavior
^^^^^^^^
1. Validates the input target exists and is a shared library
2. Copies the original library to ``${CMAKE_BINARY_DIR}/lib/libpy-<output_name>.so``
3. Uses ``patchelf --remove-needed`` to strip specified dependencies
4. Creates an imported target that can be used like any other CMake target
5. Preserves interface include directories from the original target
Example Usage
^^^^^^^^^^^^^
.. code-block:: cmake
  # Create Python-compatible version with default dependency removal
  cudaq_sanitize_plugin_for_python(my_cudaq_plugin)
  # Remove specific dependencies
  cudaq_sanitize_plugin_for_python(my_cudaq_plugin
    REMOVE_DEPENDENCIES 
      libcudaq-mlir-runtime.so
      libproblematic-dep.so
  )
  # Use the sanitized target
  target_link_libraries(my_python_module PRIVATE py-my_cudaq_plugin)
Notes
^^^^^
This function is particularly useful for CUDA-Q plugins that need to be loaded
in Python environments where certain MLIR runtime dependencies might conflict
with Python's own runtime or other loaded modules.
#]=======================================================================]
function(cudaq_sanitize_plugin_for_python TARGET_NAME)
    # Parse arguments to accept a list of dependencies to remove
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs "REMOVE_DEPENDENCIES")
    cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Default to current behavior if no dependencies specified
    if(NOT ARGS_REMOVE_DEPENDENCIES)
        set(ARGS_REMOVE_DEPENDENCIES "libcudaq-mlir-runtime.so")
    endif()

    # Validate input
    if(NOT TARGET ${TARGET_NAME})
        message(FATAL_ERROR "Target ${TARGET_NAME} does not exist")
    endif()

    # Check that target is a shared library
    get_target_property(TARGET_TYPE ${TARGET_NAME} TYPE)
    if(NOT TARGET_TYPE STREQUAL "SHARED_LIBRARY")
        message(FATAL_ERROR "Target ${TARGET_NAME} is not a shared library")
    endif()

    # Find required tools
    find_program(PATCHELF_EXECUTABLE patchelf)
    if(NOT PATCHELF_EXECUTABLE)
        message(FATAL_ERROR "patchelf not found. Install with: apt-get install patchelf (Ubuntu/Debian) or yum install patchelf (RHEL/CentOS)")
    endif()

    # Get target properties at configure time
    get_target_property(OUTPUT_NAME ${TARGET_NAME} OUTPUT_NAME)
    if(NOT OUTPUT_NAME)
        set(OUTPUT_NAME ${TARGET_NAME})
    endif()

    # Construct library name with proper suffix
    set(LIB_SUFFIX ".so")

    set(ORIGINAL_LIB_NAME "lib${OUTPUT_NAME}${LIB_SUFFIX}")
    set(PY_LIB_NAME "libpy-${OUTPUT_NAME}${LIB_SUFFIX}")
    set(PY_LIB_PATH "${CMAKE_BINARY_DIR}/lib/${PY_LIB_NAME}")

    # Build the command list - start with the copy command
    set(ALL_COMMANDS 
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${TARGET_NAME}> ${PY_LIB_PATH}
    )

    # Add removal commands for each dependency in the specified order
    foreach(DEP ${ARGS_REMOVE_DEPENDENCIES})
        list(APPEND ALL_COMMANDS COMMAND ${PATCHELF_EXECUTABLE} --remove-needed ${DEP} ${PY_LIB_PATH})
    endforeach()

    # Create the modified library
    add_custom_command(
        OUTPUT ${PY_LIB_PATH}
        ${ALL_COMMANDS}
        DEPENDS ${TARGET_NAME}
        COMMENT "Creating Python-compatible ${PY_LIB_NAME} from ${TARGET_NAME} (removing: ${ARGS_REMOVE_DEPENDENCIES})"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        VERBATIM
    )

    # Create imported target
    add_library(py-${TARGET_NAME} SHARED IMPORTED GLOBAL)
    set_target_properties(py-${TARGET_NAME} PROPERTIES
        IMPORTED_LOCATION ${PY_LIB_PATH}
    )

    # Ensure the library gets built - this creates the actual CMake target
    add_custom_target(py-${TARGET_NAME}-build ALL DEPENDS ${PY_LIB_PATH})
    add_dependencies(py-${TARGET_NAME} py-${TARGET_NAME}-build)

    # Copy interface properties from original target
    get_target_property(INTERFACE_INCLUDE_DIRS ${TARGET_NAME} INTERFACE_INCLUDE_DIRECTORIES)
    if(INTERFACE_INCLUDE_DIRS)
        set_target_properties(py-${TARGET_NAME} PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${INTERFACE_INCLUDE_DIRS}"
        )
    endif()

    install(FILES ${PY_LIB_PATH} DESTINATION lib)

endfunction()
