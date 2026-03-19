cmake_minimum_required(VERSION 3.20)
project(zigrad_mlir_ext LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Expect caller to pass:
#  -DZG_SDK_INCLUDE=/path/to/sdk/include
#  -DZG_SDK_LIB=/path/to/sdk/lib
# Optional:
#  -DZG_LLVM_ROOT=/path/to/llvm-prefix (must contain bin/mlir-tblgen)
set(ZG_SDK_INCLUDE "" CACHE PATH "SDK include root")
set(ZG_SDK_LIB "" CACHE PATH "SDK lib dir")
set(ZG_LLVM_ROOT "" CACHE PATH "LLVM tools prefix")

if(ZG_SDK_INCLUDE STREQUAL "" OR ZG_SDK_LIB STREQUAL "")
  message(FATAL_ERROR "ZG_SDK_INCLUDE and ZG_SDK_LIB must be set")
endif()

if(ZG_LLVM_ROOT STREQUAL "")
  get_filename_component(_repo_root "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
  if(EXISTS "${_repo_root}/result-llvm/bin/mlir-tblgen")
    set(ZG_LLVM_ROOT "${_repo_root}/result-llvm")
  endif()
endif()

if(ZG_LLVM_ROOT STREQUAL "" OR NOT EXISTS "${ZG_LLVM_ROOT}/bin/mlir-tblgen")
  message(FATAL_ERROR "ZG_LLVM_ROOT must point to an LLVM prefix with bin/mlir-tblgen")
endif()

set(MLIR_TBLGEN "${ZG_LLVM_ROOT}/bin/mlir-tblgen")
set(MLIR_PDLL "${ZG_LLVM_ROOT}/bin/mlir-pdll")

function(find_versioned_library out_var base_name)
  set(candidates)

  file(GLOB sdk_libs
    "${ZG_SDK_LIB}/lib${base_name}.so"
    "${ZG_SDK_LIB}/lib${base_name}.so.*"
  )
  list(APPEND candidates ${sdk_libs})

  if(EXISTS "${ZG_LLVM_ROOT}/lib")
    file(GLOB llvm_libs
      "${ZG_LLVM_ROOT}/lib/lib${base_name}.so"
      "${ZG_LLVM_ROOT}/lib/lib${base_name}.so.*"
    )
    list(APPEND candidates ${llvm_libs})
  endif()

  list(LENGTH candidates n)
  if(n EQUAL 0)
    message(FATAL_ERROR "Could not find shared library for ${base_name}")
  endif()

  list(SORT candidates COMPARE STRING ORDER DESCENDING)
  list(GET candidates 0 picked)
  set(${out_var} "${picked}" PARENT_SCOPE)
endfunction()

set(GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/zigrad")
file(MAKE_DIRECTORY "${GEN_DIR}")

set(ZG_TD_FILE "${CMAKE_CURRENT_SOURCE_DIR}/zigrad/ZigradOps.td")

add_custom_command(
  OUTPUT
    "${GEN_DIR}/ZigradOps.h.inc"
    "${GEN_DIR}/ZigradOps.cpp.inc"
  COMMAND "${MLIR_TBLGEN}" "${ZG_TD_FILE}" --gen-op-decls
          -I "${ZG_LLVM_ROOT}/include"
          -I "${ZG_SDK_INCLUDE}"
          -I "${CMAKE_CURRENT_SOURCE_DIR}"
          -o "${GEN_DIR}/ZigradOps.h.inc"
  COMMAND "${MLIR_TBLGEN}" "${ZG_TD_FILE}" --gen-op-defs
          -I "${ZG_LLVM_ROOT}/include"
          -I "${ZG_SDK_INCLUDE}"
          -I "${CMAKE_CURRENT_SOURCE_DIR}"
          -o "${GEN_DIR}/ZigradOps.cpp.inc"
  DEPENDS "${ZG_TD_FILE}"
)

add_custom_target(zigrad_tblgen
  DEPENDS
    "${GEN_DIR}/ZigradOps.h.inc"
    "${GEN_DIR}/ZigradOps.cpp.inc"
)

# --- PDLL pattern compilation ---
set(PDLL_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/zigrad/patterns")
file(MAKE_DIRECTORY "${PDLL_GEN_DIR}")

set(DOT_ADD_PDLL "${CMAKE_CURRENT_SOURCE_DIR}/zigrad/patterns/dot_add.pdll")

add_custom_command(
  OUTPUT "${PDLL_GEN_DIR}/dot_add.pdll.cpp.inc"
  COMMAND "${MLIR_PDLL}" "${DOT_ADD_PDLL}" -x=cpp
          -o "${PDLL_GEN_DIR}/dot_add.pdll.cpp.inc"
  DEPENDS "${DOT_ADD_PDLL}"
  COMMENT "Compiling dot_add.pdll -> C++"
)

add_custom_target(zigrad_pdll
  DEPENDS "${PDLL_GEN_DIR}/dot_add.pdll.cpp.inc"
)

find_versioned_library(MLIR_C_LIB "MLIR-C")
find_versioned_library(STABLEHLO_CAPI_LIB "StablehloCAPI")
find_versioned_library(MLIR_IR_LIB "MLIRIR")
find_versioned_library(MLIR_PASS_LIB "MLIRPass")
find_versioned_library(MLIR_REWRITE_LIB "MLIRRewrite")
find_versioned_library(MLIR_TRANSFORMS_LIB "MLIRTransforms")
find_versioned_library(MLIR_FUNC_LIB "MLIRFuncDialect")
find_versioned_library(MLIR_PDL_LIB "MLIRPDLDialect")
find_versioned_library(MLIR_PDL_INTERP_LIB "MLIRPDLInterpDialect")
find_versioned_library(MLIR_PDL_TO_PDL_INTERP_LIB "MLIRPDLToPDLInterp")

add_library(zigrad_mlir_ext SHARED
  mlir_ext.cc
  zigrad/ZigradDialect.cpp
  zigrad/ZigradKernelLegalizePass.cpp
  zigrad/mirage/MirageKernelSelectPass.cpp
  zigrad/PdllKernelSelectPass.cpp
)
add_dependencies(zigrad_mlir_ext zigrad_tblgen zigrad_pdll)
target_compile_options(zigrad_mlir_ext PRIVATE -fno-rtti)

target_include_directories(zigrad_mlir_ext PRIVATE
  "${ZG_LLVM_ROOT}/include"
  "${ZG_SDK_INCLUDE}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/generated"
)

target_link_libraries(zigrad_mlir_ext PRIVATE
  "${MLIR_C_LIB}"
  "${STABLEHLO_CAPI_LIB}"
  "${MLIR_IR_LIB}"
  "${MLIR_PASS_LIB}"
  "${MLIR_REWRITE_LIB}"
  "${MLIR_TRANSFORMS_LIB}"
  "${MLIR_FUNC_LIB}"
  "${MLIR_PDL_LIB}"
  "${MLIR_PDL_INTERP_LIB}"
  "${MLIR_PDL_TO_PDL_INTERP_LIB}"
  stdc++
  dl
)

set_target_properties(zigrad_mlir_ext PROPERTIES
  OUTPUT_NAME "zigrad_mlir_ext"
  BUILD_RPATH "${ZG_SDK_LIB};${ZG_LLVM_ROOT}/lib"
  INSTALL_RPATH "\$ORIGIN"
)
