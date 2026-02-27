cmake_minimum_required(VERSION 3.20)
project(zigrad_mlir_ext LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Expect caller to pass:
#  -DZG_SDK_INCLUDE=/path/to/sdk/include
#  -DZG_SDK_LIB=/path/to/sdk/lib
set(ZG_SDK_INCLUDE "" CACHE PATH "SDK include root")
set(ZG_SDK_LIB "" CACHE PATH "SDK lib dir")

if(ZG_SDK_INCLUDE STREQUAL "" OR ZG_SDK_LIB STREQUAL "")
  message(FATAL_ERROR "ZG_SDK_INCLUDE and ZG_SDK_LIB must be set")
endif()

add_library(zigrad_mlir_ext SHARED mlir_ext.cc)
target_include_directories(zigrad_mlir_ext PRIVATE "${ZG_SDK_INCLUDE}")
target_link_directories(zigrad_mlir_ext PRIVATE "${ZG_SDK_LIB}")

# Link against MLIR and StableHLO C API shared libraries from the SDK.
target_link_libraries(zigrad_mlir_ext PRIVATE
  MLIR-C
  StablehloCAPI
  stdc++
  dl
)

set_target_properties(zigrad_mlir_ext PROPERTIES OUTPUT_NAME "zigrad_mlir_ext")
