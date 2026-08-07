cmake_minimum_required(VERSION 3.25)

project(zigrad_mirage_adapter LANGUAGES CXX)

include(GNUInstallDirs)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(MIRAGE_ROOT "" CACHE PATH "Mirage installation prefix")

find_path(
  MIRAGE_INCLUDE_DIR
  NAMES mirage/kernel/graph.h
  HINTS "${MIRAGE_ROOT}"
  PATH_SUFFIXES include
  REQUIRED
)
find_library(
  MIRAGE_RUNTIME_LIBRARY
  NAMES mirage_runtime
  HINTS "${MIRAGE_ROOT}"
  PATH_SUFFIXES lib
  REQUIRED
)
find_path(
  NLOHMANN_JSON_INCLUDE_DIR
  NAMES nlohmann/json.hpp
  REQUIRED
)
find_package(CUDAToolkit REQUIRED)

add_library(zigrad_mirage SHARED mirage.cc)
target_compile_features(zigrad_mirage PRIVATE cxx_std_17)
target_compile_definitions(
  zigrad_mirage
  PRIVATE
    MIRAGE_BACKEND_USE_CUDA
    MIRAGE_FINGERPRINT_USE_CUDA
    MIRAGE_USE_FORMAL_VERIFIER
)
target_include_directories(
  zigrad_mirage
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE
    "${MIRAGE_INCLUDE_DIR}"
    "${NLOHMANN_JSON_INCLUDE_DIR}"
)
target_link_libraries(
  zigrad_mirage
  PRIVATE
    "${MIRAGE_RUNTIME_LIBRARY}"
    CUDA::cudart
)
set_target_properties(
  zigrad_mirage
  PROPERTIES
    CXX_EXTENSIONS OFF
    OUTPUT_NAME zigrad_mirage
)

install(
  TARGETS zigrad_mirage
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
install(
  DIRECTORY include/
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
