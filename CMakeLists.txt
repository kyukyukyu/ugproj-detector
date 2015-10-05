cmake_minimum_required(VERSION 3.1)
project(ugproj)

# For policy CMP0042
if(APPLE)
    set(CMAKE_MACOSX_RPATH 1)
endif(APPLE)

# Project Output Paths
SET (MAINFOLDER ${PROJECT_SOURCE_DIR})
SET (EXECUTABLE_OUTPUT_PATH "${MAINFOLDER}/bin")
SET (LIBRARY_OUTPUT_PATH "${MAINFOLDER}/lib")

# Add custom module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/tools/share/cmake/Modules/cmake-modules/")

find_package(OpenCV REQUIRED)
find_package(Boost
    COMPONENTS filesystem math_tr1 program_options system
    REQUIRED)
find_package(Eigen REQUIRED)

# Load uricamic/flandmark
set(BUILD_CPP_EXAMPLES OFF CACHE BOOL "flandmark: Build CPP examples")
set(BUILD_MATLAB_BINDINGS OFF CACHE BOOL "flandmark: Build MATLAB bindings")
add_subdirectory(thirdparty/flandmark)
find_package(flandmark REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${FLANDMARK_INCLUDE_DIRS})
link_directories(${FLANDMARK_LINK_DIRS})

add_subdirectory(src)
