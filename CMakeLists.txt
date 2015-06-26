cmake_minimum_required(VERSION 3.1)
project(UndergraduateProj)

# Project Output Paths
SET (MAINFOLDER ${PROJECT_SOURCE_DIR})
SET (EXECUTABLE_OUTPUT_PATH "${MAINFOLDER}/bin")
SET (LIBRARY_OUTPUT_PATH "${MAINFOLDER}/lib")

# Add custom module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/tools/share/cmake/Modules/cmake-modules/")

find_package(OpenCV REQUIRED)
find_package(Boost
    COMPONENTS system filesystem program_options math_tr1
    REQUIRED)
find_package(Eigen REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
include_directories(${EIGEN_INCLUDE_DIRS})

add_subdirectory(src)
