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
    COMPONENTS filesystem math_tr1 program_options system
    REQUIRED)
find_package(Eigen REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
include_directories(${EIGEN_INCLUDE_DIRS})

# Build jbeder/yaml-cpp
add_subdirectory(thirdparty/yaml-cpp)
include_directories("${MAINFOLDER}/thirdparty/yaml-cpp/include")

add_subdirectory(src)
