# Register source files
list(APPEND project_SRCS "structure.hpp")
list(APPEND project_SRCS "structure.cpp")
list(APPEND project_SRCS "main.cpp")

cmake_minimum_required(VERSION 2.8)
project(UndergraduateProj)

# Add custom module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(OpenCV REQUIRED)
find_package(Boost
    COMPONENTS system filesystem program_options
    REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

add_executable(FaceDetector ${project_SRCS})
target_link_libraries(FaceDetector ${OpenCV_LIBS})
target_link_libraries(FaceDetector ${Boost_LIBRARIES})
