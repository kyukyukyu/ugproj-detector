list(APPEND Celiu_Optflow_HEADERS "gaussian_pyramid.h")
list(APPEND Celiu_Optflow_HEADERS "image.h")
list(APPEND Celiu_Optflow_HEADERS "image_file.h")
list(APPEND Celiu_Optflow_HEADERS "image_pyramid.h")
list(APPEND Celiu_Optflow_HEADERS "image_util.h")
list(APPEND Celiu_Optflow_HEADERS "optical_flow.h")
list(APPEND Celiu_Optflow_SRCS "opfl_test.cc")
list(APPEND Celiu_Optflow_SRCS "optical_flow.cc")
list(APPEND Celiu_Optflow_SRCS "gaussian_pyramid.cc")
list(APPEND Celiu_Optflow_SRCS "image_file.cc")

cmake_minimum_required(VERSION 2.8)
project(celiu_optflow_lib)

add_library(celiu_optflow ${Celiu_Optflow_SRCS})

find_package(Glog REQUIRED)
find_package(PNG REQUIRED)
find_package(Eigen REQUIRED)
set(Dependencies_INCLUDE_DIRS
    ${GLOG_INCLUDE_DIRS}
    ${PNG_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIRS}
)
include_directories(${Dependencies_INCLUDE_DIRS})
set(Celiu_Optflow_Dependencies_INCLUDE_DIRS ${Dependencies_INCLUDE_DIRS} PARENT_SCOPE)

target_link_libraries(celiu_optflow ${GLOG_LIBRARIES})
target_link_libraries(celiu_optflow ${PNG_LIBRARY})

install(TARGETS celiu_optflow DESTINATION lib)
install(FILES ${Celiu_Optflow_HEADERS} DESTINATION include)
