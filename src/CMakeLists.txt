# Register source files
list(APPEND project_SRCS "structure.cc")
list(APPEND project_SRCS "detector/detector.cc")
list(APPEND project_SRCS "associator/associator.cc")
list(APPEND project_SRCS "clusterer/clusterer.cc")
list(APPEND project_SRCS "clusterer/visualizer.cc")
list(APPEND project_SRCS "file_io.cc")
list(APPEND project_SRCS "face_tracker.cc")
list(APPEND project_SRCS "main.cc")

set(project_LIBS ${Boost_LIBRARIES} ${OpenCV_LIBS})
set(project_BIN ${PROJECT_NAME})

add_executable(${project_BIN} ${project_SRCS})
target_link_libraries(${project_BIN} ${project_LIBS})
set_target_properties(${project_BIN} PROPERTIES
    OUTPUT_NAME ${project_BIN}
    # Since Ce Liu's optical flow is not used here, this macro should be
    # defined.
    COMPILE_DEFINITIONS "UGPROJ_SUPPRESS_CELIU")

install(TARGETS ${project_BIN} DESTINATION bin)

# We use C++11.
set_target_properties(${project_BIN} PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON)
