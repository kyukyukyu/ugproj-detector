# Register source files
list(APPEND project_SRCS "structure.hpp")
list(APPEND project_SRCS "structure.cpp")
list(APPEND project_SRCS "detector/detector.hpp")
list(APPEND project_SRCS "detector/detector.cpp")
list(APPEND project_SRCS "associator/associator.hpp")
list(APPEND project_SRCS "associator/associator.cpp")
list(APPEND project_SRCS "optflow/flow_to_color.hpp")
list(APPEND project_SRCS "optflow/flow_to_color.cpp")
list(APPEND project_SRCS "optflow/colorcode.h")
list(APPEND project_SRCS "optflow/colorcode.cpp")
list(APPEND project_SRCS "optflow/manager.hpp")
list(APPEND project_SRCS "optflow/manager.cpp")
list(APPEND project_SRCS "main.cpp")
set(project_LIBS ${Boost_LIBRARIES} ${OpenCV_LIBS} celiu_optflow)
set(project_BIN ${PROJECT_NAME})

# Load celiu-optflow library
add_subdirectory(celiu-optflow)
include_directories(${Celiu_Optflow_Dependencies_INCLUDE_DIRS})

add_executable(${project_BIN} ${project_SRCS})
target_link_libraries(${project_BIN} ${project_LIBS})
set_target_properties(${project_BIN} PROPERTIES OUTPUT_NAME ${project_BIN})

# Register source files for lucas-kanade
list(APPEND lk_SRCS "structure.cpp")
list(APPEND lk_SRCS "detector/detector.cpp")
list(APPEND lk_SRCS "associator/associator.cpp")
list(APPEND lk_SRCS "file_io.cc")
list(APPEND lk_SRCS "face_tracker.cc")
list(APPEND lk_SRCS "lucas_kanade.cc")
set(lk_LIBS ${Boost_LIBRARIES} ${OpenCV_LIBS})
set(lk_BIN lucas-kanade)

add_executable(${lk_BIN} ${lk_SRCS})
target_link_libraries(${lk_BIN} ${project_LIBS})
set_target_properties(${lk_BIN} PROPERTIES
    OUTPUT_NAME ${lk_BIN}
    # Since Ce Liu's optical flow is not used here, this macro should be
    # defined.
    COMPILE_DEFINITIONS "UGPROJ_SUPPRESS_CELIU")

install(TARGETS ${project_BIN} DESTINATION bin)

# We use C++11.
foreach(t ${project_BIN} ${lk_BIN})
    set_target_properties(${t} PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON)
endforeach()
