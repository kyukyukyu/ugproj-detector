# Register source files
list(APPEND project_SRCS "structure.hpp")
list(APPEND project_SRCS "structure.cpp")
list(APPEND project_SRCS "main.cpp")
set(project_LIBS ${Boost_LIBRARIES} ${OpenCV_LIBS} celiu_optflow)
set(project_BIN ${PROJECT_NAME})

# Load celiu-optflow library
add_subdirectory(celiu-optflow)
include_directories(${Celiu_Optflow_Dependencies_INCLUDE_DIRS})

add_executable(${project_BIN} ${project_SRCS})
target_link_libraries(${project_BIN} ${project_LIBS})
set_target_properties(${project_BIN} PROPERTIES OUTPUT_NAME ${project_BIN})

install(TARGETS ${project_BIN} DESTINATION bin)
