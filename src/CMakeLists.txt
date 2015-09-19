# Register common source files
list(APPEND common_SRCS_RELATIVE "structure.cc")
list(APPEND common_SRCS_RELATIVE "file_io.cc")

foreach(filename IN LISTS common_SRCS_RELATIVE)
    list(APPEND common_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/${filename}")
endforeach(filename)

add_subdirectory(tracker)
add_subdirectory(clusterer)
