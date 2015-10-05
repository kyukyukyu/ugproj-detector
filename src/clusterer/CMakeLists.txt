# Register source files
list(APPEND clusterer_SRCS "clusterer.cc")
list(APPEND clusterer_SRCS "vectorizer.cc")
list(APPEND clusterer_SRCS "visualizer.cc")
list(APPEND clusterer_SRCS "main.cc")

set(clusterer_LIBS ${Boost_LIBRARIES} ${OpenCV_LIBS} ${FLANDMARK_LIBRARIES})
set(clusterer_BIN clusterer)

add_executable(${clusterer_BIN} ${common_SRCS} ${clusterer_SRCS})
target_link_libraries(${clusterer_BIN} ${clusterer_LIBS})
set_target_properties(${clusterer_BIN} PROPERTIES
    OUTPUT_NAME ${clusterer_BIN})

install(TARGETS ${clusterer_BIN} DESTINATION bin)

# We use C++11.
set_target_properties(${clusterer_BIN} PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON)
