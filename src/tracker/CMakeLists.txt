# Register source files
list(APPEND tracker_SRCS "detector.cc")
list(APPEND tracker_SRCS "associator.cc")
list(APPEND tracker_SRCS "face_tracker.cc")
list(APPEND tracker_SRCS "main.cc")

set(tracker_LIBS ${Boost_LIBRARIES} ${OpenCV_LIBS} ${FLANDMARK_LIBRARIES})
set(tracker_BIN tracker)

add_executable(${tracker_BIN} ${common_SRCS} ${tracker_SRCS})
target_link_libraries(${tracker_BIN} ${tracker_LIBS})
set_target_properties(${tracker_BIN} PROPERTIES
    OUTPUT_NAME ${tracker_BIN})

install(TARGETS ${tracker_BIN} DESTINATION bin)

# We use C++11.
set_target_properties(${tracker_BIN} PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON)
