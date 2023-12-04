add_library(
    ${LIBRARY_NAME} SteepestDescent.cpp BFGS.cpp LineSearchNocedal.cpp BaseAlgorithm.cpp
)

target_include_directories(
    ${LIBRARY_NAME}
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)