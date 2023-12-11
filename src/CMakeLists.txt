add_library(
    ${LIBRARY_NAME} Function.cpp
                    BaseAlgorithm.cpp
                    SteepestDescent.cpp 
                    BFGS.cpp 
                    LineSearch.cpp 
                    LineSearchBackTrack.cpp
                    LineSearchNocedal.cpp 
)

target_include_directories(
    ${LIBRARY_NAME}
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)
