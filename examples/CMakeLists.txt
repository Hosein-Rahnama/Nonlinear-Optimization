add_executable(MatrixMultiplication MatrixMultiplication.cpp)
target_include_directories(
    MatrixMultiplication
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

add_executable(Rosenbrock Rosenbrock.cpp)
target_link_libraries(
    Rosenbrock
    PRIVATE ${LIBRARY_NAME}
)