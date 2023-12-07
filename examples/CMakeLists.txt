add_executable(LinearAlgebra LinearAlgebra.cpp)
target_include_directories(
    LinearAlgebra
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

add_executable(Rosenbrock Rosenbrock.cpp)
target_link_libraries(
    Rosenbrock
    PRIVATE ${LIBRARY_NAME}
)
