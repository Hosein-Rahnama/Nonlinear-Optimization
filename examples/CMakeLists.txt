set(EXAMPLE "LinearAlgebra")
add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
target_include_directories(
    ${EXAMPLE}
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

set(EXAMPLE "Rosenbrock")
add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
target_link_libraries(
    ${EXAMPLE}
    PRIVATE ${LIBRARY_NAME}
)

set(EXAMPLE "Quadratic")
add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
target_link_libraries(
    ${EXAMPLE}
    PRIVATE ${LIBRARY_NAME}
)

set(EXAMPLE "Trigonometric")
add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
target_link_libraries(
    ${EXAMPLE}
    PRIVATE ${LIBRARY_NAME}
)

set(EXAMPLE "Chebyquad")
add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
target_link_libraries(
    ${EXAMPLE}
    PRIVATE ${LIBRARY_NAME}
)
