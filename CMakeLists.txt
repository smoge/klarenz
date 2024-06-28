cmake_minimum_required(VERSION 3.16)
project(Klarenz_TinySynth VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(tinysynth)

find_program(STACK_EXECUTABLE stack)
if(NOT STACK_EXECUTABLE)
    message(FATAL_ERROR "Stack not found.")
endif()

add_custom_target(klarenz
    COMMAND ${STACK_EXECUTABLE} build
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/klarenz
    COMMENT "Building Klarenz")

add_custom_target(test_klarenz
    COMMAND ${STACK_EXECUTABLE} test
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/klarenz
    COMMENT "Testing Klarenz")

# Make the C++ build depend on the Haskell build
# add_dependencies(tinysynth klarenz)

# Add a custom target to build everything
# add_custom_target(build_all
#     DEPENDS tinysynth klarenz
#     COMMENT "Building all components")

# Add a custom target to run all tests
# add_custom_target(test_all
#     DEPENDS test_klarenz
#     COMMAND ${CMAKE_CTEST_COMMAND}
#     COMMENT "Running all tests")