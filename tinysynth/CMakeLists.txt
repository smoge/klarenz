cmake_minimum_required(VERSION 3.16)
project(TinySynth VERSION 0.1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Use Clang
set(CMAKE_CXX_COMPILER clang++)

# Find required packages
find_package(PkgConfig REQUIRED)

# JACK handling
pkg_check_modules(JACK jack)
if(NOT JACK_FOUND)
    message(STATUS "JACK not found via pkg-config, trying alternative method")
    find_library(JACK_LIBRARIES NAMES jack)
    find_path(JACK_INCLUDE_DIRS NAMES jack/jack.h)
    if(JACK_LIBRARIES AND JACK_INCLUDE_DIRS)
        set(JACK_FOUND TRUE)
    else()
        message(FATAL_ERROR "JACK library not found. Please install JACK development files.")
    endif()
endif()

# JACK include directory (update this path if necessary)
set(JACK_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include/external/jack2/common)
message(STATUS "Using JACK include directory: ${JACK_INCLUDE_DIR}")

# OpenGL handling
if(UNIX AND NOT APPLE)
    pkg_check_modules(GLX REQUIRED gl)
    set(OPENGL_LIBRARIES ${GLX_LIBRARIES})
    set(OPENGL_INCLUDE_DIRS ${GLX_INCLUDE_DIRS})
else()
    find_package(OpenGL REQUIRED)
endif()

find_package(glfw3 REQUIRED)

# LLVM handling
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# ImGui directory (update this path if necessary)
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include/external/imgui)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${JACK_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
    ${JACK_INCLUDE_DIR}
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${LLVM_INCLUDE_DIRS}
)

# Add ImGui source files
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# Collect all source files except main.cpp
file(GLOB_RECURSE LIB_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
list(REMOVE_ITEM LIB_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

# Create a library with all your code
add_library(TinySynth STATIC ${LIB_SOURCES} ${IMGUI_SOURCES})

# Link libraries to your library
target_link_libraries(TinySynth PRIVATE
    ${JACK_LIBRARIES}
    ${OPENGL_LIBRARIES}
    glfw
    ${LLVM_LIBRARIES}
)

# Include directories for your library
target_include_directories(TinySynth PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${JACK_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${CMAKE_CURRENT_BINARY_DIR}
    ${LLVM_INCLUDE_DIRS}
)

# Create main executable
add_executable(TinySynthExe ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(TinySynthExe PRIVATE TinySynth)

# Add test executable
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    file(GLOB_RECURSE TEST_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp
    )
    add_executable(run_tests ${TEST_SOURCES})
    target_link_libraries(run_tests PRIVATE TinySynth)
    # Add test
    enable_testing()
    add_test(NAME AllTests COMMAND run_tests)
endif()

# Check for C++20 features
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
    #include <concepts>
    #include <ranges>
    #include <span>
    int main() { return 0; }
" HAVE_CPP20_FEATURES)
if(NOT HAVE_CPP20_FEATURES)
    message(FATAL_ERROR "Your compiler does not support the required C++20 features.")
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${JACK_INCLUDE_DIR}")

# LLVM flags
llvm_map_components_to_libnames(llvm_libs support core irreader ipo passes)
target_link_libraries(TinySynth PRIVATE ${llvm_libs})