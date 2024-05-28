cmake_minimum_required(VERSION 3.10)

# Set the project name
project(cpp_mlp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include imgui backends)

set(SOURCES
    src/mlp.cc
    src/data.cc
    src/data_handler.cc
    backends/imgui_impl_glfw.cpp
    # backends/imgui_impl_opengl3_loader.cpp
    backends/imgui_impl_opengl3.cpp
    imgui/imgui_demo.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
)

add_executable(mlp ${SOURCES})