cmake_minimum_required(VERSION 3.10)

project(FastCornerDev)

find_package(Halide REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(t main.cpp)
target_link_libraries(t PRIVATE Halide::Halide Halide::Tools Halide::ImageIO ${OpenCV_LIBS})
