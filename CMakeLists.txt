# Project's name
PROJECT ( g_test C CXX )

# Specify the minimum version for CMake

cmake_minimum_required(VERSION 3.1)

add_definitions(-std=c++14 -Wall -Wextra -g)

find_package(GTest REQUIRED)
find_package (Eigen3 REQUIRED NO_MODULE)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(GSL REQUIRED)
find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS -std=c++14 -g)

# Set the output folder where your program will be created
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

# The following folder will be included
include_directories("${PROJECT_SOURCE_DIR}/include" ${GTEST_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

add_executable(tests 
    "${PROJECT_SOURCE_DIR}/src/ant.cpp"
    "${PROJECT_SOURCE_DIR}/src/tests.cpp"
    "${PROJECT_SOURCE_DIR}/src/run_ant_main.cpp"
)

add_executable(app 
    "${PROJECT_SOURCE_DIR}/src/ant.cpp"
    "${PROJECT_SOURCE_DIR}/src/hsv_to_rgb.cpp"
    "${PROJECT_SOURCE_DIR}/src/main.cpp"
    "${PROJECT_SOURCE_DIR}/src/run_ant_main.cpp"
    "${PROJECT_SOURCE_DIR}/src/display.cpp"
)

cuda_add_executable(generator 
    "${PROJECT_SOURCE_DIR}/src/ant.cpp"
    "${PROJECT_SOURCE_DIR}/src/generate.cu"
    "${PROJECT_SOURCE_DIR}/src/possible_states.cpp"
    "${PROJECT_SOURCE_DIR}/src/run_ant_main.cpp"
)

add_executable(generated_to_pic
    "${PROJECT_SOURCE_DIR}/src/ant.cpp"
    "${PROJECT_SOURCE_DIR}/src/generated_to_pic.cpp"
    "${PROJECT_SOURCE_DIR}/src/display.cpp"
    "${PROJECT_SOURCE_DIR}/src/hsv_to_rgb.cpp"
)

target_link_libraries(app ${EIGEN_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLEW_LIBRARY} pthread) 
target_link_libraries(tests ${GTEST_LIBRARIES} ${EIGEN_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLEW_LIBRARY} pthread) 
target_link_libraries(generator ${EIGEN_LIBRARIES} ${GSL_LIBRARIES} ${CUDA_LIBRARY} cnpy z) 
target_link_libraries(generated_to_pic ${EIGEN_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLEW_LIBRARY} pthread) 
