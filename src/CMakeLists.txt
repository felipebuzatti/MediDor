# Initial setup
cmake_minimum_required(VERSION 2.6)
project(TPVisao)

# Help cmake find custom modules
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_SOURCE_DIR}/../cmake"
  "${CMAKE_SOURCE_DIR}/../cmake/Modules"
)

# Find the OpenCV libraries we are using
find_package(OpenCV REQUIRED core highgui imgproc nonfree features2d video)

# Gather all the libs into one variable
set(TPVisao_LIBS 
    ${TPVisao_LIBS} 
    ${OpenCV_LIBS}
)

# Define paths to include and link
include_directories(
    ${PROJECT_SOURCE_DIR}/../include 
    ${OpenCV_INCLUDE_DIR}
)
link_directories(${PROJECT_SOURCE_DIR}/../lib)

set(TPVisao_SRC 
    ${TPVisao_SRC} 
    kalman_sfm.cpp
    main.cpp
)

set(CMAKE_CXX_FLAGS "-std=c++0x")

add_executable(TPVisao ${TPVisao_SRC})
target_link_libraries(TPVisao ${TPVisao_LIBS})
