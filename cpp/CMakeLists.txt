cmake_minimum_required(VERSION 3.10.0)
project(pointnet)

find_package(yaml-cpp REQUIRED)

file(GLOB 
CPP_SOURCES 
src/*.cpp)

add_executable(${PROJECT_NAME} 
apps/main.cpp 
${CPP_SOURCES})

include_directories(${PROJECT_NAME} 
include
${yaml-cpp_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} PRIVATE
yaml-cpp)
