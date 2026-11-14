cmake_minimum_required(VERSION 3.22)
project(mobilesam CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT QNN_SDK_ROOT)
  message(FATAL_ERROR "QNN_SDK_ROOT is not set")
endif()

add_library(mobilesam SHARED mobilesam.cpp)
target_include_directories(mobilesam PRIVATE ${QNN_SDK_ROOT}/include/QNN)
target_link_libraries(mobilesam PRIVATE log dl)

add_executable(mobilesam_test main.cpp)
target_link_libraries(mobilesam_test PRIVATE mobilesam log dl)
