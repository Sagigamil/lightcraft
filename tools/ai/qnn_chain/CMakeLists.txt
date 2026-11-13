cmake_minimum_required(VERSION 3.22)
project(qnn_chain CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT QNN_SDK_ROOT)
  message(FATAL_ERROR "QNN_SDK_ROOT is not set")
endif()

add_library(qnn_chain SHARED qnn_chain.cpp)
target_include_directories(qnn_chain PRIVATE ${QNN_SDK_ROOT}/include/QNN)
target_link_libraries(qnn_chain PRIVATE log dl)

add_executable(qnn_chain_test main.cpp)
target_link_libraries(qnn_chain_test PRIVATE qnn_chain log dl)
