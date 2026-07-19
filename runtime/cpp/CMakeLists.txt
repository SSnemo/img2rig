cmake_minimum_required(VERSION 3.16)
project(img2rig_runtime CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The runtime itself is two header files - just add runtime/cpp to your
# include path and provide a DrawLayerFn. This project only builds the tests.
add_executable(rig_test tests/rig_test.cpp)
target_include_directories(rig_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()
add_test(NAME rig_test COMMAND rig_test)
