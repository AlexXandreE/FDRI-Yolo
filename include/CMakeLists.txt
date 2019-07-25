
cmake_minimum_required(VERSION 2.8.12)

project(FDRI)

add_library(yolo yolo_cpp_dll.dll)

set_target_properties(yolo PROPERTIES LINKER_LANGUAGE C)