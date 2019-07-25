
#cmake --build . --config Release
#cmake -G "Visual Studio 14 2015 Win64" -T host=x64 .. -DBOOST_ROOT=D:\boost_1_66_0 -DBOOST_LIBRARYDIR=D:\boost_1_66_0\boost_1_66_0\stage\lib

# Minimum version recommended by Dlib -> 2.8.12
cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_BUILD_TYPE Release)

# Project name
project(FDRI)

# Include Dlib
add_subdirectory("C:/Users/Alexandre/libs/dlib" dlib_build)
add_subdirectory("C:/Users/Alexandre/libs/pugixml-1.9" pugi_build)

#set(Boost_DEBUG ON)
set(Boost_USE_STATIC_LIBS   ON)
find_package(Boost 1.66.0 REQUIRED  filesystem date_time program_options)

set(OPENSSL_ROOT_DIR "C:/Users/Alexandre/libs/openssl-1.1.1a")
set(OPENSSL_USE_STATIC_LIBS TRUE)
find_package(OpenSSL REQUIRED)

set(OpenCV_DIR "C:/Users/Alexandre/libs/opencv-4.0.0/build/")
find_package(OpenCV REQUIRED)

include_directories(./include ./src )
add_subdirectory(include)

# Executable
add_executable(FDRI FDRI.cpp ./src/DFXML_creator.cpp ./src/image_handling.cpp ./src/helper_functions.cpp)

# Additional Libraries
target_link_libraries(FDRI dlib::dlib)
target_link_libraries(FDRI Boost::filesystem Boost::date_time Boost::program_options)
target_link_libraries(FDRI pugixml)
target_link_libraries(FDRI OpenSSL::Crypto)
target_link_libraries(FDRI ${OpenCV_LIBS})
target_link_libraries(FDRI yolo)

