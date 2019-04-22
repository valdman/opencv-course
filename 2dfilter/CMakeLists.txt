cmake_minimum_required (VERSION 3.0.0)
 
project (2dfilter)
set (CMAKE_CXX_STANDARD 17)

set (SOURCE main.cpp)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable (${PROJECT_NAME} ${SOURCE})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )