cmake_minimum_required (VERSION 3.0.0)
 
project (surface-fit)
set (CMAKE_CXX_STANDARD 17)

set (SOURCE src/main.cpp)
set (OUTDIR out)

set(RUNTIME_OUTPUT_DIRECTORY ${OUTDIR}/${CMAKE_BUILD_TYPE})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${RUNTIME_OUTPUT_DIRECTORY})

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable (${PROJECT_NAME} ${SOURCE})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
