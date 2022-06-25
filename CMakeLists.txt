cmake_minimum_required(VERSION 2.8.3)
project(cam_odo_cal)

#set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
find_package(Sophus REQUIRED)
find_package(OpenCV 3.4 REQUIRED)
find_package(Ceres REQUIRED)
include_directories(
    ${CERES_INCLUDE_DIRS}
    camOdoCalib/include
    ${Sophus_INCLUDE_DIR}
    )

find_package(Eigen3)
include_directories(
  ${EIGEN3_INCLUDE_DIR}
)
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

add_library(calc_lib
    camOdoCalib/src/data_selection/data_selection.cpp)

add_library(calib_lib
    camOdoCalib/src/solver.cpp
        camOdoCalib/src/solveQyx.cpp
        )

target_link_libraries(calc_lib  ${Boost_LIBRARIES}
        ${OpenCV_LIBS}
        ${CERES_LIBRARIES}
        ${Sophus_LIBRARIES}
        )

add_executable(cam_odo_cal camOdoCalib/src/main_node.cpp )
target_link_libraries(cam_odo_cal calib_lib calc_lib
        ${CERES_LIBRARIES})


