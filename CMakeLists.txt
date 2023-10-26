cmake_minimum_required(VERSION 3.0.2)
project(ocp_tutorials)

add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
  roscpp)

find_package(pinocchio REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS})

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES 
#  CATKIN_DEPENDS 
#  DEPENDS system_lib
)

add_executable(viewer_test src/viewer_test.cpp)
target_link_libraries(viewer_test
  ${catkin_LIBRARIES}
  pinocchio::pinocchio)
