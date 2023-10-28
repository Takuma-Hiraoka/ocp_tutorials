cmake_minimum_required(VERSION 3.0.2)
project(ocp_tutorials)

add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf)

find_package(pinocchio REQUIRED)
find_package(crocoddyl REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS})

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES 
#  CATKIN_DEPENDS 
#  DEPENDS system_lib
)

add_executable(viewer_test src/viewer_test.cpp)
add_executable(whole_body_manipulation src/whole_body_manipulation.cpp)
target_link_libraries(viewer_test
  ${catkin_LIBRARIES}
  pinocchio::pinocchio)
target_link_libraries(whole_body_manipulation
  ${catkin_LIBRARIES}
  pinocchio::pinocchio
  crocoddyl::crocoddyl)
