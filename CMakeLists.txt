cmake_minimum_required(VERSION 3.0.2)
project(ocp_tutorials)

add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf)

find_package(pinocchio REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(robotoc REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS})

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES 
#  CATKIN_DEPENDS 
#  DEPENDS system_lib
)

add_executable(robotoc_gait src/robotoc_gait.cpp)
add_executable(robotoc_jump src/robotoc_jump.cpp)

target_link_libraries(robotoc_gait
  ${catkin_LIBRARIES}
  robotoc::robotoc)
target_link_libraries(robotoc_jump
  ${catkin_LIBRARIES}
  robotoc::robotoc)
