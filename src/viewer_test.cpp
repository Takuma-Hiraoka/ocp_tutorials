#include <pinocchio/fwd.hpp> //https://github.com/stack-of-tasks/pinocchio/issues/837#issuecomment-514203980
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include "pinocchio/parsers/urdf.hpp"
#include <string>
#include <math.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "joint_publisher");
  ros::NodeHandle nh;
  ros::Publisher joint_pub = nh.advertise<sensor_msgs::JointState>("joint_states", 10);

  std::string fileName;
  nh.getParam("robot_model_file", fileName);
  pinocchio::Model model;
  pinocchio::urdf::buildModel(fileName,pinocchio::JointModelFreeFlyer(),model,true);
  std::cerr << "fileName : " << fileName << std::endl; 
  ros::Rate loop_rate(10);
  int count = 0;
  while (ros::ok())
  {
    sensor_msgs::JointState js;
    js.header.stamp = ros::Time::now();
    for(pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex)model.njoints; ++joint_id)
      {
        if (model.names[joint_id] == "RARM_JOINT1"){
          js.name.push_back(model.names[joint_id]);
          js.position.push_back( -1.0 * (float)count / 40.0);
        } else {
          js.name.push_back(model.names[joint_id]);
          js.position.push_back(0.0);
        }
      }
    joint_pub.publish(js);
    count++;

    ros::spinOnce();
    loop_rate.sleep();
  }
  return 0;
}
