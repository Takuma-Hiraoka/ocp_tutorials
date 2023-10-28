#include <pinocchio/fwd.hpp> //https://github.com/stack-of-tasks/pinocchio/issues/837#issuecomment-514203980
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include <crocoddyl/core/state-base.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>
#include "crocoddyl/multibody/actuations/floating-base.hpp"

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/JointState.h>
#include <string>
#include <math.h>

class SimpleBipedGaitProbrem {
public:
  SimpleBipedGaitProbrem(boost::shared_ptr<pinocchio::Model> model_, std::string rightFoot, std::string leftFoot, Eigen::VectorXd q0_) : model(model_) {
    pinocchio::Data data_(*model);
    data = data_;
    state = boost::make_shared<crocoddyl::StateMultibody>(model);
    actuation = boost::make_shared<crocoddyl::ActuationModelFloatingBase>(state);
    rf_id = model->getFrameId(rightFoot, pinocchio::FrameType::BODY);
    lf_id = model->getFrameId(leftFoot, pinocchio::FrameType::BODY);
  };

  boost::shared_ptr<crocoddyl::ShootingProblem>createWalkingProblem(Eigen::VectorXd x0, double stepLength, double stepHeight, double timeStep, int stepKnots, int supportKnots) {
    // Compute the current foot positions
    Eigen::VectorXd q0 = x0.head(state->nq);
    pinocchio::forwardKinematics(model,data,q0);
    pinocchio::updateFramePlacements(model,data);
    Eigen::Vector3d rfPos0 = data.omf[rf_id].translation();
    Eigen::Vector3d lfPos0 = data.omf[lf_id].translation();
    Eigen::Vector3d comRef = (rfPos0 + lfPos0) / 2;
  };

protected:
  boost::shared_ptr<pinocchio::Model> model;
  pinocchio::Data data;
  boost::shared_ptr<crocoddyl::StateMultibody> state;
  boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation;
  pinocchio::FrameIndex rf_id;
  pinocchio::FrameIndex lf_id;
  bool firstStep = true;
  double mu = 0.7;
  Eigen::Matrix3d Rsurf = Eigen::Matrix3d::Identity();
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "simple_biped_gait");
  ros::NodeHandle nh;
  ros::Publisher joint_pub = nh.advertise<sensor_msgs::JointState>("joint_states", 10);
  tf::TransformBroadcaster robot_base_broadcaster;

  // Load robot
  std::string fileName;
  nh.getParam("robot_model_file", fileName);
  boost::shared_ptr<pinocchio::Model> model = boost::make_shared<pinocchio::Model>();
  pinocchio::urdf::buildModel(fileName,pinocchio::JointModelFreeFlyer(),*model.get(),false);
  Eigen::VectorXd q0 = Eigen::VectorXd::Zero(model->nq);
  q0 << 0.0, 0.0, 0.9685,
    0, 0, 0, 1,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.698132, 0.349066, 0.087266, -1.39626, 0.0, 0.0, -0.349066, // larm
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000, 0.00000, // lhand
    0.0, 0.698132, -0.349066, -0.087266, -1.39626, 0.0, 0.0, -0.349066, // rarm
    0.000000,  0.000000,  0.000000,  0.000000,  0.000000, 0.00000, // rhand
    0.000128, -0.002474, -0.488869, 1.01524, -0.526374, 0.002474, // lleg
    0.000128, -0.002474, -0.488908, 1.01524, -0.526335, 0.002474; // rleg
  Eigen::VectorXd x0(model->nq + model->nv);
  x0 << q0, Eigen::VectorXd::Zero(model->nv);

  std::vector<Eigen::VectorXd> xs;
  xs.push_back(x0);
  ros::Rate loop_rate(10.0);
  int count = 0;
  while (ros::ok())
  {
    Eigen::VectorXd q = xs[count];
    // base
    geometry_msgs::TransformStamped baseState;
    baseState.header.stamp = ros::Time::now();
    baseState.header.frame_id = "world";
    baseState.child_frame_id  = "BODY";
    baseState.transform.translation.x = q[0];
    baseState.transform.translation.y = q[1];
    baseState.transform.translation.z = q[2];
    baseState.transform.rotation.x = q[3];
    baseState.transform.rotation.y = q[4];
    baseState.transform.rotation.z = q[5];
    baseState.transform.rotation.w = q[6];
    robot_base_broadcaster.sendTransform(baseState);

    // joint angle
    sensor_msgs::JointState js;
    js.header.stamp = ros::Time::now();
    for(pinocchio::JointIndex joint_id = 2; joint_id < (pinocchio::JointIndex)model->njoints; ++joint_id) // world と root
      {
        js.name.push_back(model->names[joint_id]);
        js.position.push_back(q[7 - 2 + joint_id]);
      }
    joint_pub.publish(js);
    if (count < xs.size()-1) count++;

    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}

