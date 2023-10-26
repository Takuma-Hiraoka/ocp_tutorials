#include <pinocchio/fwd.hpp> //https://github.com/stack-of-tasks/pinocchio/issues/837#issuecomment-514203980
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include <crocoddyl/core/state-base.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>
#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/core/activations/weighted-quadratic.hpp"
#include "crocoddyl/multibody/residuals/frame-placement.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/core/activations/weighted-quadratic-barrier.hpp"
#include "crocoddyl/multibody/friction-cone.hpp"
#include "crocoddyl/multibody/residuals/contact-friction-cone.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <string>
#include <math.h>

boost::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> createActionModel(Eigen::Vector3d target, boost::shared_ptr<pinocchio::Model> model, boost::shared_ptr<crocoddyl::StateMultibody> state, boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation, std::vector<pinocchio::FrameIndex> ee_ids, Eigen::VectorXd x0){

  // Creating a double-support contact (feet support)
  boost::shared_ptr<crocoddyl::ContactModelMultiple> contacts = boost::make_shared<crocoddyl::ContactModelMultiple>(state, actuation->get_nu());
  boost::shared_ptr<crocoddyl::ContactModelAbstract> lf_contact =
    boost::make_shared<crocoddyl::ContactModel6D>(
                                                  state, ee_ids[0], pinocchio::SE3::Identity(),
                                                  pinocchio::LOCAL_WORLD_ALIGNED, actuation->get_nu(),
                                                  Eigen::Vector2d(0., 0.));
  contacts->addContact("lf_contact", lf_contact);
  boost::shared_ptr<crocoddyl::ContactModelAbstract> rf_contact =
    boost::make_shared<crocoddyl::ContactModel6D>(
                                                  state, ee_ids[1], pinocchio::SE3::Identity(),
                                                  pinocchio::LOCAL_WORLD_ALIGNED, actuation->get_nu(),
                                                  Eigen::Vector2d(0., 0.));
  contacts->addContact("rf_contact", rf_contact);

  // Define the cost sum (cost manager)
  boost::shared_ptr<crocoddyl::CostModelSum> costs =
      boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());

  // Adding the hand-placement cost
  Eigen::VectorXd w_hand = Eigen::VectorXd::Zero(6);
  w_hand << 1.0, 1.0, 1.0, 0.0001*0.0001, 0.0001*0.0001, 0.0001*0.0001;
  pinocchio::SE3 lh_Mref = pinocchio::SE3(Eigen::Matrix3d::Identity(), target);
  boost::shared_ptr<crocoddyl::ActivationModelAbstract> activation_hand =
      boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(w_hand);
  boost::shared_ptr<crocoddyl::CostModelAbstract> lh_cost =
      boost::make_shared<crocoddyl::CostModelResidual>(
          state, activation_hand,
          boost::make_shared<crocoddyl::ResidualModelFramePlacement>(
                                                                     state, ee_ids[2], lh_Mref, actuation->get_nu()));
  costs->addCost("lh_goal", lh_cost, 1e2);

  // Adding state and control regularization terms
  Eigen::VectorXd w_x(model->nv*2);
  w_x.head<3>().fill(0.);
  w_x.segment(3,6).fill(pow(10.,2));
  w_x.segment(6, model->nv - 6).fill(pow(0.01, 2));
  w_x.segment(model->nv, model->nv).fill(pow(10, 2));
  boost::shared_ptr<crocoddyl::ActivationModelAbstract> activation_xreg =
      boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(w_x);
  boost::shared_ptr<crocoddyl::CostModelAbstract> x_reg_cost =
      boost::make_shared<crocoddyl::CostModelResidual>(
          state, activation_xreg,
          boost::make_shared<crocoddyl::ResidualModelState>(
                                                            state, x0, actuation->get_nu()));
  boost::shared_ptr<crocoddyl::CostModelAbstract> u_reg_cost =
      boost::make_shared<crocoddyl::CostModelResidual>(
          state, 
          boost::make_shared<crocoddyl::ResidualModelControl>(
                                                            state, actuation->get_nu()));
  costs->addCost("xReg", x_reg_cost, 1e-3);
  costs->addCost("uReg", u_reg_cost, 1e-4);

  // Adding the state limits penalization
  boost::shared_ptr<crocoddyl::ActivationModelAbstract> activation_xbounds =
    boost::make_shared<
          crocoddyl::ActivationModelQuadraticBarrier>(
                                                              crocoddyl::ActivationBounds(state->get_lb(), state->get_ub()));
  boost::shared_ptr<crocoddyl::CostModelAbstract> x_bounds =
      boost::make_shared<crocoddyl::CostModelResidual>(
          state, activation_xbounds,
          boost::make_shared<crocoddyl::ResidualModelState>(
                                                            state, 0 * x0, actuation->get_nu()));
  costs->addCost("xBounds", x_bounds, 1.0);

  // Adding the friction cone penalization
  Eigen::Matrix3d nsurf = Eigen::Matrix3d::Identity();
  double mu = 0.7;
  crocoddyl::FrictionCone cone(nsurf, mu, 4, false);
  boost::shared_ptr<crocoddyl::ActivationModelAbstract> activation_friction =
    boost::make_shared<
          crocoddyl::ActivationModelQuadraticBarrier>(
                                                              crocoddyl::ActivationBounds(cone.get_lb(), cone.get_ub()));
  boost::shared_ptr<crocoddyl::CostModelAbstract> lf_friction =
      boost::make_shared<crocoddyl::CostModelResidual>(
          state, activation_friction,
          boost::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                    state, ee_ids[0], cone, actuation->get_nu()));
  boost::shared_ptr<crocoddyl::CostModelAbstract> rf_friction =
      boost::make_shared<crocoddyl::CostModelResidual>(
          state, activation_friction,
          boost::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                    state, ee_ids[1], cone, actuation->get_nu()));
  costs->addCost("lf_friction", lf_friction, 1e1);
  costs->addCost("rf_friction", rf_friction, 1e1);

  // Creating the action model
  boost::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics>
      dmodel = boost::make_shared<
          crocoddyl::DifferentialActionModelContactFwdDynamics>(
          state, actuation, contacts, costs);
  return dmodel;
}

std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > createSequence(std::vector<boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> > dmodels, double dt, int N)
{
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > seq;
  for (int i=0; i< dmodels.size(); i++)
    {
      for (int j=0; j<N;j++)
        {
          seq.push_back(boost::make_shared<crocoddyl::IntegratedActionModelEuler>(dmodels[i], dt));
        }
      seq.push_back(boost::make_shared<crocoddyl::IntegratedActionModelEuler>(dmodels[i], 0.0));
    }
  return seq;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "whole_body_manipulation");
  ros::NodeHandle nh;
  ros::Publisher joint_pub = nh.advertise<sensor_msgs::JointState>("joint_states", 10);

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
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(model->nq + model->nv);
  for (int i=0; i< model->nq; i++) x0[i] = q0[i];

  // Declaring the foot and hand names
  std::string rf_name = "RLEG_LINK5";
  std::string lf_name = "lLEG_LINK5";
  std::string lh_name = "HANDBASE_L";

  // Getting the frame ids
  pinocchio::FrameIndex rf_id =model->getFrameId(rf_name, pinocchio::FrameType::BODY);
  pinocchio::FrameIndex lf_id =model->getFrameId(lf_name, pinocchio::FrameType::BODY);
  pinocchio::FrameIndex lh_id =model->getFrameId(lh_name, pinocchio::FrameType::BODY);
  std::vector<pinocchio::FrameIndex> ee_ids = std::vector<pinocchio::FrameIndex>{rf_id, lf_id, lh_id};

  // Define the robot's state and actuation
  boost::shared_ptr<crocoddyl::StateMultibody> state = boost::make_shared<crocoddyl::StateMultibody>(model);
  boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation = boost::make_shared<crocoddyl::ActuationModelFloatingBase>(state);

  double dt = 5e-2;
  int N = 20;
  std::vector<Eigen::Vector3d> targets;
  targets.push_back(Eigen::Vector3d(0.4,0.1,1.2));
  targets.push_back(Eigen::Vector3d(0.6,0.1,1.2));
  targets.push_back(Eigen::Vector3d(0.6,-0.1,1.2));
  targets.push_back(Eigen::Vector3d(0.4,-0.1,1.2));

  std::vector<boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> > dmodels;
  for(int i=0; i<targets.size(); i++)
    {
      dmodels.push_back(createActionModel(targets[i], model, state, actuation, ee_ids, x0));
    }
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > seqs = createSequence(dmodels, dt, N);

  ros::Rate loop_rate(10);
  int count = 0;
  while (ros::ok())
  {
    sensor_msgs::JointState js;
    js.header.stamp = ros::Time::now();
    for(pinocchio::JointIndex joint_id = 2; joint_id < (pinocchio::JointIndex)model->njoints; ++joint_id) // world と root
      {
        js.name.push_back(model->names[joint_id]);
        js.position.push_back(q0[7 - 2 + joint_id]);
      }
    joint_pub.publish(js);
    count++;

    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
