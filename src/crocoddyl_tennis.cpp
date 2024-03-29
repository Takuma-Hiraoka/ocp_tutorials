#include <pinocchio/fwd.hpp> //https://github.com/stack-of-tasks/pinocchio/issues/837#issuecomment-514203980
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include <crocoddyl/core/state-base.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>
#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/residuals/com-position.hpp"
#include "crocoddyl/core/activations/weighted-quadratic.hpp"
#include "crocoddyl/multibody/residuals/frame-velocity.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/core/activations/weighted-quadratic-barrier.hpp"
#include "crocoddyl/multibody/friction-cone.hpp"
#include "crocoddyl/multibody/residuals/contact-friction-cone.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/core/utils/timer.hpp"
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/JointState.h>
#include <string>
#include <math.h>

boost::shared_ptr<crocoddyl::ActionModelAbstract> createActionModel(std::vector<pinocchio::Motion::Vector6> target, boost::shared_ptr<pinocchio::Model> model, boost::shared_ptr<crocoddyl::StateMultibody> state, boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation, Eigen::VectorXd x0, double dt){

  // Creating a double-support contact (feet support)
  boost::shared_ptr<crocoddyl::ContactModelMultiple> contacts = boost::make_shared<crocoddyl::ContactModelMultiple>(state, actuation->get_nu());
  boost::shared_ptr<crocoddyl::ContactModelAbstract> lf_contact =
    boost::make_shared<crocoddyl::ContactModel6D>(
                                                  state, model->getFrameId("lleg_end_coords", pinocchio::FrameType::BODY), pinocchio::SE3::Identity(),
                                                  pinocchio::LOCAL_WORLD_ALIGNED, actuation->get_nu(),
                                                  Eigen::Vector2d(0., 0.));
  contacts->addContact("lf_contact", lf_contact);
  boost::shared_ptr<crocoddyl::ContactModelAbstract> rf_contact =
    boost::make_shared<crocoddyl::ContactModel6D>(
                                                  state, model->getFrameId("rleg_end_coords", pinocchio::FrameType::BODY), pinocchio::SE3::Identity(),
                                                  pinocchio::LOCAL_WORLD_ALIGNED, actuation->get_nu(),
                                                  Eigen::Vector2d(0., 0.));
  contacts->addContact("rf_contact", rf_contact);

  // Define the cost sum (cost manager)
  boost::shared_ptr<crocoddyl::CostModelSum> costs =
      boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());

  if (target.size() != 0) {
    // Adding the hand-placement cost
    Eigen::VectorXd w_hand = Eigen::VectorXd::Zero(6);
    w_hand << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
    boost::shared_ptr<crocoddyl::ActivationModelAbstract> activation_hand =
      boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(w_hand);
    boost::shared_ptr<crocoddyl::CostModelAbstract> rh_cost =
      boost::make_shared<crocoddyl::CostModelResidual>(
                                                       state, activation_hand,
                                                       boost::make_shared<crocoddyl::ResidualModelFrameVelocity>(
                                                                                                                 state, model->getFrameId("rarm_end_coords", pinocchio::FrameType::BODY), (pinocchio::Motion)target[0],
                                                                                                                 pinocchio::ReferenceFrame::WORLD, actuation->get_nu()));
    costs->addCost("rh_goal", rh_cost, 1e0);
  }

  Eigen::VectorXd q0 = x0.head(model->nq);
  pinocchio::Data data(*model);
  Eigen::Vector3d comTask=pinocchio::centerOfMass(*model, data, q0);
  boost::shared_ptr<crocoddyl::CostModelAbstract> comTrack =
      boost::make_shared<crocoddyl::CostModelResidual>(
                                                       state, boost::make_shared<crocoddyl::ResidualModelCoMPosition>(
                                                                                                                      state, comTask, actuation->get_nu()));
  costs->addCost("comTrack", comTrack, 1e0);

  // Adding state and control regularization terms
  Eigen::VectorXd w_x(model->nv*2);
  w_x.head<3>().fill(0.);
  w_x.segment(3,6).fill(pow(10.,2));
  w_x.segment(6, model->nv - 6).fill(pow(0.01, 2));
  w_x.segment(model->nv, model->nv).fill(pow(0.01, 2));
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
  costs->addCost("xReg", x_reg_cost, 1e-4);
  costs->addCost("uReg", u_reg_cost, 1e-7);

  // Adding the state limits penalization
  boost::shared_ptr<crocoddyl::ActivationModelAbstract> activation_xbounds =
    boost::make_shared<
          crocoddyl::ActivationModelQuadraticBarrier>(
                                                      crocoddyl::ActivationBounds(state->get_lb().tail(2*model->nv), state->get_ub().tail(2*model->nv))); // 浮遊リンク自由度をクオータニオンで表すため？にget_lbではルートリンク回転自由度分が4自由度分になる。がResidualModelStateのコンストラクタではnu*2の自由度が作られるので、nrが一致せずエラーとなる。どうせ浮遊リンク自由度には拘束がない(inf)なので、一つ飛ばす。
  boost::shared_ptr<crocoddyl::CostModelAbstract> x_bounds =
      boost::make_shared<crocoddyl::CostModelResidual>(
          state, activation_xbounds,
          boost::make_shared<crocoddyl::ResidualModelState>(
                                                            state, 0 * x0, actuation->get_nu()));
  costs->addCost("xBounds", x_bounds, 1e2);

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
                    state, model->getFrameId("lleg_end_coords", pinocchio::FrameType::BODY), cone, actuation->get_nu()));
  boost::shared_ptr<crocoddyl::CostModelAbstract> rf_friction =
      boost::make_shared<crocoddyl::CostModelResidual>(
          state, activation_friction,
          boost::make_shared<crocoddyl::ResidualModelContactFrictionCone>(
                    state, model->getFrameId("rleg_end_coords", pinocchio::FrameType::BODY), cone, actuation->get_nu()));
  costs->addCost("lf_friction", lf_friction, 1e2);
  costs->addCost("rf_friction", rf_friction, 1e2);

  // Creating the action model
  boost::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics>
      dmodel = boost::make_shared<
          crocoddyl::DifferentialActionModelContactFwdDynamics>(
          state, actuation, contacts, costs);
  return boost::make_shared<crocoddyl::IntegratedActionModelEuler>(dmodel, dt);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "crocoddyl_tennis");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  ros::Publisher joint_pub = nh.advertise<sensor_msgs::JointState>("joint_states", 10);
  tf::TransformBroadcaster robot_base_broadcaster;
  double timeStep = 0.03;
  int totalKnots = 20;
  int impactKnots = 12;
  double viewer_ratio = 1.0;
  int num_iter = 100;
  double stop_th = 0.0000001;
  int num_thread = 8;
  pnh.getParam("timeStep", timeStep);
  pnh.getParam("totalKnots", totalKnots);
  pnh.getParam("impactKnots", impactKnots);
  pnh.getParam("viewer_ratio", viewer_ratio);
  pnh.getParam("num_iter", num_iter);
  pnh.getParam("stop_th", stop_th);
  pnh.getParam("num_thread", num_thread);

  // Load robot
  std::string fileName;
  nh.getParam("robot_model_file", fileName);
  boost::shared_ptr<pinocchio::Model> model = boost::make_shared<pinocchio::Model>();
  pinocchio::urdf::buildModel(fileName,pinocchio::JointModelFreeFlyer(),*model.get(),false);
  Eigen::VectorXd q0 = Eigen::VectorXd::Zero(model->nq);
  q0 << 0.0, 0.0, 0.9685,
    0, 0, 0, 1,
    0.083776, 0.001745, -0.174533, 0.959931, 0.0,
    0.10472, 0.0, 1.39626, 0.0, -0.698132, 0.785398, 0.0, 0.0, // larm
    0.000000,  1.5708,  0.000000,  0.000000,  0.000000, 0.00000, // lhand
    -0.064702, -0.849579, -0.853632, 0.048467, -2.06924, -0.525079, -0.555221, 0.03388, // rarm
    -1.5708,  1.5708,  0.785398,  0.785398,  1.5708, 0.00000, // rhand
    0.876403, 0.483304, -0.75777, 0.928995, -0.152605, -0.479589, // lleg
    0.104042, -0.30241, -0.724128, 1.34373, -0.604591, 0.293463; // rleg
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(model->nq + model->nv);
  for (int i=0; i< model->nq; i++) x0[i] = q0[i];

  // Define the robot's state and actuation
  boost::shared_ptr<crocoddyl::StateMultibody> state = boost::make_shared<crocoddyl::StateMultibody>(model);
  boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation = boost::make_shared<crocoddyl::ActuationModelFloatingBase>(state);

  std::vector<pinocchio::Motion::Vector6> targets;
  Eigen::Matrix<double, 6, 1> target;
  target << 0, 10, 0, 0, 0, 0;
  targets.push_back(target);

  std::vector<pinocchio::Motion::Vector6> targets_null;

  std::vector<Eigen::VectorXd> xs_init;
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > models;
  for (int i=0; i<impactKnots -1; i++)
    {
      models.push_back(createActionModel(targets_null, model, state, actuation, x0, timeStep));
      xs_init.push_back(x0);
    }

  Eigen::VectorXd q1 = Eigen::VectorXd::Zero(model->nq);
  q1 << 0.0, 0.0, 0.9685,
    0, 0, 0, 1,
    0.081179, 0.246468, 0.40752, 0.0, 0.0,
    0.087266, 0.349066, 0.872665, 0.0, -1.74533, 0.785398, 0.0, 0.0, // larm
    0.000000,  1.5708,  0.000000,  0.000000,  0.000000, 0.00000, // lhand
    0.0, -0.538118, -0.31069, 0.310039, -1.37469, -1.72688, -0.762861, -0.081875, // rarm
    -1.5708,  1.5708,  0.785398,  0.785398,  1.5708, 0.00000, // rhand
    0.612324,  0.419347, -0.930735, 1.30957, -0.36699, -0.402607, // lleg
    -0.159379, -0.428674,  -0.686747, 1.37095,  -0.662756,  0.43296; // rleg
  Eigen::VectorXd x1 = Eigen::VectorXd::Zero(model->nq + model->nv);
  for (int i=0; i< model->nq; i++) x1[i] = q1[i];

  models.push_back(createActionModel(targets, model, state, actuation, x1, timeStep));
  for (int i=impactKnots; i<totalKnots; i++)
    {
      models.push_back(createActionModel(targets_null, model, state, actuation, x1, timeStep));
      xs_init.push_back(x1);
    }

  boost::shared_ptr<crocoddyl::ActionModelAbstract> terminal = models.back();
  models.pop_back();
  boost::shared_ptr<crocoddyl::ShootingProblem> problem =
    boost::make_shared<crocoddyl::ShootingProblem>(x0, models,terminal);
  crocoddyl::SolverFDDP solver(problem);
  solver.set_th_stop(stop_th);
  std::vector<Eigen::VectorXd> us_init = solver.get_problem()->quasiStatic_xs(xs_init);
  xs_init.push_back(x1);

  solver.get_problem()->set_nthreads(num_thread);

  crocoddyl::Timer timer;
  std::cerr << "Problem solved: " << solver.solve(xs_init, us_init, num_iter, false) << std::endl;
  double time = timer.get_duration();
  std::cerr << "total calculation time:" << time << std::endl;
  std::cerr << "Number of iterations: " << solver.get_iter() << std::endl;
  std::cerr << "time per iterate:" << time / solver.get_iter() << std::endl;
  std::cerr << "Total cost: " << solver.get_cost() << std::endl;
  std::cerr << "Gradient norm: " << solver.get_stop() << std::endl;
  std::vector<Eigen::VectorXd> xs = solver.get_xs();
  std::vector<Eigen::VectorXd> us = solver.get_us();

  ros::Rate loop_rate(1.0 / timeStep / viewer_ratio);
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
