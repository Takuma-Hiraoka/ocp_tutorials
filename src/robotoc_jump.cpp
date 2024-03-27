#include <string>
#include <memory>
#include <pinocchio/fwd.hpp> //https://github.com/stack-of-tasks/pinocchio/issues/837#issuecomment-514203980
#include "pinocchio/parsers/urdf.hpp"

#include "Eigen/Core"
#include "robotoc/solver/ocp_solver.hpp"
#include "robotoc/ocp/ocp.hpp"
#include "robotoc/robot/robot.hpp"
#include "robotoc/planner/contact_sequence.hpp"
#include "robotoc/cost/cost_function.hpp"
#include "robotoc/cost/configuration_space_cost.hpp"
#include "robotoc/cost/task_space_3d_cost.hpp"
#include "robotoc/cost/task_space_6d_cost.hpp"
#include "robotoc/cost/com_cost.hpp"
#include "robotoc/cost/periodic_swing_foot_ref.hpp"
#include "robotoc/cost/periodic_com_ref.hpp"
#include "robotoc/constraints/constraints.hpp"
#include "robotoc/constraints/joint_position_lower_limit.hpp"
#include "robotoc/constraints/joint_position_upper_limit.hpp"
#include "robotoc/constraints/joint_velocity_lower_limit.hpp"
#include "robotoc/constraints/joint_velocity_upper_limit.hpp"
#include "robotoc/constraints/joint_torques_lower_limit.hpp"
#include "robotoc/constraints/joint_torques_upper_limit.hpp"
#include "robotoc/constraints/friction_cone.hpp"
#include "robotoc/solver/solver_options.hpp"
 

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/JointState.h>

int main(int argc, char *argv[])
{
  ros::init(argc, argv, "robotoc_jump");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  ros::Publisher joint_pub = nh.advertise<sensor_msgs::JointState>("joint_states", 10);
  tf::TransformBroadcaster robot_base_broadcaster;

  double jumpLength = 0.5;
  double timeStep = 0.03;
  int flyingKnots = 10;
  int groundKnots = 25;
  double state_w = 0.01;
  double root_w = 500;
  double waist_w = 0.01;
  double arm_w = 0.01;
  double viewer_ratio = 1.0;
  int num_thread = 8;
  int num_iter = 100;
  double stop_th = 0.0000001;

  pnh.getParam("jumpLength", jumpLength);
  pnh.getParam("timeStep", timeStep);
  pnh.getParam("flyingKnots", flyingKnots);
  pnh.getParam("groundKnots", groundKnots);
  pnh.getParam("root_w", root_w);
  pnh.getParam("waist_w", waist_w);
  pnh.getParam("arm_w", arm_w);
  pnh.getParam("state_w", state_w);
  pnh.getParam("viewer_ratio", viewer_ratio);
  pnh.getParam("num_thread", num_thread);
  pnh.getParam("num_iter", num_iter);
  pnh.getParam("stop_th", stop_th);

  // Load robot
  std::string fileName;
  nh.getParam("robot_model_file", fileName);

  // for visualize
  boost::shared_ptr<pinocchio::Model> model = boost::make_shared<pinocchio::Model>();
  pinocchio::urdf::buildModel(fileName,pinocchio::JointModelFreeFlyer(),*model.get(),false);

  robotoc::RobotModelInfo model_info;
  model_info.urdf_path = fileName;
  model_info.base_joint_type = robotoc::BaseJointType::FloatingBase;
  const double baumgarte_time_step = 0.04;
  // surface_contactsなら最後水平になる？
  model_info.surface_contacts = {robotoc::ContactModelInfo("lleg_end_coords", baumgarte_time_step),
                               robotoc::ContactModelInfo("rleg_end_coords", baumgarte_time_step)};
  robotoc::Robot robot(model_info);

  const double t0 = 0;
  auto cost = std::make_shared<robotoc::CostFunction>();
  Eigen::VectorXd q_standing(Eigen::VectorXd::Zero(robot.dimq()));
  q_standing << 0.0, 0.0, 0.9685,
    0, 0, 0, 1,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.523599,  0.349066,  0.087266, -1.396263, 0.000000,  0.000000, -0.349066, // larm
    0.523599, -0.349066, -0.087266, -1.396263,  0.000000,  0.000000, -0.349066, // rarm
    0.00027, -0.001485, -0.740046,  1.33015, -0.590108, 0.001485, // lleg
    0.00027, -0.001485, -0.740046,  1.33015, -0.590108, 0.001485; // rleg
  Eigen::VectorXd q_weight(Eigen::VectorXd::Zero(robot.dimv()));
  q_weight.head<3>().fill(0.);
  q_weight.segment(3,3).fill(pow(root_w,2));
  q_weight.segment(6, 3).fill(pow(waist_w, 2));
  q_weight.segment(6 + 3, 2 + 14).fill(pow(arm_w, 2));
  q_weight.segment(6 + 3 + 2 + 14, robot.dimv() - 6 - 3 - 2 - 14).fill(pow(state_w, 2));

  Eigen::VectorXd v_weight(Eigen::VectorXd::Zero(robot.dimv()));
  v_weight.fill(1.0e-02);
  Eigen::VectorXd a_weight(Eigen::VectorXd::Zero(robot.dimv()));
  a_weight.fill(1.0e-05);
  Eigen::VectorXd u_weight = Eigen::VectorXd::Constant(robot.dimu(), 1e-5);
  auto config_cost = std::make_shared<robotoc::ConfigurationSpaceCost>(robot);
  config_cost->set_q_ref(q_standing);
  config_cost->set_q_weight(q_weight);
  config_cost->set_q_weight_terminal(q_weight);
  config_cost->set_q_weight_impact(q_weight);
  config_cost->set_v_weight(v_weight);
  config_cost->set_v_weight_terminal(v_weight);
  config_cost->set_v_weight_impact(v_weight);
  config_cost->set_a_weight(a_weight);
  config_cost->set_u_weight(u_weight);
  cost->add("config_cost", config_cost);

  robot.updateFrameKinematics(q_standing);
  const Eigen::Vector3d RF = robot.framePosition("rleg_end_coords");
  const Eigen::Vector3d LF = robot.framePosition("lleg_end_coords");

  const double barrier_param = 1.0e-03;
  const double fraction_to_boundary_rule = 0.995;
  auto constraints          = std::make_shared<robotoc::Constraints>(barrier_param, fraction_to_boundary_rule);
  auto joint_position_lower  = std::make_shared<robotoc::JointPositionLowerLimit>(robot);
  auto joint_position_upper  = std::make_shared<robotoc::JointPositionUpperLimit>(robot);
  auto joint_velocity_lower  = std::make_shared<robotoc::JointVelocityLowerLimit>(robot);
  auto joint_velocity_upper  = std::make_shared<robotoc::JointVelocityUpperLimit>(robot);
  auto joint_torques_lower   = std::make_shared<robotoc::JointTorquesLowerLimit>(robot);
  auto joint_torques_upper   = std::make_shared<robotoc::JointTorquesUpperLimit>(robot);
  auto friction_cone         = std::make_shared<robotoc::FrictionCone>(robot);
  constraints->add("joint_position_lower", joint_position_lower);
  constraints->add("joint_position_upper", joint_position_upper);
  constraints->add("joint_velocity_lower", joint_velocity_lower);
  constraints->add("joint_velocity_upper", joint_velocity_upper);
  constraints->add("joint_torques_lower", joint_torques_lower);
  constraints->add("joint_torques_upper", joint_torques_upper);
  constraints->add("friction_cone", friction_cone);
  auto contact_sequence = std::make_shared<robotoc::ContactSequence>(robot);
  const double mu = 0.7;
  const std::unordered_map<std::string, double> friction_coefficients = {{"lleg_end_coords", mu},
                                                                         {"rleg_end_coords", mu}};
  std::unordered_map<std::string, Eigen::Vector3d> contact_positions = {{"lleg_end_coords", LF},
                                                                        {"rleg_end_coords", RF}};
  auto contact_status_standing = robot.createContactStatus();
  contact_status_standing.activateContacts(std::vector<std::string>({"lleg_end_coords", "rleg_end_coords"}));
  contact_status_standing.setContactPlacements(contact_positions);
  contact_status_standing.setFrictionCoefficients(friction_coefficients);
  contact_sequence->init(contact_status_standing);
  auto contact_status_flying = robot.createContactStatus();
  contact_sequence->push_back(contact_status_flying, t0+timeStep*groundKnots, true);

  contact_positions["rleg_end_coords"].noalias() += Eigen::Vector3d(jumpLength, 0, 0);
  contact_positions["lleg_end_coords"].noalias() += Eigen::Vector3d(jumpLength, 0, 0);
  contact_status_standing.setContactPlacements(contact_positions);
  contact_sequence->push_back(contact_status_standing, t0+timeStep*(groundKnots + flyingKnots), true);

  auto sto_cost = std::make_shared<robotoc::STOCostFunction>();
  const std::vector<double> minimum_times = {0.2, 0.2, 0.5};
  auto sto_constraints = std::make_shared<robotoc::STOConstraints>(minimum_times);

  // std::cerr << contact_sequence << std::endl;
  const double T = t0 + timeStep * (groundKnots*2 + flyingKnots);
  const int N = T / timeStep;
  robotoc::OCP ocp(robot, cost, constraints, sto_cost, sto_constraints, contact_sequence, T, N);
  auto solver_options = robotoc::SolverOptions();
  solver_options.nthreads = num_thread;
  solver_options.max_iter = num_iter;
  solver_options.enable_benchmark = true;
  solver_options.kkt_tol = stop_th;
  solver_options.initial_sto_reg_iter = 0;
  robotoc::OCPSolver ocp_solver(ocp, solver_options);

  const double t = 0;
  const Eigen::VectorXd q(q_standing); // initial state.
  const Eigen::VectorXd v(Eigen::VectorXd::Zero(robot.dimv())); // initial state.
 
  ocp_solver.discretize(t); // discretizes the optimal control problem.
  ocp_solver.setSolution("q", q); // set the initial guess of the solution.
  ocp_solver.setSolution("v", v); // set the initial guess of the solution.
  Eigen::Vector3d f_init;
  f_init << 0, 0, 0.5*robot.totalWeight();
  ocp_solver.setSolution("f", f_init); // set the initial guess of the solution.
  
  ocp_solver.initConstraints(); // initialize the slack and dual variables of the primal-dual interior point method.
  std::cerr << "Initial KKT error: " << ocp_solver.KKTError(t, q, v) << std::endl;
  ocp_solver.solve(t, q, v);
  std::cerr << "KKT error after convergence: " << ocp_solver.KKTError(t, q, v) << std::endl;
  std::cerr << ocp_solver.getSolverStatistics() << std::endl; // print solver statistics

  std::vector<Eigen::VectorXd> xs = ocp_solver.getSolution("q");
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
    // robotはフレームIDのみを持ちjointIDを持たない
    // どうせpinocchioなのでpinocchioモデルを作ってしまう
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
