#include <pinocchio/fwd.hpp> //https://github.com/stack-of-tasks/pinocchio/issues/837#issuecomment-514203980
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include <crocoddyl/core/state-base.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>
#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"
#include "crocoddyl/multibody/actions/contact-invdyn.hpp"
#include "crocoddyl/multibody/residuals/com-position.hpp"
#include "crocoddyl/multibody/wrench-cone.hpp"
#include "crocoddyl/multibody/residuals/contact-wrench-cone.hpp"
#include "crocoddyl/multibody/residuals/frame-velocity.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/core/activations/weighted-quadratic.hpp"
#include "crocoddyl/multibody/residuals/frame-placement.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/core/residuals/joint-effort.hpp"
#include "crocoddyl/core/activations/weighted-quadratic-barrier.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/core/solvers/intro.hpp"
#include "crocoddyl/core/utils/timer.hpp"

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/JointState.h>
#include <string>
#include <math.h>

class SimpleBipedGaitProblem {
public:
  SimpleBipedGaitProblem(boost::shared_ptr<pinocchio::Model> model_, std::string rightFoot, std::string leftFoot, Eigen::VectorXd q0_, bool fwddyn_, double com_track_w_, double feet_track_w_, double root_w_, double waist_w_, double arm_w_, double state_w_, int num_steps_) : model(model_), q0(q0_), fwddyn(fwddyn_), com_track_w(com_track_w_), feet_track_w(feet_track_w_), root_w(root_w_), waist_w(waist_w_), arm_w(arm_w_), state_w(state_w_), num_steps(num_steps_){
    pinocchio::Data data_(*model);
    data = data_;
    state = boost::make_shared<crocoddyl::StateMultibody>(model);
    actuation = boost::make_shared<crocoddyl::ActuationModelFloatingBase>(state);
    rf_id = model->getFrameId(rightFoot, pinocchio::FrameType::BODY);
    lf_id = model->getFrameId(leftFoot, pinocchio::FrameType::BODY);
  };

  boost::shared_ptr<crocoddyl::ShootingProblem> createWalkingProblem(Eigen::VectorXd x0, double stepLength, double stepHeight, double timeStep, int stepKnots, int supportKnots) {
    // Compute the current foot positions
    Eigen::VectorXd q0 = x0.head(model->nq);
    pinocchio::forwardKinematics(*model,data,q0);
    pinocchio::updateFramePlacements(*model,data);
    Eigen::Vector3d rfPos0 = data.oMf[rf_id].translation();
    Eigen::Vector3d lfPos0 = data.oMf[lf_id].translation();
    std::vector<std::pair<pinocchio::FrameIndex, Eigen::Vector3d>> footPos0;
    footPos0.push_back(std::pair<pinocchio::FrameIndex, Eigen::Vector3d>(rf_id, rfPos0));
    footPos0.push_back(std::pair<pinocchio::FrameIndex, Eigen::Vector3d>(lf_id, lfPos0));
    Eigen::Vector3d comRef = (rfPos0 + lfPos0) / 2;
    comRef[2] = pinocchio::centerOfMass(*model, data, q0)[2];

    std::vector<pinocchio::FrameIndex> lf_ids;
    lf_ids.push_back(lf_id);
    std::vector<pinocchio::FrameIndex> rf_ids;
    rf_ids.push_back(rf_id);

    std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> loco3dModel;

    // We defined the problem as:
    std::vector<pinocchio::FrameIndex> rf_lf_ids;
    rf_lf_ids.push_back(rf_id);
    rf_lf_ids.push_back(lf_id);
    for (int i=0;i<supportKnots; i++) {
      loco3dModel.push_back(this->createSwingFootModel(timeStep, rf_lf_ids, comRef, footPos0));
    }

    if (this->num_steps >= 1) {
      std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> rStep;
      if (firstStep) {
        rStep = this->createFootStepModels(comRef, footPos0, 0.5 * stepLength, stepHeight, timeStep, stepKnots, lf_ids, rf_ids);
        firstStep = false;
      } else {
        rStep = this->createFootStepModels(comRef, footPos0, stepLength, stepHeight, timeStep, stepKnots, lf_ids, rf_ids);
      }
      loco3dModel.insert(loco3dModel.end(), rStep.begin(), rStep.end());
    }

    if (this->num_steps >= 2) {
      for (int i=0;i<supportKnots; i++) {
        loco3dModel.push_back(this->createSwingFootModel(timeStep, rf_lf_ids, comRef, footPos0));
      }
    }

    if (this->num_steps >= 3) {
      std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> lStep = this->createFootStepModels(comRef, footPos0, stepLength, stepHeight, timeStep, stepKnots, rf_ids, lf_ids);

      loco3dModel.insert(loco3dModel.end(), lStep.begin(), lStep.end());
    }

    if (this->num_steps >= 4) {
      for (int i=0;i<supportKnots; i++) {
        loco3dModel.push_back(this->createSwingFootModel(timeStep, rf_lf_ids, comRef, footPos0));
      }
    }

    if (this->num_steps >= 5) {
      std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> rStep2 = this->createFootStepModels(comRef, footPos0, stepLength, stepHeight, timeStep, stepKnots, lf_ids, rf_ids);

      loco3dModel.insert(loco3dModel.end(), rStep2.begin(), rStep2.end());
    }

    if (this->num_steps >= 6) {
      for (int i=0;i<supportKnots; i++) {
        loco3dModel.push_back(this->createSwingFootModel(timeStep, rf_lf_ids, comRef, footPos0));
      }
    }

    if (this->num_steps >= 7) {
      std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> lStep2 = this->createFootStepModels(comRef, footPos0, stepLength, stepHeight, timeStep, stepKnots, rf_ids, lf_ids);

      loco3dModel.insert(loco3dModel.end(), lStep2.begin(), lStep2.end());
    }

    if (this->num_steps >= 8) {
      for (int i=0;i<supportKnots; i++) {
        loco3dModel.push_back(this->createSwingFootModel(timeStep, rf_lf_ids, comRef, footPos0));
      }
    }

    boost::shared_ptr<crocoddyl::ActionModelAbstract> terminal = loco3dModel.back();
    loco3dModel.pop_back();

    return boost::make_shared<crocoddyl::ShootingProblem>(x0, loco3dModel,terminal);
  };

  boost::shared_ptr<crocoddyl::ActionModelAbstract> createSwingFootModel(double timeStep, std::vector<pinocchio::FrameIndex> supportFootIds, Eigen::Vector3d comTask, std::vector<std::pair<pinocchio::FrameIndex, Eigen::Vector3d>> footTask) {
    // Creating a 6D multi-contact model, and then including the supporting
    // foot
    int nu;
    if (this->fwddyn) {
      nu = actuation->get_nu();
    } else {
      nu = state->get_nv() + 6 * supportFootIds.size();
    }
    boost::shared_ptr<crocoddyl::ContactModelMultiple> contactModel = boost::make_shared<crocoddyl::ContactModelMultiple>(state, nu);
    for (int i=0; i< supportFootIds.size(); i++) {
      boost::shared_ptr<crocoddyl::ContactModelAbstract> supportContactModel =
        boost::make_shared<crocoddyl::ContactModel6D>(
                                                      state, supportFootIds[i], pinocchio::SE3::Identity(),
                                                      pinocchio::LOCAL_WORLD_ALIGNED, nu,
                                                      Eigen::Vector2d(0., 50.));
      contactModel->addContact(model->frames[supportFootIds[i]].name + "_contact", supportContactModel);
    }

    // Creating the cost model for a contact phase
    boost::shared_ptr<crocoddyl::CostModelSum> costModel =
      boost::make_shared<crocoddyl::CostModelSum>(state, nu);
    if (comTask != Eigen::Vector3d::Zero()) {
      boost::shared_ptr<crocoddyl::CostModelAbstract> comTrack =
      boost::make_shared<crocoddyl::CostModelResidual>(
          state, boost::make_shared<crocoddyl::ResidualModelCoMPosition>(
                     state, comTask, nu));
      costModel->addCost("comTrack", comTrack, this->com_track_w);
    }
    for (int i=0; i< supportFootIds.size(); i++) {
      crocoddyl::WrenchCone cone(Rsurf, mu, Eigen::Vector2d(0.1, 0.05));
      boost::shared_ptr<crocoddyl::CostModelAbstract> wrenchCone =
      boost::make_shared<crocoddyl::CostModelResidual>(
          state, boost::make_shared<
          crocoddyl::ActivationModelQuadraticBarrier>(
                                                      crocoddyl::ActivationBounds(cone.get_lb(), cone.get_ub())),
          boost::make_shared<crocoddyl::ResidualModelContactWrenchCone>(
                                                                        state, supportFootIds[i], cone, nu, this->fwddyn));
      costModel->addCost(model->frames[supportFootIds[i]].name + "_wrenchCone", wrenchCone, 1e1);
    }
    for (int i=0; i<footTask.size(); i++) {
      boost::shared_ptr<crocoddyl::CostModelAbstract> footTrack =
      boost::make_shared<crocoddyl::CostModelResidual>(
          state, boost::make_shared<crocoddyl::ResidualModelFramePlacement>(
                                                                            state, footTask[i].first, pinocchio::SE3(Eigen::Matrix3d::Identity(), footTask[i].second), nu));
      costModel->addCost(model->frames[footTask[i].first].name + "_footTrack", footTrack, this->feet_track_w);

    }

    Eigen::VectorXd stateWeights(model->nv*2);
    stateWeights.head<3>().fill(0.);
    stateWeights.segment(3,3).fill(pow(this->root_w,2));
    stateWeights.segment(6, 3).fill(pow(this->waist_w, 2));
    stateWeights.segment(6 + 3, 2 + 16 + 12).fill(pow(this->arm_w, 2));
    stateWeights.segment(6 + 3 + 2 + 16 + 12, model->nv - 6 - 3 - 2 - 16 - 12).fill(pow(this->state_w, 2));
    stateWeights.segment(model->nv, model->nv).fill(pow(10, 2));
    boost::shared_ptr<crocoddyl::ActivationModelAbstract> stateActivation =
      boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(stateWeights);
    Eigen::VectorXd x0(model->nq + model->nv);
    x0 << q0, Eigen::VectorXd::Zero(model->nv);
    boost::shared_ptr<crocoddyl::CostModelAbstract> stateReg =
      boost::make_shared<crocoddyl::CostModelResidual>(
                                                       state, stateActivation,
                                                       boost::make_shared<crocoddyl::ResidualModelState>(state, x0, nu));
    boost::shared_ptr<crocoddyl::CostModelAbstract> ctrlReg;
    if (this->fwddyn) {
      ctrlReg =
        boost::make_shared<crocoddyl::CostModelResidual>(
                                                         state, 
                                                         boost::make_shared<crocoddyl::ResidualModelControl>(state, nu));
    } else {
      ctrlReg =
        boost::make_shared<crocoddyl::CostModelResidual>(
                                                         state, 
                                                         boost::make_shared<crocoddyl::ResidualModelJointEffort>(state, actuation, nu));
    }
    costModel->addCost("stateReg", stateReg, 1e1);
    costModel->addCost("ctrlReg", ctrlReg, 1e-1);

    // Creating the action model for the KKT dynamics with simpletic Euler
    // integration scheme
    boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> dmodel;
    if (this->fwddyn) {
      dmodel = boost::make_shared<
          crocoddyl::DifferentialActionModelContactFwdDynamics>(
                                                                state, actuation, contactModel, costModel, 0.0, true);
    } else {
      dmodel = boost::make_shared<
          crocoddyl::DifferentialActionModelContactInvDynamics>(
                                                                state, actuation, contactModel, costModel);
    }

    return boost::make_shared<crocoddyl::IntegratedActionModelEuler>(dmodel, timeStep);
  };

std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> createFootStepModels (Eigen::Vector3d& comPos0, std::vector<std::pair<pinocchio::FrameIndex, Eigen::Vector3d>>& feetPos0, double stepLength, double stepHeight, int timeStep, int numKnots, std::vector<pinocchio::FrameIndex> supportFootIds, std::vector<pinocchio::FrameIndex> swingFootIds)
  {
    int numLegs = supportFootIds.size() + swingFootIds.size();
    double comPercentage = (double)swingFootIds.size() / (double)numLegs;

    // Action models for the foot swing
    std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> footStepModels;
    std::vector<std::pair<pinocchio::FrameIndex, Eigen::Vector3d>> footTask;
    for (int k=0; k<numKnots; k++) {
      footTask.clear();
      for (int i=0; i<swingFootIds.size(); i++) {
        double phKnots = (double)numKnots / 2;
        Eigen::Vector3d dp = Eigen::Vector3d::Zero();
        if (k < phKnots) {
          dp = Eigen::Vector3d(stepLength * (k+1) / numKnots, 0.0, stepHeight * k / phKnots);
        } else if (k == phKnots) {
          dp = Eigen::Vector3d(stepLength * (k+1) / numKnots, 0.0, stepHeight);
        } else {
          dp = Eigen::Vector3d(stepLength * (k+1) / numKnots, 0.0, stepHeight * (1.0 - (k - phKnots) / phKnots));
        }
        Eigen::Vector3d feetPos = Eigen::Vector3d::Zero();
        for (int j=0; j<feetPos0.size(); j++) {
          if (feetPos0[j].first == swingFootIds[i]) feetPos = feetPos0[j].second;
        }
        footTask.push_back(std::pair<pinocchio::FrameIndex, Eigen::Vector3d>(swingFootIds[i], feetPos + dp));
      }
      Eigen::Vector3d comTask = Eigen::Vector3d(stepLength * (k+1) / numKnots, 0.0, 0.0) * comPercentage + comPos0;
      footStepModels.push_back(this->createSwingFootModel(timeStep, supportFootIds, comTask, footTask));
    }

    // Action model for the foot switch
    footStepModels.push_back(this->createFootSwitchModel(supportFootIds, comPos0 + Eigen::Vector3d(stepLength * comPercentage, 0.0, 0.0), footTask));

    // Updating the current foot position for next step
    comPos0 += Eigen::Vector3d(stepLength * comPercentage, 0.0, 0.0);

    for (int i=0; i< swingFootIds.size(); i++) {
      for (int j=0; j<feetPos0.size(); j++) {
        if (swingFootIds[i] == feetPos0[j].first) feetPos0[j].second += Eigen::Vector3d(stepLength, 0.0, 0.0);
      }
    }

    return footStepModels;
  };

  boost::shared_ptr<crocoddyl::ActionModelAbstract> createFootSwitchModel(std::vector<pinocchio::FrameIndex> supportFootIds, Eigen::Vector3d comTask, std::vector<std::pair<pinocchio::FrameIndex, Eigen::Vector3d>> swingFootTask)
  {
    int nu;
    if (this->fwddyn) {
      nu = actuation->get_nu();
    } else {
      nu = state->get_nv() + 6 * supportFootIds.size();
    }

    boost::shared_ptr<crocoddyl::ContactModelMultiple> contactModel = boost::make_shared<crocoddyl::ContactModelMultiple>(state, nu);
    for (int i=0; i< supportFootIds.size(); i++) {
      boost::shared_ptr<crocoddyl::ContactModelAbstract> supportContactModel =
        boost::make_shared<crocoddyl::ContactModel6D>(
                                                      state, supportFootIds[i], pinocchio::SE3::Identity(),
                                                      pinocchio::LOCAL_WORLD_ALIGNED, nu,
                                                      Eigen::Vector2d(0., 50.));
      contactModel->addContact(model->frames[supportFootIds[i]].name + "_contact", supportContactModel);
    }

    // Creating the cost model for a contact phase
    boost::shared_ptr<crocoddyl::CostModelSum> costModel =
      boost::make_shared<crocoddyl::CostModelSum>(state, nu);
    if (comTask != Eigen::Vector3d::Zero()) {
      boost::shared_ptr<crocoddyl::CostModelAbstract> comTrack =
      boost::make_shared<crocoddyl::CostModelResidual>(
          state, boost::make_shared<crocoddyl::ResidualModelCoMPosition>(
                     state, comTask, nu));
      costModel->addCost("comTrack", comTrack, this->com_track_w);
    }

    for (int i=0; i< supportFootIds.size(); i++) {
      crocoddyl::WrenchCone cone(Rsurf, mu, Eigen::Vector2d(0.1, 0.05));
      boost::shared_ptr<crocoddyl::CostModelAbstract> wrenchCone =
      boost::make_shared<crocoddyl::CostModelResidual>(
          state, boost::make_shared<
          crocoddyl::ActivationModelQuadraticBarrier>(
                                                      crocoddyl::ActivationBounds(cone.get_lb(), cone.get_ub())),
          boost::make_shared<crocoddyl::ResidualModelContactWrenchCone>(
                                                                        state, supportFootIds[i], cone, nu, this->fwddyn));
      costModel->addCost(model->frames[supportFootIds[i]].name + "_wrenchCone", wrenchCone, 1e1);
    }

    for (int i=0; i<swingFootTask.size(); i++) {
      boost::shared_ptr<crocoddyl::CostModelAbstract> footTrack =
        boost::make_shared<crocoddyl::CostModelResidual>(
          state, boost::make_shared<crocoddyl::ResidualModelFramePlacement>(
                                                                            state, swingFootTask[i].first, pinocchio::SE3(Eigen::Matrix3d::Identity(), swingFootTask[i].second), nu));
      costModel->addCost(model->frames[swingFootTask[i].first].name + "_footTrack", footTrack, 1e8);
      boost::shared_ptr<crocoddyl::CostModelAbstract> impulseFootVelCost =
        boost::make_shared<crocoddyl::CostModelResidual>(
          state, boost::make_shared<crocoddyl::ResidualModelFrameVelocity>(
              state, swingFootTask[i].first, pinocchio::Motion::Zero(),
              pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, nu));
          costModel->addCost(model->frames[swingFootTask[i].first].name + "_impulseVel", impulseFootVelCost, 1e6);
    }

    Eigen::VectorXd stateWeights(model->nv*2);
    stateWeights.head<3>().fill(0.);
    stateWeights.segment(3,3).fill(pow(this->root_w,2));
    stateWeights.segment(6, 3).fill(pow(this->waist_w, 2));
    stateWeights.segment(6 + 3, 2 + 16 + 12).fill(pow(this->arm_w, 2));
    stateWeights.segment(6 + 3 + 2 + 16 + 12, model->nv - 6 - 3 - 2 - 16 - 12).fill(pow(this->state_w, 2));
    //    stateWeights.segment(6, model->nv - 6).fill(pow(this->state_w, 2));
    boost::shared_ptr<crocoddyl::ActivationModelAbstract> stateActivation =
      boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(stateWeights);
    Eigen::VectorXd x0(model->nq + model->nv);
    x0 << q0, Eigen::VectorXd::Zero(model->nv);
    boost::shared_ptr<crocoddyl::CostModelAbstract> stateReg =
      boost::make_shared<crocoddyl::CostModelResidual>(
                                                       state, stateActivation,
                                                       boost::make_shared<crocoddyl::ResidualModelState>(state, x0, nu));
    boost::shared_ptr<crocoddyl::CostModelAbstract> ctrlReg;
    if (this->fwddyn) {
      ctrlReg =
        boost::make_shared<crocoddyl::CostModelResidual>(
                                                         state, 
                                                         boost::make_shared<crocoddyl::ResidualModelControl>(state, nu));
    } else {
      ctrlReg =
        boost::make_shared<crocoddyl::CostModelResidual>(
                                                         state, 
                                                         boost::make_shared<crocoddyl::ResidualModelJointEffort>(state, actuation, nu));
    }    costModel->addCost("stateReg", stateReg, 1e1);
    costModel->addCost("ctrlReg", ctrlReg, 1e-3);

    boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> dmodel;
    if (this->fwddyn) {
      dmodel = boost::make_shared<
          crocoddyl::DifferentialActionModelContactFwdDynamics>(
                                                                state, actuation, contactModel, costModel, 0.0, true);
    } else {
      dmodel = boost::make_shared<
          crocoddyl::DifferentialActionModelContactInvDynamics>(
                                                                state, actuation, contactModel, costModel);
    }

    return boost::make_shared<crocoddyl::IntegratedActionModelEuler>(dmodel, 0.0);
  };

protected:
  boost::shared_ptr<pinocchio::Model> model;
  pinocchio::Data data;
  Eigen::VectorXd q0;
  boost::shared_ptr<crocoddyl::StateMultibody> state;
  boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation;
  pinocchio::FrameIndex rf_id;
  pinocchio::FrameIndex lf_id;
  bool firstStep = true;
  double mu = 0.7;
  Eigen::Matrix3d Rsurf = Eigen::Matrix3d::Identity();
  bool fwddyn = true;
  double com_track_w = 1e6;
  double feet_track_w = 1e6;
  double root_w = 500;
  double waist_w = 0.01;
  double arm_w = 0.01;
  double state_w = 0.01;
  int num_steps = 2;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "simple_biped_gait");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  ros::Publisher joint_pub = nh.advertise<sensor_msgs::JointState>("joint_states", 10);
  tf::TransformBroadcaster robot_base_broadcaster;
  double stepLength = 0.6;
  double stepHeight = 0.1;
  double timeStep = 0.03;
  int stepKnots = 35;
  int supportKnots = 10;
  bool fwddyn = true;
  double com_track_w = 1e6;
  double feet_track_w = 1e6;
  double state_w = 0.01;
  double root_w = 500;
  double waist_w = 0.01;
  double arm_w = 0.01;
  int num_steps = 2;
  double viewer_ratio = 1.0;
  int num_iter = 100;
  double stop_th = 0.0000001;
  int num_thread = 8;
  pnh.getParam("stepLength", stepLength);
  pnh.getParam("stepHeight", stepHeight);
  pnh.getParam("timeStep", timeStep);
  pnh.getParam("stepKnots", stepKnots);
  pnh.getParam("supportKnots", supportKnots);
  pnh.getParam("fwddyn", fwddyn);
  pnh.getParam("com_track_w", com_track_w);
  pnh.getParam("feet_track_w", feet_track_w);
  pnh.getParam("root_w", root_w);
  pnh.getParam("waist_w", waist_w);
  pnh.getParam("arm_w", arm_w);
  pnh.getParam("state_w", state_w);
  pnh.getParam("num_steps", num_steps);
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
  std::vector<Eigen::VectorXd> us;

  SimpleBipedGaitProblem gait(model, "RLEG_LINK5", "LLEG_LINK5", q0, fwddyn, com_track_w, feet_track_w, root_w, waist_w, arm_w, state_w, num_steps);
  if (fwddyn) {
    crocoddyl::SolverFDDP solver(gait.createWalkingProblem(
                                                            x0,
                                                            stepLength,
                                                            stepHeight,
                                                            timeStep,
                                                            stepKnots,
                                                            supportKnots
                                                            ));
    solver.set_th_stop(stop_th);
    std::vector<Eigen::VectorXd> xs_init;
    for (int i=0; i<solver.get_problem()->get_T(); i++) xs_init.push_back(x0);
    std::vector<Eigen::VectorXd> us_init = solver.get_problem()->quasiStatic_xs(xs_init);
    xs_init.push_back(x0);

    solver.get_problem()->set_nthreads(num_thread);
    //  std::cerr << "nthread " << omp_get_max_threads() << std::endl;

    crocoddyl::Timer timer;
    std::cerr << "Problem solved: " << solver.solve(xs_init, us_init, num_iter, false) << std::endl;
    double time = timer.get_duration();
    std::cerr << "total calculation time:" << time << std::endl;
    std::cerr << "Number of iterations: " << solver.get_iter() << std::endl;
    std::cerr << "time per iterate:" << time / solver.get_iter() << std::endl;
    std::cerr << "Total cost: " << solver.get_cost() << std::endl;
    std::cerr << "Gradient norm: " << solver.get_stop() << std::endl;
    xs = solver.get_xs();
    us = solver.get_us();

  } else {
    crocoddyl::SolverIntro solver(gait.createWalkingProblem(
                                                            x0,
                                                            stepLength,
                                                            stepHeight,
                                                            timeStep,
                                                            stepKnots,
                                                            supportKnots
                                                            ));
      solver.set_th_stop(stop_th);
      std::vector<Eigen::VectorXd> xs_init;
      for (int i=0; i<solver.get_problem()->get_T(); i++) xs_init.push_back(x0);
      std::vector<Eigen::VectorXd> us_init = solver.get_problem()->quasiStatic_xs(xs_init);
      xs_init.push_back(x0);

      solver.get_problem()->set_nthreads(num_thread);
      //  std::cerr << "nthread " << omp_get_max_threads() << std::endl;

      crocoddyl::Timer timer;
      std::cerr << "Problem solved: " << solver.solve(xs_init, us_init, num_iter, false) << std::endl;
      double time = timer.get_duration();
      std::cerr << "total calculation time:" << time << std::endl;
      std::cerr << "Number of iterations: " << solver.get_iter() << std::endl;
      std::cerr << "time per iterate:" << time / solver.get_iter() << std::endl;
      std::cerr << "Total cost: " << solver.get_cost() << std::endl;
      std::cerr << "Gradient norm: " << solver.get_stop() << std::endl;
      xs = solver.get_xs();
      us = solver.get_us();
  }

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

