/*
 * Copyright (C) 2018-2024 Michael Ferguson
 * Copyright (C) 2014-2015 Fetch Robotics Inc.
 * Copyright (C) 2013-2014 Unbounded Robotics Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Author: Michael Ferguson

#include <robot_calibration/optimization/ceres_optimizer.hpp>

#include <array>
#include <deque>
#include <memory>
#include <ceres/ceres.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#if __has_include(<urdf/model.hpp>)
#include <urdf/model.hpp>
#else
#include <urdf/model.h>
#endif
#include <kdl_parser/kdl_parser.hpp>
#include <robot_calibration_msgs/msg/calibration_data.hpp>

#include <robot_calibration/optimization/offsets.hpp>
#include <robot_calibration/util/calibration_data.hpp>
#include <robot_calibration/cost_functions/stereo_reprojection_error.hpp>
#include <robot_calibration/cost_functions/chain3d_to_camera2d_error.hpp>
#include <robot_calibration/cost_functions/chain3d_to_chain3d_error.hpp>
#include <robot_calibration/cost_functions/chain3d_to_mesh_error.hpp>
#include <robot_calibration/cost_functions/chain3d_to_plane_error.hpp>
#include <robot_calibration/cost_functions/plane_to_plane_error.hpp>
#include <robot_calibration/cost_functions/outrageous_error.hpp>
#include <robot_calibration/models/camera2d.hpp>
#include <robot_calibration/models/camera3d.hpp>
#include <robot_calibration/models/chain3d.hpp>
#include <string>
#include <map>
#include <limits>

namespace robot_calibration
{

Optimizer::Optimizer(const std::string& robot_description) :
  num_params_(0),
  num_residuals_(0)
{
  model_ = std::make_shared<urdf::Model>();
  if (!model_->initString(robot_description))
    std::cerr << "Failed to parse URDF." << std::endl;

  // Maintain consistent offset parser so we hold onto offsets
  offsets_.reset(new OptimizationOffsets());

  // Create a mesh loader
  mesh_loader_.reset(new MeshLoader(model_));
}

Optimizer::~Optimizer()
{
}

int Optimizer::optimize(OptimizationParams& params,
                        std::vector<robot_calibration_msgs::msg::CalibrationData> data,
                        rclcpp::Logger& logger,
                        bool progress_to_stdout)
{
  // Load KDL from URDF
  if (!kdl_parser::treeFromUrdfModel(*model_, tree_))
  {
    std::cerr << "Failed to construct KDL tree" << std::endl;
    return -1;
  }

  // Create models
  for (size_t i = 0; i < params.models.size(); ++i)
  {
    if (params.models[i].type == "chain3d")
    {
      RCLCPP_INFO_STREAM(logger, "Creating chain '" << params.models[i].name << "' from " <<
                                                       params.base_link << " to " <<
                                                       params.models[i].frame);
      Chain3dModel* model = new Chain3dModel(params.models[i].name, tree_, params.base_link, params.models[i].frame);
      models_[params.models[i].name] = model;
    }
    else if (params.models[i].type == "camera3d")
    {
      RCLCPP_INFO_STREAM(logger, "Creating camera3d '" << params.models[i].name << "' in frame " <<
                                                          params.models[i].frame);
      std::string param_name = params.models[i].param_name;
      if (param_name == "")
      {
        // Default to same name as sensor
        param_name = params.models[i].name;
      }
      Camera3dModel* model = new Camera3dModel(params.models[i].name, param_name, tree_, params.base_link, params.models[i].frame);
      models_[params.models[i].name] = model;
    }
    else if (params.models[i].type == "camera2d")
    {
      RCLCPP_INFO_STREAM(logger, "Creating camera2d '" << params.models[i].name << "' in frame " <<
                                                          params.models[i].frame);
      std::string param_name = params.models[i].param_name;
      if (param_name == "")
      {
        // Default to same name as sensor
        param_name = params.models[i].name;
      }
      Camera2dModel* model = new Camera2dModel(params.models[i].name, param_name, tree_, params.base_link, params.models[i].frame);
      models_[params.models[i].name] = model;
    }
    else
    {
      RCLCPP_ERROR(logger, "Unknown model type: %s", params.models[i].type.c_str());
    }
  }

  // Reset which parameters are free (offset values are retained)
  offsets_->reset();

  // Setup  parameters to calibrate
  for (size_t i = 0; i < params.free_params.size(); ++i)
  {
    offsets_->add(params.free_params[i]);
  }
  for (size_t i = 0; i < params.free_frames.size(); ++i)
  {
    offsets_->addFrame(params.free_frames[i].name,
                       params.free_frames[i].x,
                       params.free_frames[i].y,
                       params.free_frames[i].z,
                       params.free_frames[i].roll,
                       params.free_frames[i].pitch,
                       params.free_frames[i].yaw);
  }
  for (size_t i = 0; i < params.free_frames_initial_values.size(); ++i)
  {
    if (!offsets_->setFrame(params.free_frames_initial_values[i].name,
                            params.free_frames_initial_values[i].x,
                            params.free_frames_initial_values[i].y,
                            params.free_frames_initial_values[i].z,
                            params.free_frames_initial_values[i].roll,
                            params.free_frames_initial_values[i].pitch,
                            params.free_frames_initial_values[i].yaw))
    {
      RCLCPP_ERROR_STREAM(logger, "Error setting initial value for " <<
                          params.free_frames_initial_values[i].name);
    }
  }

  // Allocate space
  double* free_params = new double[offsets_->size()];
  offsets_->initialize(free_params);

  // Houston, we have a problem...
  ceres::Problem* problem = new ceres::Problem();

  // Per-observation board poses for stereo reprojection error blocks.
  // Using deque so that references remain stable as elements are added.
  std::deque<std::array<double, 6>> board_poses;

  // For each sample of data:
  for (size_t i = 0; i < data.size(); ++i)
  {
    for (size_t j = 0; j < params.error_blocks.size(); ++j)
    {
      if (params.error_blocks[j]->type == "chain3d_to_chain3d")
      {
        // This error block can process data generated by the LedFinder,
        // CheckboardFinder, or any other finder that can sample the pose
        // of one or more data points that are connected at a constant offset
        // from a link a kinematic chain (the "arm").
        auto p = std::dynamic_pointer_cast<OptimizationParams::Chain3dToChain3dParams>(params.error_blocks[j]);
        std::string a_name = p->model_a;
        std::string b_name = p->model_b;

        // Do some basic error checking for bad params
        if (a_name == "" || b_name == "" || a_name == b_name)
        {
          RCLCPP_ERROR(logger, "chain3d_to_chain3d improperly configured: model_a and model_b params must be set!");
          return 0;
        }

        // Check that this sample has the required features/observations
        if (!hasSensor(data[i], a_name) || !hasSensor(data[i], b_name))
          continue;

        // Create the block
        ceres::CostFunction * cost = Chain3dToChain3d::Create(models_[a_name],
                                                              models_[b_name],
                                                              offsets_.get(),
                                                              data[i]);

        // Output initial error
        if (progress_to_stdout)
        {
          double ** params = new double*[1];
          params[0] = free_params;
          double * residuals = new double[cost->num_residuals()];

          cost->Evaluate(params, residuals, NULL);

          std::cout << "INITIAL COST (" << i << ")" << std::endl << "  x: ";
          for (size_t k = 0; k < static_cast<size_t>(cost->num_residuals() / 3); ++k)
            std::cout << "  " << std::setw(10) << std::fixed << residuals[(3*k + 0)];
          std::cout << std::endl << "  y: ";
          for (size_t k = 0; k < static_cast<size_t>(cost->num_residuals() / 3); ++k)
            std::cout << "  " << std::setw(10) << std::fixed << residuals[(3*k + 1)];
          std::cout << std::endl << "  z: ";
          for (size_t k = 0; k < static_cast<size_t>(cost->num_residuals() / 3); ++k)
            std::cout << "  " << std::setw(10) << std::fixed << residuals[(3*k + 2)];
          std::cout << std::endl << std::endl;
        }

        problem->AddResidualBlock(cost,
                                  NULL,  // squared loss
                                  free_params);
      }
      else if (params.error_blocks[j]->type == "chain3d_to_plane")
      {
        // This error block can process data generated by the PlaneFinder
        auto p = std::dynamic_pointer_cast<OptimizationParams::Chain3dToPlaneParams>(params.error_blocks[j]);
        std::string chain_name = p->model;

        // Do some basic error checking for bad params
        if (chain_name == "")
        {
          RCLCPP_ERROR(logger, "chain3d_to_plane improperly configured: model param must be set!");
          return 0;
        }

        // Check that this sample has the required features/observations
        if (!hasSensor(data[i], chain_name))
          continue;

        // Create the block
        ceres::CostFunction * cost =
          Chain3dToPlane::Create(models_[chain_name],
                                 offsets_.get(),
                                 data[i],
                                 p->a,
                                 p->b,
                                 p->c,
                                 p->d,
                                 p->scale);

        // Output initial error
        if (progress_to_stdout)
        {
          double ** params = new double*[1];
          params[0] = free_params;
          double * residuals = new double[cost->num_residuals()];

          cost->Evaluate(params, residuals, NULL);

          std::cout << "INITIAL COST (" << i << ")" << std::endl << "  d: ";
          for (size_t k = 0; k < static_cast<size_t>(cost->num_residuals()); ++k)
            std::cout << "  " << std::setw(10) << std::fixed << residuals[(k)];
          std::cout << std::endl << std::endl;
        }

        problem->AddResidualBlock(cost,
                                  NULL /* squared loss */,
                                  free_params);
      }
      else if (params.error_blocks[j]->type == "chain3d_to_mesh")
      {
        // This error block can process data generated by the RobotFinder
        auto p = std::dynamic_pointer_cast<OptimizationParams::Chain3dToMeshParams>(params.error_blocks[j]);
        std::string chain_name = p->model;

        // Do some basic error checking for bad params
        if (chain_name == "")
        {
          RCLCPP_ERROR(logger, "chain3d_to_mesh improperly configured: model param must be set!");
          return 0;
        }

        // Check that this sample has the required features/observations
        if (!hasSensor(data[i], chain_name))
          continue;

        // Get the mesh (apply optional override resource URI from this error block)
        MeshPtr mesh = mesh_loader_->getCollisionMesh(p->link_name, p->mesh_override);
        if (!mesh)
        {
          RCLCPP_ERROR(logger, "chain3d_to_mesh improperly configured: cannot load mesh for %s", p->link_name.c_str());
          return 0;
        }

        // Create the block
        ceres::CostFunction * cost =
          Chain3dToMesh::Create(models_[chain_name],
                                offsets_.get(),
                                data[i],
                                mesh);

        // Output initial error
        if (progress_to_stdout)
        {
          double ** params = new double*[1];
          params[0] = free_params;
          double * residuals = new double[cost->num_residuals()];

          cost->Evaluate(params, residuals, NULL);

          std::cout << "INITIAL COST (" << i << ")" << std::endl << "  d: ";
          for (size_t k = 0; k < static_cast<size_t>(cost->num_residuals()); ++k)
            std::cout << "  " << std::setw(10) << std::fixed << residuals[(k)];
          std::cout << std::endl << std::endl;
        }

        problem->AddResidualBlock(cost,
                                  NULL,//new ceres::CauchyLoss(0.05),
                                  free_params);


      }
      else if (params.error_blocks[j]->type == "chain3d_to_camera2d")
      {
        // This error block can process data generated by the CheckerboardFinder2d,
        auto p = std::dynamic_pointer_cast<OptimizationParams::Chain3dToCamera2dParams>(params.error_blocks[j]);

        // Do some basic error checking for bad params
        if (p->model_3d == "" || p->model_2d == "")
        {
          RCLCPP_ERROR(logger, "chain3d_to_camera2d improperly configured: model_3d and model_2d params must be set!");
          return 0;
        }

        // Check that this sample has the required features/observations
        if (!hasSensor(data[i], p->model_3d) || !hasSensor(data[i], p->model_2d))
        {
          continue;
        }

        // Have to cast our Camera2d model
        auto camera_model = dynamic_cast<Camera2dModel*>(models_[p->model_2d]);
        if (!camera_model)
        {
          RCLCPP_ERROR(logger, "camera2d model is improperly specified");
          return 0;
        }

        // Create the block
        ceres::CostFunction * cost = Chain3dToCamera2d::Create(models_[p->model_3d],
                                                               camera_model,
                                                               p->scale,
                                                               offsets_.get(),
                                                               data[i]);

        // Output initial error
        if (progress_to_stdout)
        {
          double ** params = new double*[1];
          params[0] = free_params;
          double * residuals = new double[cost->num_residuals()];

          cost->Evaluate(params, residuals, NULL);

          std::cout << "INITIAL COST (" << i << ")" << std::endl << "  x: ";
          for (size_t k = 0; k < static_cast<size_t>(cost->num_residuals() / 2); ++k)
            std::cout << "  " << std::setw(10) << std::fixed << residuals[(2*k + 0)];
          std::cout << std::endl << "  y: ";
          for (size_t k = 0; k < static_cast<size_t>(cost->num_residuals() / 2); ++k)
            std::cout << "  " << std::setw(10) << std::fixed << residuals[(2*k + 1)];
          std::cout << std::endl << std::endl;
        }

        problem->AddResidualBlock(cost,
                                  NULL,  // squared loss
                                  free_params);
      }
      else if (params.error_blocks[j]->type == "camera2d_to_camera2d")
      {
        // Stereo reprojection error: estimates per-observation checkerboard pose
        // and computes pixel reprojection error in both cameras.
        auto p = std::dynamic_pointer_cast<OptimizationParams::Camera2dToCamera2dParams>(params.error_blocks[j]);
        std::string a_name = p->model_a;
        std::string b_name = p->model_b;

        if (a_name == "" || b_name == "" || a_name == b_name)
        {
          RCLCPP_ERROR(logger, "camera2d_to_camera2d improperly configured: model_a and model_b params must be set!");
          return 0;
        }

        if (p->points_x <= 0 || p->points_y <= 0 || p->point_size <= 0.0)
        {
          RCLCPP_ERROR(logger, "camera2d_to_camera2d requires points_x, points_y, and size parameters");
          return 0;
        }

        if (!hasSensor(data[i], a_name) || !hasSensor(data[i], b_name))
          continue;

        auto camera_a = dynamic_cast<Camera2dModel*>(models_[a_name]);
        auto camera_b = dynamic_cast<Camera2dModel*>(models_[b_name]);
        if (!camera_a || !camera_b)
        {
          RCLCPP_ERROR(logger, "camera2d_to_camera2d requires both models to be camera2d type");
          return 0;
        }

        // Build checkerboard grid points (z=0 plane, origin at first corner)
        std::vector<geometry_msgs::msg::Point> grid_points;
        for (int gy = 0; gy < p->points_y; ++gy)
        {
          for (int gx = 0; gx < p->points_x; ++gx)
          {
            geometry_msgs::msg::Point pt;
            pt.x = gx * p->point_size;
            pt.y = gy * p->point_size;
            pt.z = 0.0;
            grid_points.push_back(pt);
          }
        }

        // Allocate per-observation board pose (persists through optimization)
        board_poses.emplace_back();
        auto& pose = board_poses.back();
        pose.fill(0.0);

        // Initialize board pose via PnP using camera A's intrinsics and observed features.
        // This gives the board pose in camera A's frame, then we transform to world frame.
        {
          int sensor_idx = getSensorIndex(data[i], a_name);
          if (sensor_idx >= 0 &&
              data[i].observations[sensor_idx].features.size() == grid_points.size())
          {
            // Get camera A intrinsics from the observation's camera info (P matrix)
            const auto& cam_info = data[i].observations[sensor_idx].ext_camera_info.camera_info;
            double fx = cam_info.p[0];
            double fy = cam_info.p[5];
            double cx = cam_info.p[2];
            double cy = cam_info.p[6];

            // Build 3D object points and 2D image points for solvePnP
            std::vector<cv::Point3f> obj_pts(grid_points.size());
            std::vector<cv::Point2f> img_pts(grid_points.size());
            for (size_t k = 0; k < grid_points.size(); ++k)
            {
              obj_pts[k] = cv::Point3f(
                  static_cast<float>(grid_points[k].x),
                  static_cast<float>(grid_points[k].y),
                  static_cast<float>(grid_points[k].z));
              img_pts[k] = cv::Point2f(
                  static_cast<float>(data[i].observations[sensor_idx].features[k].point.x),
                  static_cast<float>(data[i].observations[sensor_idx].features[k].point.y));
            }

            cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
                fx, 0, cx,
                0, fy, cy,
                0, 0, 1);
            cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);  // features are from rectified image

            cv::Mat rvec, tvec;
            if (cv::solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs,
                              rvec, tvec, false, cv::SOLVEPNP_ITERATIVE))
            {
              // PnP gives board pose in camera A's frame: T_cam_board
              // We need board pose in world frame: T_world_board = T_world_cam * T_cam_board
              cv::Mat R_cv;
              cv::Rodrigues(rvec, R_cv);

              KDL::Rotation R_cam_board(
                  R_cv.at<double>(0, 0), R_cv.at<double>(0, 1), R_cv.at<double>(0, 2),
                  R_cv.at<double>(1, 0), R_cv.at<double>(1, 1), R_cv.at<double>(1, 2),
                  R_cv.at<double>(2, 0), R_cv.at<double>(2, 1), R_cv.at<double>(2, 2));
              KDL::Vector t_cam_board(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
              KDL::Frame T_cam_board(R_cam_board, t_cam_board);

              // Get camera A's FK: T_world_cam
              KDL::Frame T_world_cam = camera_a->getChainFK(*offsets_, data[i].joint_states);
              KDL::Frame T_world_board = T_world_cam * T_cam_board;

              double rx, ry, rz;
              axis_magnitude_from_rotation(T_world_board.M, rx, ry, rz);
              pose[0] = T_world_board.p.x();
              pose[1] = T_world_board.p.y();
              pose[2] = T_world_board.p.z();
              pose[3] = rx;
              pose[4] = ry;
              pose[5] = rz;
            }
            else
            {
              RCLCPP_WARN(logger, "stereo_camera_error: solvePnP failed for observation %zu", i);
            }
          }
          else
          {
            RCLCPP_WARN(logger, "stereo_camera_error: no camera data for PnP init, observation %zu", i);
          }
        }

        ceres::CostFunction* cost = StereoReprojectionError::Create(
            camera_a, camera_b, p->scale, offsets_.get(), data[i], grid_points);

        if (progress_to_stdout)
        {
          double* eval_params[2] = {free_params, pose.data()};
          double* residuals = new double[cost->num_residuals()];

          cost->Evaluate(eval_params, residuals, NULL);

          size_t n_feat = static_cast<size_t>(cost->num_residuals() / 4);
          std::cout << "STEREO INITIAL COST (" << i << ")" << std::endl << "  cam_a x: ";
          for (size_t k = 0; k < n_feat; ++k)
            std::cout << "  " << std::setw(10) << std::fixed << residuals[4 * k + 0];
          std::cout << std::endl << "  cam_a y: ";
          for (size_t k = 0; k < n_feat; ++k)
            std::cout << "  " << std::setw(10) << std::fixed << residuals[4 * k + 1];
          std::cout << std::endl << "  cam_b x: ";
          for (size_t k = 0; k < n_feat; ++k)
            std::cout << "  " << std::setw(10) << std::fixed << residuals[4 * k + 2];
          std::cout << std::endl << "  cam_b y: ";
          for (size_t k = 0; k < n_feat; ++k)
            std::cout << "  " << std::setw(10) << std::fixed << residuals[4 * k + 3];
          std::cout << std::endl << std::endl;

          delete[] residuals;
        }

        // Use Cauchy loss to be robust to outlier observations (e.g., misdetected corners).
        // Cauchy loss transitions from quadratic to logarithmic at residual = scale parameter.
        // Scale of 2.0 pixels: residuals < 2px behave as squared loss, larger ones are downweighted.
        problem->AddResidualBlock(cost,
                                  new ceres::CauchyLoss(2.0),
                                  free_params, pose.data());
      }
      else if (params.error_blocks[j]->type == "plane_to_plane")
      {
        // This error block can process data generated by the PlaneFinder,
        // CheckerboardFinder, or any other finder that returns a series of
        // planar points.
        auto p = std::dynamic_pointer_cast<OptimizationParams::PlaneToPlaneParams>(params.error_blocks[j]);
        std::string a_name = p->model_a;
        std::string b_name = p->model_b;

        // Do some basic error checking for bad params
        if (a_name == "" || b_name == "" || a_name == b_name)
        {
          RCLCPP_ERROR(logger, "plane_to_plane improperly configured: model_a and model_a params must be set!");
          return 0;
        }

        // Check that this sample has the required features/observations
        if (!hasSensor(data[i], a_name) || !hasSensor(data[i], b_name))
          continue;

        // Create the block
        ceres::CostFunction * cost =
          PlaneToPlaneError::Create(models_[a_name],
                                    models_[b_name],
                                    offsets_.get(),
                                    data[i],
                                    p->normal_scale,
                                    p->offset_scale);

        // Output initial error
        if (progress_to_stdout)
        {
          double ** params = new double*[1];
          params[0] = free_params;
          double * residuals = new double[cost->num_residuals()];

          cost->Evaluate(params, residuals, NULL);
          std::cout << "INITIAL COST (" << i << ")" << std::endl << "  a: ";
          std::cout << "  " << std::setw(10) << std::fixed << residuals[0];
          std::cout << std::endl << "  b: ";
          std::cout << "  " << std::setw(10) << std::fixed << residuals[1];
          std::cout << std::endl << "  c: ";
          std::cout << "  " << std::setw(10) << std::fixed << residuals[2];
          std::cout << std::endl << "  d: ";
          std::cout << "  " << std::setw(10) << std::fixed << residuals[3];
          std::cout << std::endl << std::endl;
        }

        problem->AddResidualBlock(cost,
                                  NULL,  // squared loss
                                  free_params);
      }
      else if (params.error_blocks[j]->type == "outrageous")
      {
        // Outrageous error block requires no particular sensors, add to every sample
        auto p = std::dynamic_pointer_cast<OptimizationParams::OutrageousParams>(params.error_blocks[j]);
        problem->AddResidualBlock(
          OutrageousError::Create(offsets_.get(),
                                  p->param,
                                  p->joint_scale,
                                  p->position_scale,
                                  p->rotation_scale),
          NULL, // squared loss
          free_params);
      }
      else
      {
        RCLCPP_ERROR(logger, "Unknown error block: %s", params.error_blocks[j]->type.c_str());
        return 0;
      }
    }
  }

  // Setup the actual optimization
  ceres::Solver::Options options;
  options.use_nonmonotonic_steps = true;
  options.function_tolerance = 1e-10;
  options.max_num_iterations = params.max_num_iterations;
  options.minimizer_progress_to_stdout = progress_to_stdout;

  // Use Schur complement if we have per-observation board poses (bundle adjustment)
  if (!board_poses.empty())
  {
    options.linear_solver_type = ceres::DENSE_SCHUR;

    // Set up parameter block ordering: per-observation poses are eliminated first
    // (group 0), global offsets stay in the reduced system (group 1)
    auto* ordering = new ceres::ParameterBlockOrdering();
    ordering->AddElementToGroup(free_params, 1);  // global offsets: keep
    for (auto& pose : board_poses)
      ordering->AddElementToGroup(pose.data(), 0);  // board poses: eliminate
    options.linear_solver_ordering.reset(ordering);
  }
  else
  {
    options.linear_solver_type = ceres::DENSE_QR;
  }

  if (progress_to_stdout)
    std::cout << "\nSolver output:" << std::endl;
  summary_.reset(new ceres::Solver::Summary());
  ceres::Solve(options, problem, summary_.get());
  if (progress_to_stdout)
    std::cout << "\n" << summary_->BriefReport() << std::endl;

  // Save some status
  num_params_ = problem->NumParameters();
  num_residuals_ = problem->NumResiduals();

  // Compute covariance and update standard deviations in offsets
  {
    // Compute covariance only for global calibration offsets (not per-observation poses)
    int global_param_size = static_cast<int>(offsets_->size());

    ceres::Covariance::Options covariance_options;
    ceres::Covariance covariance(covariance_options);

    std::vector<std::pair<const double*, const double*>> covariance_blocks;
    covariance_blocks.emplace_back(free_params, free_params);

    if (covariance.Compute(covariance_blocks, problem))
    {
      // Prepare stddev vector (default to NaN)
      std::vector<double> param_stddevs(global_param_size, std::numeric_limits<double>::quiet_NaN());

      // Extract covariance block for global offsets and compute standard deviations
      std::vector<double> covariance_matrix(global_param_size * global_param_size);
      if (covariance.GetCovarianceBlock(free_params, free_params, covariance_matrix.data()))
      {
        for (int i = 0; i < global_param_size; ++i)
        {
          double variance = covariance_matrix[i * global_param_size + i];
          double std_dev = std::sqrt(std::max(0.0, variance));
          param_stddevs[i] = std_dev;
        }

        offsets_->setStandardDeviations(param_stddevs);
        RCLCPP_INFO(logger, "Computed covariance and updated standard deviations for %d parameters", global_param_size);
      }
      else
      {
        RCLCPP_WARN(logger, "Covariance computed but failed to retrieve covariance block");
      }
    }
    else
    {
      RCLCPP_WARN(logger, "Failed to compute covariance - problem may be singular or ill-conditioned");
    }
  }

  // Note: the error blocks will be managed by scoped_ptr in cost functor
  //       which takes ownership, and so we do not need to delete them here

  // Done with our free params
  delete[] free_params;
  delete problem;

  return 0;
}

std::vector<std::string> Optimizer::getCameraNames()
{
  std::vector<std::string> camera_names;
  for (auto it = models_.begin(); it != models_.end(); ++it)
  {
    if (it->second->getType() == "Camera3dModel")
    {
       camera_names.push_back(it->first);
    }
  }
  return camera_names;
}

}  // namespace robot_calibration
