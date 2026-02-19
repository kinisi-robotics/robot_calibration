/*
 * Copyright (C) 2018-2022 Michael Ferguson
 * Copyright (C) 2015 Fetch Robotics Inc.
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

#include <ctime>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <robot_calibration_msgs/msg/calibration_data.hpp>
#include <robot_calibration_msgs/msg/capture_config.hpp>
#include <robot_calibration_msgs/srv/capture_calibration.hpp>

#include <robot_calibration/optimization/ceres_optimizer.hpp>
#include <robot_calibration/optimization/export.hpp>
#include <robot_calibration/util/calibration_data.hpp>
#include <robot_calibration/util/capture_manager.hpp>
#include <robot_calibration/util/poses_from_bag.hpp>
#include <robot_calibration/util/poses_from_yaml.hpp>

/** \mainpage
 * \section parameters Parameters of the Optimization:
 *   - joint angle offsets
 *   - frame 6DOF corrections (currently head pan frame, and camera frame)
 *   - camera intrinsics (2d & 3d)
 *
 * \section residuals Residual Blocks:
 *   - difference of reprojection through the arm and camera
 *   - residual blocks that limit offsets from growing outrageously large
 *
 * \section modules Modules:
 *   - Capture:
 *     - move joints to a particular place
 *     - wait to settle
 *     - find target (led or checkerboard)
 *     - write sample to bag file: joint angles, position of targets in camera.
 *   - Calibrate:
 *     - load urdf, samples from bag file.
 *     - create arm and camera reprojection chains.
 *     - create residual blocks.
 *     - run calibration.
 *     - write results to URDF.
 */

/*
 * usage:
 *  calibrate --manual
 *  calibrate calibration_poses.bag
 *  calibrate --from-bag calibration_data.bag
 */
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::Node::SharedPtr node = std::make_shared<rclcpp::Node>("robot_calibration");
  rclcpp::Logger logger = node->get_logger();

  // Should we be stupidly verbose?
  bool verbose = node->declare_parameter<bool>("verbose", false);

  // The calibration data
  std_msgs::msg::String description_msg;
  std::vector<robot_calibration_msgs::msg::CalibrationData> data;

  // Where should calibration data come from:
  //  --manual          manually trigger after moving robot to poses
  //  --from-bag <bag>  use a bagfile of pre-recorded data
  //  poses.yaml        load poses from a YAML file and capture each pose
  //  poses.bag         load poses from a bagfile and capture each pose
  std::string data_source("calibration_poses.yaml");
  if (argc > 1)
    data_source = argv[1];

  if (data_source.compare("--from-bag") != 0)
  {
    // No name provided for a calibration bag file, must do capture
    robot_calibration::CaptureManager capture_manager;
    if (!capture_manager.init(node))
    {
      // Error will be printed in function
      return -1;
    }

    // Save URDF for calibration/export step
    description_msg.data = capture_manager.getUrdf();

    // Load a set of calibration poses
    std::vector<robot_calibration_msgs::msg::CaptureConfig> poses;
    bool is_manual = (data_source.compare("--manual") == 0 ||
                      data_source.compare("--manual-service") == 0);
    if (!is_manual)
    {
      if (data_source.find(".yaml") != std::string::npos)
      {
        RCLCPP_INFO(logger, "Loading YAML calibration poses from %s", data_source.c_str());
        if (!robot_calibration::getPosesFromYaml(data_source, poses))
        {
          // Error will be printed in function
          return -1;
        }
      }
      else
      {
        RCLCPP_INFO(logger, "Loading bagfile calibration poses from %s", data_source.c_str());
        if (!robot_calibration::getPosesFromBag(data_source, poses))
        {
          // Error will be printed in function
          return -1;
        }
      }
    }
    else
    {
      RCLCPP_INFO(logger, "Using manual calibration mode (%s)", data_source.c_str());
    }

    // Service-driven manual mode: expose a ROS2 service to trigger each capture.
    // This replaces the stdin-based --manual mode with proper synchronous feedback.
    if (data_source.compare("--manual-service") == 0)
    {
      bool capture_done = false;

      // Services are hosted on a dedicated lightweight node, NOT on the main calibration node.
      // captureFeatures() -> find() -> waitForMsg() calls rclcpp::spin_some(node) internally
      // to receive fresh sensor data. If the services were on 'node' and we added 'node' to a
      // persistent executor, that nested spin_some(node) call would throw "already added to an
      // executor". Using a separate node avoids this: rclcpp::spin_some(service_node) only manages
      // service_node. We also spin 'node' explicitly in the loop so that sensor callbacks keep
      // firing between service calls (ensuring hasData() is satisfied and joint states stay fresh).
      auto service_node = std::make_shared<rclcpp::Node>("robot_calibration_capture_service");

      auto capture_service = service_node->create_service<robot_calibration_msgs::srv::CaptureCalibration>(
        "/capture_calibration",
        [&](const std::shared_ptr<robot_calibration_msgs::srv::CaptureCalibration::Request> request,
            std::shared_ptr<robot_calibration_msgs::srv::CaptureCalibration::Response> response)
        {
          robot_calibration_msgs::msg::CalibrationData msg;
          // Use requested feature names; empty list means capture all configured features.
          std::vector<std::string> features(request->features.begin(), request->features.end());
          if (capture_manager.captureFeatures(features, msg))
          {
            response->success = true;
            response->message = "Captured sample " + std::to_string(data.size() + 1);
            response->data = msg;
            data.push_back(msg);
            RCLCPP_INFO(logger, "Captured sample %zu via service", data.size());
          }
          else
          {
            response->success = false;
            response->message = "Failed to capture features";
            RCLCPP_WARN(logger, "Service-triggered capture failed");
          }
        });

      auto stop_service = service_node->create_service<std_srvs::srv::Trigger>(
        "/stop_calibration_capture",
        [&](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
            std::shared_ptr<std_srvs::srv::Trigger::Response> response)
        {
          capture_done = true;
          response->success = true;
          response->message = "Calibration capture stopping";
          RCLCPP_INFO(logger, "Stop requested via service");
        });

      RCLCPP_INFO(logger, "Calibration capture services ready (/capture_calibration, /stop_calibration_capture)");

      while (!capture_done && rclcpp::ok())
      {
        rclcpp::spin_some(node);
        rclcpp::spin_some(service_node);
        rclcpp::sleep_for(std::chrono::milliseconds(10));
      }

      RCLCPP_INFO(logger, "Done capturing samples");
    }
    else
    // For each pose in the capture sequence.
    for (unsigned pose_idx = 0;
         (pose_idx < poses.size() || poses.empty()) && rclcpp::ok();
         ++pose_idx)
    {
      robot_calibration_msgs::msg::CalibrationData msg;
      if (poses.empty())
      {
        // Manual calibration, wait for keypress
        RCLCPP_INFO(logger, "Press [Enter] to capture a sample... (or type 'done' and [Enter] to finish capture)");
        std::string throwaway;
        std::getline(std::cin, throwaway);
        if (throwaway.compare("done") == 0)
          break;
        if (throwaway.compare("exit") == 0)
          return 0;
        if (!rclcpp::ok())
          break;

        // Empty vector causes us to capture all features
        std::vector<std::string> features;
        if (!capture_manager.captureFeatures(features, msg))
        {
          RCLCPP_WARN(logger, "Failed to capture sample %u.", pose_idx);
          continue;
        }

        RCLCPP_INFO(logger, "Captured pose %u", pose_idx + 1);
      }
      else
      {
        // Move head/arm to pose
        if (!capture_manager.moveToState(poses[pose_idx].joint_states))
        {
          RCLCPP_WARN(logger, "Unable to move to desired state for sample %u.", pose_idx);
          continue;
        }

        // Make sure sensor data is up to date after settling
        rclcpp::sleep_for(std::chrono::milliseconds(100));

        // Get pose of the features
        if (!capture_manager.captureFeatures(poses[pose_idx].features, msg))
        {
          RCLCPP_WARN(logger, "Failed to capture sample %u.", pose_idx);
          continue;
        }

        RCLCPP_INFO(logger, "Captured pose %u of %lu", pose_idx + 1, poses.size());
      }


      // Add to samples
      data.push_back(msg);
    }

    RCLCPP_INFO(logger, "Done capturing samples");
  }
  else
  {
    // Load calibration data from bagfile
    std::string data_bag_name("/tmp/calibration_data.bag");
    if (argc > 2)
      data_bag_name = argv[2];
    RCLCPP_INFO(logger, "Loading calibration data from %s", data_bag_name.c_str());

    if (!robot_calibration::load_bag(data_bag_name, description_msg, data))
    {
      // Error will have been printed in function
      return -1;
    }
  }

  // Create instance of optimizer
  robot_calibration::OptimizationParams params;
  robot_calibration::Optimizer opt(description_msg.data);

  // Load calibration steps
  std::vector<std::string> calibration_steps =
    node->declare_parameter<std::vector<std::string>>("calibration_steps", std::vector<std::string>());
  if (calibration_steps.empty())
  {
    RCLCPP_FATAL(logger, "Parameter calibration_steps is not defined");
    return -1;
  }

  // Run calibration steps
  for (auto step : calibration_steps)
  {
    params.LoadFromROS(node, step);
    opt.optimize(params, data, logger, verbose);
    if (verbose)
    {
      std::cout << "Parameter Offsets:" << std::endl;
      std::cout << opt.getOffsets()->getOffsetYAML() << std::endl;
    }
  }

  // Write outputs
  robot_calibration::exportResults(opt, description_msg.data, data);

  RCLCPP_INFO(logger, "Done calibrating");
  rclcpp::shutdown();

  return 0;
}
