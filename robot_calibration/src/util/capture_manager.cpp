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

#include <chrono>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <robot_calibration/util/capture_manager.hpp>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("capture_manager");

namespace robot_calibration
{

CaptureManager::CaptureManager()
{
  description_valid_ = false;
}

bool CaptureManager::init(rclcpp::Node::SharedPtr node)
{
  node_ = node;

  // Publish calibration data (to be recorded by rosbag)
  data_pub_ = node->create_publisher<robot_calibration_msgs::msg::CalibrationData>("/calibration_data", 10);

  // Publish ready signal once sensors are warmed up (transient_local so late subscribers get it)
  ready_pub_ = node->create_publisher<std_msgs::msg::Bool>(
    "/calibration_ready", rclcpp::QoS(1).transient_local());

  // Subscribe to robot_description
  urdf_sub_ = node->create_subscription<std_msgs::msg::String>("/robot_description",
    rclcpp::QoS(1).transient_local(),
    std::bind(&CaptureManager::callback, this, std::placeholders::_1));

  // Create chain manager
  chain_manager_ = new ChainManager(node);

  // Load feature finders
  if (!feature_finder_loader_.load(node, finders_))
  {
    RCLCPP_FATAL(LOGGER, "Unable to load feature finders!");
    return false;
  }

  // Spin until every finder has received at least one message on its topic,
  // confirming ROS2 discovery is complete and sensor data is flowing.
  // Timeout after 60 seconds to avoid hanging indefinitely.
  RCLCPP_INFO(LOGGER, "Waiting for sensor data on all subscribed topics...");
  constexpr int64_t ready_timeout_ms = 60000;
  constexpr int64_t log_interval_ms = 2000;
  auto ready_start = std::chrono::steady_clock::now();
  int64_t last_log_ms = -log_interval_ms;  // trigger an immediate first print
  while (rclcpp::ok())
  {
    rclcpp::spin_some(node);

    std::vector<std::string> waiting_for;
    for (auto& finder : finders_)
    {
      if (!finder.second->hasData())
        waiting_for.push_back(finder.first);
    }
    if (waiting_for.empty())
      break;

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - ready_start).count();

    if (elapsed_ms - last_log_ms >= log_interval_ms)
    {
      for (const auto& name : waiting_for)
        RCLCPP_INFO(LOGGER, "  Still waiting for data from finder: %s", name.c_str());
      last_log_ms = elapsed_ms;
    }

    if (elapsed_ms >= ready_timeout_ms)
    {
      RCLCPP_ERROR(LOGGER, "Timed out waiting for sensor data - check that all topics are publishing");
      for (const auto& name : waiting_for)
        RCLCPP_ERROR(LOGGER, "  No data received from finder: %s", name.c_str());
      return false;
    }
    rclcpp::sleep_for(std::chrono::milliseconds(100));
  }

  // Signal readiness - subscribers with transient_local QoS will receive this
  // even if they subscribe after it is published.
  std_msgs::msg::Bool ready_msg;
  ready_msg.data = true;
  ready_pub_->publish(ready_msg);
  RCLCPP_INFO(LOGGER, "Calibration node ready - all sensors providing data");

  return true;
}

bool CaptureManager::moveToState(const sensor_msgs::msg::JointState& state)
{
  if (!chain_manager_->moveToState(state))
  {
    return false;
  }

  // Wait for things to settle
  chain_manager_->waitToSettle();
  return true;
}

bool CaptureManager::captureFeatures(const std::vector<std::string>& feature_names,
                                     robot_calibration_msgs::msg::CalibrationData& msg)
{
  for (auto it = finders_.begin(); it != finders_.end(); ++it)
  {
    if (feature_names.empty() ||
        std::find(feature_names.begin(), feature_names.end(), it->first) != feature_names.end())
    {
      RCLCPP_INFO(LOGGER, "Capturing features from %s", it->first.c_str());
      if (!it->second->find(&msg))
      {
        RCLCPP_WARN(LOGGER, "%s failed to capture features.", it->first.c_str());
        return false;
      }
    }
  }
  chain_manager_->getState(&msg.joint_states);
  // Publish calibration data message.
  data_pub_->publish(msg);
  return true;
}

void CaptureManager::callback(std_msgs::msg::String::ConstSharedPtr msg)
{
  description_ = msg->data;
  description_valid_ = true;
}

std::string CaptureManager::getUrdf()
{
  auto last_log = std::chrono::steady_clock::now() - std::chrono::seconds(5);
  while (!description_valid_ && rclcpp::ok())
  {
    rclcpp::spin_some(node_);
    auto now = std::chrono::steady_clock::now();
    if (now - last_log >= std::chrono::seconds(5))
    {
      RCLCPP_WARN(LOGGER, "Waiting for robot_description");
      last_log = now;
    }
    rclcpp::sleep_for(std::chrono::milliseconds(100));
  }
  return description_;
}

}  // namespace robot_calibration
