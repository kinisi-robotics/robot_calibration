/*
 * Copyright (C) 2018-2023 Michael Ferguson
 * Copyright (C) 2014-2015 Fetch Robotics Inc.
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

#include <robot_calibration/optimization/params.hpp>

namespace robot_calibration
{

namespace
{
// Idempotent parameter access: re-entrant-safe across multiple LoadFromROS
// calls on the same node. Required for a long-running service node that
// handles many optimization requests without restarting.
template <typename T>
T get_or_declare(rclcpp::Node::SharedPtr node, const std::string& name, const T& default_val)
{
  if (node->has_parameter(name))
    return node->get_parameter(name).get_value<T>();
  return node->declare_parameter<T>(name, default_val);
}
}  // namespace

OptimizationParams::OptimizationParams() :
  base_link("base_link")
{
}

bool OptimizationParams::LoadFromROS(rclcpp::Node::SharedPtr node,
                                     const std::string& parameter_ns)
{
  rclcpp::Logger logger = node->get_logger();

  // Base link should be consistent across all calibration steps
  base_link = get_or_declare<std::string>(node, "base_link", "base_link");

  max_num_iterations = get_or_declare<int>(
    node, parameter_ns + ".max_num_iterations", 1000);

  free_params = get_or_declare<std::vector<std::string>>(
    node, parameter_ns + ".free_params", std::vector<std::string>());

  free_frames.clear();
  auto free_frame_names = get_or_declare<std::vector<std::string>>(
    node, parameter_ns + ".free_frames", std::vector<std::string>());
  for (auto name : free_frame_names)
  {
    RCLCPP_INFO(logger, "Adding free frame: %s", name.c_str());
    std::string prefix = parameter_ns + "." + name;
    FreeFrameParams params;
    params.name = name;
    params.x = get_or_declare<bool>(node, prefix + ".x", false);
    params.y = get_or_declare<bool>(node, prefix + ".y", false);
    params.z = get_or_declare<bool>(node, prefix + ".z", false);
    params.roll = get_or_declare<bool>(node, prefix + ".roll", false);
    params.pitch = get_or_declare<bool>(node, prefix + ".pitch", false);
    params.yaw = get_or_declare<bool>(node, prefix + ".yaw", false);
    free_frames.push_back(params);
  }

  free_frames_initial_values.clear();
  free_frame_names = get_or_declare<std::vector<std::string>>(
    node, parameter_ns + ".free_frames_initial_values", std::vector<std::string>());
  for (auto name : free_frame_names)
  {
    RCLCPP_INFO(logger, "Adding initial values for: %s", name.c_str());
    std::string prefix = parameter_ns + "." + name + "_initial_values";
    FreeFrameInitialValue params;
    params.name = name;
    params.x = get_or_declare<double>(node, prefix + ".x", 0.0);
    params.y = get_or_declare<double>(node, prefix + ".y", 0.0);
    params.z = get_or_declare<double>(node, prefix + ".z", 0.0);
    params.roll = get_or_declare<double>(node, prefix + ".roll", 0.0);
    params.pitch = get_or_declare<double>(node, prefix + ".pitch", 0.0);
    params.yaw = get_or_declare<double>(node, prefix + ".yaw", 0.0);
    free_frames_initial_values.push_back(params);
  }

  models.clear();
  auto model_names = get_or_declare<std::vector<std::string>>(
    node, parameter_ns + ".models", std::vector<std::string>());
  for (auto name : model_names)
  {
    RCLCPP_INFO(logger, "Adding model: %s", name.c_str());
    ModelParams params;
    params.name = name;
    params.type = get_or_declare<std::string>(node, parameter_ns + "." + name + ".type", std::string());
    params.frame = get_or_declare<std::string>(node, parameter_ns + "." + name + ".frame", std::string());
    params.param_name = get_or_declare<std::string>(node, parameter_ns + "." + name + ".param_name", std::string());
    models.push_back(params);
  }

  error_blocks.clear();
  auto error_block_names = get_or_declare<std::vector<std::string>>(
    node, parameter_ns + ".error_blocks", std::vector<std::string>());
  for (auto name : error_block_names)
  {
    std::string prefix = parameter_ns + "." + name;
    std::string type = get_or_declare<std::string>(node, prefix + ".type", std::string());
    RCLCPP_INFO(logger, "Adding %s: %s", type.c_str(), name.c_str());

    if (type == "chain3d_to_chain3d")
    {
      std::shared_ptr<Chain3dToChain3dParams> params = std::make_shared<Chain3dToChain3dParams>();
      params->name = name;
      params->type = type;
      params->model_a = get_or_declare<std::string>(node, prefix + ".model_a", std::string());
      params->model_b = get_or_declare<std::string>(node, prefix + ".model_b", std::string());
      error_blocks.push_back(params);
    }
    else if (type == "chain3d_to_camera2d")
    {
      std::shared_ptr<Chain3dToCamera2dParams> params = std::make_shared<Chain3dToCamera2dParams>();
      params->name = name;
      params->type = type;
      params->model_2d = get_or_declare<std::string>(node, prefix + ".model_2d", std::string());
      params->model_3d = get_or_declare<std::string>(node, prefix + ".model_3d", std::string());
      params->scale = get_or_declare<double>(node, prefix + ".scale", 1.0);
      error_blocks.push_back(params);
    }
    else if (type == "camera2d_to_camera2d")
    {
      std::shared_ptr<Camera2dToCamera2dParams> params = std::make_shared<Camera2dToCamera2dParams>();
      params->name = name;
      params->type = type;
      params->model_a = get_or_declare<std::string>(node, prefix + ".model_a", std::string());
      params->model_b = get_or_declare<std::string>(node, prefix + ".model_b", std::string());
      params->points_x = get_or_declare<int>(node, prefix + ".points_x", 0);
      params->points_y = get_or_declare<int>(node, prefix + ".points_y", 0);
      params->point_size = get_or_declare<double>(node, prefix + ".size", 0.0);
      params->scale = get_or_declare<double>(node, prefix + ".scale", 1.0);
      error_blocks.push_back(params);
    }
    else if (type == "chain3d_to_plane")
    {
      std::shared_ptr<Chain3dToPlaneParams> params = std::make_shared<Chain3dToPlaneParams>();
      params->name = name;
      params->type = type;
      params->model = get_or_declare<std::string>(node, prefix + ".model", std::string());
      params->a = get_or_declare<double>(node, prefix + ".a", 0.0);
      params->b = get_or_declare<double>(node, prefix + ".b", 0.0);
      params->c = get_or_declare<double>(node, prefix + ".c", 1.0);
      params->d = get_or_declare<double>(node, prefix + ".d", 0.0);
      params->scale = get_or_declare<double>(node, prefix + ".scale", 1.0);
      error_blocks.push_back(params);
    }
    else if (type == "chain3d_to_plane_normal")
    {
      std::shared_ptr<Chain3dToPlaneNormalParams> params = std::make_shared<Chain3dToPlaneNormalParams>();
      params->name = name;
      params->type = type;
      params->model = get_or_declare<std::string>(node, prefix + ".model", std::string());
      params->a = get_or_declare<double>(node, prefix + ".a", 0.0);
      params->b = get_or_declare<double>(node, prefix + ".b", 0.0);
      params->c = get_or_declare<double>(node, prefix + ".c", 1.0);
      params->scale = get_or_declare<double>(node, prefix + ".scale", 1.0);
      error_blocks.push_back(params);
    }
    else if (type == "chain3d_to_mesh")
    {
      std::shared_ptr<Chain3dToMeshParams> params = std::make_shared<Chain3dToMeshParams>();
      params->name = name;
      params->type = type;
      params->model = get_or_declare<std::string>(node, prefix + ".model", std::string());
      params->link_name = get_or_declare<std::string>(node, prefix + ".link_name", std::string());
      // Optional per-error-block override resource URI for the mesh
      params->mesh_override = get_or_declare<std::string>(
        node, prefix + ".mesh_override", std::string());
      error_blocks.push_back(params);
    }
    else if (type == "plane_to_plane")
    {
      std::shared_ptr<PlaneToPlaneParams> params = std::make_shared<PlaneToPlaneParams>();
      params->name = name;
      params->type = type;
      params->model_a = get_or_declare<std::string>(node, prefix + ".model_a", std::string());
      params->model_b = get_or_declare<std::string>(node, prefix + ".model_b", std::string());
      params->normal_scale = get_or_declare<double>(node, prefix + ".normal_scale", 1.0);
      params->offset_scale = get_or_declare<double>(node, prefix + ".offset_scale", 1.0);
      error_blocks.push_back(params);
    }
    else if (type == "outrageous")
    {
      std::shared_ptr<OutrageousParams> params = std::make_shared<OutrageousParams>();
      params->name = name;
      params->type = type;
      params->param = get_or_declare<std::string>(node, prefix + ".param", std::string());
      params->joint_scale = get_or_declare<double>(node, prefix + ".joint_scale", 1.0);
      params->position_scale = get_or_declare<double>(node, prefix + ".position_scale", 1.0);
      params->rotation_scale = get_or_declare<double>(node, prefix + ".rotation_scale", 1.0);
      error_blocks.push_back(params);
    }
    else
    {
      RCLCPP_ERROR(logger, "Error block %s of type '%s' is unrecognized", name.c_str(), type.c_str());
    }
  }

  if (error_blocks.empty())
  {
    RCLCPP_ERROR(logger, "No error_blocks are defined!");
  }

  return true;
}

}  // namespace robot_calibration
