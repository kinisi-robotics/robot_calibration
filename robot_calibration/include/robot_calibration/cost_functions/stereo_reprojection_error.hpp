/*
 * Copyright (C) 2026 Kinisi Robotics
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

#ifndef ROBOT_CALIBRATION_COST_FUNCTIONS_STEREO_REPROJECTION_ERROR_HPP
#define ROBOT_CALIBRATION_COST_FUNCTIONS_STEREO_REPROJECTION_ERROR_HPP

#include <string>
#include <vector>
#include <ceres/ceres.h>
#include <robot_calibration/optimization/offsets.hpp>
#include <robot_calibration/util/calibration_data.hpp>
#include <robot_calibration/models/camera2d.hpp>
#include <robot_calibration/models/chain3d.hpp>  // for rotation_from_axis_magnitude
#include <robot_calibration_msgs/msg/calibration_data.hpp>

namespace robot_calibration
{

/**
 *  \brief Error block for stereo camera calibration using pixel reprojection
 *         error.  Estimates a per-observation checkerboard pose (as a separate
 *         Ceres parameter block) and projects known grid points through both
 *         cameras.  The residual is the pixel error in both cameras.
 *
 *         This formulation constrains scale via checkerboard geometry and is
 *         independent of arm chain FK accuracy.
 */
struct StereoReprojectionError
{
  StereoReprojectionError(Camera2dModel* camera_a,
                          Camera2dModel* camera_b,
                          double scale,
                          OptimizationOffsets* offsets,
                          robot_calibration_msgs::msg::CalibrationData& data,
                          std::vector<geometry_msgs::msg::Point> grid_points)
  {
    camera_a_ = camera_a;
    camera_b_ = camera_b;
    scale_ = scale;
    offsets_ = offsets;
    data_ = data;
    grid_points_ = std::move(grid_points);
  }

  virtual ~StereoReprojectionError() {}

  bool operator()(double const* const* free_params,
                  double* residuals) const
  {
    // Update global calibration offsets
    offsets_->update(free_params[0]);

    // Per-observation checkerboard pose: tx, ty, tz, rx, ry, rz (axis-angle)
    const double* pose = free_params[1];
    KDL::Rotation R = rotation_from_axis_magnitude(pose[3], pose[4], pose[5]);
    KDL::Frame board_pose(R, KDL::Vector(pose[0], pose[1], pose[2]));

    // Transform grid points to world frame
    std::vector<geometry_msgs::msg::PointStamped> world_points(grid_points_.size());
    for (size_t i = 0; i < grid_points_.size(); ++i)
    {
      KDL::Vector wp = board_pose * KDL::Vector(
          grid_points_[i].x, grid_points_[i].y, grid_points_[i].z);
      world_points[i].point.x = wp.x();
      world_points[i].point.y = wp.y();
      world_points[i].point.z = wp.z();
    }

    // Project through both cameras and compute pixel errors
    std::vector<geometry_msgs::msg::PointStamped> errors_a =
        camera_a_->project_pixel_error(data_, world_points, *offsets_);
    std::vector<geometry_msgs::msg::PointStamped> errors_b =
        camera_b_->project_pixel_error(data_, world_points, *offsets_);

    if (errors_a.size() != grid_points_.size() ||
        errors_b.size() != grid_points_.size())
    {
      std::cerr << "stereo_reprojection: projection size mismatch" << std::endl;
      return false;
    }

    // Residuals: 4 per feature (2 cameras × 2 pixel components)
    for (size_t i = 0; i < errors_a.size(); ++i)
    {
      residuals[4 * i + 0] = errors_a[i].point.x * scale_;
      residuals[4 * i + 1] = errors_a[i].point.y * scale_;
      residuals[4 * i + 2] = errors_b[i].point.x * scale_;
      residuals[4 * i + 3] = errors_b[i].point.y * scale_;
    }

    return true;
  }

  static ceres::CostFunction* Create(Camera2dModel* camera_a,
                                      Camera2dModel* camera_b,
                                      double scale,
                                      OptimizationOffsets* offsets,
                                      robot_calibration_msgs::msg::CalibrationData& data,
                                      std::vector<geometry_msgs::msg::Point> grid_points)
  {
    int index = getSensorIndex(data, camera_a->getName());
    if (index == -1)
    {
      std::cerr << "stereo_reprojection: sensor name doesn't match any finder" << std::endl;
      return 0;
    }

    int num_features = static_cast<int>(data.observations[index].features.size());

    ceres::DynamicNumericDiffCostFunction<StereoReprojectionError>* func;
    func = new ceres::DynamicNumericDiffCostFunction<StereoReprojectionError>(
        new StereoReprojectionError(camera_a, camera_b, scale, offsets,
                                    data, std::move(grid_points)));
    func->AddParameterBlock(offsets->size());  // global calibration offsets
    func->AddParameterBlock(6);                // per-observation board pose
    func->SetNumResiduals(num_features * 4);   // 2 cameras × 2 pixel components

    return static_cast<ceres::CostFunction*>(func);
  }

  Camera2dModel* camera_a_;
  Camera2dModel* camera_b_;
  double scale_;
  OptimizationOffsets* offsets_;
  robot_calibration_msgs::msg::CalibrationData data_;
  std::vector<geometry_msgs::msg::Point> grid_points_;
};

}  // namespace robot_calibration

#endif  // ROBOT_CALIBRATION_COST_FUNCTIONS_STEREO_REPROJECTION_ERROR_HPP
