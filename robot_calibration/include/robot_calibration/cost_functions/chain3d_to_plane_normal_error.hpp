/*
 * Copyright (C) 2018 Michael Ferguson
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

#ifndef ROBOT_CALIBRATION_COST_FUNCTIONS_CHAIN3D_TO_PLANE_NORMAL_ERROR_HPP
#define ROBOT_CALIBRATION_COST_FUNCTIONS_CHAIN3D_TO_PLANE_NORMAL_ERROR_HPP

#include <string>
#include <math.h>
#include <ceres/ceres.h>
#include <kdl/frames.hpp>
#include <robot_calibration/models/chain3d.hpp>
#include <robot_calibration/optimization/offsets.hpp>
#include <robot_calibration/util/calibration_data.hpp>
#include <robot_calibration/util/eigen_geometry.hpp>
#include <robot_calibration_msgs/msg/calibration_data.hpp>

namespace robot_calibration
{

/**
 *  \brief Error block for aligning the normal of a detected plane with a
 *         target normal. The plane is fit to the projected sensor observations
 *         via SVD (once, in the constructor), and then the normal is rotated
 *         through the kinematic chain FK on each iteration. This is useful for
 *         calibrating joint offsets so that a ground plane aligns with the
 *         expected orientation, without constraining the plane offset.
 */
struct Chain3dToPlaneNormal
{
  /**
   *  \brief This function is not used directly, instead use the Create() function.
   *  \param chain_model The model for the chain, used for FK computation.
   *  \param offsets Easy access to the free parameters.
   *  \param data The calibration data collected.
   *  \param a The target plane normal x component.
   *  \param b The target plane normal y component.
   *  \param c The target plane normal z component.
   *  \param scale The scaling factor to apply to residuals.
   */
  Chain3dToPlaneNormal(Chain3dModel* chain_model,
                       OptimizationOffsets* offsets,
                       robot_calibration_msgs::msg::CalibrationData& data,
                       double a, double b, double c,
                       double scale)
  {
    chain_model_ = chain_model;
    offsets_ = offsets;
    data_ = data;
    scale_ = scale;

    // Normalize the target normal
    double denom = sqrt((a * a) + (b * b) + (c * c));
    if (abs(denom) < 0.1)
    {
      std::cerr << "Target plane normal is extremely small: " << denom << std::endl;
    }
    a_ = a / denom;
    b_ = b / denom;
    c_ = c / denom;

    // Precompute the plane normal from sensor-frame observations (SVD, done once)
    int index = getSensorIndex(data_, chain_model_->getName());
    if (index >= 0)
    {
      std::vector<geometry_msgs::msg::PointStamped> features(
          data_.observations[index].features.begin(),
          data_.observations[index].features.end());
      Eigen::MatrixXd matrix = getMatrix(features);
      double d = 0.0;
      getPlane(matrix, sensor_normal_, d);
    }
    else
    {
      sensor_normal_ = Eigen::Vector3d(0, 0, 1);
    }
  }

  virtual ~Chain3dToPlaneNormal() {}

  bool operator()(double const * const * free_params,
                  double* residuals) const
  {
    // Update calibration offsets based on free params
    offsets_->update(free_params[0]);

    // Get FK transform from sensor frame to root frame
    KDL::Frame fk = chain_model_->getChainFK(*offsets_, data_.joint_states);

    // Rotate the precomputed sensor-frame normal into root frame
    KDL::Vector sensor_n(sensor_normal_(0), sensor_normal_(1), sensor_normal_(2));
    KDL::Vector rotated = fk.M * sensor_n;

    // Compute residuals: component-wise normal difference
    residuals[0] = (rotated.x() - a_) * scale_;
    residuals[1] = (rotated.y() - b_) * scale_;
    residuals[2] = (rotated.z() - c_) * scale_;

    return true;
  }

  /**
   *  \brief Helper factory function to create a new error block. Parameters
   *         are described in the class constructor, which this function calls.
   */
  static ceres::CostFunction* Create(Chain3dModel* a_model,
                                     OptimizationOffsets* offsets,
                                     robot_calibration_msgs::msg::CalibrationData& data,
                                     double a, double b, double c,
                                     double scale)
  {
    int index = getSensorIndex(data, a_model->getName());
    if (index == -1)
    {
      // In theory, we should never get here, because the optimizer does a check
      std::cerr << "Sensor name doesn't match any of the existing finders" << std::endl;
      return 0;
    }

    ceres::DynamicNumericDiffCostFunction<Chain3dToPlaneNormal> * func;
    func = new ceres::DynamicNumericDiffCostFunction<Chain3dToPlaneNormal>(
                    new Chain3dToPlaneNormal(a_model, offsets, data, a, b, c, scale));
    func->AddParameterBlock(offsets->size());
    func->SetNumResiduals(3);

    return static_cast<ceres::CostFunction*>(func);
  }

  Chain3dModel * chain_model_;
  OptimizationOffsets * offsets_;
  robot_calibration_msgs::msg::CalibrationData data_;
  double a_, b_, c_;
  double scale_;
  Eigen::Vector3d sensor_normal_;
};

}  // namespace robot_calibration

#endif  // ROBOT_CALIBRATION_COST_FUNCTIONS_CHAIN3D_TO_PLANE_NORMAL_ERROR_HPP
