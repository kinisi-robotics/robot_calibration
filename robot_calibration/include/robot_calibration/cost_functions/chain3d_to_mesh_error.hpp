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

#ifndef ROBOT_CALIBRATION_COST_FUNCTIONS_CHAIN3D_TO_MESH_ERROR_HPP
#define ROBOT_CALIBRATION_COST_FUNCTIONS_CHAIN3D_TO_MESH_ERROR_HPP

#include <limits>
#include <string>
#include <math.h>
#include <ceres/ceres.h>
#include <fstream>
#include <mutex>
#include <unistd.h>
#include <atomic>
#include <robot_calibration/models/camera3d.hpp> 
#include <robot_calibration/models/chain3d.hpp>
#include <robot_calibration/optimization/offsets.hpp>
#include <robot_calibration/util/calibration_data.hpp>
#include <robot_calibration/util/mesh_loader.hpp>
#include <robot_calibration_msgs/msg/calibration_data.hpp>

namespace robot_calibration
{

/**
 * \brief Get the squared distance line segment A-B for point C
 * \param a Point representing one end of the line segment
 * \param b Point representing the other end of the line segment
 * \param c Point to get distance to line segment
 *
 * Based on "Real Time Collision Detection", pg 130
 */
double distToLine(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d c)
{
  Eigen::Vector3d ab = b - a;
  Eigen::Vector3d ac = c - a;
  Eigen::Vector3d bc = c - b;

  double e = ac.dot(ab);
  if (e <= 0.0)
  {
    // Point A is closest to C
    return ac.dot(ac);
  }
  double f = ab.dot(ab);
  if (e >= f)
  {
    // Point B is closest to C
    return bc.dot(bc);
  }
  // C actually projects between
  return ac.dot(ac) - e * e / f;
}

double pointDistanceToTriangle(const Eigen::Vector3d& p,
                               const Eigen::Vector3d& A,
                               const Eigen::Vector3d& B,
                               const Eigen::Vector3d& C)
 {
  // Based on "Real-Time Collision Detection" (Ericson). Returns squared distance.
  const Eigen::Vector3d AB = B - A;
  const Eigen::Vector3d AC = C - A;
  const Eigen::Vector3d AP = p - A;

  const double d1 = AB.dot(AP);
  const double d2 = AC.dot(AP);
  if (d1 <= 0.0 && d2 <= 0.0)
  {
    // Closest to vertex A
    return AP.dot(AP);
  }

  const Eigen::Vector3d BP = p - B;
  const double d3 = AB.dot(BP);
  const double d4 = AC.dot(BP);
  if (d3 >= 0.0 && d4 <= d3)
  {
    // Closest to vertex B
    return BP.dot(BP);
  }

  // Check if P in edge region of AB, if so return projection distance
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
  {
    const double ab2 = AB.dot(AB);
    const double v = d1 / (d1 - d3);
    // Compute squared distance without constructing the closest point
    const double ap2 = AP.dot(AP);
    return ap2 - 2.0 * v * d1 + v * v * ab2;
  }

  const Eigen::Vector3d CP = p - C;
  const double d5 = AB.dot(CP);
  const double d6 = AC.dot(CP);
  if (d6 >= 0.0 && d5 <= d6)
  {
    // Closest to vertex C
    return CP.dot(CP);
  }

  // Check if P in edge region of AC
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
  {
    const double ac2 = AC.dot(AC);
    const double w = d2 / (d2 - d6);
    const double ap2 = AP.dot(AP);
    return ap2 - 2.0 * w * d2 + w * w * ac2;
  }

  // Check if P in edge region of BC
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
  {
    const Eigen::Vector3d BC = C - B;
    const double bc2 = BC.dot(BC);
    const double bp2 = BP.dot(BP);
    const double bp_dot_bc = BP.dot(BC);
    const double denom = (d4 - d3) + (d5 - d6);
    const double w = (d4 - d3) / denom;
    return bp2 - 2.0 * w * bp_dot_bc + w * w * bc2;
  }

  // P inside face region. Compute squared distance to plane.
  const Eigen::Vector3d N = AB.cross(AC);
  const double n2 = N.squaredNorm();
  if (n2 <= 1e-16)
  {
    // Degenerate triangle: fallback to edges
    const double d_ab = distToLine(A, B, p);
    const double d_bc = distToLine(B, C, p);
    const double d_ca = distToLine(C, A, p);
    return std::min(d_ab, std::min(d_bc, d_ca));
  }
  const double dist_to_plane = AP.dot(N);
  return (dist_to_plane * dist_to_plane) / n2;
}

/**
 *  \brief Error block for computing the fit between a set of projected
 *         points and a mesh (usually part of the robot body). Typically used
 *         to align sensor with the robot footprint.
 */
struct Chain3dToMesh
{
  /**
   *  \brief This function is not used direcly, instead use the Create() function.
   *  \param chain_model The model for the chain, used for reprojection.
   *  \param offsets Easy access to the free parameters.
   *  \param data The calibration data collected.
   *  \param mesh_path Path to the mesh file to test against
   */
  Chain3dToMesh(Chain3dModel* chain_model,
                OptimizationOffsets* offsets,
                robot_calibration_msgs::msg::CalibrationData& data,
                MeshPtr& mesh)
  {
    chain_model_ = chain_model;
    offsets_ = offsets;
    data_ = data;
    mesh_ = mesh;
  }

  virtual ~Chain3dToMesh() {}

  bool operator()(double const * const * free_params,
                  double* residuals) const noexcept
  {
    // Update calibration offsets based on free params
    offsets_->update(free_params[0]);

    // Project the camera observations
    std::vector<geometry_msgs::msg::PointStamped> chain_pts =
        chain_model_->project(data_, *offsets_);

    // Debug: on first invocation, write chain points and mesh vertices to a PLY and terminate
    static std::once_flag _chain_mesh_debug_once;
    std::call_once(_chain_mesh_debug_once, [this, chain_pts]() {
      std::string filename = "/tmp/chain_mesh_debug_" + std::to_string(getpid()) + ".ply";
      std::ofstream ofs(filename);
      if (ofs)
      {
        size_t n_chain = chain_pts.size();
        size_t n_mesh = mesh_->vertex_count;
        size_t n_vertices = n_chain + n_mesh;
        ofs << "ply\nformat ascii 1.0\n";
        ofs << "element vertex " << n_vertices << "\n";
        ofs << "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n";
        ofs << "end_header\n";
        // Chain points in red
        for (const auto &pt : chain_pts)
        {
          ofs << pt.point.x << " " << pt.point.y << " " << pt.point.z << " 255 0 0\n";
        }
        // Mesh vertices in blue
        for (size_t vi = 0; vi < mesh_->vertex_count; ++vi)
        {
          size_t idx = 3 * vi;
          ofs << mesh_->vertices[idx] << " " << mesh_->vertices[idx + 1] << " " << mesh_->vertices[idx + 2] << " 0 0 255\n";
        }
        ofs.close();
        std::cerr << "Wrote debug PLY to " << filename << std::endl;
      }
      else
      {
        std::cerr << "Failed to open " << filename << " for writing" << std::endl;
      }

    });

    // Compute residuals
    for (size_t pt = 0; pt < chain_pts.size() ; ++pt)
    {
      Eigen::Vector3d p(chain_pts[pt].point.x, chain_pts[pt].point.y, chain_pts[pt].point.z);

      // Find shortest distance to any line segment forming a triangle
      double dist = std::numeric_limits<double>::max();
      for (size_t t = 0; t < mesh_->triangle_count; ++t)
      {
        // Get the index of each vertex of the triangle
        int A_idx = mesh_->triangles[(3 * t) + 0];
        int B_idx = mesh_->triangles[(3 * t) + 1];
        int C_idx = mesh_->triangles[(3 * t) + 2];
        // Get the vertices
        const Eigen::Map<const Eigen::Vector3d> A(&mesh_->vertices[(3 * A_idx)]);
        const Eigen::Map<const Eigen::Vector3d> B(&mesh_->vertices[(3 * B_idx)]);
        const Eigen::Map<const Eigen::Vector3d> C(&mesh_->vertices[(3 * C_idx)]);
        dist = std::min(pointDistanceToTriangle(p, A, B, C), dist);
      }
      residuals[pt] = std::sqrt(dist);
    }
    return true;
  }

  /**
   *  \brief Helper factory function to create a new error block. Parameters
   *         are described in the class constructor, which this function calls.
   */
  static ceres::CostFunction* Create(Chain3dModel* a_model,
                                     OptimizationOffsets* offsets,
                                     robot_calibration_msgs::msg::CalibrationData& data,
                                     MeshPtr mesh)
  {
    int index = getSensorIndex(data, a_model->getName());
    if (index == -1)
    {
      // In theory, we should never get here, because the optimizer does a check
      std::cerr << "Sensor name doesn't match any of the existing finders" << std::endl;
      return 0;
    }

    ceres::DynamicNumericDiffCostFunction<Chain3dToMesh> * func;
    func = new ceres::DynamicNumericDiffCostFunction<Chain3dToMesh>(
                    new Chain3dToMesh(a_model, offsets, data, mesh));
    func->AddParameterBlock(offsets->size());
    func->SetNumResiduals(data.observations[index].features.size());

    return static_cast<ceres::CostFunction*>(func);
  }

  Chain3dModel * chain_model_;
  OptimizationOffsets * offsets_;
  robot_calibration_msgs::msg::CalibrationData data_;
  MeshPtr mesh_;
};

}  // namespace robot_calibration

#endif  // ROBOT_CALIBRATION_COST_FUNCTIONS_CHAIN3D_TO_MESH_ERROR_HPP
