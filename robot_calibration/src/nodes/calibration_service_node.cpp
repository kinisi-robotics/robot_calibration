/*
 * Long-running calibration service node.
 *
 * Replaces the `calibrate --manual-service` and `calibrate --from-bag`
 * subprocess patterns with a persistent node that exposes capture and
 * optimization over ROS 2 services. Destroys and recreates an internal
 * worker (owning a child rclcpp::Node + CaptureManager) whenever the
 * /robot_calibration/configure service is called, which sidesteps ROS 2's
 * "cannot re-declare parameter" limitation when loading a new YAML config.
 */

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <robot_calibration_msgs/msg/calibration_data.hpp>
#include <robot_calibration_msgs/srv/capture_calibration.hpp>
#include <robot_calibration_msgs/srv/configure_calibration.hpp>
#include <robot_calibration_msgs/srv/run_optimization.hpp>

#include <robot_calibration/optimization/ceres_optimizer.hpp>
#include <robot_calibration/optimization/export.hpp>
#include <robot_calibration/util/calibration_data.hpp>
#include <robot_calibration/util/capture_manager.hpp>

namespace
{
const rclcpp::Logger LOGGER = rclcpp::get_logger("calibration_service_node");

// Owns everything that depends on a specific configuration. Destroyed and
// recreated on each /configure call, which releases the child node and its
// parameters, subscriptions, and finders.
struct CalibrationWorker
{
  rclcpp::Node::SharedPtr node;
  std::shared_ptr<robot_calibration::CaptureManager> capture_manager;
};
}  // namespace

class CalibrationServiceNode : public rclcpp::Node
{
public:
  CalibrationServiceNode()
  : rclcpp::Node("robot_calibration")
  {
    configure_srv_ = create_service<robot_calibration_msgs::srv::ConfigureCalibration>(
      "/robot_calibration/configure",
      [this](
        const std::shared_ptr<robot_calibration_msgs::srv::ConfigureCalibration::Request> request,
        std::shared_ptr<robot_calibration_msgs::srv::ConfigureCalibration::Response> response)
      {
        handleConfigure(request, response);
      });

    ready_srv_ = create_service<std_srvs::srv::Trigger>(
      "/robot_calibration/ready",
      [this](
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response)
      {
        handleReady(request, response);
      });

    capture_srv_ = create_service<robot_calibration_msgs::srv::CaptureCalibration>(
      "/robot_calibration/capture",
      [this](
        const std::shared_ptr<robot_calibration_msgs::srv::CaptureCalibration::Request> request,
        std::shared_ptr<robot_calibration_msgs::srv::CaptureCalibration::Response> response)
      {
        handleCapture(request, response);
      });

    optimize_srv_ = create_service<robot_calibration_msgs::srv::RunOptimization>(
      "/robot_calibration/optimize",
      [this](
        const std::shared_ptr<robot_calibration_msgs::srv::RunOptimization::Request> request,
        std::shared_ptr<robot_calibration_msgs::srv::RunOptimization::Response> response)
      {
        handleOptimize(request, response);
      });

    RCLCPP_INFO(LOGGER,
      "Calibration service node ready — waiting for /robot_calibration/configure");
  }

  // Called by main() in a tight loop so the worker's child node stays
  // subscribed to sensor topics between service calls. Without this, the
  // child node would miss sensor messages and finders' hasData() would fail
  // on subsequent captures.
  void spinWorker()
  {
    std::shared_ptr<CalibrationWorker> worker_snapshot;
    {
      std::lock_guard<std::mutex> lock(worker_mutex_);
      worker_snapshot = worker_;
    }
    if (worker_snapshot)
    {
      rclcpp::spin_some(worker_snapshot->node);
    }
  }

private:
  void handleConfigure(
    const std::shared_ptr<robot_calibration_msgs::srv::ConfigureCalibration::Request> request,
    std::shared_ptr<robot_calibration_msgs::srv::ConfigureCalibration::Response> response)
  {
    RCLCPP_INFO(LOGGER, "Configuring with '%s'", request->config_file.c_str());

    // Drop the old worker first so its subscriptions/finders release their
    // topic handles before we bring up the new ones (avoids double-binding).
    {
      std::lock_guard<std::mutex> lock(worker_mutex_);
      worker_.reset();
    }

    rclcpp::NodeOptions options;
    options.arguments({"--ros-args", "--params-file", request->config_file});
    // Worker node name must be unique so it can be torn down and a fresh one
    // created on each /configure call without colliding in the ROS graph.
    std::string node_name = "robot_calibration_worker_" +
                            std::to_string(worker_epoch_.fetch_add(1));

    auto new_worker = std::make_shared<CalibrationWorker>();
    try
    {
      new_worker->node = std::make_shared<rclcpp::Node>(node_name, options);
    }
    catch (const std::exception& e)
    {
      response->success = false;
      response->message = std::string("Failed to create worker node: ") + e.what();
      RCLCPP_ERROR(LOGGER, "%s", response->message.c_str());
      return;
    }

    new_worker->capture_manager = std::make_shared<robot_calibration::CaptureManager>();
    if (!new_worker->capture_manager->init(new_worker->node))
    {
      response->success = false;
      response->message = "CaptureManager::init failed — check sensor topics";
      RCLCPP_ERROR(LOGGER, "%s", response->message.c_str());
      return;
    }

    {
      std::lock_guard<std::mutex> lock(worker_mutex_);
      worker_ = new_worker;
    }

    response->success = true;
    response->message = "Worker reconfigured";
    RCLCPP_INFO(LOGGER, "Reconfigure complete");
  }

  void handleReady(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    std::lock_guard<std::mutex> lock(worker_mutex_);
    if (!worker_)
    {
      response->success = false;
      response->message = "No configuration loaded — call /robot_calibration/configure first";
      return;
    }
    // CaptureManager::init() already blocks until all finders have data. If
    // the worker exists, it is ready.
    response->success = true;
    response->message = "Ready";
  }

  void handleCapture(
    const std::shared_ptr<robot_calibration_msgs::srv::CaptureCalibration::Request> request,
    std::shared_ptr<robot_calibration_msgs::srv::CaptureCalibration::Response> response)
  {
    std::lock_guard<std::mutex> lock(worker_mutex_);
    if (!worker_)
    {
      response->success = false;
      response->message = "No configuration loaded";
      return;
    }

    robot_calibration_msgs::msg::CalibrationData msg;
    std::vector<std::string> features(request->features.begin(), request->features.end());
    if (!worker_->capture_manager->captureFeatures(features, msg))
    {
      response->success = false;
      response->message = "Failed to capture features";
      RCLCPP_WARN(LOGGER, "Capture request failed (features=%zu)", features.size());
      return;
    }

    response->success = true;
    response->message = "Captured";
    response->data = msg;
  }

  void handleOptimize(
    const std::shared_ptr<robot_calibration_msgs::srv::RunOptimization::Request> request,
    std::shared_ptr<robot_calibration_msgs::srv::RunOptimization::Response> response)
  {
    // Grab a snapshot of the worker under the lock, then release the lock so
    // /capture or /ready can still be serviced while optimization runs.
    // The worker's params are read-only from the optimizer's perspective, but
    // LoadFromROS may declare extra parameters — we rely on get_or_declare
    // being idempotent (see robot_calibration/optimization/params.cpp).
    std::shared_ptr<CalibrationWorker> worker_snapshot;
    {
      std::lock_guard<std::mutex> lock(worker_mutex_);
      worker_snapshot = worker_;
    }

    if (!worker_snapshot)
    {
      response->success = false;
      response->message = "No configuration loaded";
      return;
    }

    // Load calibration data from the bag.
    std_msgs::msg::String description_msg;
    std::vector<robot_calibration_msgs::msg::CalibrationData> data;
    if (!robot_calibration::load_bag(request->bag_path, description_msg, data))
    {
      response->success = false;
      response->message = "Failed to load bag: " + request->bag_path;
      RCLCPP_ERROR(LOGGER, "%s", response->message.c_str());
      return;
    }
    if (data.empty() || description_msg.data.empty())
    {
      response->success = false;
      response->message = "Bag contains no calibration data or missing URDF";
      RCLCPP_ERROR(LOGGER, "%s", response->message.c_str());
      return;
    }

    // Read calibration steps from the worker's config.
    std::vector<std::string> calibration_steps;
    if (worker_snapshot->node->has_parameter("calibration_steps"))
    {
      calibration_steps = worker_snapshot->node->get_parameter("calibration_steps")
                            .as_string_array();
    }
    else
    {
      calibration_steps = worker_snapshot->node->declare_parameter<std::vector<std::string>>(
        "calibration_steps", std::vector<std::string>());
    }

    if (calibration_steps.empty())
    {
      response->success = false;
      response->message = "Parameter calibration_steps is not defined in loaded config";
      RCLCPP_ERROR(LOGGER, "%s", response->message.c_str());
      return;
    }

    bool verbose = false;
    if (worker_snapshot->node->has_parameter("verbose"))
    {
      verbose = worker_snapshot->node->get_parameter("verbose").as_bool();
    }
    else
    {
      verbose = worker_snapshot->node->declare_parameter<bool>("verbose", false);
    }

    // Run each calibration step.
    robot_calibration::OptimizationParams params;
    robot_calibration::Optimizer opt(description_msg.data);
    rclcpp::Logger opt_logger = worker_snapshot->node->get_logger();

    for (const auto& step : calibration_steps)
    {
      RCLCPP_INFO(LOGGER, "Running calibration step: %s", step.c_str());
      params.LoadFromROS(worker_snapshot->node, step);
      opt.optimize(params, data, opt_logger, verbose);
    }

    auto results = robot_calibration::getResultStrings(opt, description_msg.data);
    response->success = true;
    response->message = "Optimization complete";
    response->offset_yaml = results.offset_yaml;
    response->calibrated_urdf = results.calibrated_urdf;
    RCLCPP_INFO(LOGGER, "Optimization complete — returning offset YAML");
  }

  rclcpp::Service<robot_calibration_msgs::srv::ConfigureCalibration>::SharedPtr configure_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr ready_srv_;
  rclcpp::Service<robot_calibration_msgs::srv::CaptureCalibration>::SharedPtr capture_srv_;
  rclcpp::Service<robot_calibration_msgs::srv::RunOptimization>::SharedPtr optimize_srv_;

  std::mutex worker_mutex_;
  std::shared_ptr<CalibrationWorker> worker_;
  std::atomic<uint64_t> worker_epoch_{0};
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto service_node = std::make_shared<CalibrationServiceNode>();

  // Pump both the service node (for RPC requests) and the worker's child
  // node (for sensor subscriptions). Done in one thread so there is no
  // contention over the worker node — service handlers that need to spin
  // the worker can call spin_some themselves under the worker mutex.
  while (rclcpp::ok())
  {
    rclcpp::spin_some(service_node);
    service_node->spinWorker();
    rclcpp::sleep_for(std::chrono::milliseconds(10));
  }

  rclcpp::shutdown();
  return 0;
}
