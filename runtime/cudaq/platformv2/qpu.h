/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "qpu_traits.h"

#include "cudaq/host_config.h"
#include "cudaq/qis/state.h"
#include "cudaq/remote_capabilities.h"
#include "cudaq/utils/extension_point.h"

#include "common/ExecutionContext.h"
#include "common/ThunkInterface.h"

#include <queue>

/// The core abstraction for the CUDA-Q runtime is the QPU.
///
/// The QPU provides an API for the quantum abstract machine.
///
/// We consider different types of QPUs - those that run as local
/// or remote simulators, remote physical QPUs provided over the cloud,
/// or logical QPUs in a tight integration context.
///
/// To support such a wide array of QPUs, we have designed the
/// QPU via a template mixin pattern. QPU subtypes can choose
/// to support specific traits or not (e.g. a local simulation
/// QPU may not require APIs required for remote invocation, or one
/// QPU may support efficient sampling while another may not).
///
/// The QPU is therefore templated on dynamically provided
/// interface / trait template types that serve "turn on" specific
/// capabilities for the QPU, and clients can access those via
/// a dynamic cast.
///

namespace cudaq::config {
class TargetConfig;
}

namespace cudaq::v2 {

/// \struct platform_metadata
/// \brief Encapsulates platform and target configuration metadata for a QPU.
///
/// This struct holds references to the target configuration, target options,
/// and the initial configuration string for the quantum platform.
struct platform_metadata {
  /// \brief Reference to the target configuration.
  const cudaq::config::TargetConfig &target_config;
  /// \brief List of target-specific options.
  const std::vector<std::string> &target_options;
  /// \brief Initial configuration string.
  const std::string &initial_config_str;
};

/// \brief Set the current QPU for the calling thread.
/// \param idx The QPU device index.
void set_qpu(std::size_t);

/// \class device
/// \brief Represents a generic quantum device.
class device {
public:
};

/// \class qpu_handle
/// \brief Abstract base class for a quantum processing unit (QPU) handle.
///
/// Provides an interface for QPU implementations, including asynchronous
/// task management, device capabilities, and context handling. Subclasses
/// can mix in additional capabilities via template mixins.
class qpu_handle
    : public extension_point<qpu_handle, const platform_metadata &> {
protected:
  /// \brief Reference to the platform metadata.
  const platform_metadata &config;
  /// \brief Unique identifier for this QPU instance.
  std::size_t qpu_uid;
  /// \brief Static counter for unique QPU IDs.
  static std::size_t uid_counter;

  /// \struct execution_queue
  /// \brief Internal execution queue for asynchronous task management.
  struct execution_queue {
  private:
    /// \class task_base
    /// \brief Abstract base for queued tasks.
    struct task_base {
      /// \brief Execute the task.
      virtual void operator()() = 0;
      virtual ~task_base() = default;
    };

    /// \class task_impl
    /// \brief Concrete implementation of a queued task.
    /// \tparam F Callable task type.
    template <typename F>
    struct task_impl : task_base {
      F func;
      /// \brief Constructor.
      /// \param f Callable object to execute.
      task_impl(F &&f) : func(std::move(f)) {}
      /// \brief Execute the stored callable.
      void operator()() override { func(); }
    };

    std::mutex lock;    ///< Mutex for synchronizing queue access.
    std::thread thread; ///< Worker thread for executing tasks.
    std::queue<std::unique_ptr<task_base>>
        exec_queue;             ///< Queue of tasks to execute.
    std::condition_variable cv; ///< Condition variable for task notification.
    bool quit = false;          ///< Flag to signal thread termination.

    /// \brief Worker thread handler function.
    ///
    /// Waits for tasks to be enqueued and executes them sequentially.
    void handler() {
      std::unique_lock<std::mutex> l(lock);
      do {
        // Wait until we have data or a quit signal
        cv.wait(l, [this] { return (exec_queue.size() || quit); });
        // after wait, we own the lock
        if (!quit && exec_queue.size()) {
          auto op = std::move(exec_queue.front());
          exec_queue.pop();
          // unlock now that we're done messing with the queue
          l.unlock();
          (*op)();
          l.lock();
        }
      } while (!quit);
    }

  public:
    /// \brief Constructor. Starts the worker thread.
    execution_queue() { thread = std::thread(&execution_queue::handler, this); }

    /// \brief Destructor. Signals the thread to quit and joins it.
    ~execution_queue() {
      std::unique_lock<std::mutex> l(lock);
      quit = true;
      cv.notify_all();
      l.unlock();
      if (thread.joinable()) {
        thread.join();
      }
    }

    /// \brief Enqueue a task for execution.
    /// \tparam F Callable task type.
    /// \param task The function to execute asynchronously.
    template <typename F>
    void enqueue_task(F &&task) {
      std::unique_lock<std::mutex> l(lock);
      exec_queue.push(
          std::make_unique<task_impl<std::decay_t<F>>>(std::forward<F>(task)));
      cv.notify_one();
    }

    /// \brief Get the thread ID of the execution thread.
    /// \return The std::thread::id of the worker thread.
    std::thread::id getExecutionThreadId() const { return thread.get_id(); }
  };

  /// \brief Internal queue for managing asynchronous tasks.
  execution_queue taskQueue;

  /// Optional logging stream for platform output.
  // If set, the platform and its QPUs will print info log to this stream.
  // Otherwise, default output stream (std::cout) will be used.
  std::ostream *platformLogStream = nullptr;

public:
  /// \brief Construct a QPU handle with the given platform metadata.
  /// \param m Platform metadata reference.
  qpu_handle(const platform_metadata &m) : config(m), qpu_uid(uid_counter++) {}

  virtual ~qpu_handle() = default;

  /// \brief Reset the static unique ID counter for QPUs.
  static void reset_uid_counter() { uid_counter = 0; }

  /// \brief Subtype-specific handling for async task launch.
  virtual void handle_async_task_launch_impl() const {}

  /// \brief Handle any preprocessing necessary for async task launching.
  void handle_async_task_launch() const {
    set_qpu(qpu_uid);
    handle_async_task_launch_impl();
  }

  /// \brief Enqueue a task for asynchronous execution.
  /// \tparam TaskTy Callable task type.
  /// \param task The function to execute.
  template <typename TaskTy>
  void enqueue_task(TaskTy &&task) {
    taskQueue.enqueue_task([&, t = std::move(task), id = qpu_uid]() mutable {
      // This is a new thread. Handle any
      // preprocessing that needs to be done like setting the QPU id.
      handle_async_task_launch();
      // run the task
      t();
    });
  }

  /// \brief Get the remote capabilities of this QPU.
  /// \return RemoteCapabilities object.
  virtual RemoteCapabilities get_remote_capabilities() const {
    return RemoteCapabilities(/*initValues=*/false);
  }

  /// \brief Set the random seed for this QPU.
  /// \param seed Random seed value.
  virtual void set_random_seed(std::size_t seed) {}

  /// \brief Indicates if execution is remotely hosted.
  /// \return True if remote execution; false otherwise.
  virtual bool is_remote() const = 0;

  /// \brief Indicates if this QPU is a local simulator.
  /// \return True if simulator; false otherwise.
  virtual bool is_simulator() const = 0;

  /// \brief Indicates if this QPU is an emulator.
  /// \return True if emulator; false otherwise.
  virtual bool is_emulator() const = 0;

  /// \brief Check if conditional feedback is supported.
  /// \return True if supported; false otherwise.
  virtual bool supports_conditional_feedback() const = 0;

  /// \brief Check if explicit measurements are supported.
  /// \return True if supported; false otherwise.
  virtual bool supports_explicit_measurements() const = 0;

  virtual bool supports_task_distribution() const = 0;

  /// \brief Tear down the QPU, releasing resources.
  virtual void tear_down() = 0;

  /// \brief Set the execution context for this QPU.
  /// \param ctx Pointer to the execution context.
  virtual void set_execution_context(ExecutionContext *ctx) = 0;

  /// \brief Get the name of the current execution context, if any.
  /// \return Optional string containing the context name.
  virtual const std::optional<std::string> get_current_context_name() {
    return std::nullopt;
  }

  /// \brief Reset the execution context for this QPU.
  virtual void reset_execution_context() = 0;

  void resetLogStream() { platformLogStream = nullptr; }

  std::ostream *getLogStream() { return platformLogStream; }

  void setLogStream(std::ostream &logStream) { platformLogStream = &logStream; }

  /// \brief Cast this QPU to a specific capability type.
  /// \tparam CapabilityType The capability interface type.
  /// \return Pointer to the capability type if supported, nullptr otherwise.
  template <typename CapabilityType>
  CapabilityType *as() {
    return dynamic_cast<CapabilityType *>(this);
  }
};

/// \class qpu
/// \brief Concrete QPU implementation using template mixins for capabilities.
///
/// This class derives from qpu_handle and any number of capability mixins.
/// By default, it represents a local simulator QPU.
template <typename... Mixins>
class qpu : public qpu_handle, public Mixins... {
public:
  /// \brief Construct a QPU with the given platform metadata.
  /// \param m Platform metadata reference.
  qpu(const platform_metadata &m) : qpu_handle(m) {}

  /// \brief Indicates if execution is remotely hosted.
  bool is_remote() const override { return false; }

  /// \brief Indicates if this QPU is a local simulator.
  bool is_simulator() const override { return true; }

  /// \brief Indicates if this QPU is an emulator.
  bool is_emulator() const override { return false; }

  /// \brief Check if conditional feedback is supported.
  bool supports_conditional_feedback() const override { return false; }

  /// \brief Check if explicit measurements are supported.
  bool supports_explicit_measurements() const override { return true; }
  bool supports_task_distribution() const override { return false; }

  /// \brief Tear down the QPU, releasing resources.
  void tear_down() override {}
};

} // namespace cudaq::v2
