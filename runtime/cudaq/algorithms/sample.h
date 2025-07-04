/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
#include "common/SampleResult.h"
#include "cudaq/algorithms/broadcast.h"
#include "cudaq/concepts.h"
#include "cudaq/host_config.h"

namespace cudaq {
bool kernelHasConditionalFeedback(const std::string &);
namespace __internal__ {
bool isKernelGenerated(const std::string &);
}
/// @brief Return type for asynchronous sampling.
using async_sample_result = async_result<sample_result>;

#if CUDAQ_USE_STD20
/// @brief Define a combined sample function validation concept.
/// These concepts provide much better error messages than old-school SFINAE
template <typename QuantumKernel, typename... Args>
concept SampleCallValid =
    ValidArgumentsPassed<QuantumKernel, Args...> &&
    HasVoidReturnType<std::invoke_result_t<QuantumKernel, Args...>>;
#endif

namespace details {

/// @brief Take the input KernelFunctor (a lambda that captures runtime
/// arguments and invokes the quantum kernel) and invoke the sampling process.
template <typename KernelFunctor>
std::optional<sample_result>
runSampling(KernelFunctor &&wrappedKernel, quantum_platform &platform,
            const std::string &kernelName, int shots, bool explicitMeasurements,
            std::size_t qpu_id = 0, details::future *futureResult = nullptr,
            std::size_t batchIteration = 0, std::size_t totalBatchIters = 0) {

  auto hasConditionalFeebdback =
      cudaq::kernelHasConditionalFeedback(kernelName);
  if (explicitMeasurements) {
    if (!platform.supports_explicit_measurements())
      throw std::runtime_error("The sampling option `explicit_measurements` is "
                               "not supported on this target.");
    if (hasConditionalFeebdback)
      throw std::runtime_error(
          "The sampling option `explicit_measurements` is not supported on a "
          "kernel with conditional logic on a measurement result.");
  }
  // Create the execution context.
  auto ctx = std::make_unique<ExecutionContext>("sample", shots);
  ctx->kernelName = kernelName;
  ctx->batchIteration = batchIteration;
  ctx->totalIterations = totalBatchIters;
  ctx->hasConditionalsOnMeasureResults = hasConditionalFeebdback;
  ctx->explicitMeasurements = explicitMeasurements;

#ifdef CUDAQ_LIBRARY_MODE
  // If we have a kernel that has its quake code registered, we
  // won't check for if statements with the tracer.
  auto isRegistered = cudaq::__internal__::isKernelGenerated(kernelName);

  // One extra check to see if we have mid-circuit
  // measures in library mode
  if (!isRegistered && !ctx->hasConditionalsOnMeasureResults) {
    // Trace the kernel function
    ExecutionContext context("tracer");
    auto &platform = get_platform();
    platform.set_exec_ctx(&context, qpu_id);
    wrappedKernel();
    platform.reset_exec_ctx(qpu_id);
    // In trace mode, if we have a measure result
    // that is passed to an if statement, then
    // we'll have collected registernames
    if (!context.registerNames.empty()) {
      // append new register names to the main sample context
      for (std::size_t i = 0; i < context.registerNames.size(); ++i)
        ctx->registerNames.emplace_back("auto_register_" + std::to_string(i));

      ctx->hasConditionalsOnMeasureResults = true;
    }
  }
#endif

  // Indicate that this is an async exec
  ctx->asyncExec = futureResult != nullptr;

  // Set the platform and the qpu id.
  platform.set_exec_ctx(ctx.get(), qpu_id);
  platform.set_current_qpu(qpu_id);

  // Loop until all shots are returned.
  cudaq::sample_result counts;
  while (counts.get_total_shots() < static_cast<std::size_t>(shots)) {
    wrappedKernel();
    if (futureResult) {
      *futureResult = ctx->futureResult;
      return std::nullopt;
    }
    platform.reset_exec_ctx(qpu_id);
    if (counts.get_total_shots() == 0)
      counts = std::move(ctx->result); // optimize for first iteration
    else
      counts += ctx->result;

    ctx->result.clear();
    if (counts.get_total_shots() == 0) {
      if (explicitMeasurements)
        throw std::runtime_error(
            "The sampling option `explicit_measurements` is not supported on a "
            "kernel without any measurement operation.");
      printf("WARNING: this kernel invocation produced 0 shots worth "
             "of results when executed. Exiting shot loop to avoid "
             "infinite loop.");
      break;
    }
    // Reset the context for the next round,
    // don't need to reset on the last exec
    if (counts.get_total_shots() < static_cast<std::size_t>(shots)) {
      platform.set_exec_ctx(ctx.get(), qpu_id);
    }
  }
  return counts;
}

/// @brief Take the input KernelFunctor (a lambda that captures runtime
/// arguments and invokes the quantum kernel) and invoke the sampling process
/// asynchronously. Return an `async_sample_result`, clients can retrieve the
/// results at a later time via the `get()` call.
template <typename KernelFunctor>
auto runSamplingAsync(KernelFunctor &&wrappedKernel, quantum_platform &platform,
                      const std::string &kernelName, int shots,
                      bool explicitMeasurements = false,
                      std::size_t qpu_id = 0) {
  if (qpu_id >= platform.num_qpus()) {
    throw std::invalid_argument("Provided qpu_id " + std::to_string(qpu_id) +
                                " is invalid (must be < " +
                                std::to_string(platform.num_qpus()) +
                                " i.e. platform.num_qpus())");
  }

  // If we are remote, then create the sampling executor with `cudaq::future`
  // provided
  if (platform.is_remote(qpu_id)) {
    details::future futureResult;
    details::runSampling(std::forward<KernelFunctor>(wrappedKernel), platform,
                         kernelName, shots, explicitMeasurements, qpu_id,
                         &futureResult);
    return async_sample_result(std::move(futureResult));
  }

  // Otherwise we'll create our own future/promise and return it
  KernelExecutionTask task(
      [qpu_id, explicitMeasurements, shots, kernelName, &platform,
       kernel = std::forward<KernelFunctor>(wrappedKernel)]() mutable {
        return details::runSampling(kernel, platform, kernelName, shots,
                                    explicitMeasurements, qpu_id)
            .value();
      });

  return async_sample_result(
      details::future(platform.enqueueAsyncTask(qpu_id, task)));
}
} // namespace details

/// @brief Sample options to provide to the sample() / async_sample() functions
///
/// @param shots number of shots to run for the given kernel
/// @param noise noise model to use for the sample operation
/// @param explicit_measurements whether or not to form the global register
/// based on user-supplied measurement order.
struct sample_options {
  std::size_t shots = 1000;
  cudaq::noise_model noise;
  bool explicit_measurements = false;
};

/// @overload
/// @brief Sample the given quantum kernel expression and return the
/// mapping of observed bit strings to corresponding number of
/// times observed.
///
/// @param kernel the kernel expression, must contain final measurements
/// @param args the variadic concrete arguments for evaluation of the kernel.
/// @returns counts, The counts dictionary.
///
/// @details Given a quantum kernel with void return type, sample
///          the corresponding quantum circuit generated by the kernel
///          expression, returning the mapping of bits observed to number
///          of times it was observed.
#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename... Args>
  requires SampleCallValid<QuantumKernel, Args...>
#else
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_r_v<void, QuantumKernel, Args...>>>
#endif
sample_result sample(QuantumKernel &&kernel, Args &&...args) {
  // Need the code to be lowered to llvm and the kernel to be registered
  // so that we can check for conditional feedback / mid circ measurement
  if constexpr (has_name<QuantumKernel>::value) {
    static_cast<cudaq::details::kernel_builder_base &>(kernel).jitCode();
  }

  // Run this SHOTS times
  auto &platform = cudaq::get_platform();
  auto shots = platform.get_shots().value_or(1000);
  auto kernelName = cudaq::getKernelName(kernel);
  return details::runSampling(
             [&]() mutable { kernel(std::forward<Args>(args)...); }, platform,
             kernelName, shots, /*explicitMeasurements=*/false)
      .value();
}

/// @overload
/// @brief Sample the given quantum kernel expression and return the
/// mapping of observed bit strings to corresponding number of
/// times observed. Specify the number of shots.
///
/// @param shots The number of samples to collect.
/// @param kernel The kernel expression, must contain final measurements.
/// @param args The variadic concrete arguments for evaluation of the kernel.
/// @returns The counts dictionary.
///
/// @details Given a quantum kernel with void return type, sample
///          the corresponding quantum circuit generated by the kernel
///          expression, returning the mapping of bits observed to number
///          of times it was observed.
#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename... Args>
  requires SampleCallValid<QuantumKernel, Args...>
#else
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_r_v<void, QuantumKernel, Args...>>>
#endif
auto sample(std::size_t shots, QuantumKernel &&kernel, Args &&...args) {
  // Need the code to be lowered to llvm and the kernel to be registered
  // so that we can check for conditional feedback / mid circ measurement
  if constexpr (has_name<QuantumKernel>::value) {
    static_cast<cudaq::details::kernel_builder_base &>(kernel).jitCode();
  }

  // Run this SHOTS times
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);
  return details::runSampling(
             [&]() mutable { kernel(std::forward<Args>(args)...); }, platform,
             kernelName, shots, /*explicitMeasurements=*/false)
      .value();
}

/// @brief Sample the given quantum kernel expression and return the
/// mapping of observed bit strings to corresponding number of
/// times observed.
///
/// @param options Sample options.
/// @param kernel The kernel expression, must contain final measurements.
/// @param args The variadic concrete arguments for evaluation of the kernel.
/// @returns The counts dictionary.
///
/// @details Given a quantum kernel with void return type, sample
///          the corresponding quantum circuit generated by the kernel
///          expression, returning the mapping of bits observed to number
///          of times it was observed.
#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename... Args>
  requires SampleCallValid<QuantumKernel, Args...>
#else
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_r_v<void, QuantumKernel, Args...>>>
#endif
sample_result sample(const sample_options &options, QuantumKernel &&kernel,
                     Args &&...args) {

  // Need the code to be lowered to llvm and the kernel to be registered
  // so that we can check for conditional feedback / mid circ measurement
  if constexpr (has_name<QuantumKernel>::value) {
    static_cast<cudaq::details::kernel_builder_base &>(kernel).jitCode();
  }

  auto &platform = cudaq::get_platform();
  auto shots = options.shots;
  auto kernelName = cudaq::getKernelName(kernel);
  platform.set_noise(&options.noise);
  auto ret = details::runSampling(
                 [&]() mutable { kernel(std::forward<Args>(args)...); },
                 platform, kernelName, shots, options.explicit_measurements)
                 .value();
  platform.reset_noise();
  return ret;
}

/// @brief Sample the given kernel expression asynchronously and return
/// the mapping of observed bit strings to corresponding number of
/// times observed.
///
/// @param qpu_id The id of the QPU to run asynchronously on.
/// @param kernel The kernel expression, must contain final measurements.
/// @param args The variadic concrete arguments for evaluation of the kernel.
/// @returns A `std::future` containing the resultant counts
/// dictionary.
///
/// @details Given a kernel with void return type, asynchronously sample
///          the corresponding quantum circuit generated by the kernel
///          expression, returning the mapping of bits observed to number
///          of times it was observed.
#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename... Args>
  requires SampleCallValid<QuantumKernel, Args...>
#else
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_r_v<void, QuantumKernel, Args...>>>
#endif
async_sample_result sample_async(const std::size_t qpu_id,
                                 QuantumKernel &&kernel, Args &&...args) {
  // Need the code to be lowered to llvm and the kernel to be registered
  // so that we can check for conditional feedback / mid circ measurement
  if constexpr (has_name<QuantumKernel>::value) {
    static_cast<cudaq::details::kernel_builder_base &>(kernel).jitCode();
  }

  // Run this SHOTS times
  auto &platform = cudaq::get_platform();
  auto shots = platform.get_shots().value_or(1000);
  auto kernelName = cudaq::getKernelName(kernel);

#if CUDAQ_USE_STD20
  return details::runSamplingAsync(
      [&kernel, ... args = std::forward<Args>(args)]() mutable {
        kernel(std::forward<Args>(args)...);
      },
      platform, kernelName, shots, /*explicitMeasurements=*/false, qpu_id);
#else
  return details::runSamplingAsync(
      detail::make_copyable_function([&kernel,
                                      args = std::make_tuple(std::forward<Args>(
                                          args)...)]() mutable {
        std::apply(
            [&kernel](Args &&...args) { kernel(std::forward<Args>(args)...); },
            std::move(args));
      }),
      platform, kernelName, shots, /*explicitMeasurements=*/false, qpu_id);
#endif
}

/// @brief Sample the given kernel expression asynchronously and return
/// the mapping of observed bit strings to corresponding number of
/// times observed.
///
/// @param shots The number of samples to collect.
/// @param qpu_id The id of the QPU to run asynchronously on.
/// @param kernel The kernel expression, must contain final measurements.
/// @param args The variadic concrete arguments for evaluation of the kernel.
/// @returns A `std::future` containing the resultant counts
/// dictionary.
///
/// @details Given a kernel with void return type, asynchronously sample
///          the corresponding quantum circuit generated by the kernel
///          expression, returning the mapping of bits observed to number
///          of times it was observed.
#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename... Args>
  requires SampleCallValid<QuantumKernel, Args...>
#else
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_r_v<void, QuantumKernel, Args...>>>
#endif
async_sample_result sample_async(std::size_t shots, std::size_t qpu_id,
                                 QuantumKernel &&kernel, Args &&...args) {
  // Need the code to be lowered to llvm and the kernel to be registered
  // so that we can check for conditional feedback / mid circ measurement
  if constexpr (has_name<QuantumKernel>::value) {
    static_cast<cudaq::details::kernel_builder_base &>(kernel).jitCode();
  }

  // Run this SHOTS times
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);

#if CUDAQ_USE_STD20
  return details::runSamplingAsync(
      [&kernel, ... args = std::forward<Args>(args)]() mutable {
        kernel(std::forward<Args>(args)...);
      },
      platform, kernelName, shots, /*explicitMeasurements=*/false, qpu_id);
#else
  return details::runSamplingAsync(
      detail::make_copyable_function([&kernel,
                                      args = std::make_tuple(std::forward<Args>(
                                          args)...)]() mutable {
        std::apply(
            [&kernel](Args &&...args) { kernel(std::forward<Args>(args)...); },
            std::move(args));
      }),
      platform, kernelName, shots, /*explicitMeasurements=*/false, qpu_id);
#endif
}

/// @brief Sample the given kernel expression asynchronously and return
/// the mapping of observed bit strings to corresponding number of
/// times observed.
///
/// @param options Sample options.
/// @param qpu_id The id of the QPU to run asynchronously on.
/// @param kernel The kernel expression, must contain final measurements.
/// @param args The variadic concrete arguments for evaluation of the kernel.
/// @returns A `std::future` containing the resultant counts
/// dictionary.
///
/// @details Given a kernel with void return type, asynchronously sample
///          the corresponding quantum circuit generated by the kernel
///          expression, returning the mapping of bits observed to number
///          of times it was observed.
#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename... Args>
  requires SampleCallValid<QuantumKernel, Args...>
#else
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_r_v<void, QuantumKernel, Args...>>>
#endif
async_sample_result sample_async(const sample_options &options,
                                 std::size_t qpu_id, QuantumKernel &&kernel,
                                 Args &&...args) {
  // Need the code to be lowered to llvm and the kernel to be registered
  // so that we can check for conditional feedback / mid circ measurement
  if constexpr (has_name<QuantumKernel>::value) {
    static_cast<cudaq::details::kernel_builder_base &>(kernel).jitCode();
  }
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);
  platform.set_noise(&options.noise);

#if CUDAQ_USE_STD20
  auto ret = details::runSamplingAsync(
      [&kernel, ... args = std::forward<Args>(args)]() mutable {
        kernel(std::forward<Args>(args)...);
      },
      platform, kernelName, options.shots, options.explicit_measurements,
      qpu_id);
#else
  auto ret = details::runSamplingAsync(
      detail::make_copyable_function([&kernel,
                                      args = std::make_tuple(std::forward<Args>(
                                          args)...)]() mutable {
        std::apply(
            [&kernel](Args &&...args) { kernel(std::forward<Args>(args)...); },
            std::move(args));
      }),
      platform, kernelName, options.shots, options.explicit_measurements,
      qpu_id);
#endif
  platform.reset_noise();
  return ret;
}

/// @brief Sample the given kernel expression asynchronously and return
/// the mapping of observed bit strings to corresponding number of
/// times observed. Defaults to QPU id 0.
///
/// @param kernel The kernel expression, must contain final measurements.
/// @param args The variadic concrete arguments for evaluation of the kernel.
/// @returns A `std::future` containing the resultant counts
/// dictionary.
///
/// @details Given a kernel with void return type, asynchronously sample
///          the corresponding quantum circuit generated by the kernel
///          expression, returning the mapping of bits observed to number
///          of times it was observed.
#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename... Args>
  requires SampleCallValid<QuantumKernel, Args...>
#else
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_r_v<void, QuantumKernel, Args...>>>
#endif
auto sample_async(QuantumKernel &&kernel, Args &&...args) {
  return sample_async(0, std::forward<QuantumKernel>(kernel),
                      std::forward<Args>(args)...);
}

/// @brief Run the standard sample functionality over a set of N
/// argument packs. For a kernel with signature `void(Args...)`, this
/// function takes as input a set of `vector<Arg>...`, a vector for
/// each argument type in the kernel signature. The vectors must be of
/// equal length, and element `i` of each vector is used in
/// execution `i` of the standard sample function. Results are collected
/// from the execution of every argument set and returned.
#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename... Args>
  requires SampleCallValid<QuantumKernel, Args...>
#else
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_r_v<void, QuantumKernel, Args...>>>
#endif
std::vector<sample_result> sample(QuantumKernel &&kernel,
                                  ArgumentSet<Args...> &&params) {
  // Get the platform and query the number of qpus
  auto &platform = cudaq::get_platform();
  auto numQpus = platform.num_qpus();

  // Create the functor that will broadcast the sampling tasks across
  // all requested argument sets provided.
  details::BroadcastFunctorType<sample_result, Args...> functor =
      [&](std::size_t qpuId, std::size_t counter, std::size_t N,
          Args &...singleIterParameters) -> sample_result {
    auto shots = platform.get_shots().value_or(1000);
    auto kernelName = cudaq::getKernelName(kernel);
    auto ret = details::runSampling(
                   [&kernel, &singleIterParameters...]() mutable {
                     kernel(std::forward<Args>(singleIterParameters)...);
                   },
                   platform, kernelName, shots, /*explicitMeasurements=*/false,
                   qpuId, nullptr, counter, N)
                   .value();
    return ret;
  };

  // Broadcast the executions and return the results.
  return details::broadcastFunctionOverArguments<sample_result, Args...>(
      numQpus, platform, functor, params);
}

/// @brief Run the standard sample functionality over a set of N
/// argument packs. For a kernel with signature `void(Args...)`, this
/// function takes as input a set of `vector<Arg>...`, a vector for
/// each argument type in the kernel signature. The vectors must be of
/// equal length, and the element `i` of each vector is used in
/// execution `i` of the standard sample function. Results are collected
/// from the execution of every argument set and returned. This overload
/// allows the number of circuit executions (shots) to be specified.
#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename... Args>
  requires SampleCallValid<QuantumKernel, Args...>
#else
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_r_v<void, QuantumKernel, Args...>>>
#endif
std::vector<sample_result> sample(std::size_t shots, QuantumKernel &&kernel,
                                  ArgumentSet<Args...> &&params) {
  // Get the platform and query the number of QPUs
  auto &platform = cudaq::get_platform();
  auto numQpus = platform.num_qpus();

  // Create the functor that will broadcast the sampling tasks across
  // all requested argument sets provided.
  details::BroadcastFunctorType<sample_result, Args...> functor =
      [&](std::size_t qpuId, std::size_t counter, std::size_t N,
          Args &...singleIterParameters) -> sample_result {
    auto kernelName = cudaq::getKernelName(kernel);
    auto ret = details::runSampling(
                   [&kernel, &singleIterParameters...]() mutable {
                     kernel(std::forward<Args>(singleIterParameters)...);
                   },
                   platform, kernelName, shots, /*explicitMeasurements=*/false,
                   qpuId, nullptr, counter, N)
                   .value();
    return ret;
  };

  // Broadcast the executions and return the results.
  return details::broadcastFunctionOverArguments<sample_result, Args...>(
      numQpus, platform, functor, params);
}

/// @brief Run the standard sample functionality over a set of N
/// argument packs. For a kernel with signature `void(Args...)`, this
/// function takes as input a set of `vector<Arg>...`, a vector for
/// each argument type in the kernel signature. The vectors must be of
/// equal length, and the element `i` of each vector is used in
/// execution `i` of the standard sample function. Results are collected
/// from the execution of every argument set and returned. This overload
/// allows the `sample_options` to be specified.
#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename... Args>
  requires SampleCallValid<QuantumKernel, Args...>
#else
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_r_v<void, QuantumKernel, Args...>>>
#endif
std::vector<sample_result> sample(const sample_options &options,
                                  QuantumKernel &&kernel,
                                  ArgumentSet<Args...> &&params) {
  // Get the platform and query the number of qpus
  auto &platform = cudaq::get_platform();
  auto numQpus = platform.num_qpus();
  auto shots = options.shots;

  platform.set_noise(&options.noise);

  // Create the functor that will broadcast the sampling tasks across
  // all requested argument sets provided.
  details::BroadcastFunctorType<sample_result, Args...> functor =
      [&, explicit_mz = options.explicit_measurements](
          std::size_t qpuId, std::size_t counter, std::size_t N,
          Args &...singleIterParameters) -> sample_result {
    auto kernelName = cudaq::getKernelName(kernel);
    auto ret = details::runSampling(
                   [&kernel, &singleIterParameters...]() mutable {
                     kernel(std::forward<Args>(singleIterParameters)...);
                   },
                   platform, kernelName, shots, explicit_mz, qpuId, nullptr,
                   counter, N)
                   .value();
    return ret;
  };

  // Broadcast the executions and return the results.
  auto ret = details::broadcastFunctionOverArguments<sample_result, Args...>(
      numQpus, platform, functor, params);

  platform.reset_noise();
  return ret;
}

/// @brief Run the standard sample functionality over a set of N
/// argument packs. For a kernel with signature `void(Args...)`, this
/// function takes as input a set of `vector<Arg>...`, a vector for
/// each argument type in the kernel signature. The vectors must be of
/// equal length, and the element `i` of each vector is used in
/// execution `i` of the standard sample function. Results are collected
/// from the execution of every argument set and returned.
#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename... Args>
  requires SampleCallValid<QuantumKernel, Args...>
#else
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_r_v<void, QuantumKernel, Args...>>>
#endif
[[deprecated("Use sample() overload instead")]] std::vector<sample_result>
sample_n(QuantumKernel &&kernel, ArgumentSet<Args...> &&params) {
  // Get the platform and query the number of qpus
  auto &platform = cudaq::get_platform();
  auto numQpus = platform.num_qpus();

  // Create the functor that will broadcast the sampling tasks across
  // all requested argument sets provided.
  details::BroadcastFunctorType<sample_result, Args...> functor =
      [&](std::size_t qpuId, std::size_t counter, std::size_t N,
          Args &...singleIterParameters) -> sample_result {
    auto shots = platform.get_shots().value_or(1000);
    auto kernelName = cudaq::getKernelName(kernel);
    auto ret = details::runSampling(
                   [&kernel, &singleIterParameters...]() mutable {
                     kernel(std::forward<Args>(singleIterParameters)...);
                   },
                   platform, kernelName, shots, /*explicitMeasurements=*/false,
                   qpuId, nullptr, counter, N)
                   .value();
    return ret;
  };

  // Broadcast the executions and return the results.
  return details::broadcastFunctionOverArguments<sample_result, Args...>(
      numQpus, platform, functor, params);
}

/// @brief Run the standard sample functionality over a set of N
/// argument packs. For a kernel with signature `void(Args...)`, this
/// function takes as input a set of `vector<Arg>...`, a vector for
/// each argument type in the kernel signature. The vectors must be of
/// equal length, and the element `i` of each vector is used in
/// execution `i` of the standard sample function. Results are collected
/// from the execution of every argument set and returned. This overload
/// allows the number of circuit executions (shots) to be specified.
#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename... Args>
  requires SampleCallValid<QuantumKernel, Args...>
#else
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_r_v<void, QuantumKernel, Args...>>>
#endif
[[deprecated("Use sample() overload instead")]] std::vector<sample_result>
sample_n(std::size_t shots, QuantumKernel &&kernel,
         ArgumentSet<Args...> &&params) {
  // Get the platform and query the number of qpus
  auto &platform = cudaq::get_platform();
  auto numQpus = platform.num_qpus();

  // Create the functor that will broadcast the sampling tasks across
  // all requested argument sets provided.
  details::BroadcastFunctorType<sample_result, Args...> functor =
      [&](std::size_t qpuId, std::size_t counter, std::size_t N,
          Args &...singleIterParameters) -> sample_result {
    auto kernelName = cudaq::getKernelName(kernel);
    auto ret = details::runSampling(
                   [&kernel, &singleIterParameters...]() mutable {
                     kernel(std::forward<Args>(singleIterParameters)...);
                   },
                   platform, kernelName, shots, /*explicitMeasurements=*/false,
                   qpuId, nullptr, counter, N)
                   .value();
    return ret;
  };

  // Broadcast the executions and return the results.
  return details::broadcastFunctionOverArguments<sample_result, Args...>(
      numQpus, platform, functor, params);
}
} // namespace cudaq
