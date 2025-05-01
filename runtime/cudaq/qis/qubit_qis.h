/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/CustomOp.h"
#include "common/SampleResult.h"
#include "cudaq/host_config.h"
#include "cudaq/operators.h"
#include "cudaq/platformv2/gates.h"
#include "cudaq/platformv2/platform.h"
#include "cudaq/qis/modifiers.h"
#include "cudaq/qis/pauli_word.h"
#include "cudaq/qis/qarray.h"
#include "cudaq/qis/qkernel.h"
#include "cudaq/qis/qreg.h"
#include "cudaq/qis/qvector.h"
#include "cudaq/utils/cudaq_utils.h"
#include <algorithm>
#include <cstring>
#include <functional>

#define __qpu__ __attribute__((annotate("quantum")))

// This file describes the API for a default qubit logical instruction
// set for CUDA-Q kernels.

namespace cudaq {
using measure_result = bool;
using SpinMeasureResult = std::pair<double, sample_result>;

// Convert a qubit to its unique id representation
inline std::size_t qubitToQuditInfo(qubit &q) { return q.id(); }

// Define the common single qubit operations.
namespace qubit_op {
#define ConcreteQubitOp(NAME)                                                  \
  struct NAME##Op {                                                            \
    static const std::string name() { return #NAME; }                          \
  };

ConcreteQubitOp(h) ConcreteQubitOp(x) ConcreteQubitOp(y) ConcreteQubitOp(z)
    ConcreteQubitOp(s) ConcreteQubitOp(t) ConcreteQubitOp(rx)
        ConcreteQubitOp(ry) ConcreteQubitOp(rz) ConcreteQubitOp(r1)
            ConcreteQubitOp(u3)

} // namespace qubit_op

namespace details {
void warn(const std::string_view msg);

// --------------------------
// Useful C++17 compliant concept checks (note we re-implement
// std::remove_cvref since its a C++20 thing)
template <typename T>
using remove_cvref = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename T>
using IsQubitType = std::is_same<remove_cvref<T>, cudaq::qubit>;

template <typename T>
using IsQvectorType = std::is_same<remove_cvref<T>, cudaq::qvector<>>;

template <typename T>
using IsQviewType = std::is_same<remove_cvref<T>, cudaq::qview<>>;

template <typename T>
using IsQarrayType = std::is_base_of<cudaq::qarray_base, remove_cvref<T>>;
// --------------------------

/// @brief Map provided qubit arguments to a vector of QuditInfo.
template <typename... QuantumT>
void qubitsToQuditInfos(const std::tuple<QuantumT...> &quantumTuple,
                        std::vector<std::size_t> &qubits) {
  cudaq::tuple_for_each(quantumTuple, [&](auto &&element) {
    using T = decltype(element);
    if constexpr (IsQubitType<T>::value) {
      qubits.push_back(qubitToQuditInfo(element));
    } else if constexpr (IsQvectorType<T>::value || IsQviewType<T>::value ||
                         IsQarrayType<T>::value) {
      for (auto &q : element)
        qubits.push_back(qubitToQuditInfo(q));
    }
  });
}

/// @brief Search through the qubit arguments and see which ones are negated.
template <typename... QuantumT>
void findQubitNegations(const std::tuple<QuantumT...> &quantumTuple,
                        std::vector<bool> &qubitIsNegated) {
  cudaq::tuple_for_each(quantumTuple, [&](auto &&element) {
    using T = decltype(element);
    if constexpr (IsQubitType<T>::value) {
      qubitIsNegated.push_back(element.is_negative());
    } else if constexpr (IsQvectorType<T>::value || IsQviewType<T>::value ||
                         IsQarrayType<T>::value) {
      for (auto &q : element)
        qubitIsNegated.push_back(q.is_negative());
    }
    return;
  });
}

template <size_t N, typename Tuple, size_t... Indices>
auto tuple_slice_impl(Tuple &&tuple, std::index_sequence<Indices...>) {
  return std::make_tuple(std::get<Indices>(std::forward<Tuple>(tuple))...);
}

template <size_t N, typename... Args>
auto tuple_slice(std::tuple<Args...> &&tuple) {
  return tuple_slice_impl<N>(std::forward<std::tuple<Args...>>(tuple),
                             std::make_index_sequence<N>{});
}

template <size_t N, typename Tuple, size_t... Indices>
auto tuple_slice_last_impl(Tuple &&tuple, std::index_sequence<Indices...>) {
  constexpr size_t M = std::tuple_size_v<std::remove_reference_t<Tuple>> - N;
  return std::forward_as_tuple(
      std::get<M + Indices>(std::forward<Tuple>(tuple))...);
}

template <size_t N, typename... Args>
auto tuple_slice_last(std::tuple<Args...> &&tuple) {
  return tuple_slice_last_impl<N>(std::forward<std::tuple<Args...>>(tuple),
                                  std::make_index_sequence<N>{});
}

// 1. Type traits for container detection
template <typename T>
struct is_fixed_size_container : std::false_type {};

template <>
struct is_fixed_size_container<cudaq::qarray_base> : std::true_type {};

// 2. Compile-time qubit counting logic
template <typename T>
constexpr std::size_t count_qubits_compile_time() {
  if constexpr (details::IsQubitType<T>::value) {
    return 1;
  } else if constexpr (details::IsQarrayType<T>::value) {
    return std::tuple_size<std::decay_t<T>>::value;
  } else {
    return 0; // Dynamic containers handled at runtime
  }
}

template <typename Tuple, std::size_t... Is>
constexpr std::size_t sum_targets_impl(std::index_sequence<Is...>) {
  return (count_qubits_compile_time<std::tuple_element_t<Is, Tuple>>() + ...);
}

// Type trait to check if T has a static constexpr integer 'num_parameters'
template <typename T, typename = void>
struct has_num_parameters : std::false_type {};

template <typename T>
struct has_num_parameters<T, std::void_t<decltype(T::num_parameters)>>
    : std::bool_constant<std::is_integral_v<decltype(T::num_parameters)>> {};

template <typename T>
inline constexpr bool has_num_parameters_v = has_num_parameters<T>::value;

inline v2::simulation_trait *get_simulation_qpu() {
  auto *sim = v2::get_qpu().as<v2::simulation_trait>();
  if (!sim)
    throw std::runtime_error(
        "cannot run local, library-mode simulation with a qpu target that does "
        "not implement the simulation_trait");
  return sim;
}

inline v2::noise_trait *get_noise_qpu() {
  return v2::get_qpu().as<v2::noise_trait>();
}

inline void localApply(const std::string &name,
                       const std::vector<double> &params,
                       const std::vector<std::size_t> &controls,
                       const std::vector<std::size_t> &targets,
                       bool isAdjoint = false) {
  if (cudaq::customOpRegistry::getInstance().isOperationRegistered(name)) {
    const auto &op = cudaq::customOpRegistry::getInstance().getOperation(name);
    auto data = op.unitary(params);
    get_simulation_qpu()->apply(data, controls, targets,
                                {name, params, isAdjoint});
    return;
  }

  auto gateEnum = gates::gateNameFromString(name);
  auto matrixData = gates::getGateByName<double>(gateEnum, params);
  get_simulation_qpu()->apply(matrixData, controls, targets,
                              {name, params, isAdjoint});
}

/// @brief Generic quantum operation applicator function. Supports the
/// following signatures for a generic operation name `OP`
/// `OP(qubit(s))`
/// `OP<ctrl>(qubit..., qubit)`
/// `OP<ctrl>(qubits, qubit)`
/// `OP(scalar..., qubit(s))`
/// `OP<ctrl>(scalar..., qubit..., qubit)`
/// `OP<ctrl>(scalar..., qubits, qubit)`
/// `OP<adj>(qubit)`
/// `OP<adj>(scalar..., qubit)`
/// Control qubits can be negated. Compile errors should be thrown
/// for erroneous signatures.
template <typename mod, std::size_t NumT, std::size_t NumP,
          typename... RotationT, typename... QuantumT,
          std::size_t NumPProvided = sizeof...(RotationT),
          std::enable_if_t<NumP == NumPProvided, std::size_t> = 0>
void applyQuantumOperation(const std::string &gateName,
                           const std::tuple<RotationT...> &paramTuple,
                           const std::tuple<QuantumT...> &quantumTuple) {

  std::vector<double> parameters;
  cudaq::tuple_for_each(paramTuple,
                        [&](auto &&element) { parameters.push_back(element); });

  std::vector<std::size_t> qubits;
  qubitsToQuditInfos(quantumTuple, qubits);

  std::vector<bool> qubitIsNegated;
  findQubitNegations(quantumTuple, qubitIsNegated);

  assert(qubitIsNegated.size() == qubits.size() && "qubit mismatch");

  // Catch the case where we have multi-target broadcast, we don't allow that
  if (std::is_same_v<mod, base> && NumT > 1 && qubits.size() > NumT)
    throw std::runtime_error(
        "cudaq does not support broadcast for multi-qubit operations.");

  // Operation on correct number of targets, no controls, possible broadcast
  if ((std::is_same_v<mod, base> || std::is_same_v<mod, adj>)&&NumT == 1) {
    for (auto &qubit : qubits)
      localApply(gateName, parameters, {}, {qubit}, std::is_same_v<mod, adj>);
    return;
  }

  // Partition out the controls and targets
  std::size_t numControls = qubits.size() - NumT;
  std::vector<std::size_t> targets(qubits.begin() + numControls, qubits.end()),
      controls(qubits.begin(), qubits.begin() + numControls);

  // Apply X for any negations
  for (std::size_t i = 0; i < controls.size(); i++)
    if (qubitIsNegated[i])
      localApply("x", {}, {}, {controls[i]});

  // Apply the gate
  localApply(gateName, parameters, controls, targets, std::is_same_v<mod, adj>);

  // Reverse any negations
  for (std::size_t i = 0; i < controls.size(); i++)
    if (qubitIsNegated[i])
      localApply("x", {}, {}, {controls[i]});

  // Reset the negations
  cudaq::tuple_for_each(quantumTuple, [&](auto &&element) {
    using T = decltype(element);
    if constexpr (IsQubitType<T>::value) {
      if (element.is_negative())
        element.negate();
    } else if constexpr (IsQvectorType<T>::value || IsQviewType<T>::value ||
                         IsQarrayType<T>::value) {
      for (auto &q : element)
        if (q.is_negative())
          q.negate();
    }
  });
}

template <typename mod, std::size_t NUMT, std::size_t NUMP, typename... Args>
void genericApplicator(const std::string &gateName, Args &&...args) {
  applyQuantumOperation<mod, NUMT, NUMP>(
      gateName, tuple_slice<NUMP>(std::forward_as_tuple(args...)),
      tuple_slice_last<sizeof...(Args) - NUMP>(std::forward_as_tuple(args...)));
}

} // namespace details

#define CUDAQ_QIS_ONE_TARGET_QUBIT_V2(NAME, NUMT, NUMP)                        \
  namespace types {                                                            \
  struct NAME {                                                                \
    inline static const std::string name{#NAME};                               \
  };                                                                           \
  }                                                                            \
  template <typename mod = base, typename... Args>                             \
  void NAME(Args &&...args) {                                                  \
    details::genericApplicator<mod, NUMT, NUMP>(#NAME,                         \
                                                std::forward<Args>(args)...);  \
  }

CUDAQ_QIS_ONE_TARGET_QUBIT_V2(h, 1, 0)
CUDAQ_QIS_ONE_TARGET_QUBIT_V2(x, 1, 0)
CUDAQ_QIS_ONE_TARGET_QUBIT_V2(y, 1, 0)
CUDAQ_QIS_ONE_TARGET_QUBIT_V2(z, 1, 0)
CUDAQ_QIS_ONE_TARGET_QUBIT_V2(t, 1, 0)
CUDAQ_QIS_ONE_TARGET_QUBIT_V2(s, 1, 0)
CUDAQ_QIS_ONE_TARGET_QUBIT_V2(rx, 1, 1)
CUDAQ_QIS_ONE_TARGET_QUBIT_V2(ry, 1, 1)
CUDAQ_QIS_ONE_TARGET_QUBIT_V2(rz, 1, 1)
CUDAQ_QIS_ONE_TARGET_QUBIT_V2(r1, 1, 1)
CUDAQ_QIS_ONE_TARGET_QUBIT_V2(u3, 1, 3)
CUDAQ_QIS_ONE_TARGET_QUBIT_V2(swap, 2, 0)

// Define common 2 qubit operations.
inline void cnot(qubit &q, qubit &r) { x<cudaq::ctrl>(q, r); }
inline void cx(qubit &q, qubit &r) { x<cudaq::ctrl>(q, r); }
inline void cy(qubit &q, qubit &r) { y<cudaq::ctrl>(q, r); }
inline void cz(qubit &q, qubit &r) { z<cudaq::ctrl>(q, r); }
inline void ch(qubit &q, qubit &r) { h<cudaq::ctrl>(q, r); }
inline void cs(qubit &q, qubit &r) { s<cudaq::ctrl>(q, r); }
inline void ct(qubit &q, qubit &r) { t<cudaq::ctrl>(q, r); }
inline void ccx(qubit &q, qubit &r, qubit &s) { x<cudaq::ctrl>(q, r, s); }

// Define common 2 qubit operations with angles.
template <typename T>
void crx(T angle, qubit &q, qubit &r) {
  rx<cudaq::ctrl>(angle, q, r);
}
template <typename T>
void cry(T angle, qubit &q, qubit &r) {
  ry<cudaq::ctrl>(angle, q, r);
}
template <typename T>
void crz(T angle, qubit &q, qubit &r) {
  rz<cudaq::ctrl>(angle, q, r);
}
template <typename T>
void cr1(T angle, qubit &q, qubit &r) {
  r1<cudaq::ctrl>(angle, q, r);
}

// Define common single qubit adjoint operations.
inline void sdg(qubit &q) { s<cudaq::adj>(q); }
inline void tdg(qubit &q) { t<cudaq::adj>(q); }
// #endif

/// @brief Apply a general Pauli rotation, takes a qubit register and the size
/// must be equal to the Pauli word length.
#if CUDAQ_USE_STD20
template <typename QubitRange>
  requires(std::ranges::range<QubitRange>)
#else
template <
    typename QubitRange,
    typename = std::enable_if_t<!std::is_same_v<
        std::remove_reference_t<std::remove_cv_t<QubitRange>>, cudaq::qubit>>>
#endif
void exp_pauli(double theta, QubitRange &&qubits, const char *pauliWord) {
  std::vector<std::size_t> quditInfos;
  std::transform(qubits.begin(), qubits.end(), std::back_inserter(quditInfos),
                 [](auto &q) { return cudaq::qubitToQuditInfo(q); });
  // FIXME: it would be cleaner if we just kept it as a pauli word here
  details::get_simulation_qpu()->apply_exp_pauli(theta, {}, quditInfos,
                                                 spin_op::from_word(pauliWord));
}

/// @brief Apply a general Pauli rotation, takes a qubit register and thesize
/// must be equal to the Pauli word length.
#if CUDAQ_USE_STD20
template <typename QubitRange>
  requires(std::ranges::range<QubitRange>)
#else
template <
    typename QubitRange,
    typename = std::enable_if_t<!std::is_same_v<
        std::remove_reference_t<std::remove_cv_t<QubitRange>>, cudaq::qubit>>>
#endif
void exp_pauli(double theta, QubitRange &&qubits,
               const cudaq::pauli_word &pauliWord) {
  exp_pauli(theta, qubits, pauliWord.str().c_str());
}

/// @brief Apply a general Pauli rotation, takes a variadic set of
/// qubits, and the number of qubits must be equal to the Pauli word length.
template <typename... QubitArgs>
void exp_pauli(double theta, const char *pauliWord, QubitArgs &...qubits) {

  if (sizeof...(QubitArgs) != std::strlen(pauliWord))
    throw std::runtime_error(
        "Invalid exp_pauli call, number of qubits != size of pauliWord.");

  // Map the qubits to their unique ids and pack them into a std::array
  std::vector<std::size_t> quditInfos{qubitToQuditInfo(qubits)...};
  details::get_simulation_qpu()->apply_exp_pauli(theta, {}, quditInfos,
                                                 spin_op::from_word(pauliWord));
}

/// @brief Apply a general Pauli rotation with control qubits and a variadic set
/// of qubits. The number of qubits must be equal to the Pauli word length.
#if CUDAQ_USE_STD20
template <typename QuantumRegister, typename... QubitArgs>
  requires(std::ranges::range<QuantumRegister>)
#else
template <typename QuantumRegister, typename... QubitArgs,
          typename = std::enable_if_t<
              std::is_same_v<std::remove_reference_t<std::remove_cv_t<
                                 decltype(*QuantumRegister().begin())>>,
                             qubit>>>
#endif
void exp_pauli(QuantumRegister &ctrls, double theta, const char *pauliWord,
               QubitArgs &...qubits) {
  std::vector<std::size_t> controls;
  std::transform(ctrls.begin(), ctrls.end(), std::back_inserter(controls),
                 [](const auto &q) { return qubitToQuditInfo(q); });
  if (sizeof...(QubitArgs) != std::strlen(pauliWord))
    throw std::runtime_error(
        "Invalid exp_pauli call, number of qubits != size of pauliWord.");

  // Map the qubits to their unique ids and pack them into a std::array
  std::vector<std::size_t> quditInfos{qubitToQuditInfo(qubits)...};
  details::get_simulation_qpu()->apply_exp_pauli(theta, controls, quditInfos,
                                                 spin_op::from_word(pauliWord));
}

// /// @brief Measure an individual qubit, return 0,1 as `bool`
inline measure_result mz(qubit &q) {
  return details::get_simulation_qpu()->mz(q.id());
}

/// @brief Measure an individual qubit in `x` basis, return 0,1 as `bool`
inline measure_result mx(qubit &q) {
  h(q);
  return mz(q);
}

// Measure an individual qubit in `y` basis, return 0,1 as `bool`
inline measure_result my(qubit &q) {
  r1(-M_PI_2, q);
  h(q);
  return mz(q);
}

inline void reset(qubit &q) { details::get_simulation_qpu()->reset(q.id()); }

// Measure all qubits in the range, return vector of 0,1
#if CUDAQ_USE_STD20
template <typename QubitRange>
  requires std::ranges::range<QubitRange>
#else
template <
    typename QubitRange,
    typename = std::enable_if_t<!std::is_same_v<
        std::remove_reference_t<std::remove_cv_t<QubitRange>>, cudaq::qubit>>>
#endif
std::vector<measure_result> mz(QubitRange &q) {
  std::vector<measure_result> b;
  for (auto &qq : q) {
    b.push_back(mz(qq));
  }
  return b;
}

template <std::size_t Levels>
std::vector<measure_result> mz(const qview<Levels> &q) {
  std::vector<measure_result> b;
  for (auto &qq : q) {
    b.emplace_back(mz(qq));
  }
  return b;
}

template <typename... Qs>
std::vector<measure_result> mz(qubit &q, Qs &&...qs);

#if CUDAQ_USE_STD20
template <typename QubitRange, typename... Qs>
  requires(std::ranges::range<QubitRange>)
#else
template <
    typename QubitRange, typename... Qs,
    typename = std::enable_if_t<!std::is_same_v<
        std::remove_reference_t<std::remove_cv_t<QubitRange>>, cudaq::qubit>>>
#endif
std::vector<measure_result> mz(QubitRange &qr, Qs &&...qs) {
  std::vector<measure_result> result = mz(qr);
  auto rest = mz(std::forward<Qs>(qs)...);
  if constexpr (std::is_same_v<decltype(rest), measure_result>) {
    result.push_back(rest);
  } else {
    result.insert(result.end(), rest.begin(), rest.end());
  }
  return result;
}

template <typename... Qs>
std::vector<measure_result> mz(qubit &q, Qs &&...qs) {
  std::vector<measure_result> result = {mz(q)};
  auto rest = mz(std::forward<Qs>(qs)...);
  if constexpr (std::is_same_v<decltype(rest), measure_result>) {
    result.push_back(rest);
  } else {
    result.insert(result.end(), rest.begin(), rest.end());
  }
  return result;
}

namespace support {
// Helpers to deal with the `vector<bool>` specialized template type.
extern "C" {
void __nvqpp_initializer_list_to_vector_bool(std::vector<bool> &, char *,
                                             std::size_t);
void __nvqpp_vector_bool_to_initializer_list(void *, const std::vector<bool> &,
                                             std::vector<char *> **);
void __nvqpp_vector_bool_free_temporary_initlists(std::vector<char *> *);
}
} // namespace support

// Measure the state in the given spin_op basis.
// inline SpinMeasureResult measure(const cudaq::spin_op &term) {
//   return {false, {}}; // getExecutionManager()->measure(term);
// }

// Cast a measure register to an int64_t.
// This function is classic control code that may run on a QPU.
inline int64_t to_integer(std::vector<measure_result> bits) {
  int64_t ret = 0;
  for (std::size_t i = 0; i < bits.size(); i++) {
    if (bits[i]) {
      ret |= 1UL << i;
    }
  }
  return ret;
}

inline int64_t to_integer(std::string bitString) {
  std::reverse(bitString.begin(), bitString.end());
  return std::stoull(bitString, nullptr, 2);
}

#if CUDAQ_USE_STD20
// This concept tests if `Kernel` is a `Callable` that takes the arguments,
// `Args`, and returns `void`.
template <typename Kernel, typename... Args>
concept isCallableVoidKernel = requires(Kernel &&k, Args &&...args) {
  { k(args...) } -> std::same_as<void>;
};

template <typename T, typename Signature>
concept signature = std::is_convertible_v<T, std::function<Signature>>;

template <typename T>
concept takes_qubit = signature<T, void(qubit &)>;

template <typename T>
concept takes_qvector = signature<T, void(qvector<> &)>;
#endif

// Control the given cudaq kernel on the given control qubit
#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename... Args>
  requires isCallableVoidKernel<QuantumKernel, Args...>
#else
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_r_v<void, QuantumKernel, Args...>>>
#endif
void control(QuantumKernel &&kernel, qubit &control, Args &&...args) {
  std::vector<std::size_t> ctrls{control.id()};
  details::get_simulation_qpu()->applyControlRegion(
      ctrls, [&]() { kernel(std::forward<Args>(args)...); });
}

// Control the given cudaq kernel on the given register of control qubits
#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename QuantumRegister, typename... Args>
  requires std::ranges::range<QuantumRegister> &&
           isCallableVoidKernel<QuantumKernel, Args...>
#else
template <typename QuantumKernel, typename QuantumRegister, typename... Args,
          typename = std::enable_if_t<
              !std::is_same_v<
                  std::remove_reference_t<std::remove_cv_t<QuantumRegister>>,
                  cudaq::qubit> &&
              std::is_invocable_r_v<void, QuantumKernel, Args...>>>
#endif
void control(QuantumKernel &&kernel, QuantumRegister &&ctrl_qubits,
             Args &&...args) {
  std::vector<std::size_t> ctrls;
  for (std::size_t i = 0; i < ctrl_qubits.size(); i++) {
    ctrls.push_back(ctrl_qubits[i].id());
  }
  details::get_simulation_qpu()->applyControlRegion(
      ctrls, [&]() { kernel(std::forward<Args>(args)...); });
}

// Control the given cudaq kernel on the given list of references to control
// qubits.
#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename... Args>
  requires isCallableVoidKernel<QuantumKernel, Args...>
#else
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_r_v<void, QuantumKernel, Args...>>>
#endif
void control(QuantumKernel &&kernel,
             std::vector<std::reference_wrapper<qubit>> &&ctrl_qubits,
             Args &&...args) {
  std::vector<std::size_t> ctrls;
  for (auto &cq : ctrl_qubits)
    ctrls.push_back(cq.get().id());

  details::get_simulation_qpu()->applyControlRegion(
      ctrls, [&]() { kernel(std::forward<Args>(args)...); });
}

// Apply the adjoint of the given cudaq kernel
#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename... Args>
  requires isCallableVoidKernel<QuantumKernel, Args...>
#else
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_r_v<void, QuantumKernel, Args...>>>
#endif
void adjoint(QuantumKernel &&kernel, Args &&...args) {
  // static_assert(true, "adj not implemented yet.");
  details::get_simulation_qpu()->applyAdjointRegion(
      [&]() { kernel(std::forward<Args>(args)...); });
}

/// Instantiate this type to affect C A C^dag, where the user
/// provides cudaq Kernels C and A (compute, action).
// struct compute_action {
#if CUDAQ_USE_STD20
template <typename ComputeFunction, typename ActionFunction>
  requires isCallableVoidKernel<ComputeFunction> &&
           isCallableVoidKernel<ActionFunction>
#else
template <
    typename ComputeFunction, typename ActionFunction,
    typename = std::enable_if_t<std::is_invocable_r_v<void, ComputeFunction> &&
                                std::is_invocable_r_v<void, ActionFunction>>>
#endif
void compute_action(ComputeFunction &&c, ActionFunction &&a) {
  c();
  a();
  adjoint(c);
}

/// Instantiate this type to affect C^dag A C, where the user
/// provides cudaq Kernels C and A (compute, action).
// struct compute_dag_action {
#if CUDAQ_USE_STD20
template <typename ComputeFunction, typename ActionFunction>
  requires isCallableVoidKernel<ComputeFunction> &&
           isCallableVoidKernel<ActionFunction>
#else
template <
    typename ComputeFunction, typename ActionFunction,
    typename = std::enable_if_t<std::is_invocable_r_v<void, ComputeFunction> &&
                                std::is_invocable_r_v<void, ActionFunction>>>
#endif
void compute_dag_action(ComputeFunction &&c, ActionFunction &&a) {
  adjoint(c);
  a();
  c();
}

/// Helper function to extract a slice of a `std::vector<T>` to be used within
/// CUDA-Q kernels.
#if CUDAQ_USE_STD20
template <typename T>
  requires(std::is_arithmetic_v<T>)
#else
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
#endif
std::vector<T> slice_vector(std::vector<T> &original, std::size_t start,
                            std::size_t count) {
  std::vector<double> ret(original.begin() + start,
                          original.begin() + start + count);
  return ret;
}

} // namespace cudaq

/// For C++17 we can't adhere to the language specification for
/// the operation modifier type. For this case, we drop the modifier
/// template parameter and users have access to a `cNAME` operation for
/// single controlled operations.
#ifdef CUDAQ_USE_STD20
#define CUDAQ_MOD_TEMPLATE template <typename mod = base, typename... Args>
#else
#define CUDAQ_MOD_TEMPLATE template <typename... Args>
#endif

namespace cudaq::details {
template <typename T, typename... RotationT, typename... QuantumT,
          std::size_t NumPProvided = sizeof...(RotationT),
          std::enable_if_t<T::num_parameters == NumPProvided, std::size_t> = 0>
void applyNoiseImpl(const std::tuple<RotationT...> &paramTuple,
                    const std::tuple<QuantumT...> &quantumTuple) {
  const auto *noiseModel = details::get_noise_qpu()->get_noise();

  // per-spec, no noise model provided, emit warning, no application
  if (!noiseModel)
    return details::warn("apply_noise called but no noise model provided.");

  std::vector<double> parameters;
  cudaq::tuple_for_each(paramTuple,
                        [&](auto &&element) { parameters.push_back(element); });
  std::vector<std::size_t> qubits;
  // auto argTuple = std::forward_as_tuple(args...);
  cudaq::tuple_for_each(quantumTuple, [&qubits](auto &&element) {
    if constexpr (details::IsQubitType<decltype(element)>::value) {
      qubits.push_back(qubitToQuditInfo(element));
    } else {
      for (auto &qq : element) {
        qubits.push_back(qubitToQuditInfo(qq));
      }
    }
  });

  if (qubits.size() != T::num_targets) {
    throw std::invalid_argument("Incorrect number of target qubits. Expected " +
                                std::to_string(T::num_targets) + ", got " +
                                std::to_string(qubits.size()));
  }

  auto channel = noiseModel->template get_channel<T>(parameters);
  // per spec - caller provides noise model, but channel not registered,
  // warning generated, no channel application.
  if (channel.empty())
    return;

  details::get_noise_qpu()->apply_noise(channel, qubits);
}
} // namespace cudaq::details

namespace cudaq {

// Apply noise with runtime vector of parameters
template <typename... Args>
constexpr bool any_float = std::disjunction_v<
    std::is_floating_point<std::remove_cv_t<std::remove_reference_t<Args>>>...>;

#if CUDAQ_USE_STD20
#ifdef CUDAQ_REMOTE_SIM
#define TARGET_OK_FOR_APPLY_NOISE false
#else
#define TARGET_OK_FOR_APPLY_NOISE true
#endif
#else
#ifdef CUDAQ_REMOTE_SIM
#define TARGET_OK_FOR_APPLY_NOISE                                              \
  typename = std::enable_if_t<std::is_same_v<T, int>>
#else
#define TARGET_OK_FOR_APPLY_NOISE                                              \
  typename = std::enable_if_t<std::is_same_v<T, T>>
#endif
#endif

#if CUDAQ_USE_STD20
template <typename T, typename... Q>
  requires(std::derived_from<T, cudaq::kraus_channel> && !any_float<Q...> &&
           TARGET_OK_FOR_APPLY_NOISE)
#else
template <typename T, typename... Q, TARGET_OK_FOR_APPLY_NOISE,
          typename = std::enable_if_t<
              std::is_base_of_v<cudaq::kraus_channel, T> &&
              std::is_convertible_v<const volatile T *,
                                    const volatile cudaq::kraus_channel *> &&
              !any_float<Q...>>>
#endif
void apply_noise(const std::vector<double> &params, Q &&...args) {
  const auto *noiseModel = details::get_noise_qpu()->get_noise();

  // per-spec, no noise model provided, emit warning, no application
  if (!noiseModel)
    return details::warn("apply_noise called but no noise model provided. "
                         "skipping kraus channel application.");

  std::vector<std::size_t> qubits;
  auto argTuple = std::forward_as_tuple(args...);
  cudaq::tuple_for_each(argTuple, [&qubits](auto &&element) {
    if constexpr (details::IsQubitType<decltype(element)>::value) {
      qubits.push_back(qubitToQuditInfo(element));
    } else {
      for (auto &qq : element) {
        qubits.push_back(qubitToQuditInfo(qq));
      }
    }
  });

  auto channel = noiseModel->template get_channel<T>(params);
  // per spec - caller provides noise model, but channel not registered,
  // warning generated, no channel application.
  if (channel.empty())
    return;

  details::get_noise_qpu()->apply_noise(channel, qubits);
}

class kraus_channel;

template <unsigned len, typename A, typename... As>
constexpr unsigned count_leading_floats() {
  // Note: don't use remove_cvref to keep this C++17 clean.
  if constexpr (std::is_floating_point_v<
                    std::remove_cv_t<std::remove_reference_t<A>>>) {
    return count_leading_floats<len + 1, As...>();
  } else {
    return len;
  }
}
template <unsigned len>
constexpr unsigned count_leading_floats() {
  return len;
}

template <typename... Args>
constexpr bool any_vector_of_float = std::disjunction_v<std::is_same<
    std::vector<double>, std::remove_cv_t<std::remove_reference_t<Args>>>...>;

#if CUDAQ_USE_STD20
template <typename T, typename... Args>
  requires(std::derived_from<T, cudaq::kraus_channel> &&
           !any_vector_of_float<Args...> && TARGET_OK_FOR_APPLY_NOISE)
#else
template <typename T, typename... Args, TARGET_OK_FOR_APPLY_NOISE,
          typename = std::enable_if_t<
              std::is_base_of_v<cudaq::kraus_channel, T> &&
              std::is_convertible_v<const volatile T *,
                                    const volatile cudaq::kraus_channel *> &&
              !any_vector_of_float<Args...>>>
#endif
void apply_noise(Args &&...args) {
  constexpr auto ctor_arity = count_leading_floats<0, Args...>();
  constexpr auto qubit_arity = sizeof...(args) - ctor_arity;

  details::applyNoiseImpl<T>(
      details::tuple_slice<ctor_arity>(std::forward_as_tuple(args...)),
      details::tuple_slice_last<qubit_arity>(std::forward_as_tuple(args...)));
}

} // namespace cudaq

#define __qop__ __attribute__((annotate("user_custom_quantum_operation")))

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b

/// Register a new custom unitary operation providing a unique name,
/// the number of target qubits, the number of rotation parameters (can be 0),
/// and the unitary matrix as a 1D row-major `std::vector<complex>`
/// representation following a MSB qubit ordering.
#define CUDAQ_REGISTER_OPERATION(NAME, NUMT, NUMP, ...)                        \
  namespace cudaq {                                                            \
  struct CONCAT(NAME, _operation) : public ::cudaq::unitary_operation {        \
    std::vector<std::complex<double>>                                          \
    unitary(const std::vector<double> &parameters) const override {            \
      [[maybe_unused]] std::complex<double> i(0, 1.);                          \
      return __VA_ARGS__;                                                      \
    }                                                                          \
    static inline const bool registered_ = []() {                              \
      cudaq::customOpRegistry::getInstance()                                   \
          .registerOperation<CONCAT(NAME, _operation)>(#NAME);                 \
      return true;                                                             \
    }();                                                                       \
  };                                                                           \
  CUDAQ_MOD_TEMPLATE                                                           \
  void NAME(Args &&...args) {                                                  \
    /* Perform registration at call site as well in case the static            \
     * initialization was not executed in the same context, e.g., remote       \
     * execution.*/                                                            \
    cudaq::customOpRegistry::getInstance()                                     \
        .registerOperation<CONCAT(NAME, _operation)>(#NAME);                   \
    details::genericApplicator<mod, NUMT, NUMP>(#NAME,                         \
                                                std::forward<Args>(args)...);  \
  }                                                                            \
  }                                                                            \
  __qop__ std::vector<std::complex<double>> CONCAT(NAME,                       \
                                                   CONCAT(_generator_, NUMT))( \
      const std::vector<double> &parameters = std::vector<double>()) {         \
    return __VA_ARGS__;                                                        \
  }
