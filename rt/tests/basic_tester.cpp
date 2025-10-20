#include "cudaq.h"

#include <gtest/gtest.h>

#ifdef CUDAQ_TEST_COMPILER_CONFIG
extern "C" {
void initialize_qpu(cudaq::heterogeneous_map &options) {
  printf("We are providing qpu config in basic_tester\n");
  options.insert("hello", "world");
}
}
#endif

TEST(RuntimeTester, checkSimple) {

  using namespace cudaq::simulator;

#ifdef CUDAQ_TEST_COMPILER_CONFIG
  EXPECT_EQ(1, cudaq::config::get_qpu_config().size());
#else
  EXPECT_EQ(0, cudaq::config::get_qpu_config().size());
#endif

  {
    auto bell = []() {
      cudaq::qubit q;
      h(q);
    };

    // Can use default qpu
    auto counts = cudaq::launch(cudaq::sample_policy{}, bell);
    counts.dump();
    EXPECT_EQ(2, counts.size());
  }
  {
    auto bell = []() {
      cudaq::qubit q;
      h(q);
    };

    // can specify the qpu
    auto counts = cudaq::launch<gpu::state_vector>(
        cudaq::sample_policy{.shots = 10}, bell);
    counts.dump();
    EXPECT_EQ(2, counts.size());
  }
  {
    auto bell = []() {
      cudaq::qubit q;
      h(q);
    };

    // can specify the qpu and configure
    gpu::state_vector qpu({{"hello", "world2"}, {"custom_int", 3}});
    auto counts = cudaq::launch(qpu, cudaq::sample_policy{.shots = 50}, bell);
    counts.dump();
    EXPECT_EQ(2, counts.size());
    EXPECT_EQ(2, qpu.get_configuration().size());
  }
}

TEST(RuntimeTester, checkBatchSampleOnMQPU) {
  using namespace cudaq::simulator;
  auto ghz = [](int i) {
    cudaq::qvector q(i);
    h(q[0]);
    for (int k = 0; k < i - 1; k++)
      x<cudaq::ctrl>(q[k], q[k + 1]);
    };
  {

    mqpu::state_vector mqpu_sv;
    EXPECT_TRUE(mqpu_sv.get_num_qpus() > 0);

    auto results = mqpu_sv.execute_batch(
        cudaq::sample_policy{}, ghz,
        {std::make_tuple(3), std::make_tuple(5), std::make_tuple(7)});

    EXPECT_EQ(3, results.size());
    EXPECT_EQ(3, results[0].begin()->first.size());
  }
  {

    mqpu::state_vector mqpu_sv(4);
    EXPECT_EQ(mqpu_sv.get_num_qpus(), 4);

    auto results = mqpu_sv.execute_batch(
        cudaq::sample_policy{}, ghz,
        {std::make_tuple(3), std::make_tuple(5), std::make_tuple(7)});

    EXPECT_EQ(3, results.size());
    EXPECT_EQ(3, results[0].begin()->first.size());
  }
  {

    auto results = cudaq::launch<mqpu::state_vector>(
        cudaq::batch::sample_policy{}, ghz,
        std::vector<std::tuple<int>>{std::make_tuple(3), std::make_tuple(5),
                                     std::make_tuple(7)});

    EXPECT_EQ(3, results.size());
    EXPECT_EQ(3, results[0].begin()->first.size());
    EXPECT_EQ(5, results[1].begin()->first.size());
  }
  {
    // should also work for gpu::state_vector
     auto results = cudaq::launch<gpu::state_vector>(
        cudaq::batch::sample_policy{}, ghz,
        std::vector<std::tuple<int>>{std::make_tuple(3), std::make_tuple(5),
                                     std::make_tuple(7)});

                                     for (auto& res : results) res.dump();
    EXPECT_EQ(3, results.size());
    EXPECT_EQ(3, results[0].begin()->first.size());
    EXPECT_EQ(5, results[1].begin()->first.size());
  }
}
