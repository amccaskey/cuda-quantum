/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"
#include "cudaq/qpu.h"
#include "mock_qpu/phantom.h"

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

#ifdef CUDAQ_TEST_COMPILER_CONFIG
extern "C" {
void initialize_qpu(cudaq::qpu_configuration &options) {
  printf("We are providing qpu config in basic_tester\n");
  options.insert({"hello", "world"});
}
}
#endif

// ============================================================================
// Mock Server Management for Phantom Remote Tests
// ============================================================================

namespace {
/// Helper class to manage the Python mock server lifecycle
class MockServerManager {
private:
  pid_t server_pid_ = -1;
  std::string server_script_path_;
  int port_ = 5000;

public:
  MockServerManager() {
    // Find the mock_phantom_server.py script
    server_script_path_ = __FILE__;
    auto pos = server_script_path_.rfind('/');
    if (pos != std::string::npos) {
      server_script_path_ = server_script_path_.substr(0, pos + 1);
    }
    server_script_path_ += "mock_qpu/mock_phantom_server.py";
  }

  bool start() {
    printf("\n[MockServer] Starting Python mock server...\n");

    // Check if Flask is available
    int flask_check = system("python3 -c 'import flask' 2>/dev/null");
    if (flask_check != 0) {
      fprintf(stderr, "[MockServer] WARNING: Flask not installed, skipping "
                      "Phantom tests\n");
      fprintf(stderr, "[MockServer] Install Flask with: pip3 install flask\n");
      return false;
    }

    // Check if script exists
    std::ifstream script_check(server_script_path_);
    if (!script_check.good()) {
      fprintf(stderr, "[MockServer] ERROR: Cannot find %s\n",
              server_script_path_.c_str());
      return false;
    }
    script_check.close();

    // Fork and start the server
    server_pid_ = fork();

    if (server_pid_ == 0) {
      // Child process - start the Python server
      // Redirect output to /dev/null to avoid cluttering test output
      freopen("/dev/null", "w", stdout);
      freopen("/dev/null", "w", stderr);

      std::string port_str = std::to_string(port_);
      execlp("python3", "python3", server_script_path_.c_str(), "--port",
             port_str.c_str(), nullptr);

      // If execlp fails
      exit(1);
    } else if (server_pid_ > 0) {
      // Parent process - wait for server to be ready
      printf("[MockServer] Started server with PID %d on port %d\n",
             server_pid_, port_);

      // Give the server time to start up
      std::this_thread::sleep_for(std::chrono::milliseconds(500));

      // Check if server is still running
      int status;
      pid_t result = waitpid(server_pid_, &status, WNOHANG);
      if (result != 0) {
        fprintf(stderr, "[MockServer] ERROR: Server failed to start\n");
        server_pid_ = -1;
        return false;
      }

      printf("[MockServer] Server ready\n");
      return true;
    } else {
      // Fork failed
      fprintf(stderr, "[MockServer] ERROR: Failed to fork server process\n");
      return false;
    }
  }

  void stop() {
    if (server_pid_ > 0) {
      printf("\n[MockServer] Stopping server (PID %d)...\n", server_pid_);

      // Send SIGTERM
      kill(server_pid_, SIGTERM);

      // Wait for process to terminate (with timeout)
      int status;
      int timeout_ms = 2000;
      int elapsed_ms = 0;

      while (elapsed_ms < timeout_ms) {
        pid_t result = waitpid(server_pid_, &status, WNOHANG);
        if (result != 0) {
          printf("[MockServer] Server stopped gracefully\n");
          server_pid_ = -1;
          return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        elapsed_ms += 100;
      }

      // If still running, force kill
      printf("[MockServer] Server didn't stop gracefully, forcing...\n");
      kill(server_pid_, SIGKILL);
      waitpid(server_pid_, &status, 0);
      server_pid_ = -1;
      printf("[MockServer] Server stopped (forced)\n");
    }
  }

  ~MockServerManager() { stop(); }
};

// Global server manager instance
static MockServerManager *g_mock_server = nullptr;

} // anonymous namespace

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
    auto counts = cudaq::launch<gpu::state_vector<>>(
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
    gpu::state_vector<> qpu(
        {std::make_pair("hello", "world2"), std::make_pair("custom_int", 3)});
    auto counts = cudaq::launch(qpu, cudaq::sample_policy{.shots = 50}, bell);
    counts.dump();
    EXPECT_EQ(2, counts.size());
    EXPECT_EQ(2, qpu.get_configuration().size());
  }
}

TEST(RuntimeTester, checkAsyncSample) {

  using namespace cudaq::simulator;

  {
    auto bell = []() {
      cudaq::qubit q;
      h(q);
    };

    // Can use default qpu with async sample policy
    auto future_counts = cudaq::launch(cudaq::async::sample_policy{}, bell);
    // Do some other work while sampling happens asynchronously
    EXPECT_TRUE(future_counts.valid());
    // Get the result
    auto counts = future_counts.get();
    counts.dump();
    EXPECT_EQ(2, counts.size());
  }
  {
    auto bell = []() {
      cudaq::qubit q;
      h(q);
    };

    // can specify the qpu with async sample policy
    auto future_counts = cudaq::launch<gpu::state_vector<>>(
        cudaq::async::sample_policy{.shots = 10}, bell);
    EXPECT_TRUE(future_counts.valid());
    auto counts = future_counts.get();
    counts.dump();
    EXPECT_EQ(2, counts.size());
  }
  {
    auto bell = []() {
      cudaq::qubit q;
      h(q);
    };

    // can specify the qpu and configure with async sample policy
    gpu::state_vector<> qpu(
        {std::make_pair("hello", "world2"), std::make_pair("custom_int", 3)});
    auto future_counts =
        cudaq::launch(qpu, cudaq::async::sample_policy{.shots = 50}, bell);
    EXPECT_TRUE(future_counts.valid());
    auto counts = future_counts.get();
    counts.dump();
    EXPECT_EQ(2, counts.size());
    EXPECT_EQ(2, qpu.get_configuration().size());
  }
  {
    auto bell = []() {
      cudaq::qubit q;
      h(q);
    };

    // Multiple async samples can run concurrently
    std::vector<std::future<cudaq::sample_result>> futures;
    for (int i = 0; i < 3; i++) {
      futures.push_back(
          cudaq::launch(cudaq::async::sample_policy{.shots = 100}, bell));
    }

    // All futures should be valid
    for (auto &future : futures) {
      EXPECT_TRUE(future.valid());
    }

    // Collect all results
    for (auto &future : futures) {
      auto counts = future.get();
      counts.dump();
      EXPECT_EQ(2, counts.size());
    }
  }
}

// ============================================================================
// Phantom Remote Test Fixture with Server Management
// ============================================================================

class PhantomRemoteTest : public ::testing::Test {
protected:
  static bool server_available_;

  static void SetUpTestSuite() {
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("  Setting up Phantom Remote Test Suite\n");
    printf("═══════════════════════════════════════════════════════════\n");

    g_mock_server = new MockServerManager();
    server_available_ = g_mock_server->start();

    if (!server_available_) {
      delete g_mock_server;
      g_mock_server = nullptr;
      printf("[MockServer] Server not available, tests will be skipped\n");
    }
  }

  static void TearDownTestSuite() {
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("  Tearing down Phantom Remote Test Suite\n");
    printf("═══════════════════════════════════════════════════════════\n");

    if (g_mock_server) {
      delete g_mock_server;
      g_mock_server = nullptr;
    }
  }

  void SetUp() override {
    if (!server_available_) {
      GTEST_SKIP() << "Mock server not available (Flask not installed?)";
    }
  }
};

bool PhantomRemoteTest::server_available_ = false;

TEST_F(PhantomRemoteTest, checkBasicJobSubmission) {
  printf("\n[Test] Basic job submission with status polling\n");

  auto bell = []() {
    cudaq::qubit q;
    h(q);
  };

  // Create phantom QPU with local mock endpoint
  cudaq::mock::phantom qpu("local");

  // Launch returns a job handle, not direct results
  auto job = cudaq::launch(qpu, cudaq::sample_policy{.shots = 1000}, bell);

  // Job should have a valid ID
  EXPECT_FALSE(job.id().empty());
  printf("Submitted job: %s\n", job.id().c_str());

  // Check status (non-blocking)
  auto status = job.status();
  EXPECT_TRUE(status == cudaq::job_status::queued ||
              status == cudaq::job_status::running ||
              status == cudaq::job_status::completed);
  printf("Job status: %s\n", cudaq::to_string(status).c_str());

  // Get results (blocking - waits for completion)
  auto counts = job.get();
  counts.dump();

  // Should have 2 measurement outcomes (0 and 1)
  EXPECT_EQ(2, counts.size());

  // Verify it's completed now
  EXPECT_TRUE(job.is_complete());
  EXPECT_EQ(cudaq::job_status::completed, job.status());
}

TEST_F(PhantomRemoteTest, checkQPUConfiguration) {
  printf("\n[Test] QPU configuration and manual polling\n");

  auto bell = []() {
    cudaq::qubit q;
    h(q);
  };

  cudaq::qpu_configuration config;
  config.insert({"endpoint", "local"});
  config.insert({"auth_token", "test-token-123"});

  // Create phantom QPU with configuration
  cudaq::mock::phantom qpu(config);

  EXPECT_EQ("phantom", qpu.name());
  EXPECT_EQ("local", qpu.endpoint());

  // Submit job with custom shots
  auto job = cudaq::launch(qpu, cudaq::sample_policy{.shots = 500}, bell);

  // Can poll manually if desired
  int poll_count = 0;
  while (!job.is_complete() && poll_count < 100) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    poll_count++;
  }

  EXPECT_TRUE(job.is_complete());

  auto counts = job.get();
  counts.dump();
  EXPECT_EQ(2, counts.size());
}

TEST_F(PhantomRemoteTest, checkConcurrentJobs) {
  printf("\n[Test] Multiple concurrent remote jobs\n");

  auto bell = []() {
    cudaq::qubit q;
    h(q);
  };

  cudaq::mock::phantom qpu("local");

  std::vector<cudaq::job<cudaq::sample_result>> jobs;
  for (int i = 0; i < 3; i++) {
    auto job = cudaq::launch(qpu, cudaq::sample_policy{.shots = 100}, bell);
    printf("Submitted job %d: %s\n", i, job.id().c_str());
    jobs.push_back(std::move(job));
  }

  // All jobs should have unique IDs
  EXPECT_EQ(3, jobs.size());
  EXPECT_NE(jobs[0].id(), jobs[1].id());
  EXPECT_NE(jobs[1].id(), jobs[2].id());

  // Wait for all jobs to complete
  for (auto &job : jobs) {
    auto counts = job.get();
    counts.dump();
    EXPECT_EQ(2, counts.size());
    EXPECT_TRUE(job.is_complete());
  }
}

TEST_F(PhantomRemoteTest, checkResultCaching) {
  printf("\n[Test] Job result caching\n");

  auto bell = []() {
    cudaq::qubit q;
    h(q);
  };

  cudaq::mock::phantom qpu("local");
  auto job = cudaq::launch(qpu, cudaq::sample_policy{.shots = 200}, bell);

  // First get() - retrieves from remote
  auto counts1 = job.get();
  EXPECT_TRUE(job.has_result());

  // Second get() - returns cached result (no network call)
  auto counts2 = job.get();
  EXPECT_EQ(counts1.size(), counts2.size());
}

TEST(RuntimeTester, checkControlledOperations) {
  using namespace cudaq;
  using namespace cudaq::simulator;

  printf("\n=== Testing Controlled Operations ===\n");

  // Test 1: Single controlled operation (CNOT using template modifier)
  {
    printf("Test 1: Single controlled operation (template modifier)\n");
    auto cnot_test = []() {
      qubit q, r;
      cudaq::x(q);                 // Prepare |10⟩
      cudaq::x<cudaq::ctrl>(q, r); // CNOT: |10⟩ → |11⟩
    };

    auto counts = launch(sample_policy{.shots = 100}, cnot_test);
    counts.dump();
    EXPECT_EQ(1, counts.size());
    EXPECT_TRUE(counts.begin()->first == "11");
  }

  // Test 2: Multi-controlled operation (Toffoli)
  {
    printf("Test 2: Multi-controlled operation (Toffoli)\n");
    auto toffoli_test = []() {
      qubit q, r, s;
      cudaq::x(q); // Prepare |101⟩
      cudaq::x(s);
      cudaq::x<cudaq::ctrl>(q, r, s); // CCNOT with q,r as controls, s as target
    };

    auto counts = launch(sample_policy{.shots = 100}, toffoli_test);
    counts.dump();
    EXPECT_EQ(1, counts.size());
    EXPECT_TRUE(counts.begin()->first == "101"); // Controls were 10, so no flip
  }

  // Test 3: Controlled SWAP (Fredkin gate)
  {
    printf("Test 3: Controlled SWAP (Fredkin gate)\n");
    auto fredkin_test = []() {
      qubit q, r, s;
      cudaq::x(q); // Control to |1⟩ (rightmost bit)
      cudaq::x(s); // s to |1⟩ (leftmost bit)
      // State is |101⟩ in q,r,s order, or "101" in string
      cudaq::cswap(q, r, s); // Controlled SWAP of r and s
    };

    auto counts = launch(sample_policy{.shots = 100}, fredkin_test);
    counts.dump();
    EXPECT_EQ(1, counts.size());
    // After cswap: control is |1⟩, so swap happens: r=1, s=0 → |011⟩
    EXPECT_TRUE(counts.begin()->first == "011");
  }

  printf("✓ All controlled operation tests passed!\n");
}

TEST(RuntimeTester, checkAdjointOperations) {
  using namespace cudaq;
  using namespace cudaq::simulator;

  printf("\n=== Testing Adjoint Operations ===\n");

  // Test 1: Single qubit adjoint (self-inverse)
  {
    printf("Test 1: Single qubit adjoint (X gate)\n");
    auto adjoint_x = []() {
      qubit q;
      cudaq::x(q);
      cudaq::x<cudaq::adj>(q); // X† = X, should return to |0⟩
    };

    auto counts = launch(sample_policy{.shots = 100}, adjoint_x);
    counts.dump();
    EXPECT_EQ(1, counts.size());
    EXPECT_TRUE(counts.begin()->first == "0");
  }

  // Test 2: CNOT is self-adjoint
  {
    printf("Test 2: CNOT self-adjoint\n");
    auto cnot_adjoint = []() {
      qubit q, r;
      cudaq::h(q);                 // Create superposition
      cudaq::x<cudaq::ctrl>(q, r); // CNOT
      cudaq::x<cudaq::ctrl>(q, r); // CNOT again (self-adjoint)
      cudaq::h(q);                 // Return to |00⟩
    };

    auto counts = launch(sample_policy{.shots = 100}, cnot_adjoint);
    counts.dump();
    EXPECT_EQ(1, counts.size());
    EXPECT_TRUE(counts.begin()->first == "00");
  }

  printf("✓ All adjoint operation tests passed!\n");
}

TEST(RuntimeTester, checkKernelControlAndAdjoint) {
  using namespace cudaq;
  using namespace cudaq::simulator;

  printf("\n=== Testing Kernel Control and Adjoint ===\n");

  // Test 1: Kernel control region
  {
    printf("Test 1: Kernel control region\n");

    auto test_control = []() {
      qubit q, r;
      cudaq::x(q); // Control qubit to |1⟩

      auto apply_h = [](qubit &target) { cudaq::h(target); };

      cudaq::control(apply_h, q, r);
    };

    auto counts = launch(sample_policy{.shots = 100}, test_control);
    counts.dump();
    EXPECT_EQ(2, counts.size()); // Controlled-H creates superposition
  }

  // Test 2: Kernel adjoint region
  {
    printf("Test 2: Kernel adjoint region\n");

    auto test_adjoint = []() {
      qubit q;

      auto prepare = [](qubit &target) {
        cudaq::h(target);
        cudaq::t(target);
      };

      prepare(q);
      cudaq::adjoint(prepare, q); // Apply adjoint
    };

    auto counts = launch(sample_policy{.shots = 100}, test_adjoint);
    counts.dump();
    EXPECT_EQ(1, counts.size());
    EXPECT_TRUE(counts.begin()->first == "0"); // Should be back to |0⟩
  }

  // Test 3: Nested control regions
  {
    printf("Test 3: Nested control regions\n");

    auto nested_control_test = []() {
      qubit q, r, s;
      cudaq::x(q);
      cudaq::x(r); // Prepare |110⟩

      auto apply_x = [](qubit &target) { cudaq::x(target); };

      // Nested control: outer control on q, inner control on r
      cudaq::control([&](qubit &target) { cudaq::control(apply_x, r, target); },
                     q, s);
    };

    auto counts = launch(sample_policy{.shots = 100}, nested_control_test);
    counts.dump();
    EXPECT_EQ(1, counts.size());
    EXPECT_TRUE(counts.begin()->first == "111");
  }

  printf("✓ All kernel control and adjoint tests passed!\n");
}
