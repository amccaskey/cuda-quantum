# Mock QPUs for Testing

This directory contains test-only QPU implementations used for validating the CUDA-Q runtime infrastructure.

## Phantom Mock QPU

**Purpose**: Test harness for remote execution infrastructure

The `phantom` QPU is a mock implementation that simulates remote quantum execution without requiring an actual remote service. It provides job-based asynchronous execution for testing purposes.

### Components

- **`phantom.h`**: Mock remote QPU implementation
  - Namespace: `cudaq::mock::phantom`
  - Inherits from: `cudaq::qpu` and `cudaq::traits::remote_trait`
  - Provides: Job-based execution model with status polling

- **`mock_phantom_server.py`**: Python Flask server for integration testing
  - Simulates a remote quantum service with HTTP endpoints
  - Can be started automatically by test fixtures
  - Endpoints: `/submit`, `/status/<job_id>`, `/results/<job_id>`

### Usage in Tests

```cpp
#include "mock_qpu/phantom.h"

// In test fixture
cudaq::mock::phantom qpu("local");
auto job = cudaq::launch(qpu, cudaq::sample_policy{.shots = 1000}, kernel);

// Poll for completion
while (!job.is_complete()) {
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

auto results = job.get();
```

### Running the Mock Server Manually

```bash
cd runtime/cudaq/experimental/tests/mock_qpu
python3 mock_phantom_server.py --port 5000
```

### Design Notes

**Why in tests/ folder?**

The phantom QPU is strictly for testing infrastructure and should not be used in production code. Placing it in the tests directory makes this clear and prevents accidental dependencies from production code.

**What's being tested?**

- Remote execution trait pattern
- Job-based asynchronous execution model
- Status polling and result retrieval
- Concurrent job submission
- Result caching

**For production remote QPUs**, implement the same interface but with actual HTTP/gRPC clients communicating with real quantum services.

## Adding New Mock QPUs

When adding new test QPUs to this directory:

1. Use the `cudaq::mock` namespace
2. Include comprehensive documentation about the test purpose
3. Mark clearly as test-only code
4. Add corresponding test cases to `basic_tester.cpp`

---

*Test infrastructure only - not for production use*

