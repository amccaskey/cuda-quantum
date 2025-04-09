#include "cudaq.h"
#include "cudaq/driver/driver.h"

__qpu__ int test(double angle) {
  cudaq::qubit q, r;
  // r1(angle, q);
  h(q);
  x<cudaq::ctrl>(q, r);
  auto b = mz(q);
  // if (b) x(r);
  return b;
}

int main() {

  using namespace cudaq;

  {
    // Allocate data on the driver
    auto devPtr = driver::malloc(sizeof(double));
    // Free the data
    driver::free(devPtr);
  }

  {
    // allocate data and set the data on the controller
    double value = 2.2;
    auto devPtr = driver::malloc(sizeof(double));
    driver::memcpy(devPtr, &value);

    double getValue = 0;
    driver::memcpy(&getValue, devPtr);
    printf("Are they the same? %lf vs %lf \n", value, getValue);

    driver::free(devPtr);
  }

  {
    // Allocate data on the 0th device
    // Host is not connected to this device, but the
    // controller is and will forward the request
    auto gpuDevPtr = driver::malloc(sizeof(double), 0);
    driver::free(gpuDevPtr);
  }

  {
    // Allocate data on the 0th device
    // Host is not connected to this device, but the
    // controller is and will forward the request
    auto gpuDevPtr = driver::malloc(sizeof(double), 0);
    double value = 2.2;
    driver::memcpy(gpuDevPtr, &value);

    double getValue = 0.;
    driver::memcpy(&getValue, gpuDevPtr);
    printf("From GPU, are they the same? %lf vs %lf \n", value, getValue);

    driver::free(gpuDevPtr);
  }

  {
    // Allocate data on the 0th device 
    // Host is not connected to this device, but the
    // controller is and will forward the request
    auto gpuDevPtr = driver::malloc(sizeof(double), 1);
    double value = 2.2;
    driver::memcpy(gpuDevPtr, &value);

    double getValue = 0.;
    driver::memcpy(&getValue, gpuDevPtr);
    printf("From GPU, are they the same? %lf vs %lf \n", value, getValue);

    driver::free(gpuDevPtr);
  }

  {
    auto i = test(2.2);
    printf("From showcase %d\n", i);
  }
}
