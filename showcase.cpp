#include "cudaq/driver/driver.h"

#include "cudaq/Support/TargetConfig.h"

// bin/nvq++ test.cpp -I ../include/ -I /usr/local/llvm/include -L /usr/local/llvm/lib/ -l LLVMSupport -lcudaq-driver

// QPU Target Config Diagram
//
// Host --- ethernet_channel --- controller -- quantum_register
//                                 |
//                                 | cuda channel
//                                 |
//                               MyA6000GPU (Device 0)     
//                          
const std::string target_config_str = R"#(

description: "Custom target for DGX-Q experimentation."

name: entangler 
controller-ip: "127.0.0.1"

config: 
  gen-target-backend: true
  platform-qpu: driver_qpu
  link-libs: ["-lcudaq-driver-qpu", "-lcudaq-driver"]

devices:
  - name: "MyA6000GPU"
    config: 
      channel: cuda_channel
      cuda_device: 0
  - name: MyA6000GPU_1
    config: 
      channel: cuda_channel 
      cuda_device: 1

)#";

int main() {

  using namespace cudaq;

  llvm::yaml::Input yin(target_config_str);
  config::TargetConfig config;
  yin >> config;
  driver::initialize(config);

  {
    // Allocate data on the driver
    auto devPtr = driver::malloc(sizeof(double));
    // Free the data
    driver::free(devPtr);
  }

  {
    // allocate data and set the data
    double value = 2.2;
    auto devPtr = driver::malloc(sizeof(double));
    driver::memcpy(devPtr, &value);

    double getValue = 0;
    driver::memcpy(&getValue, devPtr);
    printf("Are they the same? %lf vs %lf \n", value, getValue);

    driver::free(devPtr);
  }

  {
    // Allocate data on the GPU device (device 0)
    // Host is not connected to this device, but the
    // controller is and will forward the request
    auto gpuDevPtr = driver::malloc(sizeof(double), 0);
    driver::free(gpuDevPtr);
  }

  {
    // Allocate data on the GPU device (device 0)
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
    // Allocate data on the GPU device (device 0)
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
}