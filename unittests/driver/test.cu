#include <cuda.h>
#include <vector>
#include <cstdint>
#include <stdio.h>
// Kernel declaration (compiled separately)
extern "C" __global__ void incrementInt(int* ptr);

int main() {
    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction cuFunction;
    
    // Initialize CUDA driver API
    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&cuContext, 0, cuDevice);
    
    // Load compiled kernel
    cuModuleLoad(&cuModule, "unittests/driver/CMakeFiles/test_cuda_kernels.dir/test_cuda.fatbin");
    cuModuleGetFunction(&cuFunction, cuModule, "incrementInt");
    
    // Create device buffer
    CUdeviceptr d_ptr;
    cuMemAlloc(&d_ptr, sizeof(int));
    int i = 2;
    cuMemcpyHtoD(d_ptr, &i, 4);
    
    // Package single argument into buffer
    size_t argBufferSize = 0;
    std::vector<char> argBuffer;

    auto add_arg = [&](const void* val, size_t size, size_t alignment) {
        size_t offset = argBuffer.size();
        offset = (offset + alignment - 1) & ~(alignment - 1);
        printf("OFFSET IS %lu %lu %lu\n", offset, size, alignment);
        argBuffer.resize(offset + size);
        memcpy(argBuffer.data() + offset, val, size);
        argBufferSize = argBuffer.size();
    };

    // Add pointer argument with 8-byte alignment
    add_arg(&d_ptr, sizeof(d_ptr), 8);

    // Configure launch parameters
    void* config[] = {
        CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer.data(),
        CU_LAUNCH_PARAM_BUFFER_SIZE, &argBufferSize,
        CU_LAUNCH_PARAM_END
    };

    // Launch kernel with single thread
    cuLaunchKernel(cuFunction,
        1, 1, 1,  // grid dimensions
        1, 1, 1,  // block dimensions
        0,        // shared memory
        nullptr,  // stream
        nullptr,  // kernelParams
        config    // extra parameters
    );

    // Cleanup
    cuMemFree(d_ptr);
    cuModuleUnload(cuModule);
    cuCtxDestroy(cuContext);
    return 0;
}
