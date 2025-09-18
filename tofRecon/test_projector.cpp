#include <iostream>
#include <cuda_runtime.h>
#include "grid_utils.h"

// Declaration of the function from your CUDA file
extern "C" void CudaProjectorTOFBackListMode(
    float *h_detector_coords_1,  
    float *h_detector_coords_2,  
    float *h_delta_t,            
    int num_lors,
    void *d_volume,              
    const Grid &grid,    
    const float tof_sigma,       
    const float c_light,         
    const float step_size);

int main() {
    // Step 1: Define grid parameters
    int3 volume_size = {64, 64, 64};  // Example volume size
    float3 origin = {-32.0f, -32.0f, -32.0f};  // Origin at the center of the volume in grid
    float3 grid_spacing = {1.0f, 1.0f, 1.0f};  // 1 mm spacing

    Grid grid = defineGrid(volume_size, origin, grid_spacing);

    // Step 2: Initialize list mode data (LORs)
    const int num_lors = 1;  // Test with a single LOR
    float h_detector_coords_1[num_lors * 3] = {0.0f, 0.0f, -32.0f};  // Start point of LOR
    float h_detector_coords_2[num_lors * 3] = {0.0f, 0.0f, 32.0f};   // End point of LOR
    float h_delta_t[num_lors] = {0.0f};  // TOF centered (0 means no time difference)

    // Step 3: Allocate and initialize device memory for the volume
    float *d_volume;
    size_t volume_mem_size = volume_size.x * volume_size.y * volume_size.z * sizeof(float);
    cudaMalloc(&d_volume, volume_mem_size);
    // Step 4: Call the CUDA function
    float tof_sigma = 1.0f;  // Example TOF sigma
    float c_light = 0.299792;  // Speed of light in mm/ps
    float step_size = 1.0f;  // Step size for LOR marching

    CudaProjectorTOFBackListMode(
        h_detector_coords_1, h_detector_coords_2, h_delta_t, num_lors,
        d_volume, grid, tof_sigma, c_light, step_size);

    // Step 5: Copy the result back to the host
    float *h_volume = new float[volume_size.x * volume_size.y * volume_size.z];
    cudaMemcpy(h_volume, d_volume, volume_mem_size, cudaMemcpyDeviceToHost);

    // Step 6: Verify the results by printing some values
    std::cout << "Volume data after TOF backprojection:\n";
    for (int z = int(volume_size.z/2) - 2; z <  int(volume_size.z/2) + 3; ++z) {
        for (int y = int(volume_size.y/2) - 5; y < int(volume_size.y/2) + 5; ++y) {
            for (int x = int(volume_size.x/2) - 5; x < int(volume_size.x/2) + 5; ++x) {
                int idx = z * volume_size.y * volume_size.x + y * volume_size.x + x;
                std::cout << h_volume[idx] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    
    

    // Step 7: Clean up
    delete[] h_volume;
    cudaFree(d_volume);

    return 0;
}

