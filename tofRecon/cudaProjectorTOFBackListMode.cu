#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include "grid_utils.h"

__global__ void KernelProjectorTOFBackListMode(
    float *volume,
    const int *volume_size_array,
    const float *volume_spacing_array,
    const float *volume_origin_array,
    float *detector_coords_1_array,
    float *detector_coords_2_array,
    float *delta_t,
    float *tpos,
    float isoCenterMm3,
    float *axBin,
    float rebinEff_size,
    float *rebinEff,
    float *axialNorm,
    int num_lors,
    float tof_sigma,
    float c_light,
    float step_size) {

    // printf("blockIdx.x: %d\n", blockIdx.x);
    
    // Reconstruct int3 and float3 from arrays
    int3 volume_size = make_int3(volume_size_array[0], volume_size_array[1], volume_size_array[2]);
    float3 volume_spacing = make_float3(volume_spacing_array[0], volume_spacing_array[1], volume_spacing_array[2]);
    float3 volume_origin = make_float3(volume_origin_array[0], volume_origin_array[1], volume_origin_array[2]);

    // Reconstruct float3 for detector coordinates
    int lor_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (lor_index >= num_lors) return;


    float3 p1 = make_float3(detector_coords_1_array[3 * lor_index],
                            detector_coords_1_array[3 * lor_index + 1],
                            detector_coords_1_array[3 * lor_index + 2]);
                            
    float3 p2 = make_float3(detector_coords_2_array[3 * lor_index],
                            detector_coords_2_array[3 * lor_index + 1],
                            detector_coords_2_array[3 * lor_index + 2]);

    float dt = delta_t[lor_index];

    // Calculate the LOR vector
    float3 lor_vec = p1 - p2;
    float lor_length = length(lor_vec);
    float3 lor_dir = lor_vec / lor_length;

    // Estimate the most probable annihilation point using TOF
    float tof_dist = (c_light * (-dt)) / 2.0f;
    float3 midPoint = (p1 + p2) / 2.0f;
    float3 tof_point = midPoint - tof_dist * lor_dir;

    // Compute the number of steps along the LOR
    int num_steps = ceil(lor_length / step_size);
    float3 step = lor_dir * step_size;

    // March along the LOR
    float3 current_point = p2;
    for (int step_idx = 0; step_idx < num_steps; ++step_idx, current_point += step) {

        //convert current_point to detector coordinates
        //y1 + isoCenterMm(3) + lors.tpos
        float current_point_detector = current_point.z - isoCenterMm3 - tpos[lor_index];
        //interpolate corresponding rebinEff along axBin using current_point_detector
        float rebinEff_current = 0;
        for (int i = 0; i < rebinEff_size - 1; i++) {
            if (current_point_detector >= axBin[i] && current_point_detector < axBin[i+1]) {
                rebinEff_current = rebinEff[i] + (rebinEff[i+1] - rebinEff[i]) * (current_point_detector - axBin[i]) / (axBin[i+1] - axBin[i]);
                break;
            }
        }
        float correction = rebinEff_current * axialNorm[lor_index];

        
        // Convert current_point to voxel index using worldToGrid
        int3 voxel_idx = worldToGrid(current_point, Grid{volume_size, volume_origin, volume_spacing});

        // Debugging: Print information for the first LOR and first step
        // if (lor_index == 0 && step_idx == 0) {
        //     printf("LOR Index: %d\n", lor_index);
        //     printf("p1: (%f, %f, %f)\n", p1.x, p1.y, p1.z);
        //     printf("current_point: (%f, %f, %f)\n", current_point.x, current_point.y, current_point.z);
        //     printf("voxel_idx: (%d, %d, %d)\n", voxel_idx.x, voxel_idx.y, voxel_idx.z);
        //     printf("volume_size: (%d, %d, %d)\n", volume_size.x, volume_size.y, volume_size.z);
        //     printf("tof_dist: %f\n", tof_dist);
        // }

        // Ensure the voxel index is within bounds
        if (voxel_idx.x >= 0 && voxel_idx.x < volume_size.x &&
            voxel_idx.y >= 0 && voxel_idx.y < volume_size.y &&
            voxel_idx.z >= 0 && voxel_idx.z < volume_size.z) {
          
            // Calculate the distance from the voxel to the TOF point
            float distance_to_tof_point = length(current_point - tof_point);
            
            // if(lor_index == 0 && step_idx == 1) {
            //     printf("distance_to_tof_point: %f\n", distance_to_tof_point);
            // }

            // Apply the TOF Gaussian weighting with normalization
            float tof_weight = exp(-0.5f * pow(distance_to_tof_point / tof_sigma, 2)) / (sqrt(2 * M_PI) * tof_sigma);
            tof_weight *= correction;

            // Add the weighted value to the voxel
            int voxel_index = voxel_idx.z * volume_size.y * volume_size.x + voxel_idx.y * volume_size.x + voxel_idx.x;
            atomicAdd(&volume[voxel_index], tof_weight);
        }
    }
}

#define CUDA_CHECK_ERROR(call) {                                         \
    cudaError_t err = call;                                              \
    if (err != cudaSuccess) {                                            \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n";  \
        exit(EXIT_FAILURE);                                              \
    }                                                                    \
}

extern "C" __host__
void CudaProjectorTOFBackListMode(
    float *h_detector_coords_1,  
    float *h_detector_coords_2,  
    float *h_delta_t,            
    int num_lors,
    void *d_volume,              
    const Grid &grid,
    const float tof_sigma,       
    const float c_light,         
    const float step_size) {

    // Allocate and copy detector coordinates and delta t to the GPU
    
    // Allocate and copy detector coordinates and delta t to the GPU
    float *d_delta_t;
    float *d_detector_coords_1;
    float *d_detector_coords_2;

    CUDA_CHECK_ERROR(cudaMalloc(&d_detector_coords_1, num_lors * 3 * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_detector_coords_2, num_lors * 3 * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_delta_t, num_lors * sizeof(float)));


    CUDA_CHECK_ERROR(cudaMemcpy(d_detector_coords_1, h_detector_coords_1, num_lors * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_detector_coords_2, h_detector_coords_2, num_lors * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_delta_t, h_delta_t, num_lors * sizeof(float), cudaMemcpyHostToDevice));


    // Allocate and copy the small arrays to the GPU
    int volume_size_array[3] = {grid.volume_size.x, grid.volume_size.y, grid.volume_size.z};
    float volume_spacing_array[3] = {grid.grid_spacing.x, grid.grid_spacing.y, grid.grid_spacing.z};
    float volume_origin_array[3] = {grid.origin.x, grid.origin.y, grid.origin.z};

    int *d_volume_size_array;
    float *d_volume_spacing_array;
    float *d_volume_origin_array;

    CUDA_CHECK_ERROR(cudaMalloc(&d_volume_size_array, 3 * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_volume_spacing_array, 3 * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_volume_origin_array, 3 * sizeof(float)));

    CUDA_CHECK_ERROR(cudaMemcpy(d_volume_size_array, volume_size_array, 3 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_volume_spacing_array, volume_spacing_array, 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_volume_origin_array, volume_origin_array, 3 * sizeof(float), cudaMemcpyHostToDevice));



    float *volume = static_cast<float*>(d_volume);

    CUDA_CHECK_ERROR(cudaMemset(volume, 0, grid.volume_size.x * grid.volume_size.y * grid.volume_size.z * sizeof(float)));

    // Define block and grid sizes
    int threads_per_block = 256;  // Number of threads per block
    int num_blocks = (num_lors + threads_per_block - 1) / threads_per_block;

    printf("volume size array: (%d, %d, %d)\n", volume_size_array[0], volume_size_array[1], volume_size_array[2]);

    KernelProjectorTOFBackListMode <<<num_blocks, threads_per_block>>> (
        volume,
        d_volume_size_array,
        d_volume_spacing_array,
        d_volume_origin_array,
        d_detector_coords_1,
        d_detector_coords_2,
        d_delta_t,
        num_lors,
        tof_sigma,
        c_light,
        step_size
    );

    // Check for errors during kernel launch
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    CUDA_CHECK_ERROR(cudaFree(d_detector_coords_1));
    CUDA_CHECK_ERROR(cudaFree(d_detector_coords_2));
    CUDA_CHECK_ERROR(cudaFree(d_delta_t));
}
