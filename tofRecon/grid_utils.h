#ifndef GRID_UTILS_H
#define GRID_UTILS_H

#include <cuda_runtime.h>
#include <cmath>

// Define the Grid structure
struct Grid {
    int3 volume_size;       // Number of voxels in each dimension (Nx, Ny, Nz)
    float3 origin;          // Origin point of the grid in world coordinates (x0, y0, z0)
    float3 grid_spacing;    // Size of each voxel in world coordinates (dx, dy, dz)
};

// Function to define a grid
__host__ inline Grid defineGrid(int3 volume_size, float3 origin, float3 grid_spacing) {
    Grid grid;
    grid.volume_size = volume_size;
    grid.origin = origin;
    grid.grid_spacing = grid_spacing;
    return grid;
}

// Function to map world coordinates to grid voxel indices
__device__ inline int3 worldToGrid(const float3 &world_coord, const Grid &grid) {
    int3 voxel_idx;
    voxel_idx.x = static_cast<int>((world_coord.x - grid.origin.x) / grid.grid_spacing.x);
    voxel_idx.y = static_cast<int>((world_coord.y - grid.origin.y) / grid.grid_spacing.y);
    voxel_idx.z = static_cast<int>((world_coord.z - grid.origin.z) / grid.grid_spacing.z);
    return voxel_idx;
}


// Vector operation helpers
__device__ inline float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float length(const float3 &vec) {
    return sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

__device__ inline float3 operator/(const float3 &a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ inline float3 operator*(const float3 &a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ inline float3 operator*(float b, const float3 &a) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ inline float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator+( float3 &a,  float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3& operator+=(float3 &a, const float3 &b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

#endif // GRID_UTILS_H

