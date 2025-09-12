//forward_intr_extr.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(e) do {auto _e=(e); if(_e!=cudaSuccess){printf("CUDA %s @ %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__);}} while (0)
#endif

__device__ __forceinline__ float tri_sample(const float* __restrict__ vol, int Nx, int Ny, int Nz, float x, float y, float z){
    int x0 = int(floorf(x)), y0 = int(floorf(y)), z0 = int(floorf(z));
    if (x0<0 || y0<0 || z0<0 || x0>=Nx || y0>=Ny || z0>=Nz)
        return 0.0f;
    int x1 = min((x0+1), Nx-1), y1 = min((y0+1), Ny-1), z1 = min((z0+1), Nz-1);
    float dx = x - float(x0), dy = y - float(y0), dz = z - float(z0);
    auto at=[&](int xx, int yy, int zz){
        return vol[(size_t)zz*Nx*Ny + (size_t)yy*Nx + (size_t)xx];
    }
    float c000 = at(x0, y0, z0);
    float c100 = at(x1, y0, z0);
    float c010 = at(x0, y1, z0);
    float c110 = at(x1, y1, z0);
    float c001 = at(x0, y0, z1);
    float c101 = at(x1, y0, z1);
    float c011 = at(x0, y1, z1);
    float c111 = at(x1, y1, z1);

    float c00 = fmaf(dx, c100 - c000, c000);
    float c10 = fmaf(dx, c110 - c010, c010);
    float c01 = fmaf(dx, c101 - c001, c001);
    float c11 = fmaf(dx, c111 - c011, c011);
    float c0 = fmaf(dy, c10 - c00, c00);
    float c1 = fmaf(dy, c11 - c01, c01);
    return fmaf(dz, c1 - c0, c0);
}

__device__ __forceinline__ bool ray_box_intersect(float3 originalIdx, float3 dirIdx, float &tmin, float &tmax, float Nx, float Ny, float Nz){
    float3 inv = make_float3(1.f/dirIdx.x, 1.f/dirIdx.y, 1.f/dirIdx.z); //mm/vox
    float3 t0 = make_float3((0.f-originalIdx.x)*inv.x, (0.f-originalIdx.y)*inv.y, (0.f-originalIdx.z)*inv.z); //distance from source to lower planes (x y z), vox*mm/vox = mm
    float3 t1 = make_float3((Nx-originalIdx.x)*inv.x, (Ny-originalIdx.y)*inv.y, (Nz-originalIdx.z)*inv.z); //distance from source to higher planes (x y z)
    float3 tSmall = make_float3(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));
    float3 tBig = make_float3(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));
    tmin = fmaxf((fmaxf(tSmall.x, tSmall.y), fmax(tSmall.z, 0.f))); //we only care about the rays in front of the source
    tmax = fminf(fminf(tBig.x, tBig.y), tBig.z);
    return tmax > tmin; // tmin and tmax are in mm
}

__device__ __forceinline__ float3 mat3_mul_vec3(const float M[9], const float3 &v){
    return make_float3(
        fmaf(M[0], v.x, fmaf(M[1], v,y, fmaf(M[2], v.z, 0.f))),
        fmaf(M[3], v.x, fmaf(M[4], v,y, fmaf(M[5], v.z, 0.f))),
        fmaf(M[6], v.x, fmaf(M[7], v,y, fmaf(M[8], v.z, 0.f)))
    );
}

__device__ __forceinline__ float3 normlize3(const float3 &v){
    float len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    return make_float3(v.x/len, v.y/len, v.z/len);
}

__device__ __forceinline__ float3 mat3T_mul_vec3(const float M[9], const float3 &v){
    return make_float3(
        fmaf(M[0], v.x, fmaf(M[3], v,y, fmaf(M[6], v.z, 0.f))),
        fmaf(M[1], v.x, fmaf(M[4], v,y, fmaf(M[7], v.z, 0.f))),
        fmaf(M[2], v.x, fmaf(M[5], v,y, fmaf(M[8], v.z, 0.f)))
    );
}

__global__ void forward_project_intr_extr_kernel(const float* __restrict__ vol, int Nx, int Ny, int Nz, float vx, float vy, float vz, float ox, float oy, float oz,
                                                const float * __restrict__Kinv,
                                                const float * __restrict__Rwc,
                                                const float * __restrict__t,
                                                float * __restrict__ img, int H, int W, float step_mm){
    // Cache per-view matrices once per block (or use __constant__ if one view per launch)
    __shared__ float Kinv_s[9], Rwc_s[9], t_s[3];
    if (threadIdx.x == 0 && threadIdx.y == 0){
        #pragma unroll
        for (int i=0; i<9; i++){
            Kinv_s[i] = Kinv[i];
            Rwc_s[i] = Rwc[i];
        }
        for (int i=0; i<3; i++)
            t_s[i] = t[i];
    }

    __syncthreads();

    //detector coord (u,v)
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u>=W || v>=H)
        return;

    // dc ~ K^(-1) [u + 0.5, v + 0.5, 1]^T
    float3 pix = make_float3(float(u) + 0.5f, float(v) + 0.5f, 1.f);
    float3 dc = normalize3(mat3_mul_vec3(Kinv_s, pix)); //direction in camera coord     

    //ray direction in world coord dw = Rcw * dc
    float3 dw = mat3T_mul_vec3(Rwc_s, dc);
    //ray origin in world coord Cw = -Rcw * t
    float3 Cw = mat3T_mul_vec3(Rwc_s, make_float3(-t_s[0], -t_s[1], -t_s[2]));

    //Convert origin and direction from world coord to voxel index space
    auto world2idx = [&](const float3 &p){
        return make_float3((p.x - ox)/vx, (p.y - oy)/vy, (p.z - oz)/vz);
    };

    float3 originalIdx = world2idx(Cw); //ray origin in voxel index space
    float3 dirIdx = make_float3(dw.x/vx, dw.y/vy, dw.z/vz);  //direction with unit of unit 1/mm/vox = vox/mm

    float tmin, tmax;
    if (!ray_box_intersect(originalIdx, dirIdx, tmin, tmax, float(Nx), float(Ny), float(Nz))){
        img((size_t)v*W + (size_t)u) = 0.f;
        return;
    }

    float tcur = tmin;
    float sum = 0.f;
    while (tcur < tmax){
        float3 p = make_float3(
            originalIdx.x + tcur*dirIdx.x, //vox + mm*vox/mm = vox
            originalIdx.y + tcur*dirIdx.y,
            originalIdx.z + tcur*dirIdx.z
        ); // index coord of current sample point
        sum += tri_sample(vol, Nx, Ny, Nz, p.x, p.y, p.z);
        tcur += step_mm;
    }
    img[(size_t)v*W + (size_t)u] = sum * step_mm;
    
}

// ByBind launcher

static inline void check_cuda_float*const torch::Tensor& t, const char* name){
    TORCH_CHECK(t.is_cuda(), "%s must be CUDA tensor", name);
    TORCH_CHECK(t.scalar_type() == at::kFloat, "%s must be float32 tensor", name);
    TORCH_CHECK(t.is_contiguous(), "%s must be contiguous tensor", name);
}

void forward_project_intr_extr(
    torch::Tensor img,
    torch::Tensor vol,
    int Nx, int Ny, int Nz,
    float vx, float vy, float vz,
    float ox, float oy, float oz,
    torch::Tensor Kinv,
    torch::Tensor Rwc,
    torch::Tensor t,
    float step_mm
){
    check_cuda_float(img, "img");
    check_cuda_float(vol, "vol");
    check_cuda_float(Kinv, "Kinv");
    check_cuda_float(Rwc, "Rwc");
    check_cuda_float(t, "t");
    TORCH_CHECK(Kinv.numel() == 9, "Kinv must be 3x3");
    TORCH_CHECK(Rwc.numel() == 9, "Rwc must be 3x3");
    TORCH_CHECK(t.numel() == 3, "t must be 3x1");

    int H = img.size(0), W = img.size(1);
    dim3 block(16, 16); //each block has 16*16 threads
    dim3 grid((W+block.x-1)/block.x, (H+block.y-1)/block.y); //number of blocks needed
    forward_project_intr_extr<<<grid, block>>>(
        vol.data_ptr<float>(),
        Nx, Ny, Nz,
        vx, vy, vz,
        ox, oy, oz,
        Kinv.data_ptr<float>(),
        Rwc.data_ptr<float>(),
        t.data_ptr<float>(),
        img.data_ptr<float>(),
        H, W,
        step_mm
    );
    CHUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXXTENSION_NAME, m){
    m.def("forward_project_intr_extr", &forward_project_intr_extr, "Forward projection with intrinsic and extrinsic parameters (CUDA)");
}

