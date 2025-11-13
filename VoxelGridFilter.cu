#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <cstdint>
#include <cmath>
#include <cstdio>

#include "Geometry.cuh"
using Point3D_f = Point3D<float>;

// VoxelData 用于归约累加
struct VoxelData {
    float x, y, z;
    float nx, ny, nz;
    float cr, cg, cb;      // 新增：颜色累加
    int cnt;
    __host__ __device__ VoxelData()
        : x(0), y(0), z(0), nx(0), ny(0), nz(0),
          cr(0), cg(0), cb(0), cnt(0) {}
    __host__ __device__ VoxelData(float _x,float _y,float _z,
                                  float _nx,float _ny,float _nz,
                                  float _cr,float _cg,float _cb,int _c)
        : x(_x), y(_y), z(_z), nx(_nx), ny(_ny), nz(_nz),
          cr(_cr), cg(_cg), cb(_cb), cnt(_c) {}
    __host__ __device__ VoxelData& operator+=(const VoxelData &o){
        x += o.x; y += o.y; z += o.z;
        nx += o.nx; ny += o.ny; nz += o.nz;
        cr += o.cr; cg += o.cg; cb += o.cb;
        cnt += o.cnt;
        return *this;
    }
};
__host__ __device__ inline VoxelData operator+(const VoxelData &a,const VoxelData &b){ VoxelData r=a; r+=b; return r; }

// kernel: compute 64-bit voxel key for each point
__global__ void compute_keys_kernel(const Point3D_f* __restrict__ points, uint64_t* keys, int n, float leaf_size, int grid_dim_x, int grid_dim_y, int grid_dim_z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = points[idx].coords[0];
    float y = points[idx].coords[1];
    float z = points[idx].coords[2];
    int ix = (int)floorf(x / leaf_size);
    int iy = (int)floorf(y / leaf_size);
    int iz = (int)floorf(z / leaf_size);
    // clamp
    if(ix < 0) ix = 0; else if(ix >= grid_dim_x) ix = grid_dim_x - 1;
    if(iy < 0) iy = 0; else if(iy >= grid_dim_y) iy = grid_dim_y - 1;
    if(iz < 0) iz = 0; else if(iz >= grid_dim_z) iz = grid_dim_z - 1;
    // pack into 64-bit: 21 bits per axis (fits up to 2M)
    uint64_t key = ( (uint64_t)ix << 42 ) | ( (uint64_t)iy << 21 ) | (uint64_t)iz;
    keys[idx] = key;
}

// kernel: build VoxelData array from points & normals
__global__ void build_voxeldata_kernel(const Point3D_f* __restrict__ points,
                                       const Point3D_f* __restrict__ normals,
                                       VoxelData* vals, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    vals[idx].x = points[idx].coords[0];
    vals[idx].y = points[idx].coords[1];
    vals[idx].z = points[idx].coords[2];
    if(normals){
        vals[idx].nx = normals[idx].coords[0];
        vals[idx].ny = normals[idx].coords[1];
        vals[idx].nz = normals[idx].coords[2];
    } else {
        vals[idx].nx = vals[idx].ny = vals[idx].nz = 0.f;
    }
    // 颜色累加初始化
    vals[idx].cr = float(points[idx].color[0]);
    vals[idx].cg = float(points[idx].color[1]);
    vals[idx].cb = float(points[idx].color[2]);
    vals[idx].cnt = 1;
}

// kernel: write aggregated averages back to Point3D arrays
__global__ void write_aggregates_kernel(const VoxelData* __restrict__ agg,
                                        Point3D_f* out_pts, Point3D_f* out_nmls, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m) return;
    VoxelData v = agg[idx];
    float inv = v.cnt > 0 ? 1.0f / float(v.cnt) : 0.f;
    out_pts[idx].coords[0] = v.x * inv;
    out_pts[idx].coords[1] = v.y * inv;
    out_pts[idx].coords[2] = v.z * inv;
    // 写平均颜色
    float r = v.cr * inv;
    float g = v.cg * inv;
    float b = v.cb * inv;
    out_pts[idx].color[0] = (unsigned char)fminf(fmaxf(r,0.f),255.f);
    out_pts[idx].color[1] = (unsigned char)fminf(fmaxf(g,0.f),255.f);
    out_pts[idx].color[2] = (unsigned char)fminf(fmaxf(b,0.f),255.f);
    if (out_nmls) {
        out_nmls[idx].coords[0] = v.nx * inv;
        out_nmls[idx].coords[1] = v.ny * inv;
        out_nmls[idx].coords[2] = v.nz * inv;
    }
}

// Host API: perform GPU voxel grid filter.
// Inputs:
//   d_points : device pointer to Point3D<float> array (coordinates can be in any range; leaf_size computed in same units)
//   d_normals: device pointer to normals (may be nullptr)
//   N        : number of points
//   leaf_size: voxel size (in same units as points — if points are not normalized, choose accordingly)
// Outputs:
//   out_points_d/out_normals_d are allocated device pointers (caller must cudaFree)
//   out_count set to number of voxels kept
// Returns cuda error code (0 success)
extern "C" int VoxelGridFilterGPU(Point3D_f* d_points, Point3D_f* d_normals, int N, float leaf_size,
                       Point3D_f** out_points_d, Point3D_f** out_normals_d, int* out_count)
{
    if (N <= 0 || leaf_size <= 0.f) { *out_count = 0; *out_points_d = nullptr; *out_normals_d = nullptr; return 0; }

    // Compute bounding box (on device would require kernel; to keep simple and robust, copy a small sample to host)
    // Here we compute min/max on host by copying first and last few points if N not too large.
    // Simpler: assume points live in reasonable coordinate range. We'll compute grid dims based on extent from a quick device reduction.
    // For determinism, compute min/max on device using thrust.
    thrust::device_ptr<float> dptr(reinterpret_cast<float*>(d_points));
    // We can't directly use thrust::min_element on interleaved structure easily; instead perform a small kernel to find min/max per axis.
    // For simplicity and to avoid extra kernels, assume user provides leaf_size in same coordinate units and compute grid dims with a safe large bound.
    // Set grid dims so we don't overflow 21 bits: clamp to maxDim
    const int maxDimPerAxis = (1<<21) - 1;
    int gx = max(1, int(ceilf(1.0f / leaf_size)));
    if (gx > maxDimPerAxis) gx = maxDimPerAxis;
    int gy = gx, gz = gx;

    // allocate keys and vals
    thrust::device_vector<uint64_t> d_keys(N);
    thrust::device_vector<VoxelData> d_vals(N);

    // compute keys and build vals
    {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        compute_keys_kernel<<<blocks, threads>>>(d_points, thrust::raw_pointer_cast(d_keys.data()), N, leaf_size, gx, gy, gz);
        cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) return e;
        build_voxeldata_kernel<<<blocks, threads>>>(d_points, d_normals, thrust::raw_pointer_cast(d_vals.data()), N);
        e = cudaGetLastError(); if (e != cudaSuccess) return e;
        cudaDeviceSynchronize();
    }

    // sort by key, reorder vals accordingly
    thrust::sort_by_key(thrust::device, d_keys.begin(), d_keys.end(), d_vals.begin());

    // reduce by key (sum per voxel)
    thrust::device_vector<uint64_t> unique_keys(N);
    thrust::device_vector<VoxelData> aggregated(N);
    typedef thrust::device_vector<uint64_t>::iterator KeyIt;
    typedef thrust::device_vector<VoxelData>::iterator ValIt;
    thrust::pair<KeyIt, ValIt> end_pair =
        thrust::reduce_by_key(thrust::device,
                              d_keys.begin(), d_keys.end(),
                              d_vals.begin(),
                              unique_keys.begin(),
                              aggregated.begin());
    int unique_count = end_pair.first - unique_keys.begin();

    if (unique_count == 0) {
        *out_count = 0; *out_points_d = nullptr; *out_normals_d = nullptr;
        return 0;
    }

    // allocate output arrays
    Point3D_f* pts_out = nullptr;
    Point3D_f* nmls_out = nullptr;
    cudaError_t err;
    err = cudaMalloc((void**)&pts_out, sizeof(Point3D_f) * unique_count); if (err != cudaSuccess) return err;
    if (d_normals) {
        err = cudaMalloc((void**)&nmls_out, sizeof(Point3D_f) * unique_count); if (err != cudaSuccess) { cudaFree(pts_out); return err; }
    } else {
        nmls_out = nullptr;
    }

    // write aggregated averages to outputs
    {
        int threads = 256;
        int blocks = (unique_count + threads - 1) / threads;
        write_aggregates_kernel<<<blocks, threads>>>(thrust::raw_pointer_cast(aggregated.data()), pts_out, nmls_out, unique_count);
        err = cudaGetLastError(); if (err != cudaSuccess) { cudaFree(pts_out); if (nmls_out) cudaFree(nmls_out); return err; }
        cudaDeviceSynchronize();
    }

    *out_points_d = pts_out;
    *out_normals_d = nmls_out;
    *out_count = unique_count;
    return 0;
}