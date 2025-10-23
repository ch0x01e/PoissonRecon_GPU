#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include "Geometry.cuh"

// 使用项目中 Point3D<float>
using Point3D_f = Point3D<float>;

// Kernel: 归一化并朝向给定的中心点（centroid）
__global__ void normalize_and_orient_normals_kernel(Point3D_f* points, Point3D_f* normals, int n, Point3D_f centroid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float nx = normals[idx].coords[0];
    float ny = normals[idx].coords[1];
    float nz = normals[idx].coords[2];
    // 归一化
    float len = sqrtf(nx*nx + ny*ny + nz*nz);
    if (len > 1e-8f) {
        nx /= len; ny /= len; nz /= len;
    } else {
        // 若零向量，尝试用与点到中心方向一致的向量
        float vx = points[idx].coords[0] - centroid.coords[0];
        float vy = points[idx].coords[1] - centroid.coords[1];
        float vz = points[idx].coords[2] - centroid.coords[2];
        float vlen = sqrtf(vx*vx + vy*vy + vz*vz);
        if (vlen > 1e-8f) { nx = vx / vlen; ny = vy / vlen; nz = vz / vlen; }
        else { nx = 0.f; ny = 0.f; nz = 1.f; }
    }
    // 使法线朝向离质心的方向：若点到质心方向与法线点乘 < 0，则翻转
    float vx = centroid.coords[0] - points[idx].coords[0];
    float vy = centroid.coords[1] - points[idx].coords[1];
    float vz = centroid.coords[2] - points[idx].coords[2];
    float dot = nx*vx + ny*vy + nz*vz;
    if (dot < 0.f) { nx = -nx; ny = -ny; nz = -nz; }
    normals[idx].coords[0] = nx;
    normals[idx].coords[1] = ny;
    normals[idx].coords[2] = nz;
}

// Kernel: 暴力邻域平均平滑（半径 r）。仅用于点数较小时（N 小于阈值）。
// 权重为简单的计数平均（可替换为高斯权重）。
__global__ void smooth_normals_bruteforce_kernel(const Point3D_f* __restrict__ points,
                                                 Point3D_f* normals,
                                                 int n, float r2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float px = points[i].coords[0];
    float py = points[i].coords[1];
    float pz = points[i].coords[2];

    float sx = 0.f, sy = 0.f, sz = 0.f;
    int cnt = 0;
    // 暴力遍历所有点（仅在 N 较小时使用）
    for (int j = 0; j < n; ++j) {
        float dx = points[j].coords[0] - px;
        float dy = points[j].coords[1] - py;
        float dz = points[j].coords[2] - pz;
        float dist2 = dx*dx + dy*dy + dz*dz;
        if (dist2 <= r2) {
            sx += normals[j].coords[0];
            sy += normals[j].coords[1];
            sz += normals[j].coords[2];
            ++cnt;
        }
    }
    if (cnt > 0) {
        float inv = 1.0f / cnt;
        sx *= inv; sy *= inv; sz *= inv;
        // 归一化结果
        float len = sqrtf(sx*sx + sy*sy + sz*sz);
        if (len > 1e-8f) {
            normals[i].coords[0] = sx / len;
            normals[i].coords[1] = sy / len;
            normals[i].coords[2] = sz / len;
        }
    }
}

extern "C" void ImproveNormalsGPU(Point3D_f* d_points, Point3D_f* d_normals, int N,
                                  float smooth_radius, int max_bruteforce_points)
{
    if (!d_points || !d_normals || N <= 0) return;

    // 1) 计算质心（在 host 上对全部点做一次拷贝求和；也可采样）
    Point3D_f centroid_h; centroid_h.coords[0]=0; centroid_h.coords[1]=0; centroid_h.coords[2]=0;
    {
        // 为避免频繁大拷贝，当 N 很大时只采样前 min(N,200000) 点
        int sampleN = N;
        const int MAX_SAMPLE = 200000;
        if (N > MAX_SAMPLE) sampleN = MAX_SAMPLE;
        // 分配临时 host 缓冲
        Point3D_f* tmp = (Point3D_f*)malloc(sizeof(Point3D_f) * sampleN);
        if (tmp == nullptr) return;
        cudaMemcpy(tmp, d_points, sizeof(Point3D_f) * sampleN, cudaMemcpyDeviceToHost);
        double sx=0, sy=0, sz=0;
        for (int i=0;i<sampleN;++i){ sx += tmp[i].coords[0]; sy += tmp[i].coords[1]; sz += tmp[i].coords[2]; }
        free(tmp);
        centroid_h.coords[0] = float(sx / sampleN);
        centroid_h.coords[1] = float(sy / sampleN);
        centroid_h.coords[2] = float(sz / sampleN);
    }

    // 2) 归一化并朝向质心
    {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        normalize_and_orient_normals_kernel<<<blocks, threads>>>(d_points, d_normals, N, centroid_h);
        cudaError_t e = cudaGetLastError(); if (e != cudaSuccess){ fprintf(stderr,"normalize kernel err: %s\n", cudaGetErrorString(e)); return; }
        cudaDeviceSynchronize();
    }

    // 3) 可选：局部平滑（仅当点数不超过阈值）
    if (smooth_radius > 0.f && N <= max_bruteforce_points) {
        float r2 = smooth_radius * smooth_radius;
        int threads = 128;
        int blocks = (N + threads - 1) / threads;
        smooth_normals_bruteforce_kernel<<<blocks, threads>>>(d_points, d_normals, N, r2);
        cudaError_t e = cudaGetLastError(); if (e != cudaSuccess){ fprintf(stderr,"smooth kernel err: %s\n", cudaGetErrorString(e)); return; }
        cudaDeviceSynchronize();
    }

    // 完成：d_normals 已被修改（单位化和一致朝向，可能平滑）
}