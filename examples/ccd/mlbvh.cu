//
// mlbvh.cu
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "mlbvh.cuh"
#include <cmath>
#include "cuda_tools.h"
#include <cstdint>
#include <cub/device/device_radix_sort.cuh>
#include <iostream>
#include <fstream>
#include <utility>
#include "gpu_eigen_libs.cuh"

namespace
{
constexpr uint32_t INVALID_INDEX = 0xFFFFFFFFu;

__device__ inline uint32_t reserve_bounded(uint32_t* counter,
                                           uint32_t  capacity,
                                           uint32_t* overflow)
{
    const uint32_t current = atomicAdd(counter, 1u);
    if(current < capacity)
        return current;
    if(overflow)
        atomicAdd(overflow, 1u);
    return INVALID_INDEX;
}

__device__ inline uint32_t active_case(const int4& pair)
{
    if(pair.x >= 0)
        return 4u;
    if(pair.z < 0)
        return 1u;
    if(pair.w < 0)
        return 2u;
    return 3u;
}

__device__ inline void emit_active(const int4& pair,
                                   uint32_t*   cpNum,
                                   uint32_t*   caseRank,
                                   int4*       activePairs,
                                   uint32_t    capacity,
                                   uint32_t*   activeOverflow)
{
    const uint32_t output = reserve_bounded(cpNum, capacity, activeOverflow);
    if(output == INVALID_INDEX)
        return;

    const uint32_t collisionCase = active_case(pair);
    activePairs[output]           = pair;
    caseRank[output]              = atomicAdd(cpNum + collisionCase, 1u);
}
}
template <class F>
__device__ __host__
inline F __m_min(F a, F b) {
    return a > b ? b : a;
}


template <class F>
__device__ __host__
inline F __m_max(F a, F b) {
    return a > b ? a : b;
}

template <class element_type>
__global__
void _calcLeafBvs(const double3* _vertexes, const element_type* _elements, AABB* _bvs, int faceNum, int type = 0) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= faceNum) return;
    AABB _bv;

    element_type _e = _elements[idx];
    double3 _v = _vertexes[_e.x];
    _bv.combines(_v.x, _v.y, _v.z);
    _v = _vertexes[_e.y];
    _bv.combines(_v.x, _v.y, _v.z);
    if (type == 0) {
        _v = _vertexes[*((uint32_t*)(&_e) + 2)];
        _bv.combines(_v.x, _v.y, _v.z);
    }
    _bvs[idx] = _bv;
}

template <class element_type>
__global__
void _calcLeafBvs_ccd(const double3* _vertexes, const double3* _moveDir, double alpha, const element_type* _elements, AABB* _bvs, int faceNum, int type = 0) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= faceNum) return;
    AABB _bv;

    element_type _e = _elements[idx];
    double3 _v = _vertexes[_e.x];
    double3 _mvD = _moveDir[_e.x];
    _bv.combines(_v.x, _v.y, _v.z);
    _bv.combines(_v.x - _mvD.x * alpha, _v.y - _mvD.y * alpha, _v.z - _mvD.z * alpha);


    _v = _vertexes[_e.y];
    _mvD = _moveDir[_e.y];
    _bv.combines(_v.x, _v.y, _v.z);
    _bv.combines(_v.x - _mvD.x * alpha, _v.y - _mvD.y * alpha, _v.z - _mvD.z * alpha);
    if (type == 0) {
        _v = _vertexes[*((uint32_t*)(&_e) + 2)];
        _mvD = _moveDir[*((uint32_t*)(&_e) + 2)];
        _bv.combines(_v.x, _v.y, _v.z);
        _bv.combines(_v.x - _mvD.x * alpha, _v.y - _mvD.y * alpha, _v.z - _mvD.z * alpha);
    }
    _bvs[idx] = _bv;
}

__global__ void cache_swept_point_boxes(const double3*  vertices,
                                        const double3*  moveDir,
                                        double          alpha,
                                        const uint32_t* surfaceVertices,
                                        AABB*           pointBoxes,
                                        uint32_t        pointCount)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= pointCount)
        return;
    const uint32_t point = surfaceVertices[idx];
    const double3 x       = vertices[point];
    const double3 dx      = moveDir[point];
    AABB box;
    box.combines(x.x, x.y, x.z);
    box.combines(x.x - dx.x * alpha,
                 x.y - dx.y * alpha,
                 x.z - dx.z * alpha);
    pointBoxes[idx] = box;
}

template <class element_type>
__global__ void validate_cached_leaf_bounds(const double3*     vertices,
                                            const element_type* elements,
                                            const AABB*        cachedBoxes,
                                            const Node*        nodes,
                                            uint32_t           elementCount,
                                            int                type,
                                            uint32_t*          outside)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= elementCount)
        return;

    const uint32_t leaf      = idx + elementCount - 1u;
    const element_type elem  = elements[nodes[leaf].element_idx];
    AABB current;
    double3 x = vertices[elem.x];
    current.combines(x.x, x.y, x.z);
    x = vertices[elem.y];
    current.combines(x.x, x.y, x.z);
    if(type == 0)
    {
        x = vertices[reinterpret_cast<const uint32_t*>(&elem)[2]];
        current.combines(x.x, x.y, x.z);
    }

    const AABB cached = cachedBoxes[leaf];
    const bool contained =
        current.lower.x >= cached.lower.x && current.lower.y >= cached.lower.y
        && current.lower.z >= cached.lower.z && current.upper.x <= cached.upper.x
        && current.upper.y <= cached.upper.y && current.upper.z <= cached.upper.z;
    if(!contained)
        atomicExch(outside, 1u);
}

__global__ void validate_cached_point_bounds(const double3*  vertices,
                                             const uint32_t* surfaceVertices,
                                             const AABB*     cachedBoxes,
                                             uint32_t        pointCount,
                                             uint32_t*       outside)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= pointCount)
        return;
    const double3 x = vertices[surfaceVertices[idx]];
    const AABB box  = cachedBoxes[idx];
    const bool contained = x.x >= box.lower.x && x.y >= box.lower.y
                           && x.z >= box.lower.z && x.x <= box.upper.x
                           && x.y <= box.upper.y && x.z <= box.upper.z;
    if(!contained)
        atomicExch(outside, 1u);
}

template <class element_type>
void calcLeafBvs(const double3* _vertexes, const element_type* _faces, AABB* _bvs, const int& faceNum, const int& type) {
    int numbers = faceNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calcLeafBvs <<<blockNum, threadNum >>> (_vertexes, _faces, _bvs + numbers - 1, faceNum, type);
}

template <class element_type>
void calcLeafBvs_fullCCD(const double3* _vertexes, const double3* _moveDir, const double& alpha, const element_type* _faces, AABB* _bvs, const int& faceNum, const int& type) {
    int numbers = faceNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calcLeafBvs_ccd << <blockNum, threadNum >> > (_vertexes, _moveDir, alpha, _faces, _bvs + numbers - 1, faceNum, type);
}

__device__ __host__
inline AABB merge(const AABB& lhs, const AABB& rhs) noexcept
{
    AABB merged;
    merged.upper.x = __m_max(lhs.upper.x, rhs.upper.x);
    merged.upper.y = __m_max(lhs.upper.y, rhs.upper.y);
    merged.upper.z = __m_max(lhs.upper.z, rhs.upper.z);
    merged.lower.x = __m_min(lhs.lower.x, rhs.lower.x);
    merged.lower.y = __m_min(lhs.lower.y, rhs.lower.y);
    merged.lower.z = __m_min(lhs.lower.z, rhs.lower.z);
    return merged;
}

__device__ __host__
inline bool overlap(const AABB& lhs, const AABB& rhs, const double& gapL) noexcept
{
    if ((rhs.lower.x - lhs.upper.x) >= gapL || (lhs.lower.x - rhs.upper.x) >= gapL) return false;
    if ((rhs.lower.y - lhs.upper.y) >= gapL || (lhs.lower.y - rhs.upper.y) >= gapL) return false;
    if ((rhs.lower.z - lhs.upper.z) >= gapL || (lhs.lower.z - rhs.upper.z) >= gapL) return false;
    return true;
}

__device__ __host__
inline double3 centroid(const AABB& box) noexcept
{
    double3 c;
    c.x = (box.upper.x + box.lower.x) * 0.5;
    c.y = (box.upper.y + box.lower.y) * 0.5;
    c.z = (box.upper.z + box.lower.z) * 0.5;
    return c;
}

__device__ __host__
inline std::uint32_t expand_bits(std::uint32_t v) noexcept
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __host__
inline std::uint32_t morton_code(double x, double y, double z, double resolution = 1024.0) noexcept
{
    x = __m_min(__m_max(x * resolution, 0.0), resolution - 1.0);
    y = __m_min(__m_max(y * resolution, 0.0), resolution - 1.0);
    z = __m_min(__m_max(z * resolution, 0.0), resolution - 1.0);

    const std::uint32_t xx = expand_bits(static_cast<std::uint32_t>(x));
    const std::uint32_t yy = expand_bits(static_cast<std::uint32_t>(y));
    const std::uint32_t zz = expand_bits(static_cast<std::uint32_t>(z));

    std::uint32_t mchash = ((xx << 2) + (yy << 1) + zz);

    return mchash;
}

__device__ __host__
void AABB::combines(const double& x, const double& y, const double& z)
{
    lower = make_double3(__m_min(lower.x, x), __m_min(lower.y, y), __m_min(lower.z, z));
    upper = make_double3(__m_max(upper.x, x), __m_max(upper.y, y), __m_max(upper.z, z));
}

__device__ __host__
void AABB::combines(const double& x, const double& y, const double& z, const double& xx, const double& yy, const double& zz)
{
    lower = make_double3(__m_min(lower.x, x), __m_min(lower.y, y), __m_min(lower.z, z));
    upper = make_double3(__m_max(upper.x, xx), __m_max(upper.y, yy), __m_max(upper.z, zz));
}

__host__ __device__
void AABB::combines(const AABB& aabb) {
    lower = make_double3(__m_min(lower.x, aabb.lower.x), __m_min(lower.y, aabb.lower.y), __m_min(lower.z, aabb.lower.z));
    upper = make_double3(__m_max(upper.x, aabb.upper.x), __m_max(upper.y, aabb.upper.y), __m_max(upper.z, aabb.upper.z));
}

__host__ __device__
double3 AABB::center() {
    return make_double3((upper.x + lower.x) * 0.5, (upper.y + lower.y) * 0.5, (upper.z + lower.z) * 0.5);
}

__device__ __host__
AABB::AABB()
{
    lower = make_double3(1e32, 1e32, 1e32);
    upper = make_double3(-1e32, -1e32, -1e32);
}

__device__
inline int common_upper_bits(const unsigned long long int lhs, const unsigned long long int rhs) noexcept
{
    return ::__clzll(lhs ^ rhs);
}


__device__
inline uint2 determine_range(const uint64_t* node_code,
    const unsigned int num_leaves, unsigned int idx)
{
    if (idx == 0)
    {
        return make_uint2(0, num_leaves - 1);
    }

    // determine direction of the range
    const uint64_t self_code = node_code[idx];
    const int L_delta = common_upper_bits(self_code, node_code[idx - 1]);
    const int R_delta = common_upper_bits(self_code, node_code[idx + 1]);
    const int d = (R_delta > L_delta) ? 1 : -1;

    // Compute upper bound for the length of the range

    const int delta_min = __m_min(L_delta, R_delta);
    int l_max = 2;
    int delta = -1;
    int i_tmp = idx + d * l_max;
    if (0 <= i_tmp && i_tmp < num_leaves)
    {
        delta = common_upper_bits(self_code, node_code[i_tmp]);
    }
    while (delta > delta_min)
    {
        l_max <<= 1;
        i_tmp = idx + d * l_max;
        delta = -1;
        if (0 <= i_tmp && i_tmp < num_leaves)
        {
            delta = common_upper_bits(self_code, node_code[i_tmp]);
        }
    }

    // Find the other end by binary search
    int l = 0;
    int t = l_max >> 1;
    while (t > 0)
    {
        i_tmp = idx + (l + t) * d;
        delta = -1;
        if (0 <= i_tmp && i_tmp < num_leaves)
        {
            delta = common_upper_bits(self_code, node_code[i_tmp]);
        }
        if (delta > delta_min)
        {
            l += t;
        }
        t >>= 1;
    }
    unsigned int jdx = idx + l * d;
    if (d < 0)
    {
        unsigned int temp_jdx = jdx;
        jdx = idx;
        idx = temp_jdx;
    }
    return make_uint2(idx, jdx);
}

__device__
inline unsigned int find_split(const uint64_t* node_code, const unsigned int num_leaves,
    const unsigned int first, const unsigned int last) noexcept
{
    const uint64_t first_code = node_code[first];
    const uint64_t last_code = node_code[last];
    if (first_code == last_code)
    {
        return (first + last) >> 1;
    }
    const int delta_node = common_upper_bits(first_code, last_code);

    // binary search...
    int split = first;
    int stride = last - first;
    do
    {
        stride = (stride + 1) >> 1;
        const int middle = split + stride;
        if (middle < last)
        {
            const int delta = common_upper_bits(first_code, node_code[middle]);
            if (delta > delta_node)
            {
                split = middle;
            }
        }
    } while (stride > 1);

    return split;
}

__device__
void _d_PP(const double3& v0, const double3& v1, double& d)
{
    d = __GEIGEN__::__squaredNorm3(__GEIGEN__::__minus(v0, v1));
}

__device__
void _d_PT(const double3& v0, const double3& v1, const double3& v2, const double3& v3, double& d)
{
    double3 b = __GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v2, v1), __GEIGEN__::__minus(v3, v1));
    double3 test = __GEIGEN__::__minus(v0, v1);
    double aTb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__minus(v0, v1), b);//(v0 - v1).dot(b);
    //printf("%f   %f   %f          %f   %f   %f   %f\n", b.x, b.y, b.z, test.x, test.y, test.z, aTb);
    d = aTb * aTb / __GEIGEN__::__squaredNorm3(b);
}

__device__
void _d_PE(const double3& v0, const double3& v1, const double3& v2, double& d)
{
    d = __GEIGEN__::__squaredNorm3(__GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v1, v0), __GEIGEN__::__minus(v2, v0))) / __GEIGEN__::__squaredNorm3(__GEIGEN__::__minus(v2, v1));
}

__device__
void _d_EE(const double3& v0, const double3& v1, const double3& v2, const double3& v3, double& d)
{
    double3 b = __GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v1, v0), __GEIGEN__::__minus(v3, v2));//(v1 - v0).cross(v3 - v2);
    double aTb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__minus(v2, v0), b);//(v2 - v0).dot(b);
    d = aTb * aTb / __GEIGEN__::__squaredNorm3(b);
}


__device__
void _d_EEParallel(const double3& v0, const double3& v1, const double3& v2, const double3& v3, double& d)
{
    double3 b = __GEIGEN__::__v_vec_cross(__GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v1, v0), __GEIGEN__::__minus(v2, v0)), __GEIGEN__::__minus(v1, v0));
    double aTb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__minus(v2, v0), b);//(v2 - v0).dot(b);
    d = aTb * aTb / __GEIGEN__::__squaredNorm3(b);
}

__device__
double _compute_epx(const double3& v0, const double3& v1, const double3& v2, const double3& v3) {
    return 0 * __GEIGEN__::__squaredNorm3(__GEIGEN__::__minus(v0, v1)) * __GEIGEN__::__squaredNorm3(__GEIGEN__::__minus(v2, v3));
}

__device__
double _compute_epx_cp(const double3& v0, const double3& v1, const double3& v2, const double3& v3) {
    return 0 * __GEIGEN__::__squaredNorm3(__GEIGEN__::__minus(v0, v1)) * __GEIGEN__::__squaredNorm3(__GEIGEN__::__minus(v2, v3));
}

__device__
int _dType_PT(const double3& v0, const double3& v1, const double3& v2, const double3& v3)
{
    double3 basis0 = __GEIGEN__::__minus(v2, v1);
    double3 basis1 = __GEIGEN__::__minus(v3, v1);
    double3 basis2 = __GEIGEN__::__minus(v0, v1);

    const double3 nVec = __GEIGEN__::__v_vec_cross(basis0, basis1);

    basis1 = __GEIGEN__::__v_vec_cross(basis0, nVec);
    __GEIGEN__::Matrix3x3d D, D1, D2;

    __GEIGEN__::__set_Mat_val(D, basis0.x, basis1.x, nVec.x, basis0.y, basis1.y, nVec.y, basis0.z, basis1.z, nVec.z);
    __GEIGEN__::__set_Mat_val(D1, basis2.x, basis1.x, nVec.x, basis2.y, basis1.y, nVec.y, basis2.z, basis1.z, nVec.z);
    __GEIGEN__::__set_Mat_val(D2, basis0.x, basis2.x, nVec.x, basis0.y, basis2.y, nVec.y, basis0.z, basis2.z, nVec.z);

    double2 param[3];
    param[0].x = __GEIGEN__::__Determiant_output(D1) / __GEIGEN__::__Determiant_output(D);
    param[0].y = __GEIGEN__::__Determiant_output(D2) / __GEIGEN__::__Determiant_output(D);

    if (param[0].x > 0 && param[0].x < 1 && param[0].y >= 0) {
        return 3; // PE v1v2
    }
    else {
        basis0 = __GEIGEN__::__minus(v3, v2);
        basis1 = __GEIGEN__::__v_vec_cross(basis0, nVec);
        basis2 = __GEIGEN__::__minus(v0, v2);

        __GEIGEN__::__set_Mat_val(D, basis0.x, basis1.x, nVec.x, basis0.y, basis1.y, nVec.y, basis0.z, basis1.z, nVec.z);
        __GEIGEN__::__set_Mat_val(D1, basis2.x, basis1.x, nVec.x, basis2.y, basis1.y, nVec.y, basis2.z, basis1.z, nVec.z);
        __GEIGEN__::__set_Mat_val(D2, basis0.x, basis2.x, nVec.x, basis0.y, basis2.y, nVec.y, basis0.z, basis2.z, nVec.z);

        param[1].x = __GEIGEN__::__Determiant_output(D1) / __GEIGEN__::__Determiant_output(D);
        param[1].y = __GEIGEN__::__Determiant_output(D2) / __GEIGEN__::__Determiant_output(D);

        if (param[1].x > 0.0 && param[1].x < 1.0 && param[1].y >= 0.0) {
            return 4; // PE v2v3
        }
        else {
            basis0 = __GEIGEN__::__minus(v1, v3);
            basis1 = __GEIGEN__::__v_vec_cross(basis0, nVec);
            basis2 = __GEIGEN__::__minus(v0, v3);

            __GEIGEN__::__set_Mat_val(D, basis0.x, basis1.x, nVec.x, basis0.y, basis1.y, nVec.y, basis0.z, basis1.z, nVec.z);
            __GEIGEN__::__set_Mat_val(D1, basis2.x, basis1.x, nVec.x, basis2.y, basis1.y, nVec.y, basis2.z, basis1.z, nVec.z);
            __GEIGEN__::__set_Mat_val(D2, basis0.x, basis2.x, nVec.x, basis0.y, basis2.y, nVec.y, basis0.z, basis2.z, nVec.z);

            param[2].x = __GEIGEN__::__Determiant_output(D1) / __GEIGEN__::__Determiant_output(D);
            param[2].y = __GEIGEN__::__Determiant_output(D2) / __GEIGEN__::__Determiant_output(D);

            if (param[2].x > 0.0 && param[2].x < 1.0 && param[2].y >= 0.0) {
                return 5; // PE v3v1
            }
            else {
                if (param[0].x <= 0.0 && param[2].x >= 1.0) {
                    return 0; // PP v1
                }
                else if (param[1].x <= 0.0 && param[0].x >= 1.0) {
                    return 1; // PP v2
                }
                else if (param[2].x <= 0.0 && param[1].x >= 1.0) {
                    return 2; // PP v3
                }
                else {
                    return 6; // PT
                }
            }
        }
    }
}

__device__
int _dType_EE(const double3& v0, const double3& v1, const double3& v2, const double3& v3)
{
    double3 u = __GEIGEN__::__minus(v1, v0);
    double3 v = __GEIGEN__::__minus(v3, v2);
    double3 w = __GEIGEN__::__minus(v0, v2);

    double a = __GEIGEN__::__squaredNorm3(u);
    double b = __GEIGEN__::__v_vec_dot(u, v);
    double c = __GEIGEN__::__squaredNorm3(v);
    double d = __GEIGEN__::__v_vec_dot(u, w);
    double e = __GEIGEN__::__v_vec_dot(v, w);

    double D = a * c - b * b; // always >= 0
    double tD = D; // tc = tN / tD, default tD = D >= 0
    double sN, tN;
    int defaultCase = 8;
    sN = (b * e - c * d);
    if (sN <= 0.0) { // sc < 0 => the s=0 edge is visible
        tN = e;
        tD = c;
        defaultCase = 2;
    }
    else if (sN >= D) { // sc > 1  => the s=1 edge is visible
        tN = e + b;
        tD = c;
        defaultCase = 5;
    }
    else {
        tN = (a * e - b * d);
        if (tN > 0.0 && tN < tD && (__GEIGEN__::__v_vec_dot(w, __GEIGEN__::__v_vec_cross(u, v)) == 0.0 || __GEIGEN__::__squaredNorm3(__GEIGEN__::__v_vec_cross(u, v)) < 1.0e-20 * a * c)) {
            if (sN < D / 2) {
                tN = e;
                tD = c;
                defaultCase = 2;
            }
            else {
                tN = e + b;
                tD = c;
                defaultCase = 5;
            }
        }
    }

    if (tN <= 0.0) {
        if (-d <= 0.0) {
            return 0;
        }
        else if (-d >= a) {
            return 3;
        }
        else {
            return 6;
        }
    }
    else if (tN >= tD) {
        if ((-d + b) <= 0.0) {
            return 1;
        }
        else if ((-d + b) >= a) {
            return 4;
        }
        else {
            return 7;
        }
    }

    return defaultCase;
}


__device__ inline void _checkPTintersection(const double3* _vertexes,
                                            uint32_t       id0,
                                            uint32_t       id1,
                                            uint32_t       id2,
                                            uint32_t       id3,
                                            double         dHat,
                                            uint32_t*      cpNum,
                                            uint32_t*      caseRank,
                                            int4*          activePairs,
                                            uint32_t       activeCapacity,
                                            uint32_t*      activeOverflow) noexcept
{
    const double3 v0 = _vertexes[id0];
    const double3 v1 = _vertexes[id1];
    const double3 v2 = _vertexes[id2];
    const double3 v3 = _vertexes[id3];

    const int dtype = _dType_PT(v0, v1, v2, v3);
    double    d     = 100.0;
    int4      pair;
    switch(dtype)
    {
        case 0:
            _d_PP(v0, v1, d);
            pair = make_int4(-static_cast<int>(id0) - 1, id1, -1, -1);
            break;
        case 1:
            _d_PP(v0, v2, d);
            pair = make_int4(-static_cast<int>(id0) - 1, id2, -1, -1);
            break;
        case 2:
            _d_PP(v0, v3, d);
            pair = make_int4(-static_cast<int>(id0) - 1, id3, -1, -1);
            break;
        case 3:
            _d_PE(v0, v1, v2, d);
            pair = make_int4(-static_cast<int>(id0) - 1, id1, id2, -1);
            break;
        case 4:
            _d_PE(v0, v2, v3, d);
            pair = make_int4(-static_cast<int>(id0) - 1, id2, id3, -1);
            break;
        case 5:
            _d_PE(v0, v3, v1, d);
            pair = make_int4(-static_cast<int>(id0) - 1, id3, id1, -1);
            break;
        case 6:
            _d_PT(v0, v1, v2, v3, d);
            pair = make_int4(-static_cast<int>(id0) - 1, id1, id2, id3);
            break;
        default:
            return;
    }

    if(d < dHat)
        emit_active(pair, cpNum, caseRank, activePairs, activeCapacity, activeOverflow);
}

__device__ inline void _checkEEintersection(const double3* _vertexes,
                                            const double3* _rest_vertexes,
                                            uint32_t       id0,
                                            uint32_t       id1,
                                            uint32_t       id2,
                                            uint32_t       id3,
                                            uint32_t       obj_idx,
                                            double         dHat,
                                            uint32_t*      cpNum,
                                            uint32_t*      caseRank,
                                            int4*          activePairs,
                                            uint32_t       activeCapacity,
                                            uint32_t*      activeOverflow) noexcept
{
    const double3 v0 = _vertexes[id0];
    const double3 v1 = _vertexes[id1];
    const double3 v2 = _vertexes[id2];
    const double3 v3 = _vertexes[id3];

    const int dtype = _dType_EE(v0, v1, v2, v3);
    double    d     = 100.0;
    int4      pair;
    switch(dtype)
    {
        case 0:
            _d_PP(v0, v2, d);
            pair = make_int4(-static_cast<int>(id0) - 1, id2, -1, -1);
            break;
        case 1:
            _d_PP(v0, v3, d);
            pair = make_int4(-static_cast<int>(id0) - 1, id3, -1, -1);
            break;
        case 2:
            _d_PE(v0, v2, v3, d);
            pair = make_int4(-static_cast<int>(id0) - 1, id2, id3, -1);
            break;
        case 3:
            _d_PP(v1, v2, d);
            pair = make_int4(-static_cast<int>(id1) - 1, id2, -1, -1);
            break;
        case 4:
            _d_PP(v1, v3, d);
            pair = make_int4(-static_cast<int>(id1) - 1, id3, -1, -1);
            break;
        case 5:
            _d_PE(v1, v2, v3, d);
            pair = make_int4(-static_cast<int>(id1) - 1, id2, id3, -1);
            break;
        case 6:
            _d_PE(v2, v0, v1, d);
            pair = make_int4(-static_cast<int>(id2) - 1, id0, id1, -1);
            break;
        case 7:
            _d_PE(v3, v0, v1, d);
            pair = make_int4(-static_cast<int>(id3) - 1, id0, id1, -1);
            break;
        case 8:
            _d_EE(v0, v1, v2, v3, d);
            pair = make_int4(id0, id1, id2, id3);
            break;
        default:
            return;
    }

    if(d >= dHat)
        return;

    // Preserve the original parallel-edge marker for PP/PE cases. It is not
    // part of packed Python output, but downstream collision energies use it.
    if(dtype != 8)
    {
        const bool reverse = dtype >= 6;
        const double3 a0   = reverse ? v2 : v0;
        const double3 a1   = reverse ? v3 : v1;
        const double3 b0   = reverse ? v0 : v2;
        const double3 b1   = reverse ? v1 : v3;
        const uint32_t a0i = reverse ? id2 : id0;
        const uint32_t a1i = reverse ? id3 : id1;
        const uint32_t b0i = reverse ? id0 : id2;
        const uint32_t b1i = reverse ? id1 : id3;
        const double crossSquared = __GEIGEN__::__squaredNorm3(
            __GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(a0, a1),
                                      __GEIGEN__::__minus(b0, b1)));
        const double eps = _compute_epx_cp(_rest_vertexes[a0i],
                                           _rest_vertexes[a1i],
                                           _rest_vertexes[b0i],
                                           _rest_vertexes[b1i]);
        pair.w = crossSquared < eps ? -static_cast<int>(obj_idx) - 2 : -1;
    }

    emit_active(pair, cpNum, caseRank, activePairs, activeCapacity, activeOverflow);
}

__global__
void _reduct_max_box(AABB* _leafBoxes, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ AABB tep[];

    if (idx >= number) return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    AABB temp = _leafBoxes[idx];

    __threadfence();

    double xmin = temp.lower.x, ymin = temp.lower.y, zmin = temp.lower.z;
    double xmax = temp.upper.x, ymax = temp.upper.y, zmax = temp.upper.z;
    //printf("%f   %f    %f   %f   %f    %f\n", xmin, ymin, zmin, xmax, ymax, zmax);
    //printf("%f   %f    %f\n", xmax, ymax, zmax);
    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        warpNum = ((number - idof + 31) >> 5);
        if (warpId == warpNum - 1) {
            tidNum = number - idof - (warpNum - 1) * 32;
        }
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < tidNum; i = (i << 1)) {
        temp.combines(__shfl_down_sync(0xFFFFFFFF, xmin, i), __shfl_down_sync(0xFFFFFFFF, ymin, i), __shfl_down_sync(0xFFFFFFFF, zmin, i),
          __shfl_down_sync(0xFFFFFFFF, xmax, i), __shfl_down_sync(0xFFFFFFFF, ymax, i), __shfl_down_sync(0xFFFFFFFF, zmax, i));
        if (warpTid + i < tidNum) {
            xmin = temp.lower.x, ymin = temp.lower.y, zmin = temp.lower.z;
            xmax = temp.upper.x, ymax = temp.upper.y, zmax = temp.upper.z;
        }
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        xmin = temp.lower.x, ymin = temp.lower.y, zmin = temp.lower.z;
        xmax = temp.upper.x, ymax = temp.upper.y, zmax = temp.upper.z;
        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            temp.combines(__shfl_down_sync(0xFFFFFFFF, xmin, i), __shfl_down_sync(0xFFFFFFFF, ymin, i), __shfl_down_sync(0xFFFFFFFF, zmin, i),
              __shfl_down_sync(0xFFFFFFFF, xmax, i), __shfl_down_sync(0xFFFFFFFF, ymax, i), __shfl_down_sync(0xFFFFFFFF, zmax, i));
            if (threadIdx.x + i < warpNum) {
                xmin = temp.lower.x, ymin = temp.lower.y, zmin = temp.lower.z;
                xmax = temp.upper.x, ymax = temp.upper.y, zmax = temp.upper.z;
            }
        }
    }
    if (threadIdx.x == 0) {
        _leafBoxes[blockIdx.x] = temp;
    }
}



namespace
{
constexpr uint32_t BVH_THREADS = 256u;

__global__ void reduce_boxes_safe(const AABB* input, AABB* output, uint32_t number)
{
    __shared__ double lowerX[BVH_THREADS];
    __shared__ double lowerY[BVH_THREADS];
    __shared__ double lowerZ[BVH_THREADS];
    __shared__ double upperX[BVH_THREADS];
    __shared__ double upperY[BVH_THREADS];
    __shared__ double upperZ[BVH_THREADS];

    const uint32_t tid = threadIdx.x;
    const uint32_t idx = blockIdx.x * blockDim.x + tid;
    if(idx < number)
    {
        const AABB value = input[idx];
        lowerX[tid] = value.lower.x;
        lowerY[tid] = value.lower.y;
        lowerZ[tid] = value.lower.z;
        upperX[tid] = value.upper.x;
        upperY[tid] = value.upper.y;
        upperZ[tid] = value.upper.z;
    }
    else
    {
        lowerX[tid] = lowerY[tid] = lowerZ[tid] = 1e32;
        upperX[tid] = upperY[tid] = upperZ[tid] = -1e32;
    }
    __syncthreads();

    for(uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(tid < stride)
        {
            lowerX[tid] = __m_min(lowerX[tid], lowerX[tid + stride]);
            lowerY[tid] = __m_min(lowerY[tid], lowerY[tid + stride]);
            lowerZ[tid] = __m_min(lowerZ[tid], lowerZ[tid + stride]);
            upperX[tid] = __m_max(upperX[tid], upperX[tid + stride]);
            upperY[tid] = __m_max(upperY[tid], upperY[tid + stride]);
            upperZ[tid] = __m_max(upperZ[tid], upperZ[tid + stride]);
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        AABB result;
        result.lower = make_double3(lowerX[0], lowerY[0], lowerZ[0]);
        result.upper = make_double3(upperX[0], upperY[0], upperZ[0]);
        output[blockIdx.x] = result;
    }
}

__device__ inline double normalized_axis(double offset, double extent)
{
    return extent > 0.0 && isfinite(extent) ? offset / extent : 0.5;
}

__global__ void calc_morton_keys(uint64_t* keys, const AABB* bvs, uint32_t number)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    const AABB sceneBox = bvs[0];
    const double3 extent = make_double3(sceneBox.upper.x - sceneBox.lower.x,
                                        sceneBox.upper.y - sceneBox.lower.y,
                                        sceneBox.upper.z - sceneBox.lower.z);
    AABB leafBox = bvs[idx + number - 1];
    const double3 center = leafBox.center();
    const uint64_t morton = morton_code(
        normalized_axis(center.x - sceneBox.lower.x, extent.x),
        normalized_axis(center.y - sceneBox.lower.y, extent.y),
        normalized_axis(center.z - sceneBox.lower.z, extent.z));
    keys[idx] = (morton << 32) | static_cast<uint64_t>(idx);
}

__global__ void calc_leaf_nodes(Node* nodes, const uint64_t* keys, uint32_t number)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    if(idx < number - 1)
    {
        nodes[idx].left_idx    = INVALID_INDEX;
        nodes[idx].right_idx   = INVALID_INDEX;
        nodes[idx].parent_idx  = INVALID_INDEX;
        nodes[idx].element_idx = INVALID_INDEX;
    }

    const uint32_t leaf       = idx + number - 1;
    nodes[leaf].left_idx      = INVALID_INDEX;
    nodes[leaf].right_idx     = INVALID_INDEX;
    nodes[leaf].parent_idx    = INVALID_INDEX;
    nodes[leaf].element_idx   = static_cast<uint32_t>(keys[idx]);
}

__global__ void calc_internal_nodes(Node* nodes, const uint64_t* keys, uint32_t number)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number - 1)
        return;

    const uint2 range = determine_range(keys, number, idx);
    const uint32_t split = find_split(keys, number, range.x, range.y);

    uint32_t left  = split;
    uint32_t right = split + 1;
    if(__m_min(range.x, range.y) == split)
        left += number - 1;
    if(__m_max(range.x, range.y) == split + 1)
        right += number - 1;

    nodes[idx].left_idx  = left;
    nodes[idx].right_idx = right;
    nodes[left].parent_idx  = idx;
    nodes[right].parent_idx = idx;
}

__global__ void calc_internal_boxes(
    const Node* nodes, AABB* bvs, uint32_t* flags, uint32_t number)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    idx += number - 1;
    uint32_t parent = nodes[idx].parent_idx;
    while(parent != INVALID_INDEX)
    {
        if(atomicCAS(flags + parent, INVALID_INDEX, 0u) == INVALID_INDEX)
            return;

        bvs[parent] = merge(bvs[nodes[parent].left_idx], bvs[nodes[parent].right_idx]);
        __threadfence();
        parent = nodes[parent].parent_idx;
    }
}

__global__ void calc_escape_links(const Node* nodes, int32_t* escape, uint32_t nodeCount)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= nodeCount)
        return;

    uint32_t node = idx;
    int32_t next = -1;
    while(node != 0)
    {
        const uint32_t parent = nodes[node].parent_idx;
        if(parent == INVALID_INDEX)
            break;
        if(node == nodes[parent].left_idx)
        {
            next = static_cast<int32_t>(nodes[parent].right_idx);
            break;
        }
        node = parent;
    }
    escape[idx] = next;
}

__global__ void reorder_leaf_boxes(
    const uint64_t* keys, AABB* leafBoxes, const AABB* unsorted, uint32_t number)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    leafBoxes[idx] = unsorted[static_cast<uint32_t>(keys[idx])];
}

__device__ inline bool same_nonzero_mesh(const uint32_t* meshIndices,
                                         uint32_t a,
                                         uint32_t b,
                                         uint32_t c,
                                         uint32_t d)
{
    if(!meshIndices)
        return false;
    const uint32_t mesh = meshIndices[a];
    return mesh != 0u && mesh == meshIndices[b] && mesh == meshIndices[c]
           && mesh == meshIndices[d];
}

__device__ inline bool all_fixed(
    const int* btype, uint32_t a, uint32_t b, uint32_t c, uint32_t d)
{
    return btype && btype[a] >= 2 && btype[b] >= 2 && btype[c] >= 2 && btype[d] >= 2;
}

__device__ inline void append_candidate(int2          pair,
                                        int2*         candidates,
                                        uint32_t*     candidateNum,
                                        uint32_t      capacity,
                                        uint32_t*     candidateOverflow)
{
    const uint32_t output =
        reserve_bounded(candidateNum, capacity, candidateOverflow);
    if(output != INVALID_INDEX)
        candidates[output] = pair;
}

__global__ void query_face_candidates(const int*      btype,
                                      const uint32_t* meshIndices,
                                      const double3*  vertices,
                                      const double3*  moveDir,
                                      double          alpha,
                                      const uint3*    faces,
                                      const uint32_t* surfaceVertices,
                                      const AABB*     bvs,
                                      const Node*     nodes,
                                      const int32_t*  escape,
                                      int2*           candidates,
                                      uint32_t*       candidateNum,
                                      uint32_t        candidateCapacity,
                                      uint32_t*       candidateOverflow,
                                      double          dHat,
                                      uint32_t        surfaceCount,
                                      bool            swept)
{
    uint32_t surfaceIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(surfaceIdx >= surfaceCount)
        return;

    const uint32_t point = surfaceVertices[surfaceIdx];
    const double3 x = vertices[point];
    AABB query;
    query.lower = x;
    query.upper = x;
    if(swept)
    {
        const double3 dx = moveDir[point];
        query.combines(x.x - dx.x * alpha, x.y - dx.y * alpha, x.z - dx.z * alpha);
    }

    const double gap = sqrt(dHat);
    int32_t node = 0;
    while(node != -1)
    {
        const Node current = nodes[node];
        if(!overlap(query, bvs[node], gap))
        {
            node = escape[node];
            continue;
        }

        if(current.element_idx == INVALID_INDEX)
        {
            node = static_cast<int32_t>(current.left_idx);
            continue;
        }

        const uint32_t faceId = current.element_idx;
        const uint3 face = faces[faceId];
        if(point != face.x && point != face.y && point != face.z
           && !all_fixed(btype, point, face.x, face.y, face.z)
           && !same_nonzero_mesh(meshIndices, point, face.x, face.y, face.z))
        {
            append_candidate(make_int2(-static_cast<int>(point) - 1,
                                       static_cast<int>(faceId)),
                             candidates,
                             candidateNum,
                             candidateCapacity,
                             candidateOverflow);
        }
        node = escape[node];
    }
}

__global__ void query_edge_candidates(const int*      btype,
                                      const uint32_t* meshIndices,
                                      const uint2*    edges,
                                      const AABB*     bvs,
                                      const Node*     nodes,
                                      const int32_t*  escape,
                                      int2*           candidates,
                                      uint32_t*       candidateNum,
                                      uint32_t        candidateCapacity,
                                      uint32_t*       candidateOverflow,
                                      double          dHat,
                                      uint32_t        edgeCount)
{
    uint32_t leaf = blockIdx.x * blockDim.x + threadIdx.x;
    if(leaf >= edgeCount)
        return;

    leaf += edgeCount - 1;
    const AABB query = bvs[leaf];
    const uint32_t selfId = nodes[leaf].element_idx;
    const uint2 self = edges[selfId];
    const double gap = sqrt(dHat);

    int32_t node = 0;
    while(node != -1)
    {
        const Node current = nodes[node];
        if(!overlap(query, bvs[node], gap))
        {
            node = escape[node];
            continue;
        }

        if(current.element_idx == INVALID_INDEX)
        {
            node = static_cast<int32_t>(current.left_idx);
            continue;
        }

        const uint32_t otherId = current.element_idx;
        const uint2 other = edges[otherId];
        if(otherId > selfId && self.x != other.x && self.x != other.y
           && self.y != other.x && self.y != other.y
           && !all_fixed(btype, self.x, self.y, other.x, other.y)
           && !same_nonzero_mesh(meshIndices, self.x, self.y, other.x, other.y))
        {
            append_candidate(make_int2(static_cast<int>(selfId),
                                       static_cast<int>(otherId)),
                             candidates,
                             candidateNum,
                             candidateCapacity,
                             candidateOverflow);
        }
        node = escape[node];
    }
}

__global__ void process_cached_candidates(const double3* vertices,
                                          const double3* restVertices,
                                          const uint3*   faces,
                                          const uint2*   edges,
                                          const int2*    candidates,
                                          uint32_t       candidateCount,
                                          double         dHat,
                                          uint32_t*      cpNum,
                                          uint32_t*      caseRank,
                                          int4*          activePairs,
                                          uint32_t       activeCapacity,
                                          uint32_t*      activeOverflow)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= candidateCount)
        return;

    const int2 candidate = candidates[idx];
    if(candidate.x < 0)
    {
        const uint32_t point = static_cast<uint32_t>(-candidate.x - 1);
        const uint3 face = faces[candidate.y];
        _checkPTintersection(vertices,
                             point,
                             face.x,
                             face.y,
                             face.z,
                             dHat,
                             cpNum,
                             caseRank,
                             activePairs,
                             activeCapacity,
                             activeOverflow);
    }
    else
    {
        const uint2 first  = edges[candidate.x];
        const uint2 second = edges[candidate.y];
        _checkEEintersection(vertices,
                             restVertices,
                             first.x,
                             first.y,
                             second.x,
                             second.y,
                             static_cast<uint32_t>(candidate.y),
                             dHat,
                             cpNum,
                             caseRank,
                             activePairs,
                             activeCapacity,
                             activeOverflow);
    }
}

__global__ void expand_cached_candidates(const int2*  candidates,
                                         const uint3* faces,
                                         const uint2* edges,
                                         int4*        expandedPairs,
                                         uint32_t     candidateCount)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= candidateCount)
        return;

    const int2 candidate = candidates[idx];
    if(candidate.x < 0)
    {
        const uint3 face = faces[candidate.y];
        expandedPairs[idx] = make_int4(candidate.x,
                                       static_cast<int>(face.x),
                                       static_cast<int>(face.y),
                                       static_cast<int>(face.z));
    }
    else
    {
        const uint2 first  = edges[candidate.x];
        const uint2 second = edges[candidate.y];
        expandedPairs[idx] = make_int4(static_cast<int>(first.x),
                                       static_cast<int>(first.y),
                                       static_cast<int>(second.x),
                                       static_cast<int>(second.y));
    }
}

__global__ void scatter_packed_cases(const int4*     activePairs,
                                     const uint32_t* caseRank,
                                     const uint32_t* cpNum,
                                     uint32_t*       packedOutput,
                                     uint32_t        activeCount)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= activeCount)
        return;

    const int4 pair = activePairs[idx];
    const uint32_t rank = caseRank[idx];
    const uint32_t collisionCase = active_case(pair);
    const uint32_t ppBase = 0u;
    const uint32_t peBase = 2u * cpNum[1];
    const uint32_t ptBase = peBase + 3u * cpNum[2];
    const uint32_t eeBase = ptBase + 4u * cpNum[3];

    if(collisionCase == 1u)
    {
        const uint32_t output = ppBase + 2u * rank;
        packedOutput[output]     = static_cast<uint32_t>(-pair.x - 1);
        packedOutput[output + 1] = static_cast<uint32_t>(pair.y);
    }
    else if(collisionCase == 2u)
    {
        const uint32_t output = peBase + 3u * rank;
        packedOutput[output]     = static_cast<uint32_t>(-pair.x - 1);
        packedOutput[output + 1] = static_cast<uint32_t>(pair.y);
        packedOutput[output + 2] = static_cast<uint32_t>(pair.z);
    }
    else if(collisionCase == 3u)
    {
        const uint32_t output = ptBase + 4u * rank;
        packedOutput[output]     = static_cast<uint32_t>(-pair.x - 1);
        packedOutput[output + 1] = static_cast<uint32_t>(pair.y);
        packedOutput[output + 2] = static_cast<uint32_t>(pair.z);
        packedOutput[output + 3] = static_cast<uint32_t>(pair.w);
    }
    else
    {
        const uint32_t output = eeBase + 4u * rank;
        packedOutput[output]     = static_cast<uint32_t>(pair.x);
        packedOutput[output + 1] = static_cast<uint32_t>(pair.y);
        packedOutput[output + 2] = static_cast<uint32_t>(pair.z);
        packedOutput[output + 3] = static_cast<uint32_t>(pair.w);
    }
}

AABB calculate_scene(AABB* bvs, AABB* temporary, uint32_t number)
{
    AABB hostScene;
    if(number == 0)
        return hostScene;

    CUDA_SAFE_CALL(cudaMemcpy(temporary,
                              bvs + number - 1,
                              number * sizeof(AABB),
                              cudaMemcpyDeviceToDevice));

    const AABB* input = temporary;
    AABB* output = bvs;
    uint32_t count = number;
    while(count > 1)
    {
        const uint32_t blocks = (count + BVH_THREADS - 1) / BVH_THREADS;
        reduce_boxes_safe<<<blocks, BVH_THREADS>>>(input, output, count);
        count = blocks;
        if(count > 1)
        {
            const AABB* nextInput = output;
            output = output == bvs ? temporary : bvs;
            input = nextInput;
        }
    }

    if(number == 1)
        output = temporary;
    CUDA_SAFE_CALL(cudaMemcpy(bvs, output, sizeof(AABB), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(&hostScene, output, sizeof(AABB), cudaMemcpyDeviceToHost));

    // Sorting needs the original, unsorted leaf order after reduction used the
    // temporary buffer as ping-pong storage.
    CUDA_SAFE_CALL(cudaMemcpy(temporary,
                              bvs + number - 1,
                              number * sizeof(AABB),
                              cudaMemcpyDeviceToDevice));
    return hostScene;
}

void build_topology(lbvh* bvh, uint32_t number)
{
    if(number == 0)
        return;

    const uint32_t blocks = (number + BVH_THREADS - 1) / BVH_THREADS;
    calc_morton_keys<<<blocks, BVH_THREADS>>>(bvh->_MChash, bvh->_bvs, number);
    bvh->radixSortMorton(number);
    reorder_leaf_boxes<<<blocks, BVH_THREADS>>>(
        bvh->_MChash, bvh->_bvs + number - 1, bvh->_tempLeafBox, number);
    calc_leaf_nodes<<<blocks, BVH_THREADS>>>(bvh->_nodes, bvh->_MChash, number);

    if(number > 1)
    {
        const uint32_t internalBlocks =
            (number - 1 + BVH_THREADS - 1) / BVH_THREADS;
        calc_internal_nodes<<<internalBlocks, BVH_THREADS>>>(
            bvh->_nodes, bvh->_MChash, number);
        CUDA_SAFE_CALL(cudaMemset(
            bvh->_flags, 0xFF, (number - 1) * sizeof(uint32_t)));
        calc_internal_boxes<<<blocks, BVH_THREADS>>>(
            bvh->_nodes, bvh->_bvs, bvh->_flags, number);
    }

    const uint32_t nodeCount = 2u * number - 1u;
    const uint32_t nodeBlocks = (nodeCount + BVH_THREADS - 1) / BVH_THREADS;
    calc_escape_links<<<nodeBlocks, BVH_THREADS>>>(
        bvh->_nodes, bvh->_escape, nodeCount);
}

uint32_t shared_pointer_count(const lbvh* obj)
{
    return static_cast<uint32_t>(obj->_collisionPair != nullptr)
           + static_cast<uint32_t>(obj->_caseRank != nullptr)
           + static_cast<uint32_t>(obj->_cpNum != nullptr)
           + static_cast<uint32_t>(obj->_packedOutput != nullptr)
           + static_cast<uint32_t>(obj->_candidatePairs != nullptr)
           + static_cast<uint32_t>(obj->_candidateNum != nullptr)
           + static_cast<uint32_t>(obj->_overflowCount != nullptr);
}

lbvh* initialized_shared_owner(lbvh_f* faceObj, lbvh_e* edgeObj)
{
    lbvh* face = faceObj && faceObj->_initialized
                     ? static_cast<lbvh*>(faceObj)
                     : nullptr;
    lbvh* edge = edgeObj && edgeObj->_initialized
                     ? static_cast<lbvh*>(edgeObj)
                     : nullptr;
    if(!face)
        return edge;
    if(!edge)
        return face;
    return shared_pointer_count(edge) > shared_pointer_count(face) ? edge : face;
}

void launch_face_query(lbvh_f* obj,
                       const double3* moveDir,
                       double alpha,
                       double dHat,
                       bool swept)
{
    if(!obj || obj->face_number == 0 || obj->vert_number == 0
       || !obj->_candidatePairs || !obj->_candidateNum)
        return;
    const uint32_t blocks = (obj->vert_number + BVH_THREADS - 1) / BVH_THREADS;
    query_face_candidates<<<blocks, BVH_THREADS>>>(obj->_btype,
                                                   obj->_meshIndices,
                                                   obj->_vertexes,
                                                   moveDir,
                                                   alpha,
                                                   obj->_faces,
                                                   obj->_surfVerts,
                                                   obj->_bvs,
                                                   obj->_nodes,
                                                   obj->_escape,
                                                   obj->_candidatePairs,
                                                   obj->_candidateNum,
                                                   obj->_maxCandidatePairs,
                                                   obj->_overflowCount,
                                                   dHat,
                                                   obj->vert_number,
                                                   swept);
}

void launch_edge_query(lbvh_e* obj, double dHat)
{
    if(!obj || obj->edge_number == 0 || !obj->_candidatePairs || !obj->_candidateNum)
        return;
    const uint32_t blocks = (obj->edge_number + BVH_THREADS - 1) / BVH_THREADS;
    query_edge_candidates<<<blocks, BVH_THREADS>>>(obj->_btype,
                                                   obj->_meshIndices,
                                                   obj->_edges,
                                                   obj->_bvs,
                                                   obj->_nodes,
                                                   obj->_escape,
                                                   obj->_candidatePairs,
                                                   obj->_candidateNum,
                                                   obj->_maxCandidatePairs,
                                                   obj->_overflowCount,
                                                   dHat,
                                                   obj->edge_number);
}

double scene_diagonal_squared(const AABB& scene)
{
    const double x = scene.upper.x - scene.lower.x;
    const double y = scene.upper.y - scene.lower.y;
    const double z = scene.upper.z - scene.lower.z;
    if(!isfinite(x) || !isfinite(y) || !isfinite(z))
        return 0.0;
    return x * x + y * y + z * z;
}
}

void lbvh::radixSortMorton(uint32_t number)
{
    if(number < 2)
        return;

    size_t required = 0;
    CUDA_SAFE_CALL(cub::DeviceRadixSort::SortKeys(
        nullptr, required, _MChash, _MChash_sorted, number));
    if(required > _sort_temp_bytes)
    {
        if(_sort_temp_storage)
            CUDA_SAFE_CALL(cudaFree(_sort_temp_storage));
        CUDA_SAFE_CALL(cudaMalloc(&_sort_temp_storage, required));
        _sort_temp_bytes = required;
    }
    CUDA_SAFE_CALL(cub::DeviceRadixSort::SortKeys(
        _sort_temp_storage, required, _MChash, _MChash_sorted, number));
    std::swap(_MChash, _MChash_sorted);
}

void lbvh::MALLOC_DEVICE_MEM(uint32_t primitiveNumber, uint32_t pointNumber)
{
    if(primitiveNumber > 0)
    {
        CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&_MChash),
                                  primitiveNumber * sizeof(uint64_t)));
        CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&_MChash_sorted),
                                  primitiveNumber * sizeof(uint64_t)));
        CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&_nodes),
                                  (2u * primitiveNumber - 1u) * sizeof(Node)));
        CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&_bvs),
                                  (2u * primitiveNumber - 1u) * sizeof(AABB)));
        CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&_escape),
                                  (2u * primitiveNumber - 1u) * sizeof(int32_t)));
        CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&_tempLeafBox),
                                  primitiveNumber * sizeof(AABB)));
        if(primitiveNumber > 1)
            CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&_flags),
                                      (primitiveNumber - 1u) * sizeof(uint32_t)));
    }
    if(pointNumber > 0)
        CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&_sweptPointBox),
                                  pointNumber * sizeof(AABB)));
}

void lbvh::FREE_DEVICE_MEM()
{
    // Python finalizers may run after PyCUDA has destroyed the context. CUDA
    // teardown is therefore best-effort and must never call the fatal macro.
    if(_MChash)
        (void)cudaFree(_MChash);
    if(_MChash_sorted)
        (void)cudaFree(_MChash_sorted);
    if(_nodes)
        (void)cudaFree(_nodes);
    if(_bvs)
        (void)cudaFree(_bvs);
    if(_flags)
        (void)cudaFree(_flags);
    if(_escape)
        (void)cudaFree(_escape);
    if(_tempLeafBox)
        (void)cudaFree(_tempLeafBox);
    if(_sweptPointBox)
        (void)cudaFree(_sweptPointBox);
    if(_sort_temp_storage)
        (void)cudaFree(_sort_temp_storage);

    _MChash = nullptr;
    _MChash_sorted = nullptr;
    _nodes = nullptr;
    _bvs = nullptr;
    _flags = nullptr;
    _escape = nullptr;
    _tempLeafBox = nullptr;
    _sweptPointBox = nullptr;
    _sort_temp_storage = nullptr;
    _sort_temp_bytes = 0;
}

lbvh::~lbvh()
{
    FREE_DEVICE_MEM();
}

void lbvh_f::init(int*       btype,
                  double3*   vertices,
                  uint3*     faces,
                  uint32_t*  surfaceVertices,
                  int4*      activePairs,
                  uint32_t*  caseRank,
                  uint32_t*  cpNum,
                  uint32_t*  meshIndices,
                  int2*      candidates,
                  uint32_t*  candidateNum,
                  uint32_t*  packedOutput,
                  uint32_t*  overflowCount,
                  uint32_t   maxCandidatePairs,
                  uint32_t   maxActivePairs,
                  uint32_t   faceNum,
                  uint32_t   vertNum)
{
    FREE_DEVICE_MEM();
    _btype = btype;
    _vertexes = vertices;
    _faces = faces;
    _surfVerts = surfaceVertices;
    _collisionPair = activePairs;
    _caseRank = caseRank;
    _cpNum = cpNum;
    _meshIndices = meshIndices;
    _candidatePairs = candidates;
    _candidateNum = candidateNum;
    _packedOutput = packedOutput;
    _overflowCount = overflowCount;
    _maxCandidatePairs = maxCandidatePairs;
    _maxActivePairs = maxActivePairs;
    face_number = faceNum;
    vert_number = vertNum;
    MALLOC_DEVICE_MEM(face_number, face_number > 0 ? vert_number : 0u);
    _initialized = true;
}

void lbvh_e::init(int*       btype,
                  double3*   vertices,
                  double3*   restVertices,
                  uint2*     edges,
                  int4*      activePairs,
                  uint32_t*  caseRank,
                  uint32_t*  cpNum,
                  uint32_t*  meshIndices,
                  int2*      candidates,
                  uint32_t*  candidateNum,
                  uint32_t*  packedOutput,
                  uint32_t*  overflowCount,
                  uint32_t   maxCandidatePairs,
                  uint32_t   maxActivePairs,
                  uint32_t   edgeNum,
                  uint32_t   vertNum)
{
    FREE_DEVICE_MEM();
    _btype = btype;
    _vertexes = vertices;
    _rest_vertexes = restVertices;
    _edges = edges;
    _collisionPair = activePairs;
    _caseRank = caseRank;
    _cpNum = cpNum;
    _meshIndices = meshIndices;
    _candidatePairs = candidates;
    _candidateNum = candidateNum;
    _packedOutput = packedOutput;
    _overflowCount = overflowCount;
    _maxCandidatePairs = maxCandidatePairs;
    _maxActivePairs = maxActivePairs;
    edge_number = edgeNum;
    vert_number = vertNum;
    MALLOC_DEVICE_MEM(edge_number);
    _initialized = true;
}

AABB* lbvh_f::getSceneSize()
{
    if(face_number == 0)
    {
        scene = AABB();
        return nullptr;
    }
    const uint32_t blocks = (face_number + BVH_THREADS - 1) / BVH_THREADS;
    _calcLeafBvs<<<blocks, BVH_THREADS>>>(
        _vertexes, _faces, _bvs + face_number - 1, face_number, 0);
    scene = calculate_scene(_bvs, _tempLeafBox, face_number);
    return _bvs;
}

double lbvh_f::Construct(double3* vertices)
{
    _vertexes = vertices;
    if(face_number == 0)
    {
        scene = AABB();
        return 0.0;
    }
    const uint32_t blocks = (face_number + BVH_THREADS - 1) / BVH_THREADS;
    _calcLeafBvs<<<blocks, BVH_THREADS>>>(
        _vertexes, _faces, _bvs + face_number - 1, face_number, 0);
    scene = calculate_scene(_bvs, _tempLeafBox, face_number);
    build_topology(this, face_number);
    return 0.0;
}

double lbvh_f::ConstructFullCCD(
    double3* vertices, const double3* moveDir, double alpha)
{
    _vertexes = vertices;
    if(face_number == 0)
    {
        scene = AABB();
        return 0.0;
    }
    const uint32_t blocks = (face_number + BVH_THREADS - 1) / BVH_THREADS;
    _calcLeafBvs_ccd<<<blocks, BVH_THREADS>>>(_vertexes,
                                              moveDir,
                                              alpha,
                                              _faces,
                                              _bvs + face_number - 1,
                                              face_number,
                                              0);
    scene = calculate_scene(_bvs, _tempLeafBox, face_number);
    build_topology(this, face_number);
    if(vert_number > 0)
    {
        const uint32_t pointBlocks =
            (vert_number + BVH_THREADS - 1) / BVH_THREADS;
        cache_swept_point_boxes<<<pointBlocks, BVH_THREADS>>>(_vertexes,
                                                              moveDir,
                                                              alpha,
                                                              _surfVerts,
                                                              _sweptPointBox,
                                                              vert_number);
    }
    return 0.0;
}

double lbvh_e::Construct(double3* vertices)
{
    _vertexes = vertices;
    if(edge_number == 0)
    {
        scene = AABB();
        return 0.0;
    }
    const uint32_t blocks = (edge_number + BVH_THREADS - 1) / BVH_THREADS;
    _calcLeafBvs<<<blocks, BVH_THREADS>>>(
        _vertexes, _edges, _bvs + edge_number - 1, edge_number, 1);
    scene = calculate_scene(_bvs, _tempLeafBox, edge_number);
    build_topology(this, edge_number);
    return 0.0;
}

double lbvh_e::ConstructFullCCD(
    double3* vertices, const double3* moveDir, double alpha)
{
    _vertexes = vertices;
    if(edge_number == 0)
    {
        scene = AABB();
        return 0.0;
    }
    const uint32_t blocks = (edge_number + BVH_THREADS - 1) / BVH_THREADS;
    _calcLeafBvs_ccd<<<blocks, BVH_THREADS>>>(_vertexes,
                                              moveDir,
                                              alpha,
                                              _edges,
                                              _bvs + edge_number - 1,
                                              edge_number,
                                              1);
    scene = calculate_scene(_bvs, _tempLeafBox, edge_number);
    build_topology(this, edge_number);
    return 0.0;
}

void lbvh_f::SelfCollitionDetect(double dHat)
{
    launch_face_query(this, nullptr, 0.0, dHat, false);
}

void lbvh_e::SelfCollitionDetect(double dHat)
{
    launch_edge_query(this, dHat);
}

void lbvh_f::SelfCollitionFullDetect(
    double dHat, const double3* moveDir, double alpha)
{
    launch_face_query(this, moveDir, alpha, dHat, true);
}

void lbvh_e::SelfCollitionFullDetect(
    double dHat, const double3*, double)
{
    launch_edge_query(this, dHat);
}

extern "C"
{
uint32_t mlbvh_api_version()
{
    return 3u;
}

lbvh_f* create_lbvh_f()
{
    return new lbvh_f();
}

lbvh_e* create_lbvh_e()
{
    return new lbvh_e();
}

void destroy_lbvh_f(lbvh_f* obj)
{
    delete obj;
}

void destroy_lbvh_e(lbvh_e* obj)
{
    delete obj;
}

void lbvh_f_init(lbvh_f*   obj,
                 int*      btype,
                 double3*  vertices,
                 uint3*    faces,
                 uint32_t* surfaceVertices,
                 int4*     activePairs,
                 uint32_t* caseRank,
                 uint32_t* cpNum,
                 uint32_t* meshIndices,
                 int2*     candidates,
                 uint32_t* candidateNum,
                 uint32_t* packedOutput,
                 uint32_t* overflowCount,
                 uint32_t  maxCandidatePairs,
                 uint32_t  maxActivePairs,
                 uint32_t  faceNum,
                 uint32_t  vertNum)
{
    obj->init(btype,
              vertices,
              faces,
              surfaceVertices,
              activePairs,
              caseRank,
              cpNum,
              meshIndices,
              candidates,
              candidateNum,
              packedOutput,
              overflowCount,
              maxCandidatePairs,
              maxActivePairs,
              faceNum,
              vertNum);
}

void lbvh_e_init(lbvh_e*   obj,
                 int*      btype,
                 double3*  vertices,
                 double3*  restVertices,
                 uint2*    edges,
                 int4*     activePairs,
                 uint32_t* caseRank,
                 uint32_t* cpNum,
                 uint32_t* meshIndices,
                 int2*     candidates,
                 uint32_t* candidateNum,
                 uint32_t* packedOutput,
                 uint32_t* overflowCount,
                 uint32_t  maxCandidatePairs,
                 uint32_t  maxActivePairs,
                 uint32_t  edgeNum,
                 uint32_t  vertNum)
{
    obj->init(btype,
              vertices,
              restVertices,
              edges,
              activePairs,
              caseRank,
              cpNum,
              meshIndices,
              candidates,
              candidateNum,
              packedOutput,
              overflowCount,
              maxCandidatePairs,
              maxActivePairs,
              edgeNum,
              vertNum);
}

void lbvh_f_construct(lbvh_f* obj, double3* vertices)
{
    obj->Construct(vertices);
}

void lbvh_e_construct(lbvh_e* obj, double3* vertices)
{
    obj->Construct(vertices);
}

void lbvh_f_construct_full_ccd(
    lbvh_f* obj, double3* vertices, const double3* moveDir, double alpha)
{
    obj->ConstructFullCCD(vertices, moveDir, alpha);
}

void lbvh_e_construct_full_ccd(
    lbvh_e* obj, double3* vertices, const double3* moveDir, double alpha)
{
    obj->ConstructFullCCD(vertices, moveDir, alpha);
}

void lbvh_reset_candidate_cache(lbvh_f* faceObj, lbvh_e* edgeObj)
{
    lbvh* shared = initialized_shared_owner(faceObj, edgeObj);
    if(!shared)
        return;
    if(shared->_candidateNum)
        CUDA_SAFE_CALL(cudaMemset(shared->_candidateNum, 0, sizeof(uint32_t)));
    if(shared->_overflowCount)
        CUDA_SAFE_CALL(cudaMemset(shared->_overflowCount, 0, sizeof(uint32_t)));
}

uint32_t lbvh_cached_bounds_contain(
    lbvh_f* faceObj, lbvh_e* edgeObj, const double3* currentVertices)
{
    lbvh* shared = initialized_shared_owner(faceObj, edgeObj);
    if(!shared || !currentVertices || !shared->_overflowCount)
        return 0u;

    const bool checkFaces = faceObj && faceObj->_initialized
                            && faceObj->face_number > 0
                            && faceObj->vert_number > 0;
    const bool checkEdges = edgeObj && edgeObj->_initialized
                            && edgeObj->edge_number > 0;
    if(checkFaces
       && (!faceObj->_faces || !faceObj->_surfVerts || !faceObj->_bvs
           || !faceObj->_nodes || !faceObj->_sweptPointBox))
        return 0u;
    if(checkEdges && (!edgeObj->_edges || !edgeObj->_bvs || !edgeObj->_nodes))
        return 0u;

    uint32_t* outside = shared->_overflowCount + 1;
    CUDA_SAFE_CALL(cudaMemset(outside, 0, sizeof(uint32_t)));
    if(checkFaces)
    {
        const uint32_t pointBlocks =
            (faceObj->vert_number + BVH_THREADS - 1) / BVH_THREADS;
        validate_cached_point_bounds<<<pointBlocks, BVH_THREADS>>>(
            currentVertices,
            faceObj->_surfVerts,
            faceObj->_sweptPointBox,
            faceObj->vert_number,
            outside);
        const uint32_t faceBlocks =
            (faceObj->face_number + BVH_THREADS - 1) / BVH_THREADS;
        validate_cached_leaf_bounds<<<faceBlocks, BVH_THREADS>>>(currentVertices,
                                                                 faceObj->_faces,
                                                                 faceObj->_bvs,
                                                                 faceObj->_nodes,
                                                                 faceObj->face_number,
                                                                 0,
                                                                 outside);
    }
    if(checkEdges)
    {
        const uint32_t edgeBlocks =
            (edgeObj->edge_number + BVH_THREADS - 1) / BVH_THREADS;
        validate_cached_leaf_bounds<<<edgeBlocks, BVH_THREADS>>>(currentVertices,
                                                                 edgeObj->_edges,
                                                                 edgeObj->_bvs,
                                                                 edgeObj->_nodes,
                                                                 edgeObj->edge_number,
                                                                 1,
                                                                 outside);
    }

    uint32_t hostOutside = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&hostOutside,
                              outside,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
    return hostOutside == 0u ? 1u : 0u;
}

void lbvh_f_append_proximity_candidates(
    lbvh_f* obj, double3* vertices, double dHat)
{
    if(!obj)
        return;
    obj->Construct(vertices);
    obj->SelfCollitionDetect(dHat);
}

void lbvh_e_append_proximity_candidates(
    lbvh_e* obj, double3* vertices, double dHat)
{
    if(!obj)
        return;
    obj->Construct(vertices);
    obj->SelfCollitionDetect(dHat);
}

void lbvh_f_append_swept_candidates(lbvh_f*        obj,
                                    double3*       vertices,
                                    const double3* moveDir,
                                    double         alpha,
                                    double         dHat)
{
    if(!obj)
        return;
    obj->ConstructFullCCD(vertices, moveDir, alpha);
    obj->SelfCollitionFullDetect(dHat, moveDir, alpha);
}

void lbvh_e_append_swept_candidates(lbvh_e*        obj,
                                    double3*       vertices,
                                    const double3* moveDir,
                                    double         alpha,
                                    double         dHat)
{
    if(!obj)
        return;
    obj->ConstructFullCCD(vertices, moveDir, alpha);
    obj->SelfCollitionFullDetect(dHat, moveDir, alpha);
}

void lbvh_refilter_cached_candidates(
    lbvh_f* faceObj, lbvh_e* edgeObj, double3* currentVertices, double dHat)
{
    lbvh* shared = initialized_shared_owner(faceObj, edgeObj);
    if(!shared || !shared->_cpNum)
        return;

    const bool faceInitialized = faceObj && faceObj->_initialized;
    const bool edgeInitialized = edgeObj && edgeObj->_initialized;
    if(faceInitialized)
        faceObj->_vertexes = currentVertices;
    if(edgeInitialized)
        edgeObj->_vertexes = currentVertices;

    CUDA_SAFE_CALL(cudaMemset(shared->_cpNum, 0, 5u * sizeof(uint32_t)));
    uint32_t* activeOverflow =
        shared->_overflowCount ? shared->_overflowCount + 1 : nullptr;
    if(activeOverflow)
        CUDA_SAFE_CALL(cudaMemset(activeOverflow, 0, sizeof(uint32_t)));

    if(!shared->_candidateNum || !shared->_candidatePairs
       || !shared->_collisionPair || !shared->_caseRank)
        return;

    uint32_t candidateCount = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&candidateCount,
                              shared->_candidateNum,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
    if(candidateCount == 0)
        return;
    if(candidateCount > shared->_maxCandidatePairs)
        candidateCount = shared->_maxCandidatePairs;

    const uint32_t blocks = (candidateCount + BVH_THREADS - 1) / BVH_THREADS;
    process_cached_candidates<<<blocks, BVH_THREADS>>>(
        currentVertices,
        edgeInitialized ? edgeObj->_rest_vertexes : nullptr,
        faceInitialized ? faceObj->_faces : nullptr,
        edgeInitialized ? edgeObj->_edges : nullptr,
        shared->_candidatePairs,
        candidateCount,
        dHat,
        shared->_cpNum,
        shared->_caseRank,
        shared->_collisionPair,
        shared->_maxActivePairs,
        activeOverflow);
}

void lbvh_scatter_packed_cases(lbvh_f* faceObj, lbvh_e* edgeObj)
{
    lbvh* shared = initialized_shared_owner(faceObj, edgeObj);
    if(!shared || !shared->_cpNum || !shared->_collisionPair
       || !shared->_caseRank || !shared->_packedOutput)
        return;

    uint32_t activeCount = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&activeCount,
                              shared->_cpNum,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
    if(activeCount == 0)
        return;
    if(activeCount > shared->_maxActivePairs)
        activeCount = shared->_maxActivePairs;

    const uint32_t blocks = (activeCount + BVH_THREADS - 1) / BVH_THREADS;
    scatter_packed_cases<<<blocks, BVH_THREADS>>>(shared->_collisionPair,
                                                  shared->_caseRank,
                                                  shared->_cpNum,
                                                  shared->_packedOutput,
                                                  activeCount);
}

void lbvh_expand_cached_candidates(lbvh_f*  faceObj,
                                   lbvh_e*  edgeObj,
                                   int4*    expandedPairs,
                                   uint32_t candidateCount)
{
    lbvh* shared = initialized_shared_owner(faceObj, edgeObj);
    if(!shared || !expandedPairs || !shared->_candidatePairs || candidateCount == 0)
        return;

    candidateCount = __m_min(candidateCount, shared->_maxCandidatePairs);
    const bool faceInitialized = faceObj && faceObj->_initialized;
    const bool edgeInitialized = edgeObj && edgeObj->_initialized;
    const uint32_t blocks = (candidateCount + BVH_THREADS - 1) / BVH_THREADS;
    expand_cached_candidates<<<blocks, BVH_THREADS>>>(
        shared->_candidatePairs,
        faceInitialized ? faceObj->_faces : nullptr,
        edgeInitialized ? edgeObj->_edges : nullptr,
        expandedPairs,
        candidateCount);
}

void lbvh_f_self_collision_detect(lbvh_f* obj, double dHat)
{
    if(obj)
        obj->SelfCollitionDetect(dHat);
}

void lbvh_e_self_collision_detect(lbvh_e* obj, double dHat)
{
    if(obj)
        obj->SelfCollitionDetect(dHat);
}

void lbvh_f_self_collision_full_detect(
    lbvh_f* obj, double dHat, const double3* moveDir, double alpha)
{
    if(obj)
        obj->SelfCollitionFullDetect(dHat, moveDir, alpha);
}

void lbvh_e_self_collision_full_detect(
    lbvh_e* obj, double dHat, const double3* moveDir, double alpha)
{
    if(obj)
        obj->SelfCollitionFullDetect(dHat, moveDir, alpha);
}

double scene_size_f(lbvh_f* obj)
{
    if(!obj || obj->face_number == 0)
        return 0.0;
    obj->getSceneSize();
    return scene_diagonal_squared(obj->scene);
}

double scene_size_e(lbvh_e* obj)
{
    if(!obj || obj->edge_number == 0)
        return 0.0;
    obj->Construct(obj->_vertexes);
    return scene_diagonal_squared(obj->scene);
}
}
