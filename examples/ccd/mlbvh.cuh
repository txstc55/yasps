//
// mlbvh.cuh
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#pragma once
#ifndef _MLBVH_CUH_
#define _MLBVH_CUH_

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

struct AABB
{
  public:
    double3 upper;
    double3 lower;

    __host__ __device__ AABB();
    __host__ __device__ void combines(const double& x, const double& y, const double& z);
    __host__ __device__ void combines(const double& x,
                                      const double& y,
                                      const double& z,
                                      const double& xx,
                                      const double& yy,
                                      const double& zz);
    __host__ __device__ void combines(const AABB& aabb);
    __host__ __device__ double3 center();
};

struct Node
{
  public:
    uint32_t parent_idx;
    uint32_t left_idx;
    uint32_t right_idx;
    uint32_t element_idx;
};

class lbvh
{
  public:
    uint32_t vert_number = 0;
    double3* _vertexes   = nullptr;

    AABB*     _bvs           = nullptr;
    AABB*     _tempLeafBox   = nullptr;
    AABB*     _sweptPointBox = nullptr;
    Node*     _nodes         = nullptr;
    uint64_t* _MChash        = nullptr;
    uint64_t* _MChash_sorted = nullptr;
    uint32_t* _flags         = nullptr;
    int32_t*  _escape        = nullptr;

    void*  _sort_temp_storage = nullptr;
    size_t _sort_temp_bytes   = 0;

    int4*     _collisionPair = nullptr;
    uint32_t* _caseRank      = nullptr;
    uint32_t* _cpNum         = nullptr;
    uint32_t* _packedOutput  = nullptr;

    int2*     _candidatePairs = nullptr;
    uint32_t* _candidateNum   = nullptr;
    uint32_t* _overflowCount  = nullptr;

    int*      _btype       = nullptr;
    uint32_t* _meshIndices = nullptr;
    AABB      scene;

    uint32_t _maxCandidatePairs = 0;
    uint32_t _maxActivePairs    = 0;
    bool     _initialized       = false;

  public:
    lbvh() = default;
    virtual ~lbvh();

    void MALLOC_DEVICE_MEM(uint32_t primitiveNumber, uint32_t pointNumber = 0);
    void FREE_DEVICE_MEM();
    void radixSortMorton(uint32_t number);
};

class lbvh_f : public lbvh
{
  public:
    uint32_t  face_number = 0;
    uint3*    _faces      = nullptr;
    uint32_t* _surfVerts  = nullptr;

    void init(int*       btype,
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
              uint32_t   vertNum);

    double Construct(double3* vertices);
    double ConstructFullCCD(double3* vertices, const double3* moveDir, double alpha);
    AABB* getSceneSize();

    void SelfCollitionDetect(double dHat);
    void SelfCollitionFullDetect(double dHat, const double3* moveDir, double alpha);
};

class lbvh_e : public lbvh
{
  public:
    double3* _rest_vertexes = nullptr;
    uint32_t edge_number    = 0;
    uint2*   _edges         = nullptr;

    void init(int*       btype,
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
              uint32_t   vertNum);

    double Construct(double3* vertices);
    double ConstructFullCCD(double3* vertices, const double3* moveDir, double alpha);

    void SelfCollitionDetect(double dHat);
    void SelfCollitionFullDetect(double dHat, const double3* moveDir, double alpha);
};

extern "C"
{
uint32_t mlbvh_api_version();
lbvh_f* create_lbvh_f();
lbvh_e* create_lbvh_e();
void destroy_lbvh_f(lbvh_f* obj);
void destroy_lbvh_e(lbvh_e* obj);

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
                 uint32_t  vertNum);

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
                 uint32_t  vertNum);

void lbvh_f_construct(lbvh_f* obj, double3* vertices);
void lbvh_e_construct(lbvh_e* obj, double3* vertices);
void lbvh_f_construct_full_ccd(
    lbvh_f* obj, double3* vertices, const double3* moveDir, double alpha);
void lbvh_e_construct_full_ccd(
    lbvh_e* obj, double3* vertices, const double3* moveDir, double alpha);

// Append operations do not reset the shared cache. Reset once, append face
// candidates, then append edge candidates.
void lbvh_reset_candidate_cache(lbvh_f* faceObj, lbvh_e* edgeObj);
void lbvh_f_append_proximity_candidates(lbvh_f* obj, double3* vertices, double dHat);
void lbvh_e_append_proximity_candidates(lbvh_e* obj, double3* vertices, double dHat);
void lbvh_f_append_swept_candidates(lbvh_f*        obj,
                                    double3*       vertices,
                                    const double3* moveDir,
                                    double         alpha,
                                    double         dHat);
void lbvh_e_append_swept_candidates(lbvh_e*        obj,
                                    double3*       vertices,
                                    const double3* moveDir,
                                    double         alpha,
                                    double         dHat);

// Re-filtering reuses the CCD cache and current trial positions. It resets
// cpNum[0..4] and active overflow before launching.
void lbvh_refilter_cached_candidates(
    lbvh_f* faceObj, lbvh_e* edgeObj, double3* currentVertices, double dHat);
uint32_t lbvh_cached_bounds_contain(
    lbvh_f* faceObj, lbvh_e* edgeObj, const double3* currentVertices);
void lbvh_scatter_packed_cases(lbvh_f* faceObj, lbvh_e* edgeObj);
void lbvh_expand_cached_candidates(lbvh_f*   faceObj,
                                   lbvh_e*   edgeObj,
                                   int4*     expandedPairs,
                                   uint32_t  candidateCount);

// Compatibility names. These calls append to the current cache.
void lbvh_f_self_collision_detect(lbvh_f* obj, double dHat);
void lbvh_e_self_collision_detect(lbvh_e* obj, double dHat);
void lbvh_f_self_collision_full_detect(
    lbvh_f* obj, double dHat, const double3* moveDir, double alpha);
void lbvh_e_self_collision_full_detect(
    lbvh_e* obj, double dHat, const double3* moveDir, double alpha);

double scene_size_f(lbvh_f* obj);
double scene_size_e(lbvh_e* obj);
}

__device__ void _d_PP(const double3& v0, const double3& v1, double& d);
__device__ void _d_PT(
    const double3& v0, const double3& v1, const double3& v2, const double3& v3, double& d);
__device__ void _d_PE(const double3& v0, const double3& v1, const double3& v2, double& d);
__device__ void _d_EE(
    const double3& v0, const double3& v1, const double3& v2, const double3& v3, double& d);
__device__ void _d_EEParallel(
    const double3& v0, const double3& v1, const double3& v2, const double3& v3, double& d);
__device__ double _compute_epx(
    const double3& v0, const double3& v1, const double3& v2, const double3& v3);

#endif
