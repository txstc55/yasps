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
#include <cstdint>
#include <cuda_runtime.h>
extern "C" {
struct AABB {
public:
    double3 upper;
    double3 lower;
    __host__ __device__  AABB();
    __host__ __device__  void combines(const double& x, const double& y, const double& z);
    __host__ __device__  void combines(const double& x, const double& y, const double& z, const double& xx, const double& yy, const double& zz);
    __host__ __device__  void combines(const AABB& aabb);
    __host__ __device__  double3 center();
};

struct Node {
public:
    uint32_t parent_idx;
    uint32_t left_idx;
    uint32_t right_idx;
    uint32_t element_idx;
};

class lbvh {
public:
    uint32_t vert_number;
    double3* _vertexes;
    AABB* _bvs;
    AABB* _tempLeafBox;
    Node* _nodes;
    uint64_t* _MChash;
    uint32_t* _indices;
    int4* _collisionPair;
    int4* _ccd_collisionPair;
    uint32_t* _cpNum;
    uint32_t* _flags;
    AABB scene;
    int* _btype;
    uint32_t* _meshIndices;
public:
    lbvh() {}
    ~lbvh();
    void MALLOC_DEVICE_MEM(const int& number);
    void FREE_DEVICE_MEM();
    //void Construct();
};


class lbvh_f : public lbvh {
public:
    uint32_t face_number;
    uint3* _faces;
    uint32_t* _surfVerts;
public:
    void init(int* _btype, double3* _mVerts, uint3* _mFaces, uint32_t* _mSurfVert, int4* _mCollisonPairs, int4* _ccd_mCollisonPairs, uint32_t* _mcpNum, uint32_t* _meshIndices, int faceNum, int vertNum);
    // cpnum, the first one is the number of collision pairs, 2 for pp, 3 for pe, 4 for pt here, but ee for edge bvh_e
    double Construct(double3* _mVerts);
    AABB* getSceneSize(); // fuck this one
    double ConstructFullCCD(double3* _mVerts, const double3* moveDir, const double& alpha);
    void SelfCollitionDetect(double dHat); // only check for local without moving direction, for checking substeps
    void SelfCollitionFullDetect(double dHat, const double3* moveDir, const double& alpha); // check when moving, what's tha largest moving size
    void SeparateCasesCCD(uint2* pp_indices, uint3* pe_indices, uint4* pt_indices, uint* counts);
    void SeparateCasesCD(uint2* pp_indices, uint3* pe_indices, uint4* pt_indices, uint* counts);
};


// for collision pairs
// if the first one is positive, then it has to be ee
// if the first one is negative, check the third one, if the third one is negative, then it is pp
// otherwise, if the third one is positive, check the forth one, if the forth one is negative, then it is pe
// else, it is pt


class lbvh_e : public lbvh{
public:
    double3* _rest_vertexes;
    uint32_t edge_number;
    uint2* _edges;
public:
    void init(int* _btype, double3* _mVerts, double3* _rest_vertexes, uint2* _mEdges, int4* _mCollisonPairs, int4* _ccd_mCollisonPairs, uint32_t* _mcpNum, uint32_t* _meshIndices, int edgeNum, int vertNum);
    double Construct(double3* _mVerts);
    double ConstructFullCCD(double3* _mVerts, const double3* moveDir, const double& alpha);
    void SelfCollitionDetect(double dHat);
    void SelfCollitionFullDetect(double dHat, const double3* moveDir, const double& alpha);
    void SeparateCasesCCD(uint2* pp_indices, uint3* pe_indices, uint4* ee_indices, uint* counts);
    void SeparateCasesCD(uint2* pp_indices, uint3* pe_indices, uint4* ee_indices, uint* counts);
};

lbvh_f* create_lbvh_f();
lbvh_e* create_lbvh_e();
void lbvh_f_init(lbvh_f* obj, int* _btype, double3* _mVerts, uint3* _mFaces, uint32_t* _mSurfVert, int4* _mCollisonPairs, int4* _ccd_mCollisonPairs, uint32_t* _mcpNum, uint32_t* _meshIndices, int faceNum, int vertNum);
void lbvh_e_init(lbvh_e* obj, int* _btype, double3* _mVerts, double3* _rest_vertexes, uint2* _mEdges, int4* _mCollisonPairs, int4* _ccd_mCollisonPairs, uint32_t* _mcpNum, uint32_t* _meshIndices, int edgeNum, int vertNum);
void lbvh_f_construct(lbvh_f* obj, double3* _mVerts);
void lbvh_e_construct(lbvh_e* obj, double3* _mVerts);
void lbvh_f_construct_full_ccd(lbvh_f* obj, double3* _mVerts, const double3* moveDir, const double& alpha);
void lbvh_e_construct_full_ccd(lbvh_e* obj, double3* _mVerts, const double3* moveDir, const double& alpha);
void lbvh_f_self_collision_detect(lbvh_f* obj, double dHat);
void lbvh_e_self_collision_detect(lbvh_e* obj, double dHat);
void lbvh_f_self_collision_full_detect(lbvh_f* obj, double dHat, const double3* moveDir, const double& alpha);
void lbvh_e_self_collision_full_detect(lbvh_e* obj, double dHat, const double3* moveDir, const double& alpha);
void destroy_lbvh_f(lbvh_f* obj);
void destroy_lbvh_e(lbvh_e* obj);
double scene_size_f(lbvh_f* obj);
double scene_size_e(lbvh_e* obj);


void lbvh_f_separate_cases_ccd(lbvh_f* obj, uint2* pp_indices, uint3* pe_indices, uint4* pt_indices, uint32_t* count);
void lbvh_e_separate_cases_ccd(lbvh_e* obj, uint2* pp_indices, uint3* pe_indices, uint4* ee_indices, uint32_t* count);
void lbvh_f_separate_cases_cd(lbvh_f* obj, uint2* pp_indices, uint3* pe_indices, uint4* pt_indices, uint32_t* count);
void lbvh_e_separate_cases_cd(lbvh_e* obj, uint2* pp_indices, uint3* pe_indices, uint4* ee_indices, uint32_t* count);

__device__
void _d_PP(const double3& v0, const double3& v1, double& d);

__device__
void _d_PT(const double3& v0, const double3& v1, const double3& v2, const double3& v3, double& d);

__device__
void _d_PE(const double3& v0, const double3& v1, const double3& v2, double& d);

__device__
void _d_EE(const double3& v0, const double3& v1, const double3& v2, const double3& v3, double& d);

__device__
void _d_EEParallel(const double3& v0, const double3& v1, const double3& v2, const double3& v3, double& d);

__device__
double _compute_epx(const double3& v0, const double3& v1, const double3& v2, const double3& v3);

#endif
}
