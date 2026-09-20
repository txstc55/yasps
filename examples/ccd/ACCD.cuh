//
// ACCD.cuh
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#pragma once
#include <cuda_runtime.h>
extern "C" {
__device__
double point_triangle_ccd(
    const double3& _p,
    const double3& _t0,
    const double3& _t1,
    const double3& _t2,
    const double3& _dp,
    const double3& _dt0,
    const double3& _dt1,
    const double3& _dt2,
    double eta, double thickness);

__device__
double edge_edge_ccd(
    const double3& _ea0,
    const double3& _ea1,
    const double3& _eb0,
    const double3& _eb1,
    const double3& _dea0,
    const double3& _dea1,
    const double3& _deb0,
    const double3& _deb1,
    double eta, double thickness);

__device__
double doCCDVF(const double3& _p,
    const double3& _t0,
    const double3& _t1,
    const double3& _t2,
    const double3& _dp,
    const double3& _dt0,
    const double3& _dt1,
    const double3& _dt2,
    double errorRate, double thickness);

double self_largestFeasibleStepSize(
  double slackness,
  const double3* _vertexes,
  const int4* _ccd_collisonPairs,
  const double3* _moveDir,
  double* mqueue,
  int numbers);

double self_largestFeasibleStepSizeCompact(
  double slackness,
  const double3* _vertexes,
  const int2* _ccd_candidatePairs,
  const uint3* _faces,
  const uint2* _edges,
  const double3* _moveDir,
  double* mqueue,
  int numbers);
}
