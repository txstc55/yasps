//
// ACCD.cu
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "ACCD.cuh"
#include "gpu_eigen_libs.cuh"
#include <cub/block/block_reduce.cuh>
#include <cfloat>
#include <cmath>
#include <stdio.h>
const static int default_threads = 256;
template <class F>
__device__ __host__
inline F __m_max(F a, F b) {
    return a > b ? a : b;
}

template <class F>
__device__ __host__
inline F __m_min(F a, F b) {
    return a > b ? b : a;
}

extern "C" {
__device__
int _dType_point_triangle(const double3& v0, const double3& v1, const double3& v2, const double3& v3)
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
int _dType_edge_edge(const double3& v0, const double3& v1, const double3& v2, const double3& v3)
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

__device__ __forceinline__
double point_point_distance(const double3& v0, const double3& v1)
{
    return __GEIGEN__::__squaredNorm3(__GEIGEN__::__minus(v0, v1));
}

__device__ __forceinline__
double point_triangle_distance(const double3& v0, const double3& v1, const double3& v2, const double3& v3)
{
    double3 b = __GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v2, v1), __GEIGEN__::__minus(v3, v1));
    //double3 test = __GEIGEN__::__minus(v0, v1);
    double aTb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__minus(v0, v1), b);//(v0 - v1).dot(b);
    //printf("%f   %f   %f          %f   %f   %f   %f\n", b.x, b.y, b.z, test.x, test.y, test.z, aTb);
    return aTb * aTb / __GEIGEN__::__squaredNorm3(b);
}

__device__ __forceinline__
double point_edge_distance(const double3& v0, const double3& v1, const double3& v2)
{
    return __GEIGEN__::__squaredNorm3(__GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v1, v0), __GEIGEN__::__minus(v2, v0))) / __GEIGEN__::__squaredNorm3(__GEIGEN__::__minus(v2, v1));
}

__device__ __forceinline__
double edge_edge_distance(const double3& v0, const double3& v1, const double3& v2, const double3& v3)
{
    double3 b = __GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v1, v0), __GEIGEN__::__minus(v3, v2));//(v1 - v0).cross(v3 - v2);
    //if(__GEIGEN__::__norm(b) <1e-6)
    //    b = __GEIGEN__::__v_vec_cross(__GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v1, v0), __GEIGEN__::__minus(v2, v0)), __GEIGEN__::__minus(v1, v0));
    double aTb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__minus(v2, v0), b);//(v2 - v0).dot(b);
    return aTb * aTb / __GEIGEN__::__squaredNorm3(b);
}


__device__ __forceinline__
double _d_EEParallel(const double3& v0, const double3& v1, const double3& v2, const double3& v3)
{
    double3 b = __GEIGEN__::__v_vec_cross(__GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v1, v0), __GEIGEN__::__minus(v2, v0)), __GEIGEN__::__minus(v1, v0));
    double aTb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__minus(v2, v0), b);//(v2 - v0).dot(b);
    return aTb * aTb / __GEIGEN__::__squaredNorm3(b);
}

__device__
double edge_edge_distance_unclassified(
    const double3& ea0,
    const double3& ea1,
    const double3& eb0,
    const double3& eb1)
{
    switch (_dType_edge_edge(ea0, ea1, eb0, eb1)) {
    case 0:
        return point_point_distance(ea0, eb0);
    case 1:
        return point_point_distance(ea0, eb1);
    case 2:
        return point_edge_distance(ea0, eb0, eb1);
    case 3:
        return point_point_distance(ea1, eb0);
    case 4:
        return point_point_distance(ea1, eb1);
    case 5:
        return point_edge_distance(ea1, eb0, eb1);
    case 6:
        return point_edge_distance(eb0, ea0, ea1);
    case 7:
        return point_edge_distance(eb1, ea0, ea1);
    case 8:
        return edge_edge_distance(ea0, ea1, eb0, eb1);
    default:
        return 1e32;
    }
}

__device__
double point_triangle_distance_unclassified(
    const double3& p,
    const double3& t0,
    const double3& t1,
    const double3& t2)
{
    switch (_dType_point_triangle(p, t0, t1, t2)) {
    case 0:
        return point_point_distance(p, t0);
    case 1:
        return point_point_distance(p, t1);
    case 2:
        return point_point_distance(p, t2);
    case 3:
        return point_edge_distance(p, t0, t1);
    case 4:
        return point_edge_distance(p, t1, t2);
    case 5:
        return point_edge_distance(p, t2, t0);
    case 6:
        return point_triangle_distance(p, t0, t1, t2);
    default:
        return 1e32;
    }
}

__device__ __forceinline__
double3 _ccd_point_line_axis(
    const double3& point, const double3& a, const double3& b)
{
    double3 edge = __GEIGEN__::__minus(b, a);
    double3 offset = __GEIGEN__::__minus(point, a);
    double length2 = __GEIGEN__::__squaredNorm3(edge);
    if (!(length2 > 0.0) || !isfinite(length2))
        return make_double3(0.0, 0.0, 0.0);
    return __GEIGEN__::__minus(offset, __GEIGEN__::__s_vec_multiply3(
        edge, __GEIGEN__::__v_vec_dot(offset, edge) / length2));
}

// Bounded geometric fallback for linearly moving convex primitives.
// Helpers use ordinary arithmetic only to choose an axis; every acceptance
// checks original input coordinates/motions with directed-rounding bounds.
// A failed proof, exhausted budget, or unrepresentable subdivision returns
// only the contiguous prefix that has already been proved separated.

struct _CCDIntervalGeometry {
    double3 p0, p1, p2, p3;
    double3 d0, d1, d2, d3;
};

struct _CCDProjectionInterval {
    double lo, hi;
};

__device__ __forceinline__
double3 _ccd_interval_position(const double3& p, const double3& d, double t)
{
    // This position is only used to choose a witness axis, never in its proof.
    return make_double3(fma(t, d.x, p.x), fma(t, d.y, p.y), fma(t, d.z, p.z));
}

__device__ __forceinline__
double3 _ccd_interval_axis(bool pt, int index,
    const double3& p0, const double3& p1,
    const double3& p2, const double3& p3)
{
    if (pt) {
        switch (index) {
        case 0: return __GEIGEN__::__v_vec_cross(
            __GEIGEN__::__minus(p2, p1), __GEIGEN__::__minus(p3, p1));
        case 1: return _ccd_point_line_axis(p0, p1, p2);
        case 2: return _ccd_point_line_axis(p0, p2, p3);
        case 3: return _ccd_point_line_axis(p0, p3, p1);
        case 4: return __GEIGEN__::__minus(p0, p1);
        case 5: return __GEIGEN__::__minus(p0, p2);
        default: return __GEIGEN__::__minus(p0, p3);
        }
    }
    switch (index) {
    case 0: return __GEIGEN__::__v_vec_cross(
        __GEIGEN__::__minus(p1, p0), __GEIGEN__::__minus(p3, p2));
    case 1: return _ccd_point_line_axis(p0, p2, p3);
    case 2: return _ccd_point_line_axis(p1, p2, p3);
    case 3: return _ccd_point_line_axis(p2, p0, p1);
    case 4: return _ccd_point_line_axis(p3, p0, p1);
    case 5: return __GEIGEN__::__minus(p0, p2);
    case 6: return __GEIGEN__::__minus(p0, p3);
    case 7: return __GEIGEN__::__minus(p1, p2);
    default: return __GEIGEN__::__minus(p1, p3);
    }
}

__device__ __forceinline__
bool _ccd_interval_normalize_axis(double3& axis, double& normUpper)
{
    if (!isfinite(axis.x) || !isfinite(axis.y) || !isfinite(axis.z))
        return false;
    double scale = fmax(fabs(axis.x), fmax(fabs(axis.y), fabs(axis.z)));
    if (!(scale > 0.0))
        return false;
    axis = make_double3(axis.x / scale, axis.y / scale, axis.z / scale);
    // Interpret the resulting doubles as the exact chosen axis. Its norm is
    // rounded upward so multiplying by a target distance cannot understate it.
    normUpper = __dsqrt_ru(__dadd_ru(__dmul_ru(axis.x, axis.x),
        __dadd_ru(__dmul_ru(axis.y, axis.y), __dmul_ru(axis.z, axis.z))));
    return normUpper > 0.0 && isfinite(normUpper);
}

__device__ __forceinline__
_CCDProjectionInterval _ccd_interval_coordinate(
    double a, double b, double da, double db, double t, double n)
{
    // t is nonnegative. These intervals enclose (a-b)+t*(da-db) in exact
    // arithmetic on the ORIGINAL supplied doubles, including subtraction error.
    double lo = __dadd_rd(__dsub_rd(a, b), __dmul_rd(t, __dsub_rd(da, db)));
    double hi = __dadd_ru(__dsub_ru(a, b), __dmul_ru(t, __dsub_ru(da, db)));
    if (n >= 0.0)
        return {__dmul_rd(n, lo), __dmul_ru(n, hi)};
    return {__dmul_rd(n, hi), __dmul_ru(n, lo)};
}

__device__ __forceinline__
_CCDProjectionInterval _ccd_interval_projection(
    const double3& axis, const double3& a, const double3& b,
    const double3& da, const double3& db, double t)
{
    _CCDProjectionInterval x = _ccd_interval_coordinate(a.x, b.x, da.x, db.x, t, axis.x);
    _CCDProjectionInterval y = _ccd_interval_coordinate(a.y, b.y, da.y, db.y, t, axis.y);
    _CCDProjectionInterval z = _ccd_interval_coordinate(a.z, b.z, da.z, db.z, t, axis.z);
    return {__dadd_rd(x.lo, __dadd_rd(y.lo, z.lo)),
            __dadd_ru(x.hi, __dadd_ru(y.hi, z.hi))};
}

__device__ __forceinline__
void _ccd_interval_include_projection(_CCDProjectionInterval& range,
    const _CCDProjectionInterval& item)
{
    // A NaN must invalidate the witness, never be hidden by fmin/fmax.
    if (!isfinite(item.lo) || !isfinite(item.hi)) {
        range.lo = -INFINITY;
        range.hi = INFINITY;
        return;
    }
    range.lo = fmin(range.lo, item.lo);
    range.hi = fmax(range.hi, item.hi);
}

__device__ __forceinline__
void _ccd_interval_include_time(_CCDProjectionInterval& range, bool pt,
    const _CCDIntervalGeometry& g, const double3& axis, double t)
{
    if (pt) {
        _ccd_interval_include_projection(range, _ccd_interval_projection(axis, g.p0, g.p1, g.d0, g.d1, t));
        _ccd_interval_include_projection(range, _ccd_interval_projection(axis, g.p0, g.p2, g.d0, g.d2, t));
        _ccd_interval_include_projection(range, _ccd_interval_projection(axis, g.p0, g.p3, g.d0, g.d3, t));
    } else {
        _ccd_interval_include_projection(range, _ccd_interval_projection(axis, g.p0, g.p2, g.d0, g.d2, t));
        _ccd_interval_include_projection(range, _ccd_interval_projection(axis, g.p0, g.p3, g.d0, g.d3, t));
        _ccd_interval_include_projection(range, _ccd_interval_projection(axis, g.p1, g.p2, g.d1, g.d2, t));
        _ccd_interval_include_projection(range, _ccd_interval_projection(axis, g.p1, g.p3, g.d1, g.d3, t));
    }
}

__device__ __forceinline__
double _ccd_interval_axis_gap(bool pt, const _CCDIntervalGeometry& g,
    double3 axis, double startTime, double endTime)
{
    double normUpper;
    if (!_ccd_interval_normalize_axis(axis, normUpper))
        return 0.0;
    _CCDProjectionInterval range = {INFINITY, -INFINITY};
    _ccd_interval_include_time(range, pt, g, axis, startTime);
    if (endTime != startTime)
        _ccd_interval_include_time(range, pt, g, axis, endTime);
    // Either orientation is a valid separating axis. Strictly positive gap
    // excludes touching endpoints. Linear projection and convexity then cover
    // every time and every point of both primitives in the complete interval.
    if (range.lo > 0.0)
        return __ddiv_rd(range.lo, normUpper);
    if (range.hi < 0.0)
        return __ddiv_rd(-range.hi, normUpper);
    return 0.0;
}

// A time-varying primitive normal avoids subdividing a grazing rotating face
// into hundreds of fixed-axis intervals. Bernstein coefficient hulls enclose
// the cubic signed volume and quadratic normal over the entire time interval.
// All operands come from original input positions and directions.
struct _CCDIntervalVector {
    _CCDProjectionInterval x, y, z;
};

__device__ __forceinline__
bool _ccd_poly_valid(const _CCDProjectionInterval& a)
{
    return isfinite(a.lo) && isfinite(a.hi) && a.lo <= a.hi;
}

__device__ __forceinline__
_CCDProjectionInterval _ccd_poly_invalid()
{
    return {-INFINITY, INFINITY};
}

__device__ __forceinline__
_CCDProjectionInterval _ccd_poly_add(
    const _CCDProjectionInterval& a, const _CCDProjectionInterval& b)
{
    if (!_ccd_poly_valid(a) || !_ccd_poly_valid(b))
        return _ccd_poly_invalid();
    return {__dadd_rd(a.lo, b.lo), __dadd_ru(a.hi, b.hi)};
}

__device__ __forceinline__
_CCDProjectionInterval _ccd_poly_sub(
    const _CCDProjectionInterval& a, const _CCDProjectionInterval& b)
{
    if (!_ccd_poly_valid(a) || !_ccd_poly_valid(b))
        return _ccd_poly_invalid();
    return {__dsub_rd(a.lo, b.hi), __dsub_ru(a.hi, b.lo)};
}

__device__ __forceinline__
_CCDProjectionInterval _ccd_poly_mul(
    const _CCDProjectionInterval& a, const _CCDProjectionInterval& b)
{
    if (!_ccd_poly_valid(a) || !_ccd_poly_valid(b))
        return _ccd_poly_invalid();
    double lo = fmin(fmin(__dmul_rd(a.lo, b.lo), __dmul_rd(a.lo, b.hi)),
                     fmin(__dmul_rd(a.hi, b.lo), __dmul_rd(a.hi, b.hi)));
    double hi = fmax(fmax(__dmul_ru(a.lo, b.lo), __dmul_ru(a.lo, b.hi)),
                     fmax(__dmul_ru(a.hi, b.lo), __dmul_ru(a.hi, b.hi)));
    return {lo, hi};
}

__device__ __forceinline__
_CCDProjectionInterval _ccd_poly_div_positive(
    const _CCDProjectionInterval& a, double divisor)
{
    if (!_ccd_poly_valid(a))
        return _ccd_poly_invalid();
    return {__ddiv_rd(a.lo, divisor), __ddiv_ru(a.hi, divisor)};
}

__device__ __forceinline__
_CCDIntervalVector _ccd_poly_difference_at_time(
    const double3& a, const double3& b,
    const double3& da, const double3& db, double t)
{
    return {_ccd_interval_coordinate(a.x, b.x, da.x, db.x, t, 1.0),
            _ccd_interval_coordinate(a.y, b.y, da.y, db.y, t, 1.0),
            _ccd_interval_coordinate(a.z, b.z, da.z, db.z, t, 1.0)};
}

__device__ __forceinline__
_CCDIntervalVector _ccd_poly_cross(
    const _CCDIntervalVector& a, const _CCDIntervalVector& b)
{
    return {_ccd_poly_sub(_ccd_poly_mul(a.y, b.z), _ccd_poly_mul(a.z, b.y)),
            _ccd_poly_sub(_ccd_poly_mul(a.z, b.x), _ccd_poly_mul(a.x, b.z)),
            _ccd_poly_sub(_ccd_poly_mul(a.x, b.y), _ccd_poly_mul(a.y, b.x))};
}

__device__ __forceinline__
_CCDIntervalVector _ccd_poly_half_sum(
    const _CCDIntervalVector& a, const _CCDIntervalVector& b)
{
    return {_ccd_poly_div_positive(_ccd_poly_add(a.x, b.x), 2.0),
            _ccd_poly_div_positive(_ccd_poly_add(a.y, b.y), 2.0),
            _ccd_poly_div_positive(_ccd_poly_add(a.z, b.z), 2.0)};
}

__device__ __forceinline__
_CCDProjectionInterval _ccd_poly_dot(
    const _CCDIntervalVector& a, const _CCDIntervalVector& b)
{
    return _ccd_poly_add(_ccd_poly_mul(a.x, b.x),
        _ccd_poly_add(_ccd_poly_mul(a.y, b.y), _ccd_poly_mul(a.z, b.z)));
}

__device__ __forceinline__
double _ccd_poly_vector_norm_upper(const _CCDIntervalVector& a)
{
    if (!_ccd_poly_valid(a.x) || !_ccd_poly_valid(a.y) || !_ccd_poly_valid(a.z))
        return INFINITY;
    double x = fmax(fabs(a.x.lo), fabs(a.x.hi));
    double y = fmax(fabs(a.y.lo), fabs(a.y.hi));
    double z = fmax(fabs(a.z.lo), fabs(a.z.hi));
    return __dsqrt_ru(__dadd_ru(__dmul_ru(x, x),
        __dadd_ru(__dmul_ru(y, y), __dmul_ru(z, z))));
}

__device__ __noinline__
double _ccd_interval_moving_normal_gap(bool pt, const _CCDIntervalGeometry& g,
    double startTime, double endTime)
{
    if (!(startTime >= 0.0) || !(endTime >= startTime) || !isfinite(endTime))
        return 0.0;
    _CCDIntervalVector u0, u1, v0, v1, w0, w1;
    if (pt) {
        u0 = _ccd_poly_difference_at_time(g.p2, g.p1, g.d2, g.d1, startTime);
        u1 = _ccd_poly_difference_at_time(g.p2, g.p1, g.d2, g.d1, endTime);
        v0 = _ccd_poly_difference_at_time(g.p3, g.p1, g.d3, g.d1, startTime);
        v1 = _ccd_poly_difference_at_time(g.p3, g.p1, g.d3, g.d1, endTime);
        w0 = _ccd_poly_difference_at_time(g.p0, g.p1, g.d0, g.d1, startTime);
        w1 = _ccd_poly_difference_at_time(g.p0, g.p1, g.d0, g.d1, endTime);
    } else {
        u0 = _ccd_poly_difference_at_time(g.p1, g.p0, g.d1, g.d0, startTime);
        u1 = _ccd_poly_difference_at_time(g.p1, g.p0, g.d1, g.d0, endTime);
        v0 = _ccd_poly_difference_at_time(g.p3, g.p2, g.d3, g.d2, startTime);
        v1 = _ccd_poly_difference_at_time(g.p3, g.p2, g.d3, g.d2, endTime);
        w0 = _ccd_poly_difference_at_time(g.p2, g.p0, g.d2, g.d0, startTime);
        w1 = _ccd_poly_difference_at_time(g.p2, g.p0, g.d2, g.d0, endTime);
    }
    _CCDIntervalVector q0 = _ccd_poly_cross(u0, v0);
    _CCDIntervalVector q1 = _ccd_poly_half_sum(
        _ccd_poly_cross(u0, v1), _ccd_poly_cross(u1, v0));
    _CCDIntervalVector q2 = _ccd_poly_cross(u1, v1);
    _CCDProjectionInterval c0 = _ccd_poly_dot(w0, q0);
    _CCDProjectionInterval c1a = _ccd_poly_dot(w0, q1);
    _CCDProjectionInterval c1 = _ccd_poly_div_positive(_ccd_poly_add(
        _ccd_poly_add(c1a, c1a), _ccd_poly_dot(w1, q0)), 3.0);
    _CCDProjectionInterval c2a = _ccd_poly_dot(w1, q1);
    _CCDProjectionInterval c2 = _ccd_poly_div_positive(_ccd_poly_add(
        _ccd_poly_add(c2a, c2a), _ccd_poly_dot(w0, q2)), 3.0);
    _CCDProjectionInterval c3 = _ccd_poly_dot(w1, q2);
    if (!_ccd_poly_valid(c0) || !_ccd_poly_valid(c1) ||
        !_ccd_poly_valid(c2) || !_ccd_poly_valid(c3))
        return 0.0;
    double minVolume = fmin(fmin(c0.lo, c1.lo), fmin(c2.lo, c3.lo));
    double maxVolume = fmax(fmax(c0.hi, c1.hi), fmax(c2.hi, c3.hi));
    double volumeLower = minVolume > 0.0 ? minVolume :
        (maxVolume < 0.0 ? -maxVolume : 0.0);
    if (!(volumeLower > 0.0))
        return 0.0;
    // The quadratic Bernstein basis is nonnegative and sums to one, so
    // convexity of the norm bounds every normal by the largest control norm.
    double normUpper = fmax(_ccd_poly_vector_norm_upper(q0),
        fmax(_ccd_poly_vector_norm_upper(q1), _ccd_poly_vector_norm_upper(q2)));
    if (!(normUpper > 0.0) || !isfinite(normUpper))
        return 0.0;
    return __ddiv_rd(volumeLower, normUpper);
}

__device__ __forceinline__
bool _ccd_interval_moving_normal(bool pt, const _CCDIntervalGeometry& g,
    double target, double startTime, double endTime)
{
    return target >= 0.0 && isfinite(target) &&
        _ccd_interval_moving_normal_gap(pt, g, startTime, endTime) > target;
}

__device__ __forceinline__
bool _ccd_interval_certified_geometry(bool pt, const _CCDIntervalGeometry& g,
    double target, double startTime, double endTime)
{
    if (!(target >= 0.0) || !isfinite(target) || !(startTime >= 0.0) ||
        !(endTime >= startTime) || !isfinite(endTime))
        return false;
    double mid = startTime + (endTime - startTime) * 0.5;
    double3 p0 = _ccd_interval_position(g.p0, g.d0, mid);
    double3 p1 = _ccd_interval_position(g.p1, g.d1, mid);
    double3 p2 = _ccd_interval_position(g.p2, g.d2, mid);
    double3 p3 = _ccd_interval_position(g.p3, g.d3, mid);
    int axes = pt ? 7 : 9;
    for (int i = 0; i < axes; ++i) {
        if (_ccd_interval_axis_gap(pt, g,
                _ccd_interval_axis(pt, i, p0, p1, p2, p3), startTime, endTime) > target)
            return true;
    }
    return false;
}

// Public helper for validating a fast ACCD proposal: target=0, startTime=0,
// endTime=proposal. A false result means unresolved, not necessarily collision.
__device__ __forceinline__
bool _ccd_interval_certified(bool pt,
    const double3& p0, const double3& p1, const double3& p2, const double3& p3,
    const double3& d0, const double3& d1, const double3& d2, const double3& d3,
    double target, double startTime, double endTime)
{
    _CCDIntervalGeometry g = {p0, p1, p2, p3, d0, d1, d2, d3};
    return _ccd_interval_certified_geometry(pt, g, target, startTime, endTime);
}

// At most 128 interval tests, each trying a moving-normal certificate and
// at most 7 PT or 9 EE fixed axes. The greedy
// left-to-right walk keeps O(1) state; it never allocates a subdivision stack.
// Keeping the fallback out of line limits register growth in the common path.
__device__ __noinline__
double _ccd_interval_step(bool pt,
    const double3& p0, const double3& p1, const double3& p2, const double3& p3,
    const double3& d0, const double3& d1, const double3& d2, const double3& d3,
    double eta, double thickness, double maxTime)
{
    if (!(eta >= 0.0 && eta < 1.0) || !(thickness >= 0.0) ||
        !isfinite(thickness) || !(maxTime > 0.0) || !isfinite(maxTime))
        return 0.0;
    _CCDIntervalGeometry g = {p0, p1, p2, p3, d0, d1, d2, d3};
    double initialGap = 0.0;
    int axes = pt ? 7 : 9;
    for (int i = 0; i < axes; ++i)
        initialGap = fmax(initialGap, _ccd_interval_axis_gap(pt, g,
            _ccd_interval_axis(pt, i, p0, p1, p2, p3), 0.0, 0.0));
    if (!(initialGap > thickness))
        return 0.0;
    // This separation lower bound is independent of the legacy distance
    // classifier, including its nearly-parallel edge overestimation. Retain
    // a positive fraction of the proven initial clearance when eta>0.
    double clearance = __dsub_rd(initialGap, thickness);
    double target = __dadd_ru(thickness, __dmul_ru(eta, clearance));
    if (!isfinite(target))
        return 0.0;
    double prefix = 0.0;
    double width = maxTime;
    constexpr int maxIntervalTests = 128;
    for (int test = 0; test < maxIntervalTests; ++test) {
        double end = fmin(maxTime, prefix + width);
        if (!(end > prefix))
            break;
        if (_ccd_interval_moving_normal(pt, g, target, prefix, end) ||
            _ccd_interval_certified_geometry(pt, g, target, prefix, end)) {
            prefix = end;
            if (prefix >= maxTime)
                return maxTime;
            width = fmin(width + width, maxTime - prefix);
        } else {
            width *= 0.5;
            if (!(prefix + width > prefix))
                break;
        }
    }
    return prefix;
}

// A fixed axis separates two moving convex primitives for the whole interval
// if every vertex-pair projection exceeds the target at both endpoints:
// the projections are linear in time, and convex combinations preserve the
// bound. Keep ACCD's initial target gap, rather than testing only for contact,
// so near misses that approach closer than that gap still run through ACCD.
__device__ __forceinline__
bool _ccd_point_triangle_separated(
    const double3& p, const double3& t0, const double3& t1, const double3& t2,
    const double3& dp, const double3& dt0, const double3& dt1, const double3& dt2,
    double target, double maxTime)
{
    double3 axis;
    switch (_dType_point_triangle(p, t0, t1, t2)) {
    case 0: axis = __GEIGEN__::__minus(p, t0); break;
    case 1: axis = __GEIGEN__::__minus(p, t1); break;
    case 2: axis = __GEIGEN__::__minus(p, t2); break;
    case 3: axis = _ccd_point_line_axis(p, t0, t1); break;
    case 4: axis = _ccd_point_line_axis(p, t1, t2); break;
    case 5: axis = _ccd_point_line_axis(p, t2, t0); break;
    case 6:
        axis = __GEIGEN__::__v_vec_cross(
            __GEIGEN__::__minus(t1, t0), __GEIGEN__::__minus(t2, t0));
        break;
    default: return false;
    }
    _CCDIntervalGeometry g = {p, t0, t1, t2, dp, dt0, dt1, dt2};
    return _ccd_interval_axis_gap(true, g, axis, 0.0, maxTime) > target;
}

__device__ __forceinline__
bool _ccd_edge_edge_separated(
    const double3& a0, const double3& a1, const double3& b0, const double3& b1,
    const double3& da0, const double3& da1, const double3& db0, const double3& db1,
    double target, double maxTime)
{
    double3 axis;
    switch (_dType_edge_edge(a0, a1, b0, b1)) {
    case 0: axis = __GEIGEN__::__minus(a0, b0); break;
    case 1: axis = __GEIGEN__::__minus(a0, b1); break;
    case 2: axis = _ccd_point_line_axis(a0, b0, b1); break;
    case 3: axis = __GEIGEN__::__minus(a1, b0); break;
    case 4: axis = __GEIGEN__::__minus(a1, b1); break;
    case 5: axis = _ccd_point_line_axis(a1, b0, b1); break;
    case 6: axis = _ccd_point_line_axis(b0, a0, a1); break;
    case 7: axis = _ccd_point_line_axis(b1, a0, a1); break;
    case 8:
        axis = __GEIGEN__::__v_vec_cross(
            __GEIGEN__::__minus(a1, a0), __GEIGEN__::__minus(b1, b0));
        break;
    default: return false;
    }
    _CCDIntervalGeometry g = {a0, a1, b0, b1, da0, da1, db0, db1};
    return _ccd_interval_axis_gap(false, g, axis, 0.0, maxTime) > target;
}

__device__
double edge_edge_ccd_fast(
    const double3& _ea0,
    const double3& _ea1,
    const double3& _eb0,
    const double3& _eb1,
    const double3& _dea0,
    const double3& _dea1,
    const double3& _deb0,
    const double3& _deb1,
    double eta, double thickness, double maxTime, int maxIterations, bool& certified)
{
    double3 ea0 = _ea0, ea1 = _ea1, eb0 = _eb0, eb1 = _eb1, dea0 = _dea0, dea1 = _dea1, deb0 = _deb0, deb1 = _deb1;
    double3 temp0 = __GEIGEN__::__add(dea0, dea1);
    double3 temp1 = __GEIGEN__::__add(deb0, deb1);
    double3 mov = __GEIGEN__::__s_vec_multiply3(__GEIGEN__::__add(temp0, temp1), -0.25);

    dea0 = __GEIGEN__::__add(dea0, mov);
    dea1 = __GEIGEN__::__add(dea1, mov);
    deb0 = __GEIGEN__::__add(deb0, mov);
    deb1 = __GEIGEN__::__add(deb1, mov);

    double max_disp_mag = sqrt(__m_max(__GEIGEN__::__squaredNorm3(dea0), __GEIGEN__::__squaredNorm3(dea1))) + sqrt(__m_max(__GEIGEN__::__squaredNorm3(deb0), __GEIGEN__::__squaredNorm3(deb1)));
    if (max_disp_mag == 0)
        return maxTime;
    if (!(eta >= 0.0 && eta < 1.0) || !isfinite(max_disp_mag))
        return 0.0;

    // The legacy distance classifier downgrades coplanar interior edge pairs
    // to point-edge cases. Detect an initial crossing before that downgrade.
    double3 initial_u = __GEIGEN__::__minus(ea1, ea0);
    double3 initial_v = __GEIGEN__::__minus(eb1, eb0);
    double3 initial_w = __GEIGEN__::__minus(ea0, eb0);
    double3 initial_n = __GEIGEN__::__v_vec_cross(initial_u, initial_v);
    double initial_n2 = __GEIGEN__::__squaredNorm3(initial_n);
    if (initial_n2 > 0.0 && isfinite(initial_n2) &&
        __GEIGEN__::__v_vec_dot(initial_w, initial_n) == 0.0) {
        double initial_s = __GEIGEN__::__v_vec_dot(
            __GEIGEN__::__v_vec_cross(initial_v, initial_w), initial_n) / initial_n2;
        double initial_t = __GEIGEN__::__v_vec_dot(
            __GEIGEN__::__v_vec_cross(initial_u, initial_w), initial_n) / initial_n2;
        if (initial_s >= 0.0 && initial_s <= 1.0 &&
            initial_t >= 0.0 && initial_t <= 1.0)
            return 0.0;
    }

    double dist2_cur = edge_edge_distance_unclassified(ea0, ea1, eb0, eb1);
    // Initial contact and invalid/degenerate distances cannot certify a
    // positive step. In particular, do not substitute endpoint distances for
    // an edge-edge contact: those can be positive while the segments touch.
    if (!isfinite(dist2_cur) || !(dist2_cur > thickness * thickness))
        return 0.0;

    double dFunc = dist2_cur - thickness * thickness;
    double dist_cur = sqrt(dist2_cur);
    double gap = eta * dFunc / (dist_cur + thickness);
    if (dist_cur > thickness && _ccd_edge_edge_separated(
            ea0, ea1, eb0, eb1, _dea0, _dea1, _deb0, _deb1, thickness + gap, maxTime)) {
        certified = true;
        return maxTime;
    }
    double toc = 0.0;
    int count = 0;
    while (true) {
        // A negative result is an internal first-pass marker, never a step.
        if (maxIterations > 0 && count >= maxIterations)
            return -1.0;
        ++count;
        double toc_lower_bound = (1 - eta) * dFunc / ((dist_cur + thickness) * max_disp_mag);
        if (!isfinite(toc_lower_bound) || !(toc_lower_bound > 0.0) ||
            !(toc + toc_lower_bound > toc))
            return toc;
        ea0 = __GEIGEN__::__add(ea0, __GEIGEN__::__s_vec_multiply3(dea0, toc_lower_bound));
        ea1 = __GEIGEN__::__add(ea1, __GEIGEN__::__s_vec_multiply3(dea1, toc_lower_bound));
        eb0 = __GEIGEN__::__add(eb0, __GEIGEN__::__s_vec_multiply3(deb0, toc_lower_bound));
        eb1 = __GEIGEN__::__add(eb1, __GEIGEN__::__s_vec_multiply3(deb1, toc_lower_bound));

        dist2_cur = edge_edge_distance_unclassified(ea0, ea1, eb0, eb1);
        if (!isfinite(dist2_cur) || !(dist2_cur > thickness * thickness))
            return toc;
        dFunc = dist2_cur - thickness * thickness;
        dist_cur = sqrt(dist2_cur);
        if (toc && (dFunc / (dist_cur + thickness) < gap)) {
            break;
        }
        toc += toc_lower_bound;
        if (toc >= maxTime)
            return maxTime;
    }
    return toc;
}

__device__
double point_triangle_ccd_fast(
    const double3& _p,
    const double3& _t0,
    const double3& _t1,
    const double3& _t2,
    const double3& _dp,
    const double3& _dt0,
    const double3& _dt1,
    const double3& _dt2,
    double eta, double thickness, double maxTime, int maxIterations, bool& certified)
{
    double3 p = _p, t0 = _t0, t1 = _t1, t2 = _t2, dp = _dp, dt0 = _dt0, dt1 = _dt1, dt2 = _dt2;

    double3 temp0 = __GEIGEN__::__add(dt0, dt1);
    double3 temp1 = __GEIGEN__::__add(dt2, dp);
    double3 mov = __GEIGEN__::__s_vec_multiply3(__GEIGEN__::__add(temp0, temp1), -0.25);

    dt0 = __GEIGEN__::__add(dt0, mov);
    dt1 = __GEIGEN__::__add(dt1, mov);
    dt2 = __GEIGEN__::__add(dt2, mov);
    dp = __GEIGEN__::__add(dp, mov);

    double disp_mag2_vec0 = __GEIGEN__::__squaredNorm3(dt0);
    double disp_mag2_vec1 = __GEIGEN__::__squaredNorm3(dt1);
    double disp_mag2_vec2 = __GEIGEN__::__squaredNorm3(dt2);

    double max_disp_mag = __GEIGEN__::__norm(dp) + sqrt(__m_max(disp_mag2_vec0, __m_max(disp_mag2_vec1, disp_mag2_vec2)));
    if (max_disp_mag == 0)
        return maxTime;
    if (!(eta >= 0.0 && eta < 1.0) || !isfinite(max_disp_mag))
        return 0.0;

    double dist2_cur = point_triangle_distance_unclassified(p, t0, t1, t2);
    if (!isfinite(dist2_cur) || !(dist2_cur > thickness * thickness))
        return 0.0;
    double dist_cur = sqrt(dist2_cur);
    double gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
    if (dist_cur > thickness && _ccd_point_triangle_separated(
            p, t0, t1, t2, _dp, _dt0, _dt1, _dt2, thickness + gap, maxTime)) {
        certified = true;
        return maxTime;
    }
    double toc = 0.0;
    int count = 0;
    while (true) {
        // A negative result is an internal first-pass marker, never a step.
        if (maxIterations > 0 && count >= maxIterations)
            return -1.0;
        ++count;
        double toc_lower_bound = (1 - eta) * (dist2_cur - thickness * thickness) / ((dist_cur + thickness) * max_disp_mag);
        // Stop at the last certified time if arithmetic cannot advance it.
        if (!isfinite(toc_lower_bound) || !(toc_lower_bound > 0.0) ||
            !(toc + toc_lower_bound > toc))
            return toc;

        p = __GEIGEN__::__add(p, __GEIGEN__::__s_vec_multiply3(dp, toc_lower_bound));
        t0 = __GEIGEN__::__add(t0, __GEIGEN__::__s_vec_multiply3(dt0, toc_lower_bound));
        t1 = __GEIGEN__::__add(t1, __GEIGEN__::__s_vec_multiply3(dt1, toc_lower_bound));
        t2 = __GEIGEN__::__add(t2, __GEIGEN__::__s_vec_multiply3(dt2, toc_lower_bound));

        dist2_cur = point_triangle_distance_unclassified(p, t0, t1, t2);
        if (!isfinite(dist2_cur) || !(dist2_cur > thickness * thickness))
            return toc;
        dist_cur = sqrt(dist2_cur);
        if (toc && ((dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap)) {
            break;
        }

        toc += toc_lower_bound;
        if (toc >= maxTime) {
            return maxTime;
        }
    }
    return toc;
}


// Fast advancement proposes a step. A directed-rounding geometric witness
// verifies every accepted positive proposal. Unresolved first-pass proposals
// retain the negative marker; refinement uses bounded geometric subdivision.
__device__
double edge_edge_ccd_limited(
    const double3& p0, const double3& p1, const double3& p2, const double3& p3,
    const double3& d0, const double3& d1, const double3& d2, const double3& d3,
    double eta, double thickness, double maxTime, int maxIterations)
{
    bool certified = false;
    double proposal = edge_edge_ccd_fast(p0, p1, p2, p3, d0, d1, d2, d3,
        eta, thickness, maxTime, maxIterations > 0 ? maxIterations : 8, certified);
    if (proposal == 0.0)
        return 0.0;
    if (proposal > 0.0 && (certified || _ccd_interval_certified(false,
            p0, p1, p2, p3, d0, d1, d2, d3, thickness, 0.0, proposal)))
        return proposal;
    if (maxIterations > 0)
        return -1.0;
    return _ccd_interval_step(false, p0, p1, p2, p3, d0, d1, d2, d3,
        eta, thickness, maxTime);
}

__device__
double point_triangle_ccd_limited(
    const double3& p0, const double3& p1, const double3& p2, const double3& p3,
    const double3& d0, const double3& d1, const double3& d2, const double3& d3,
    double eta, double thickness, double maxTime, int maxIterations)
{
    bool certified = false;
    double proposal = point_triangle_ccd_fast(p0, p1, p2, p3, d0, d1, d2, d3,
        eta, thickness, maxTime, maxIterations > 0 ? maxIterations : 8, certified);
    if (proposal == 0.0)
        return 0.0;
    if (proposal > 0.0 && (certified || _ccd_interval_certified(true,
            p0, p1, p2, p3, d0, d1, d2, d3, thickness, 0.0, proposal)))
        return proposal;
    if (maxIterations > 0)
        return -1.0;
    return _ccd_interval_step(true, p0, p1, p2, p3, d0, d1, d2, d3,
        eta, thickness, maxTime);
}

// Preserve the device API for callers needing an individual full-interval CCD.
__device__ double edge_edge_ccd(
    const double3& a0, const double3& a1, const double3& b0, const double3& b1,
    const double3& da0, const double3& da1, const double3& db0, const double3& db1,
    double eta, double thickness)
{
    return edge_edge_ccd_limited(a0, a1, b0, b1, da0, da1, db0, db1,
                                 eta, thickness, 1.0, 0);
}

__device__ double point_triangle_ccd(
    const double3& p, const double3& a, const double3& b, const double3& c,
    const double3& dp, const double3& da, const double3& db, const double3& dc,
    double eta, double thickness)
{
    return point_triangle_ccd_limited(p, a, b, c, dp, da, db, dc,
                                      eta, thickness, 1.0, 0);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


typedef struct {
    double3 ad, bd, cd, pd;
    double3 a0, b0, c0, p0;
} NewtonCheckData;

inline __device__ bool _insideTriangle(double3 a, double3 b, double3 c, double3 p)
{
    double3 n, da, db, dc;
    double wa, wb, wc;

    double3 ba = __GEIGEN__::__minus(b, a);
    double3 ca = __GEIGEN__::__minus(c, a);

    n = __GEIGEN__::__v_vec_cross(ba, ca);//cross(ba, ca);

    da = __GEIGEN__::__minus(a, p);
    db = __GEIGEN__::__minus(b, p);
    dc = __GEIGEN__::__minus(c, p);
    //da = a - p, db = b - p, dc = c - p;
    if ((wa = __GEIGEN__::__v_vec_dot(__GEIGEN__::__v_vec_cross(db, dc), n)) < 0.0f) return false;
    if ((wb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__v_vec_cross(dc, da), n)) < 0.0f) return false;
    if ((wc = __GEIGEN__::__v_vec_dot(__GEIGEN__::__v_vec_cross(da, db), n)) < 0.0f) return false;

    //Compute barycentric coordinates
    double area2 = __GEIGEN__::__v_vec_dot(n, n);
    wa /= area2, wb /= area2, wc /= area2;


    return true;
}


inline __device__ int solveQuadric(double c[3], double s[2], const double& errorRate)
{
    double p, q, D;

    // make sure we have a d2 equation

    if (c[2] < errorRate) {
        if ((c[1]) < errorRate)
            return 0;
        s[0] = -c[0] / c[1];
        return 1;
    }

    // normal for: x^2 + px + q
    p = c[1] / (2.0f * c[2]);
    q = c[0] / c[2];
    D = p * p - q;

    if ((D)< errorRate)
    {
        // one float root
        s[0] = s[1] = -p;
        return 1;
    }

    if (D < 0.0f)
        // no real root
        return 0;

    else
    {
        // two real roots
        double sqrt_D = sqrt(D);
        s[0] = sqrt_D - p;
        s[1] = -sqrt_D - p;
        return 2;
    }
}
inline __device__ int solveCubic(double c[4], double s[3], const double& errorRate)
{
    int	i, num;
    double	sub,
        A, B, C,
        sq_A, p, q,
        cb_p, D;

    if (c[3]< errorRate) {
        return solveQuadric(c, s, errorRate);
    }

    A = c[2] / c[3];
    B = c[1] / c[3];
    C = c[0] / c[3];

    sq_A = A * A;
    double ONE_DIV_3 = 1.0 / 3;
    p = ONE_DIV_3 * (-ONE_DIV_3 * sq_A + B);
    q = 0.5f * (2.0f / 27.0f * A * sq_A - ONE_DIV_3 * A * B + C);

    // use Cardano's formula

    cb_p = p * p * p;
    D = q * q + cb_p;
    double My_PI = 3.1415926535897932;
    if ((D)< errorRate)
    {
        if ((q)< errorRate)
        {
            // one triple solution
            s[0] = 0.0f;
            num = 1;
        }
        else
        {
            // one single and one float solution
            double u = cbrt(-q);
            s[0] = 2.0f * u;
            s[1] = -u;
            num = 2;
        }
    }
    else
        if (D < 0.0f)
        {
            // casus irreductibilis: three real solutions
            double phi = ONE_DIV_3 * acos(-q / sqrt(-cb_p));
            double t = 2.0f * sqrt(-p);
            s[0] = t * cos(phi);
            s[1] = -t * cos(phi + My_PI / 3.0f);
            s[2] = -t * cos(phi - My_PI / 3.0f);
            num = 3;
        }
        else
        {
            // one real solution
            double sqrt_D = sqrt(D);
            double u = cbrt(sqrt_D + fabs(q));
            if (q > 0.0f)
                s[0] = -u + p / u;
            else
                s[0] = u - p / u;
            num = 1;
        }

    // resubstitute
    sub = ONE_DIV_3 * A;
    for (i = 0; i < num; i++)
        s[i] -= sub;
    return num;
}

inline __device__ void _equateCubic_VF(
    double3 a0, double3 ad, double3 b0, double3 bd,
    double3 c0, double3 cd, double3 p0, double3 pd,
    double& a, double& b, double& c, double& d, const double& thickness)
{
    double3 dab, dac, dap;
    double3 oab, oac, oap;
    double3 dabXdac, dabXoac, oabXdac, oabXoac;

    dab =__GEIGEN__::__minus(bd, ad), dac =__GEIGEN__::__minus(cd , ad), dap =__GEIGEN__::__minus(pd , ad);
    oab =__GEIGEN__::__minus(b0, a0), oac =__GEIGEN__::__minus(c0 , a0), oap =__GEIGEN__::__minus(p0 , a0);

    dabXdac = __GEIGEN__::__v_vec_cross(dab, dac);
    dabXoac = __GEIGEN__::__v_vec_cross(dab, oac);
    oabXdac = __GEIGEN__::__v_vec_cross(oab, dac);
    oabXoac = __GEIGEN__::__v_vec_cross(oab, oac);

    a = __GEIGEN__::__v_vec_dot(dap, dabXdac);
    b = __GEIGEN__::__v_vec_dot(oap, dabXdac) + __GEIGEN__::__v_vec_dot(dap,__GEIGEN__::__add(dabXoac , oabXdac));
    c = __GEIGEN__::__v_vec_dot(dap, oabXoac) + __GEIGEN__::__v_vec_dot(oap,__GEIGEN__::__add(dabXoac , oabXdac));
    d = thickness * __GEIGEN__::__v_vec_dot(oap, oabXoac);
}

inline __device__ double IntersectVF(double3 ta0, double3 tb0, double3 tc0,
    double3 ad, double3 bd, double3 cd,
    double3 q0, double3 qd,
    const double& errorRate, const double& thickness)
{
    double collisionTime = 1.0;

    double a, b, c, d; /* cubic polynomial coefficients */
    _equateCubic_VF(ta0, ad, tb0, bd, tc0, cd, q0, qd, a, b, c, d, thickness);

    //if ((a) < errorRate && (b) < errorRate && (c) < errorRate && (d) < errorRate)
    //    return 1.0;

    double roots[3];
    double coeffs[4];
    coeffs[3] = a, coeffs[2] = b, coeffs[1] = c, coeffs[0] = d;
    int num;// = solveCubic(coeffs, roots, errorRate);

    __GEIGEN__::__NewtonSolverForCubicEquation(a, b, c, d, roots, num, errorRate);

    if (num == 0)
        return 1.0;

    for (int i = 0; i < num; i++) {
        double r = roots[i];
        if (r < 0 || r > 1) continue;

        if (_insideTriangle(
            __GEIGEN__::__add(__GEIGEN__::__s_vec_multiply3(ad, r), ta0),
            __GEIGEN__::__add(__GEIGEN__::__s_vec_multiply3(bd, r), tb0),
            __GEIGEN__::__add(__GEIGEN__::__s_vec_multiply3(cd, r), tc0),
            __GEIGEN__::__add(__GEIGEN__::__s_vec_multiply3(qd, r), q0))) {
            if (collisionTime > r) {
                collisionTime = r;
            }
        }
    }

    return collisionTime;
}

__device__
double doCCDVF(const double3& _p,
    const double3& _t0,
    const double3& _t1,
    const double3& _t2,
    const double3& _dp,
    const double3& _dt0,
    const double3& _dt1,
    const double3& _dt2,
    double errorRate, double thickness)
{
    double ret = IntersectVF(_t0, _t1, _t2, _dt0, _dt1, _dt2, _p, _dp, errorRate, thickness);

    return ret;
}

// A pair certified through the requested horizon cannot constrain the minimum.
// Encode it as 1 so a full sweep preserves its exact caller-supplied bound.
__device__ __forceinline__
double _ccd_step_reciprocal(double step, double maxTime)
{
    return step >= maxTime ? 1.0 : __ddiv_ru(1.0, step);
}

// Round reciprocals upward so the host inverse cannot exceed a certified step.
// Magnitude is the largest reciprocal of a completed step (at least 1).
// The sign carries "some pair unfinished" through BOTH reduction levels.
struct StepBoundMax {
    __device__ double operator()(double a, double b) const {
        const double magnitude = __m_max(fabs(a), fabs(b));
        return (a < 0.0 || b < 0.0) ? -magnitude : magnitude;
    }
};

__global__
void _reduct_min_selfTimeStep_to_double(const double3* vertexes, const int4* _ccd_collitionPairs, const double3* moveDir, double* minStepSizes, double slackness, int number, double maxTime, int maxIterations) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    double temp = 0.0;
    if (idx < number) {
        double CCDDistRatio = 1.0 - slackness;
        int4 MMCVIDI = _ccd_collitionPairs[idx];

        if (MMCVIDI.x < 0) {
            MMCVIDI.x = -MMCVIDI.x - 1;

            double temp1 = point_triangle_ccd_limited(vertexes[MMCVIDI.x],
                vertexes[MMCVIDI.y],
                vertexes[MMCVIDI.z],
                vertexes[MMCVIDI.w],
                __GEIGEN__::__s_vec_multiply3(moveDir[MMCVIDI.x], -1),
                __GEIGEN__::__s_vec_multiply3(moveDir[MMCVIDI.y], -1),
                __GEIGEN__::__s_vec_multiply3(moveDir[MMCVIDI.z], -1),
                __GEIGEN__::__s_vec_multiply3(moveDir[MMCVIDI.w], -1), CCDDistRatio, 0, maxTime, maxIterations);

            temp = _ccd_step_reciprocal(temp1, maxTime);
        }
        else {
            temp = _ccd_step_reciprocal(edge_edge_ccd_limited(vertexes[MMCVIDI.x],
                vertexes[MMCVIDI.y],
                vertexes[MMCVIDI.z],
                vertexes[MMCVIDI.w],
                __GEIGEN__::__s_vec_multiply3(moveDir[MMCVIDI.x], -1),
                __GEIGEN__::__s_vec_multiply3(moveDir[MMCVIDI.y], -1),
                __GEIGEN__::__s_vec_multiply3(moveDir[MMCVIDI.z], -1),
                __GEIGEN__::__s_vec_multiply3(moveDir[MMCVIDI.w], -1), CCDDistRatio, 0, maxTime, maxIterations), maxTime);
        }
    }

    using BlockReduce = cub::BlockReduce<double, default_threads>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    double blockMax = BlockReduce(temp_storage).Reduce(temp, StepBoundMax());

    if (threadIdx.x == 0) {
        minStepSizes[blockIdx.x] = blockMax;
    }
}

__global__
void _reduct_min_selfTimeStepCompact_to_double(
    const double3* vertexes,
    const int2* _ccd_candidatePairs,
    const uint3* faces,
    const uint2* edges,
    const double3* moveDir,
    double* minStepSizes,
    double slackness,
    int number, double maxTime, int maxIterations) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    double temp = 0.0;
    if (idx < number) {
        double CCDDistRatio = 1.0 - slackness;
        int2 candidate = _ccd_candidatePairs[idx];

        if (candidate.x < 0) {
            int point = -candidate.x - 1;
            uint3 face = faces[candidate.y];

            double temp1 = point_triangle_ccd_limited(vertexes[point],
                vertexes[face.x],
                vertexes[face.y],
                vertexes[face.z],
                __GEIGEN__::__s_vec_multiply3(moveDir[point], -1),
                __GEIGEN__::__s_vec_multiply3(moveDir[face.x], -1),
                __GEIGEN__::__s_vec_multiply3(moveDir[face.y], -1),
                __GEIGEN__::__s_vec_multiply3(moveDir[face.z], -1), CCDDistRatio, 0, maxTime, maxIterations);

            temp = _ccd_step_reciprocal(temp1, maxTime);
        }
        else {
            uint2 edge0 = edges[candidate.x];
            uint2 edge1 = edges[candidate.y];

            temp = _ccd_step_reciprocal(edge_edge_ccd_limited(vertexes[edge0.x],
                vertexes[edge0.y],
                vertexes[edge1.x],
                vertexes[edge1.y],
                __GEIGEN__::__s_vec_multiply3(moveDir[edge0.x], -1),
                __GEIGEN__::__s_vec_multiply3(moveDir[edge0.y], -1),
                __GEIGEN__::__s_vec_multiply3(moveDir[edge1.x], -1),
                __GEIGEN__::__s_vec_multiply3(moveDir[edge1.y], -1), CCDDistRatio, 0, maxTime, maxIterations), maxTime);
        }
    }

    using BlockReduce = cub::BlockReduce<double, default_threads>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    double blockMax = BlockReduce(temp_storage).Reduce(temp, StepBoundMax());

    if (threadIdx.x == 0) {
        minStepSizes[blockIdx.x] = blockMax;
    }
}


__global__
void _reduct_max_double(double* _double1Dim, int number) {
    double temp = 0.0;
    for (int idx = threadIdx.x; idx < number; idx += blockDim.x) {
        temp = StepBoundMax()(temp, _double1Dim[idx]);
    }

    using BlockReduce = cub::BlockReduce<double, default_threads>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    double blockMax = BlockReduce(temp_storage).Reduce(temp, StepBoundMax());

    if (threadIdx.x == 0) {
        _double1Dim[0] = blockMax;
    }
}


// Most candidates finish within eight iterations. Complete those first so a
// few grazing pairs cannot hold up every GPU block while searching to t=1.
// If any are unfinished, F is a provisional bound from completed pairs.
// Recheck through F with eight fast iterations and at most 128 geometric
// interval tests. A work limit returns only a certified prefix, never F by default.
double self_largestFeasibleStepSize(
  double slackness,
  const double3* _vertexes,
  const int4* _ccd_collisonPairs,
  const double3* _moveDir,
  double* mqueue,
  int numbers) {
    if (numbers < 1) return 1.0;
    const int blockNum = 1 + (numbers - 1) / default_threads;
    double horizon = 1.0;
    for (int pass = 0; pass < 2; ++pass) {
        _reduct_min_selfTimeStep_to_double<<<blockNum, default_threads>>>(
            _vertexes, _ccd_collisonPairs, _moveDir, mqueue, slackness,
            numbers, horizon, pass == 0 ? 8 : 0);
        if (blockNum > 1)
            _reduct_max_double<<<1, default_threads>>>(mqueue, blockNum);
        double result;
        cudaMemcpy(&result, mqueue, sizeof(double), cudaMemcpyDeviceToHost);
        horizon = __m_min(horizon, 1.0 / fabs(result));
        if (result >= 0.0 || horizon == 0.0)
            return horizon;
    }
    return horizon;
}

double self_largestFeasibleStepSizeCompact(
  double slackness,
  const double3* _vertexes,
  const int2* _ccd_candidatePairs,
  const uint3* _faces,
  const uint2* _edges,
  const double3* _moveDir,
  double* mqueue,
  int numbers) {
    return self_largestFeasibleStepSizeCompactWithBound(slackness, _vertexes,
        _ccd_candidatePairs, _faces, _edges, _moveDir, mqueue, numbers, 1.0);
}

double self_largestFeasibleStepSizeCompactWithBound(
  double slackness,
  const double3* _vertexes,
  const int2* _ccd_candidatePairs,
  const uint3* _faces,
  const uint2* _edges,
  const double3* _moveDir,
  double* mqueue,
  int numbers,
  double maxTime) {
    if (!(maxTime > 0.0) || !isfinite(maxTime)) return 0.0;
    maxTime = __m_min(maxTime, 1.0);
    if (numbers < 1) return maxTime;
    const int blockNum = 1 + (numbers - 1) / default_threads;
    double horizon = maxTime;
    for (int pass = 0; pass < 2; ++pass) {
        _reduct_min_selfTimeStepCompact_to_double<<<blockNum, default_threads>>>(
            _vertexes, _ccd_candidatePairs, _faces, _edges, _moveDir, mqueue,
            slackness, numbers, horizon, pass == 0 ? 8 : 0);
        if (blockNum > 1)
            _reduct_max_double<<<1, default_threads>>>(mqueue, blockNum);
        double result;
        cudaMemcpy(&result, mqueue, sizeof(double), cudaMemcpyDeviceToHost);
        horizon = __m_min(horizon, 1.0 / fabs(result));
        if (result >= 0.0 || horizon == 0.0)
            return horizon;
    }
    return horizon;
}
}
