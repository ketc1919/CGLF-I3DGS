//
// The P3P solver in this file is a CUDA port of the Lambda Twist P3P solver
// from Viktor Larsson's reference implementation, used under the BSD 3-Clause
// License (reproduced below):
//   https://github.com/vlarsson/lambdatwist  (lambdatwist/p3p.cc)
// Algorithm: M. Persson and K. Nordberg, "Lambda Twist: An Accurate Fast Robust
// Perspective Three Point (P3P) Solver", ECCV 2018.
//
// Changes from the original: ported from Eigen to CUDA device code (the
// polynomial / eigen / depth recovery is kept in double precision, the returned
// geometry is single precision), and the returned pose is the world-to-camera
// [R | t] such that lambda * x = R * X + t with positive depth lambda.
//
// This BSD-3-Clause license covers only the P3P solver (p3p_solve_device and its
// helpers). The RANSAC harness kernels below (batchP3P, p3p_select_best,
// p3p_inliers_single) are based on OpenCV's usac module
// (https://github.com/opencv/opencv/tree/4.x/modules/calib3d/src/usac), used
// under the Apache License 2.0 (a copy is provided in licenses/Apache-2.0.txt),
// as with poses/ransac.cu.
//
// ---------------------------------------------------------------------------
// Copyright (c) 2020, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include "cuda_utils.cuh"

// ---------------------------------------------------------------------------
// Minimal double-precision vector / 3x3 matrix helpers, local to this file.
// Lambda Twist is numerically sensitive, so the solver runs in double and only
// the final pose is cast down to the single-precision Matx33f / float3 output.
// ---------------------------------------------------------------------------
namespace {

struct Vec3d { double x, y, z; };

__device__ inline Vec3d make_vec3d(const float3& v) { return {(double)v.x, (double)v.y, (double)v.z}; }
__device__ inline Vec3d operator-(Vec3d a, Vec3d b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__device__ inline Vec3d operator*(double s, Vec3d a) { return {s * a.x, s * a.y, s * a.z}; }

__device__ inline double dotd(Vec3d a, Vec3d b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline double sqnormd(Vec3d a) { return dotd(a, a); }
__device__ inline Vec3d crossd(Vec3d a, Vec3d b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// Row-major 3x3 double matrix.
struct Mat3d {
    double m[3][3];
    __device__ inline double& operator()(int i, int j) { return m[i][j]; }
    __device__ inline double operator()(int i, int j) const { return m[i][j]; }
};

// Build a matrix from its three columns.
__device__ inline Mat3d cols(Vec3d c0, Vec3d c1, Vec3d c2) {
    Mat3d A;
    A.m[0][0] = c0.x; A.m[0][1] = c1.x; A.m[0][2] = c2.x;
    A.m[1][0] = c0.y; A.m[1][1] = c1.y; A.m[1][2] = c2.y;
    A.m[2][0] = c0.z; A.m[2][1] = c1.z; A.m[2][2] = c2.z;
    return A;
}

__device__ inline Vec3d col(const Mat3d& A, int j) { return {A.m[0][j], A.m[1][j], A.m[2][j]}; }

// Sum of the element-wise product of two matrices (Eigen: (A.array()*B.array()).sum()).
__device__ inline double elemwise_dot(const Mat3d& A, const Mat3d& B) {
    double s = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            s += A.m[i][j] * B.m[i][j];
    return s;
}

__device__ inline Mat3d matmul(const Mat3d& A, const Mat3d& B) {
    Mat3d R;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R.m[i][j] = A.m[i][0] * B.m[0][j] + A.m[i][1] * B.m[1][j] + A.m[i][2] * B.m[2][j];
    return R;
}

__device__ inline Vec3d matvec(const Mat3d& A, Vec3d v) {
    return {A.m[0][0] * v.x + A.m[0][1] * v.y + A.m[0][2] * v.z,
            A.m[1][0] * v.x + A.m[1][1] * v.y + A.m[1][2] * v.z,
            A.m[2][0] * v.x + A.m[2][1] * v.y + A.m[2][2] * v.z};
}

// Inverse via cofactors (double precision).
__device__ inline Mat3d inv3(const Mat3d& A) {
    double a = A.m[0][0], b = A.m[0][1], c = A.m[0][2];
    double d = A.m[1][0], e = A.m[1][1], f = A.m[1][2];
    double g = A.m[2][0], h = A.m[2][1], i = A.m[2][2];
    double A00 = e * i - f * h;
    double A01 = -(d * i - f * g);
    double A02 = d * h - e * g;
    double det = a * A00 + b * A01 + c * A02;
    double inv_det = 1.0 / det;
    Mat3d R;
    R.m[0][0] = A00 * inv_det;
    R.m[0][1] = (c * h - b * i) * inv_det;
    R.m[0][2] = (b * f - c * e) * inv_det;
    R.m[1][0] = A01 * inv_det;
    R.m[1][1] = (a * i - c * g) * inv_det;
    R.m[1][2] = (c * d - a * f) * inv_det;
    R.m[2][0] = A02 * inv_det;
    R.m[2][1] = (b * g - a * h) * inv_det;
    R.m[2][2] = (a * e - b * d) * inv_det;
    return R;
}

// ---------------------------------------------------------------------------
// Eigen decomposition of a symmetric 3x3 matrix with a known zero eigenvalue.
// (from lambdatwist compute_eig3x3known0; only the first two eigenvectors are
// needed, so the third column of E is left unset.)
// ---------------------------------------------------------------------------
__device__ inline void compute_eig3x3known0(const Mat3d& M, Mat3d& E, double& sig1, double& sig2) {
    // In the original paper there is a missing minus sign here (for M(0,0)).
    double p1 = -M(0, 0) - M(1, 1) - M(2, 2);
    double p0 = -M(0, 1) * M(0, 1) - M(0, 2) * M(0, 2) - M(1, 2) * M(1, 2) +
                M(0, 0) * (M(1, 1) + M(2, 2)) + M(1, 1) * M(2, 2);

    double disc = sqrt(p1 * p1 / 4.0 - p0);
    double tmp = -p1 / 2.0;
    sig1 = tmp + disc;
    sig2 = tmp - disc;

    if (fabs(sig1) < fabs(sig2)) {
        double t = sig1; sig1 = sig2; sig2 = t;
    }

    double c = sig1 * sig1 + M(0, 0) * M(1, 1) - sig1 * (M(0, 0) + M(1, 1)) - M(0, 1) * M(0, 1);
    double a1 = (sig1 * M(0, 2) + M(0, 1) * M(1, 2) - M(0, 2) * M(1, 1)) / c;
    double a2 = (sig1 * M(1, 2) + M(0, 1) * M(0, 2) - M(0, 0) * M(1, 2)) / c;
    double n = 1.0 / sqrt(1.0 + a1 * a1 + a2 * a2);
    E(0, 0) = a1 * n; E(1, 0) = a2 * n; E(2, 0) = n;

    c = sig2 * sig2 + M(0, 0) * M(1, 1) - sig2 * (M(0, 0) + M(1, 1)) - M(0, 1) * M(0, 1);
    a1 = (sig2 * M(0, 2) + M(0, 1) * M(1, 2) - M(0, 2) * M(1, 1)) / c;
    a2 = (sig2 * M(1, 2) + M(0, 1) * M(0, 2) - M(0, 0) * M(1, 2)) / c;
    n = 1.0 / sqrt(1.0 + a1 * a1 + a2 * a2);
    E(0, 1) = a1 * n; E(1, 1) = a2 * n; E(2, 1) = n;
}

// ---------------------------------------------------------------------------
// A few Newton steps refining the three depths (from lambdatwist refine_lambda).
// ---------------------------------------------------------------------------
__device__ inline void refine_lambda(double& lambda1, double& lambda2, double& lambda3,
                                     const double a12, const double a13, const double a23,
                                     const double b12, const double b13, const double b23) {
    for (int iter = 0; iter < 5; ++iter) {
        double r1 = (lambda1 * lambda1 - 2.0 * lambda1 * lambda2 * b12 + lambda2 * lambda2 - a12);
        double r2 = (lambda1 * lambda1 - 2.0 * lambda1 * lambda3 * b13 + lambda3 * lambda3 - a13);
        double r3 = (lambda2 * lambda2 - 2.0 * lambda2 * lambda3 * b23 + lambda3 * lambda3 - a23);
        if (fabs(r1) + fabs(r2) + fabs(r3) < 1e-10)
            return;
        double x11 = lambda1 - lambda2 * b12, x12 = lambda2 - lambda1 * b12;
        double x21 = lambda1 - lambda3 * b13, x23 = lambda3 - lambda1 * b13;
        double x32 = lambda2 - lambda3 * b23, x33 = lambda3 - lambda2 * b23;
        double detJ = 0.5 / (x11 * x23 * x32 + x12 * x21 * x33); // half minus inverse determinant
        // Closed form of the (sparse) Jacobian inverse.
        lambda1 += (-x23 * x32 * r1 - x12 * x33 * r2 + x12 * x23 * r3) * detJ;
        lambda2 += (-x21 * x33 * r1 + x11 * x33 * r2 - x11 * x23 * r3) * detJ;
        lambda3 += (x21 * x32 * r1 - x11 * x32 * r2 - x12 * x21 * r3) * detJ;
    }
}

__device__ inline bool finite_pose(const Mat3d& R, Vec3d t) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (!isfinite(R.m[i][j])) return false;
    return isfinite(t.x) && isfinite(t.y) && isfinite(t.z);
}

} // namespace

// ---------------------------------------------------------------------------
// p3p_solve_device (Lambda Twist, from lambdatwist p3p)
//
// Pw: 3 world points. bc: 3 unit bearing vectors K^-1 x / ||K^-1 x||.
// Writes up to 4 world-to-camera (R, t) solutions such that
// lambda * bc = R * Pw + t with positive depth. Returns the solution count.
// ---------------------------------------------------------------------------
__device__ inline int p3p_solve_device(
    const float3 Pw[3], const float3 bc[3], Matx33f outR[4], float3 outT[4])
{
    const Vec3d X0 = make_vec3d(Pw[0]);
    const Vec3d X1 = make_vec3d(Pw[1]);
    const Vec3d X2 = make_vec3d(Pw[2]);
    const Vec3d x0 = make_vec3d(bc[0]);
    const Vec3d x1 = make_vec3d(bc[1]);
    const Vec3d x2 = make_vec3d(bc[2]);

    const Vec3d dX12 = X0 - X1;
    const Vec3d dX13 = X0 - X2;
    const Vec3d dX23 = X1 - X2;

    const double a12 = sqnormd(dX12);
    const double b12 = dotd(x0, x1);
    const double a13 = sqnormd(dX13);
    const double b13 = dotd(x0, x2);
    const double a23 = sqnormd(dX23);
    const double b23 = dotd(x1, x2);

    const double a23b12 = a23 * b12;
    const double a12b23 = a12 * b23;
    const double a23b13 = a23 * b13;
    const double a13b23 = a13 * b23;

    Mat3d D1, D2;
    D1(0, 0) = a23;     D1(0, 1) = -a23b12;    D1(0, 2) = 0.0;
    D1(1, 0) = -a23b12; D1(1, 1) = a23 - a12;  D1(1, 2) = a12b23;
    D1(2, 0) = 0.0;     D1(2, 1) = a12b23;     D1(2, 2) = -a12;

    D2(0, 0) = a23;     D2(0, 1) = 0.0;        D2(0, 2) = -a23b13;
    D2(1, 0) = 0.0;     D2(1, 1) = -a13;       D2(1, 2) = a13b23;
    D2(2, 0) = -a23b13; D2(2, 1) = a13b23;     D2(2, 2) = a23 - a13;

    const Mat3d DX1 = cols(crossd(col(D1, 1), col(D1, 2)),
                           crossd(col(D1, 2), col(D1, 0)),
                           crossd(col(D1, 0), col(D1, 1)));
    const Mat3d DX2 = cols(crossd(col(D2, 1), col(D2, 2)),
                           crossd(col(D2, 2), col(D2, 0)),
                           crossd(col(D2, 0), col(D2, 1)));

    // Coefficients of p(gamma) = det(D1 + gamma*D2).
    double c3 = dotd(col(D2, 0), col(DX2, 0));
    double c2 = elemwise_dot(D1, DX2);
    double c1 = elemwise_dot(D2, DX1);
    double c0 = dotd(col(D1, 0), col(DX1, 0));

    // Closed-form cubic root solver.
    const double c3inv = 1.0 / c3;
    c2 *= c3inv; c1 *= c3inv; c0 *= c3inv;

    double ca = c1 - c2 * c2 / 3.0;
    double cb = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
    double cc = cb * cb / 4.0 + ca * ca * ca / 27.0;
    double gamma;
    if (cc > 0) {
        cc = sqrt(cc);
        cb *= -0.5;
        gamma = cbrt(cb + cc) + cbrt(cb - cc) - c2 / 3.0;
    } else {
        cc = 3.0 * cb / (2.0 * ca) * sqrt(-3.0 / ca);
        gamma = 2.0 * sqrt(-ca / 3.0) * cos(acos(cc) / 3.0) - c2 / 3.0;
    }

    // A single Newton step on the cubic equation.
    double f = gamma * gamma * gamma + c2 * gamma * gamma + c1 * gamma + c0;
    double df = 3.0 * gamma * gamma + 2.0 * c2 * gamma + c1;
    gamma = gamma - f / df;

    Mat3d D0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            D0(i, j) = D1(i, j) + gamma * D2(i, j);

    Mat3d E;
    double sig1, sig2;
    compute_eig3x3known0(D0, E, sig1, sig2);

    double s = sqrt(-sig2 / sig1);

    Mat3d XX = inv3(cols(dX12, dX13, crossd(dX12, dX13)));

    const double TOL_DOUBLE_ROOT = 1e-12;

    int outcount = 0;
    for (int s_flip = 0; s_flip < 2; ++s_flip, s = -s) {
        // [u1 u2 u3] * [lambda1; lambda2; lambda3] = 0
        double u1 = E(0, 0) - s * E(0, 1);
        double u2 = E(1, 0) - s * E(1, 1);
        double u3 = E(2, 0) - s * E(2, 1);

        // Depending on which is larger we solve for either lambda1 or lambda2.
        // The case u1 = u2 = 0 is degenerate and can be ignored.
        bool switch_12 = fabs(u1) < fabs(u2);

        double qa, qb, qc, w0, w1;
        if (switch_12) {
            // solve for lambda2
            w0 = -u1 / u2;
            w1 = -u3 / u2;
            qa = -a13 * w1 * w1 + 2.0 * a13b23 * w1 - a13 + a23;
            qb = 2.0 * a13b23 * w0 - 2.0 * a23b13 - 2.0 * a13 * w0 * w1;
            qc = -a13 * w0 * w0 + a23;
        } else {
            // solve for lambda1 (default case in the paper)
            w0 = -u2 / u1;
            w1 = -u3 / u1;
            qa = (a13 - a12) * w1 * w1 + 2.0 * a12 * b13 * w1 - a12;
            qb = -2.0 * a13 * b12 * w1 + 2.0 * a12 * b13 * w0 - 2.0 * w0 * w1 * (a12 - a13);
            qc = (a13 - a12) * w0 * w0 - 2.0 * a13 * b12 * w0 + a13;
        }

        double b2m4ac = qb * qb - 4.0 * qa * qc;
        // If b2m4ac is (near) zero we have a double root; allow slightly
        // negative discriminants and clip to zero.
        if (b2m4ac < -TOL_DOUBLE_ROOT)
            continue;
        double sq = sqrt(fmax(0.0, b2m4ac));
        double tau = (qb > 0) ? (2.0 * qc) / (-qb - sq) : (2.0 * qc) / (-qb + sq);

        for (int tau_flip = 0; tau_flip < 2; ++tau_flip, tau = qc / (qa * tau)) {
            if (tau > 0) {
                double lambda1, lambda2, lambda3;
                if (switch_12) {
                    lambda1 = sqrt(a13 / (tau * (tau - 2.0 * b13) + 1.0));
                    lambda3 = tau * lambda1;
                    lambda2 = w0 * lambda1 + w1 * lambda3;
                    // tau > 0 and lambda1 > 0, so only lambda2 needs checking.
                    if (lambda2 < 0) {
                        if (b2m4ac < TOL_DOUBLE_ROOT) break;
                        continue;
                    }
                } else {
                    lambda2 = sqrt(a23 / (tau * (tau - 2.0 * b23) + 1.0));
                    lambda3 = tau * lambda2;
                    lambda1 = w0 * lambda2 + w1 * lambda3;
                    if (lambda1 < 0) {
                        if (b2m4ac < TOL_DOUBLE_ROOT) break;
                        continue;
                    }
                }

                refine_lambda(lambda1, lambda2, lambda3, a12, a13, a23, b12, b13, b23);
                Vec3d v1 = lambda1 * x0 - lambda2 * x1;
                Vec3d v2 = lambda1 * x0 - lambda3 * x2;
                Mat3d YY = cols(v1, v2, crossd(v1, v2));
                Mat3d R = matmul(YY, XX);
                Vec3d t = lambda1 * x0 - matvec(R, X0);

                if (finite_pose(R, t) && outcount < 4) {
                    outR[outcount] = Matx33f(
                        (float)R(0, 0), (float)R(0, 1), (float)R(0, 2),
                        (float)R(1, 0), (float)R(1, 1), (float)R(1, 2),
                        (float)R(2, 0), (float)R(2, 1), (float)R(2, 2));
                    outT[outcount] = make_float3((float)t.x, (float)t.y, (float)t.z);
                    outcount++;
                }
            }

            if (b2m4ac < TOL_DOUBLE_ROOT) {
                // Double root: skip the second tau.
                break;
            }
        }
    }

    return outcount;
}

// RANSAC harness kernels below are based on OpenCV's usac module, used under
// the Apache License 2.0 (see the file header and licenses/Apache-2.0.txt).

// ---------------------------------------------------------------------------
// Kernel: batchP3P
// ---------------------------------------------------------------------------
extern "C" __global__ void batchP3P(
    const float3* cam_rays,
    const float3* Pw_world,
    Matx33f* Rs,           // preallocated length batchSize * 4
    float3* Ts,            // preallocated length batchSize * 4
    int* solCounts,        // length batchSize
    bool* isValid,         // length batchSize
    int batchSize,
    int n_pairs
) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= batchSize) return;

    // Pick 3 distinct indices in [0,N-1] based on index
    int indices[3];
    pickDistinctIndices(indices, 3, n_pairs, batch + 1);
    int i0 = indices[0];
    int i1 = indices[1];
    int i2 = indices[2];

    float3 Pw[3];
    Pw[0] = Pw_world[i0];
    Pw[1] = Pw_world[i1];
    Pw[2] = Pw_world[i2];

    float3 bc[3];
    bc[0] = normalize(cam_rays[i0]);
    bc[1] = normalize(cam_rays[i1]);
    bc[2] = normalize(cam_rays[i2]);

    // Basic degeneracy check: area of world triangle
    float3 v01 = Pw[1] - Pw[0];
    float3 v02 = Pw[2] - Pw[0];
    float3 tri_n = cross(v01, v02);
    float area2 = norm(tri_n);
    const float EPS_AREA = 1e-8f;
    if (area2 < EPS_AREA) {
        solCounts[batch] = 0;
        isValid[batch] = false;
        for (int k = 0; k < 4; k++) {
            Rs[batch*4 + k] = Matx33f();  // Zero matrix
            Ts[batch*4 + k] = make_float3(0.0f, 0.0f, 0.0f);
        }
        return;
    }

    Matx33f outR[4];
    float3 outT[4];
    int nsol = p3p_solve_device(Pw, bc, outR, outT);

    solCounts[batch] = nsol;
    isValid[batch] = (nsol > 0);
    for (int s = 0; s < 4; ++s) {
        int idx = batch*4 + s;
        if (s < nsol) {
            Rs[idx] = outR[s];
            Ts[idx] = outT[s];
        } else {
            Rs[idx] = Matx33f();  // Zero matrix
            Ts[idx] = make_float3(0.0f, 0.0f, 0.0f);
        }
    }
}

// ---------------------------------------------------------------------------
// p3p_select_best
//
// One block per hypothesis (N blocks total), 128 threads = 4 warps.
// Warp w (threadIdx.x/32) evaluates solution (hyp*4+w): each lane strides
// over all n_pts points, counts inliers, then warp-reduces with shfl.
// Lane 0 writes to shared memory; thread 0 picks the argmax and writes the
// winning (R,t,count) to best_Rs[hyp], best_Ts[hyp], best_counts[hyp].
// ---------------------------------------------------------------------------
#ifdef __HIPCC__
#define WARP_SHFL_DOWN(val, offset) __shfl_down(val, offset)
#else
#define WARP_SHFL_DOWN(val, offset) __shfl_down_sync(0xffffffff, val, offset)
#endif

extern "C" __global__ void p3p_select_best(
    int             n_pts,
    const float3*   pts_world,      // [n_pts]  world 3-D points
    const float2*   pts2d,          // [n_pts]  2-D pixel observations
    const Matx33f*  Rs,             // [N*4]    rotation matrices from batchP3P
    const float3*   Ts,             // [N*4]    translation vectors
    float           focal,          // isotropic focal length
    float           cx,             // principal point x
    float           cy,             // principal point y
    float           max_error_sq,   // reprojection threshold (pixels^2)
    Matx33f*        best_Rs,        // [N]  output: best R per hypothesis
    float3*         best_Ts,        // [N]  output: best T per hypothesis
    int*            best_counts,    // [N]  output: best inlier count
    int             N
) {
    int hyp     = blockIdx.x;
    if (hyp >= N) return;

    int warp_id = threadIdx.x >> 5;   // 0..3
    int lane    = threadIdx.x & 31;

    int sol_idx = hyp * 4 + warp_id;

    Matx33f R = Rs[sol_idx];
    float3  T = Ts[sol_idx];

    // Degenerate/unused solution: all-zero rotation diagonal
    bool valid_sol = (R(0,0) != 0.0f || R(1,1) != 0.0f || R(2,2) != 0.0f);

    int my_count = 0;
    if (valid_sol) {
        for (int j = lane; j < n_pts; j += 32) {
            float3 pw  = pts_world[j];
            float  cx_ = R(0,0)*pw.x + R(0,1)*pw.y + R(0,2)*pw.z + T.x;
            float  cy_ = R(1,0)*pw.x + R(1,1)*pw.y + R(1,2)*pw.z + T.y;
            float  cz_ = R(2,0)*pw.x + R(2,1)*pw.y + R(2,2)*pw.z + T.z;
            if (cz_ > 0.0f) {
                float inv_z = focal / cz_;
                float px    = cx_ * inv_z + cx;
                float py    = cy_ * inv_z + cy;
                float2 p2d  = pts2d[j];
                float  dx   = px - p2d.x;
                float  dy   = py - p2d.y;
                if (dx*dx + dy*dy < max_error_sq) my_count++;
            }
        }
    }

    // Warp reduction: sum inlier counts across the 32 lanes
    for (int offset = 16; offset > 0; offset >>= 1)
        my_count += WARP_SHFL_DOWN(my_count, offset);

    __shared__ int shm_counts[4];
    if (lane == 0) shm_counts[warp_id] = my_count;
    __syncthreads();

    // Thread 0 picks argmax and writes outputs
    if (threadIdx.x == 0) {
        int best = 0;
        for (int s = 1; s < 4; s++) {
            if (shm_counts[s] > shm_counts[best]) best = s;
        }
        best_Rs    [hyp] = Rs[hyp * 4 + best];
        best_Ts    [hyp] = Ts[hyp * 4 + best];
        best_counts[hyp] = shm_counts[best];
    }
}

// ---------------------------------------------------------------------------
// p3p_inliers_single
//
// One thread per point.  Computes reprojection error for a single (R,t) and
// writes a bool inlier mask -- used for the final mask of the best model.
// ---------------------------------------------------------------------------
extern "C" __global__ void p3p_inliers_single(
    int             n_pts,
    const float3*   pts_world,      // [n_pts]
    const float2*   pts2d,          // [n_pts]
    const Matx33f*  R_ptr,          // [1]  single rotation
    const float3*   T_ptr,          // [1]  single translation
    float           focal,
    float           cx,
    float           cy,
    float           max_error_sq,
    bool*           inliers         // [n_pts]  output
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pts) return;

    Matx33f R  = *R_ptr;
    float3  T  = *T_ptr;
    float3  pw = pts_world[i];

    float cx_ = R(0,0)*pw.x + R(0,1)*pw.y + R(0,2)*pw.z + T.x;
    float cy_ = R(1,0)*pw.x + R(1,1)*pw.y + R(1,2)*pw.z + T.y;
    float cz_ = R(2,0)*pw.x + R(2,1)*pw.y + R(2,2)*pw.z + T.z;

    bool is_inlier = false;
    if (cz_ > 0.0f) {
        float inv_z = focal / cz_;
        float px    = cx_ * inv_z + cx;
        float py    = cy_ * inv_z + cy;
        float2 p2d  = pts2d[i];
        float  dx   = px - p2d.x;
        float  dy   = py - p2d.y;
        is_inlier   = (dx*dx + dy*dy < max_error_sq);
    }
    inliers[i] = is_inlier;
}
