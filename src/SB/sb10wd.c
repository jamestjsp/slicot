/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB10WD - H2 optimal controller matrices from state feedback and output injection
 *
 * Computes controller matrices (AK, BK, CK, DK) from state feedback matrix F
 * and output injection matrix H as determined by SB10VD.
 */

#include "slicot.h"
#include "slicot_blas.h"

void sb10wd(
    const i32 n,
    const i32 m,
    const i32 np,
    const i32 ncon,
    const i32 nmeas,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    const f64* d,
    const i32 ldd,
    const f64* f,
    const i32 ldf,
    const f64* h,
    const i32 ldh,
    const f64* tu,
    const i32 ldtu,
    const f64* ty,
    const i32 ldty,
    f64* ak,
    const i32 ldak,
    f64* bk,
    const i32 ldbk,
    f64* ck,
    const i32 ldck,
    f64* dk,
    const i32 lddk,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 negone = -1.0;

    i32 m1 = m - ncon;
    i32 m2 = ncon;
    i32 np1 = np - nmeas;
    i32 np2 = nmeas;

    *info = 0;

    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (np < 0) {
        *info = -3;
    } else if (ncon < 0 || m1 < 0 || m2 > np1) {
        *info = -4;
    } else if (nmeas < 0 || np1 < 0 || np2 > m1) {
        *info = -5;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -9;
    } else if (ldc < (np > 1 ? np : 1)) {
        *info = -11;
    } else if (ldd < (np > 1 ? np : 1)) {
        *info = -13;
    } else if (ldf < (m2 > 1 ? m2 : 1)) {
        *info = -15;
    } else if (ldh < (n > 1 ? n : 1)) {
        *info = -17;
    } else if (ldtu < (m2 > 1 ? m2 : 1)) {
        *info = -19;
    } else if (ldty < (np2 > 1 ? np2 : 1)) {
        *info = -21;
    } else if (ldak < (n > 1 ? n : 1)) {
        *info = -23;
    } else if (ldbk < (n > 1 ? n : 1)) {
        *info = -25;
    } else if (ldck < (m2 > 1 ? m2 : 1)) {
        *info = -27;
    } else if (lddk < (m2 > 1 ? m2 : 1)) {
        *info = -29;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || m == 0 || np == 0 || m1 == 0 || m2 == 0 ||
        np1 == 0 || np2 == 0) {
        return;
    }

    // Compute transpose of D22*F. BK is used as workspace.
    // D22 = D(NP1+1:NP, M1+1:M) -> d + np1 + m1*ldd
    // Result: BK = (D22 * F)^T = F^T * D22^T
    SLC_DGEMM("T", "T", &n, &np2, &m2, &one,
              f, &ldf,
              d + np1 + m1 * ldd, &ldd,
              &zero, bk, &ldbk);

    // AK = A (copy)
    SLC_DLACPY("F", &n, &n, a, &lda, ak, &ldak);

    // AK = AK + H * C2, where C2 = C(NP1+1:NP, :) -> c + np1
    SLC_DGEMM("N", "N", &n, &n, &np2, &one,
              h, &ldh,
              c + np1, &ldc,
              &one, ak, &ldak);

    // AK = AK + B2 * F, where B2 = B(:, M1+1:M) -> b + m1*ldb
    SLC_DGEMM("N", "N", &n, &n, &m2, &one,
              b + m1 * ldb, &ldb,
              f, &ldf,
              &one, ak, &ldak);

    // AK = AK + H * (D22*F)^T = AK + H * BK^T
    SLC_DGEMM("N", "T", &n, &n, &np2, &one,
              h, &ldh,
              bk, &ldbk,
              &one, ak, &ldak);

    // BK = -H * TY
    SLC_DGEMM("N", "N", &n, &np2, &np2, &negone,
              h, &ldh,
              ty, &ldty,
              &zero, bk, &ldbk);

    // CK = TU * F
    SLC_DGEMM("N", "N", &m2, &n, &m2, &one,
              tu, &ldtu,
              f, &ldf,
              &zero, ck, &ldck);

    // DK = 0
    SLC_DLASET("F", &m2, &np2, &zero, &zero, dk, &lddk);
}
