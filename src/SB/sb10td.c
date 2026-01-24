/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB10TD - H2 optimal discrete-time controller transformation
 *
 * Computes the matrices of the H2 optimal discrete-time controller:
 *
 *          | AK | BK |
 *      K = |----|----|
 *          | CK | DK |
 *
 * from the matrices of the controller for the normalized system,
 * as determined by SB10SD.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <math.h>

void sb10td(
    const i32 n,
    const i32 m,
    const i32 np,
    const i32 ncon,
    const i32 nmeas,
    const f64* d,
    const i32 ldd,
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
    f64* rcond,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

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
    } else if (ldd < (np > 1 ? np : 1)) {
        *info = -7;
    } else if (ldtu < (m2 > 1 ? m2 : 1)) {
        *info = -9;
    } else if (ldty < (np2 > 1 ? np2 : 1)) {
        *info = -11;
    } else if (ldak < (n > 1 ? n : 1)) {
        *info = -13;
    } else if (ldbk < (n > 1 ? n : 1)) {
        *info = -15;
    } else if (ldck < (m2 > 1 ? m2 : 1)) {
        *info = -17;
    } else if (lddk < (m2 > 1 ? m2 : 1)) {
        *info = -19;
    } else {
        i32 t1 = n * m2;
        i32 t2 = n * np2;
        i32 t3 = m2 * np2;
        i32 t4 = m2 * m2 + 4 * m2;
        i32 minwrk = t1;
        if (t2 > minwrk) minwrk = t2;
        if (t3 > minwrk) minwrk = t3;
        if (t4 > minwrk) minwrk = t4;
        if (minwrk < 1) minwrk = 1;

        if (ldwork < minwrk) {
            *info = -24;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || m == 0 || np == 0 || m1 == 0 || m2 == 0 || np1 == 0 || np2 == 0) {
        *rcond = one;
        return;
    }

    f64 toll = tol;
    if (toll <= zero) {
        toll = sqrt(SLC_DLAMCH("Epsilon"));
    }

    // BKHAT = BK * TY
    SLC_DGEMM("N", "N", &n, &np2, &np2, &one, bk, &ldbk, ty, &ldty, &zero, dwork, &n);
    SLC_DLACPY("Full", &n, &np2, dwork, &n, bk, &ldbk);

    // CKHAT = TU * CK
    SLC_DGEMM("N", "N", &m2, &n, &m2, &one, tu, &ldtu, ck, &ldck, &zero, dwork, &m2);
    SLC_DLACPY("Full", &m2, &n, dwork, &m2, ck, &ldck);

    // DKHAT = TU * DK * TY
    SLC_DGEMM("N", "N", &m2, &np2, &m2, &one, tu, &ldtu, dk, &lddk, &zero, dwork, &m2);
    SLC_DGEMM("N", "N", &m2, &np2, &np2, &one, dwork, &m2, ty, &ldty, &zero, dk, &lddk);

    // Compute Im2 + DKHAT*D22
    i32 iwrk = m2 * m2;
    SLC_DLASET("Full", &m2, &m2, &zero, &one, dwork, &m2);
    SLC_DGEMM("N", "N", &m2, &m2, &np2, &one, dk, &lddk, &d[np1 + m1 * ldd], &ldd, &one, dwork, &m2);

    // Compute 1-norm of (Im2 + DKHAT*D22)
    f64 anorm = SLC_DLANGE("1", &m2, &m2, dwork, &m2, &dwork[iwrk]);

    // LU factorization
    i32 info2;
    SLC_DGETRF(&m2, &m2, dwork, &m2, iwork, &info2);
    if (info2 > 0) {
        *info = 1;
        return;
    }

    // Estimate condition number
    SLC_DGECON("1", &m2, dwork, &m2, &anorm, rcond, &dwork[iwrk], &iwork[m2], &info2);

    if (*rcond < toll) {
        *info = 1;
        return;
    }

    // CK = inv(Im2 + DKHAT*D22) * CKHAT
    SLC_DGETRS("N", &m2, &n, dwork, &m2, iwork, ck, &ldck, &info2);

    // DK = inv(Im2 + DKHAT*D22) * DKHAT
    SLC_DGETRS("N", &m2, &np2, dwork, &m2, iwork, dk, &lddk, &info2);

    // AK = AK - BKHAT * D22 * CK
    // First compute temp = BKHAT * D22 in dwork
    SLC_DGEMM("N", "N", &n, &m2, &np2, &one, bk, &ldbk, &d[np1 + m1 * ldd], &ldd, &zero, dwork, &n);
    // Then AK = AK - temp * CK
    f64 minusone = -one;
    SLC_DGEMM("N", "N", &n, &n, &m2, &minusone, dwork, &n, ck, &ldck, &one, ak, &ldak);

    // BK = BKHAT - BKHAT * D22 * DK
    // dwork still contains BKHAT * D22
    SLC_DGEMM("N", "N", &n, &np2, &m2, &minusone, dwork, &n, dk, &lddk, &one, bk, &ldbk);
}
