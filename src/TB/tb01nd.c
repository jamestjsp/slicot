/*
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "slicot.h"
#include "slicot_blas.h"

/**
 * @brief Reduce (A,C) pair to observer Hessenberg form.
 *
 * Reduces the pair (A,C) to lower or upper observer Hessenberg form
 * using (and optionally accumulating) unitary state-space transformations.
 *
 * The transformation is: A_out = U' * A * U, C_out = C * U
 *
 * @param[in] jobu 'N': don't form U; 'I': U = identity then accumulate;
 *                 'U': update given U
 * @param[in] uplo 'U': upper observer Hessenberg; 'L': lower
 * @param[in] n State dimension (order of A). N >= 0.
 * @param[in] p Output dimension (rows of C). 0 <= P <= N.
 * @param[in,out] a State matrix (lda, n). On exit: U' * A * U.
 * @param[in] lda Leading dimension of a. LDA >= max(1,N).
 * @param[in,out] c Output matrix (ldc, n). On exit: C * U.
 * @param[in] ldc Leading dimension of c. LDC >= max(1,P).
 * @param[in,out] u Transformation matrix (ldu, n).
 * @param[in] ldu Leading dimension of u.
 * @param[out] dwork Workspace of size max(N, P-1).
 * @param[out] info 0 = success, < 0 = -i means i-th argument invalid.
 */
void tb01nd(const char* jobu, const char* uplo, i32 n, i32 p,
            f64* a, i32 lda, f64* c, i32 ldc,
            f64* u, i32 ldu, f64* dwork, i32* info)
{
    const f64 dbl0 = 0.0;
    const f64 dbl1 = 1.0;

    bool ljoba, ljobi, luplo;
    i32 ii, j, n1, nj, p1;
    i32 par1, par2, par3, par4, par5, par6;
    f64 dz;

    *info = 0;
    luplo = (*uplo == 'U' || *uplo == 'u');
    ljobi = (*jobu == 'I' || *jobu == 'i');
    ljoba = ljobi || (*jobu == 'U' || *jobu == 'u');

    if (!ljoba && !(*jobu == 'N' || *jobu == 'n')) {
        *info = -1;
    } else if (!luplo && !(*uplo == 'L' || *uplo == 'l')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (p < 0 || p > n) {
        *info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldc < (p > 1 ? p : 1)) {
        *info = -8;
    } else if ((!ljoba && ldu < 1) || (ljoba && ldu < (n > 1 ? n : 1))) {
        *info = -10;
    }

    if (*info != 0) {
        return;
    }

    if (ljobi) {
        SLC_DLASET("Full", &n, &n, &dbl0, &dbl1, u, &ldu);
    }

    if (n == 0 || p == 0) {
        return;
    }

    p1 = p + 1;
    n1 = n - 1;

    // First loop: transformations involving both C and A
    // DO J = 1, MIN(P, N1)
    i32 min_pn1 = (p < n1) ? p : n1;
    for (j = 1; j <= min_pn1; j++) {
        nj = n - j;
        if (luplo) {
            // Upper observer Hessenberg
            par1 = p - j + 1;    // Row index in C
            par2 = nj + 1;       // Column index (first of reflector)
            par3 = 1;            // Column index (start of reflector vector)
            par4 = p - j;        // For updating remaining C
            par5 = nj;           // For zeroing elements in C
        } else {
            // Lower observer Hessenberg
            par1 = j;            // Row index in C
            par2 = j;            // Column index (first)
            par3 = j + 1;        // Column index (start of vector)
            par4 = p;            // For updating remaining C
            par5 = n;            // For zeroing elements in C
        }

        // Generate Householder reflector in row PAR1 of C
        // DLARFG operates on a row of C (columns PAR2 to N)
        // The reflector is stored in C(PAR1, PAR3:) with scalar DZ
        i32 nj1 = nj + 1;
        SLC_DLARFG(&nj1, &c[(par1 - 1) + (par2 - 1) * ldc],
                   &c[(par1 - 1) + (par3 - 1) * ldc], &ldc, &dz);

        // Update A from left: A = (I - tau * v * v') * A
        SLC_DLATZM("Left", &nj1, &n, &c[(par1 - 1) + (par3 - 1) * ldc], &ldc, &dz,
                   &a[(par2 - 1) + 0 * lda], &a[(par3 - 1) + 0 * lda], &lda, dwork);

        // Update A from right: A = A * (I - tau * v * v')
        SLC_DLATZM("Right", &n, &nj1, &c[(par1 - 1) + (par3 - 1) * ldc], &ldc, &dz,
                   &a[0 + (par2 - 1) * lda], &a[0 + (par3 - 1) * lda], &lda, dwork);

        if (ljoba) {
            // Update U from right: U = U * (I - tau * v * v')
            SLC_DLATZM("Right", &n, &nj1, &c[(par1 - 1) + (par3 - 1) * ldc], &ldc, &dz,
                       &u[0 + (par2 - 1) * ldu], &u[0 + (par3 - 1) * ldu], &ldu, dwork);
        }

        if (j != p) {
            // Update remaining rows of C
            i32 nrows = par4 - par3 + 1;
            SLC_DLATZM("Right", &nrows, &nj1, &c[(par1 - 1) + (par3 - 1) * ldc], &ldc, &dz,
                       &c[(par3 - 1) + (par2 - 1) * ldc], &c[(par3 - 1) + (par3 - 1) * ldc],
                       &ldc, dwork);
        }

        // Zero out the reflector entries in C
        for (ii = par3; ii <= par5; ii++) {
            c[(par1 - 1) + (ii - 1) * ldc] = dbl0;
        }
    }

    // Second loop: transformations only involving A (when P < N-1)
    // DO J = P1, N1
    for (j = p1; j <= n1; j++) {
        nj = n - j;
        if (luplo) {
            // Upper observer Hessenberg
            par1 = n + p1 - j;   // Row index in A
            par2 = nj + 1;       // Column index
            par3 = 1;            // Start of reflector vector
            par4 = nj;           // For zeroing
            par5 = 1;            // For DLATZM on A rows
            par6 = n + p - j;    // For DLATZM on A rows
        } else {
            // Lower observer Hessenberg
            par1 = j - p;        // Row index in A
            par2 = j;            // Column index
            par3 = j + 1;        // Start of reflector vector
            par4 = n;            // For zeroing
            par5 = j - p + 1;    // For DLATZM on A rows
            par6 = n;            // For DLATZM on A rows
        }

        if (nj > 0) {
            // Generate Householder reflector in row PAR1 of A
            i32 nj1 = nj + 1;
            SLC_DLARFG(&nj1, &a[(par1 - 1) + (par2 - 1) * lda],
                       &a[(par1 - 1) + (par3 - 1) * lda], &lda, &dz);

            // Update A from left
            SLC_DLATZM("Left", &nj1, &n, &a[(par1 - 1) + (par3 - 1) * lda], &lda, &dz,
                       &a[(par2 - 1) + 0 * lda], &a[(par3 - 1) + 0 * lda], &lda, dwork);

            // Update A from right (only relevant rows)
            i32 nrows = par6 - par5 + 1;
            SLC_DLATZM("Right", &nrows, &nj1, &a[(par1 - 1) + (par3 - 1) * lda], &lda, &dz,
                       &a[(par5 - 1) + (par2 - 1) * lda], &a[(par5 - 1) + (par3 - 1) * lda],
                       &lda, dwork);

            if (ljoba) {
                // Update U from right
                SLC_DLATZM("Right", &n, &nj1, &a[(par1 - 1) + (par3 - 1) * lda], &lda, &dz,
                           &u[0 + (par2 - 1) * ldu], &u[0 + (par3 - 1) * ldu], &ldu, dwork);
            }

            // Zero out the reflector entries in A
            for (ii = par3; ii <= par4; ii++) {
                a[(par1 - 1) + (ii - 1) * lda] = dbl0;
            }
        }
    }
}
