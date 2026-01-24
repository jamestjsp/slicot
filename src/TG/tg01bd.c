// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"

void tg01bd(const char *jobe, const char *compq, const char *compz,
            i32 n, i32 m, i32 p, i32 ilo, i32 ihi,
            f64 *a, i32 lda, f64 *e, i32 lde,
            f64 *b, i32 ldb, f64 *c, i32 ldc,
            f64 *q, i32 ldq, f64 *z, i32 ldz,
            f64 *dwork, i32 ldwork, i32 *info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const i32 INT1 = 1;

    bool upper = (*jobe == 'U' || *jobe == 'u');
    bool inq = (*compq == 'I' || *compq == 'i');
    bool ilq = (*compq == 'V' || *compq == 'v') || inq;
    bool inz = (*compz == 'I' || *compz == 'i');
    bool ilz = (*compz == 'V' || *compz == 'v') || inz;
    bool withb = m > 0;
    bool withc = p > 0;

    *info = 0;

    if (!upper && !(*jobe == 'G' || *jobe == 'g')) {
        *info = -1;
    } else if (!ilq && !(*compq == 'N' || *compq == 'n')) {
        *info = -2;
    } else if (!ilz && !(*compz == 'N' || *compz == 'n')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (m < 0) {
        *info = -5;
    } else if (p < 0) {
        *info = -6;
    } else if (ilo < 1) {
        *info = -7;
    } else if (ihi > n || ihi < ilo - 1) {
        *info = -8;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -10;
    } else if (lde < (n > 1 ? n : 1)) {
        *info = -12;
    } else if ((withb && ldb < n) || ldb < 1) {
        *info = -14;
    } else if (ldc < (p > 1 ? p : 1)) {
        *info = -16;
    } else if ((ilq && ldq < n) || ldq < 1) {
        *info = -18;
    } else if ((ilz && ldz < n) || ldz < 1) {
        *info = -20;
    } else {
        i32 jrow_len = ihi + 1 - ilo;
        i32 jcol_len = n + 1 - ilo;
        i32 minwrk;
        if (upper) {
            minwrk = 1;
        } else {
            i32 ni = ilq ? n : jcol_len;
            minwrk = jrow_len + (ni > m ? ni : m);
            if (minwrk < 1) minwrk = 1;
        }
        if (ldwork < minwrk) {
            *info = -22;
        }
    }

    if (*info != 0) {
        return;
    }

    // Initialize Q and Z if desired
    if (inq) {
        SLC_DLASET("F", &n, &n, &ZERO, &ONE, q, &ldq);
    }
    if (inz) {
        SLC_DLASET("F", &n, &n, &ZERO, &ONE, z, &ldz);
    }

    // Quick return
    if (n <= 1) {
        dwork[0] = ONE;
        return;
    }

    i32 jrow_len = ihi + 1 - ilo;
    i32 jcol_len = n + 1 - ilo;
    i32 maxwrk = 1;
    i32 ierr;

    if (!upper) {
        // Reduce E to triangular form (QR decomposition of E)
        i32 itau = 0;
        i32 iwrk = itau + jrow_len;

        // e[ilo-1 + (ilo-1)*lde] is start of submatrix
        i32 e_offset = (ilo - 1) + (ilo - 1) * lde;
        i32 a_offset = (ilo - 1) + (ilo - 1) * lda;
        i32 b_offset = (ilo - 1);
        i32 q_offset = (ilo - 1) * ldq;

        i32 lwork_qr = ldwork - iwrk;
        SLC_DGEQRF(&jrow_len, &jcol_len, &e[e_offset], &lde,
                   &dwork[itau], &dwork[iwrk], &lwork_qr, &ierr);

        i32 opt = (i32)dwork[iwrk] + iwrk;
        if (opt > maxwrk) maxwrk = opt;

        // Apply the orthogonal transformation to matrix A
        SLC_DORMQR("L", "T", &jrow_len, &jcol_len, &jrow_len,
                   &e[e_offset], &lde, &dwork[itau],
                   &a[a_offset], &lda, &dwork[iwrk], &lwork_qr, &ierr);
        opt = (i32)dwork[iwrk] + iwrk;
        if (opt > maxwrk) maxwrk = opt;

        // Apply to B
        if (withb) {
            SLC_DORMQR("L", "T", &jrow_len, &m, &jrow_len,
                       &e[e_offset], &lde, &dwork[itau],
                       &b[b_offset], &ldb, &dwork[iwrk], &lwork_qr, &ierr);
            opt = (i32)dwork[iwrk] + iwrk;
            if (opt > maxwrk) maxwrk = opt;
        }

        // Apply to Q
        if (ilq) {
            SLC_DORMQR("R", "N", &n, &jrow_len, &jrow_len,
                       &e[e_offset], &lde, &dwork[itau],
                       &q[q_offset], &ldq, &dwork[iwrk], &lwork_qr, &ierr);
            opt = (i32)dwork[iwrk] + iwrk;
            if (opt > maxwrk) maxwrk = opt;
        }
    }

    // Zero out lower triangle of E
    if (jrow_len > 1) {
        i32 jrow_m1 = jrow_len - 1;
        i32 e_lower_offset = (ilo) + (ilo - 1) * lde;
        SLC_DLASET("L", &jrow_m1, &jrow_m1, &ZERO, &ZERO, &e[e_lower_offset], &lde);
    }

    // Reduce A and E and apply the transformations to B, C, Q and Z
    f64 cs, s, temp;

    for (i32 jcol = ilo - 1; jcol <= ihi - 3; jcol++) {
        for (i32 jrow = ihi - 1; jrow >= jcol + 2; jrow--) {
            // Step 1: rotate rows jrow-1, jrow to kill A(jrow,jcol)
            temp = a[(jrow - 1) + jcol * lda];
            SLC_DLARTG(&temp, &a[jrow + jcol * lda], &cs, &s, &a[(jrow - 1) + jcol * lda]);
            a[jrow + jcol * lda] = ZERO;

            i32 len = n - jcol - 1;
            SLC_DROT(&len, &a[(jrow - 1) + (jcol + 1) * lda], &lda,
                     &a[jrow + (jcol + 1) * lda], &lda, &cs, &s);

            i32 len2 = n + 1 - jrow;
            SLC_DROT(&len2, &e[(jrow - 1) + (jrow - 1) * lde], &lde,
                     &e[jrow + (jrow - 1) * lde], &lde, &cs, &s);

            if (withb) {
                SLC_DROT(&m, &b[jrow - 1], &ldb, &b[jrow], &ldb, &cs, &s);
            }

            if (ilq) {
                SLC_DROT(&n, &q[(jrow - 1) * ldq], &INT1, &q[jrow * ldq], &INT1, &cs, &s);
            }

            // Step 2: rotate columns jrow, jrow-1 to kill E(jrow,jrow-1)
            temp = e[jrow + jrow * lde];
            SLC_DLARTG(&temp, &e[jrow + (jrow - 1) * lde], &cs, &s, &e[jrow + jrow * lde]);
            e[jrow + (jrow - 1) * lde] = ZERO;

            i32 ihi_len = ihi;
            SLC_DROT(&ihi_len, &a[jrow * lda], &INT1, &a[(jrow - 1) * lda], &INT1, &cs, &s);

            i32 jrow_len2 = jrow;
            SLC_DROT(&jrow_len2, &e[jrow * lde], &INT1, &e[(jrow - 1) * lde], &INT1, &cs, &s);

            if (withc) {
                SLC_DROT(&p, &c[jrow * ldc], &INT1, &c[(jrow - 1) * ldc], &INT1, &cs, &s);
            }

            if (ilz) {
                SLC_DROT(&n, &z[jrow * ldz], &INT1, &z[(jrow - 1) * ldz], &INT1, &cs, &s);
            }
        }
    }

    dwork[0] = (f64)maxwrk;
}
