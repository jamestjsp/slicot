// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"

void mb03ya(bool wantt, bool wantq, bool wantz, i32 n, i32 ilo, i32 ihi,
            i32 iloq, i32 ihiq, i32 pos, f64 *a, i32 lda, f64 *b, i32 ldb,
            f64 *q, i32 ldq, f64 *z, i32 ldz, i32 *info) {
    const f64 ZERO = 0.0;
    const i32 ONE = 1;

    i32 nq = ihiq - iloq + 1;
    *info = 0;

    if (n < 0) {
        *info = -4;
    } else if (ilo < 1 || ilo > (n > 1 ? n : 1)) {
        *info = -5;
    } else if (ihi < (ilo < n ? ilo : n) || ihi > n) {
        *info = -6;
    } else if (iloq < 1 || iloq > ilo) {
        *info = -7;
    } else if (ihiq < ihi || ihiq > n) {
        *info = -8;
    } else if (pos < ilo || pos > ihi) {
        *info = -9;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -11;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -13;
    } else if (ldq < 1 || (wantq && ldq < n)) {
        *info = -15;
    } else if (ldz < 1 || (wantz && ldz < n)) {
        *info = -17;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        return;
    }

    i32 i1, i2;
    if (wantt) {
        i1 = 0;
        i2 = n - 1;
    } else {
        i1 = ilo - 1;
        i2 = ihi - 1;
    }

    f64 cs, sn, temp;
    i32 iloq0 = iloq - 1;

    for (i32 j = ilo - 1; j < pos - 1; j++) {
        temp = a[j + j * lda];
        SLC_DLARTG(&temp, &a[j + 1 + j * lda], &cs, &sn, &a[j + j * lda]);
        a[j + 1 + j * lda] = ZERO;
        i32 len = i2 - j;
        SLC_DROT(&len, &a[j + (j + 1) * lda], &lda, &a[j + 1 + (j + 1) * lda], &lda, &cs, &sn);
        i32 len2 = (j < pos - 2 ? j : pos - 2) - i1 + 2;
        if (len2 < 0) len2 = 0;
        SLC_DROT(&len2, &b[i1 + j * ldb], &ONE, &b[i1 + (j + 1) * ldb], &ONE, &cs, &sn);
        if (wantq) {
            SLC_DROT(&nq, &q[iloq0 + j * ldq], &ONE, &q[iloq0 + (j + 1) * ldq], &ONE, &cs, &sn);
        }
    }

    for (i32 j = ilo - 1; j < pos - 2; j++) {
        temp = b[j + j * ldb];
        SLC_DLARTG(&temp, &b[j + 1 + j * ldb], &cs, &sn, &b[j + j * ldb]);
        b[j + 1 + j * ldb] = ZERO;
        i32 len = i2 - j;
        SLC_DROT(&len, &b[j + (j + 1) * ldb], &ldb, &b[j + 1 + (j + 1) * ldb], &ldb, &cs, &sn);
        i32 len2 = j - i1 + 2;
        SLC_DROT(&len2, &a[i1 + j * lda], &ONE, &a[i1 + (j + 1) * lda], &ONE, &cs, &sn);
        if (wantz) {
            SLC_DROT(&nq, &z[iloq0 + j * ldz], &ONE, &z[iloq0 + (j + 1) * ldz], &ONE, &cs, &sn);
        }
    }

    for (i32 j = ihi - 1; j >= pos; j--) {
        temp = a[j + j * lda];
        SLC_DLARTG(&temp, &a[j + (j - 1) * lda], &cs, &sn, &a[j + j * lda]);
        a[j + (j - 1) * lda] = ZERO;
        sn = -sn;
        i32 len = j - i1;
        SLC_DROT(&len, &a[i1 + (j - 1) * lda], &ONE, &a[i1 + j * lda], &ONE, &cs, &sn);
        i32 start_col = (j - 1 > pos ? j - 1 : pos);
        i32 len2 = i2 - start_col + 1;
        SLC_DROT(&len2, &b[j - 1 + start_col * ldb], &ldb, &b[j + start_col * ldb], &ldb, &cs, &sn);
        if (wantz) {
            SLC_DROT(&nq, &z[iloq0 + (j - 1) * ldz], &ONE, &z[iloq0 + j * ldz], &ONE, &cs, &sn);
        }
    }

    for (i32 j = ihi - 1; j >= pos + 1; j--) {
        temp = b[j + j * ldb];
        SLC_DLARTG(&temp, &b[j + (j - 1) * ldb], &cs, &sn, &b[j + j * ldb]);
        b[j + (j - 1) * ldb] = ZERO;
        sn = -sn;
        i32 len = j - i1;
        SLC_DROT(&len, &b[i1 + (j - 1) * ldb], &ONE, &b[i1 + j * ldb], &ONE, &cs, &sn);
        i32 len2 = i2 - j + 2;
        SLC_DROT(&len2, &a[j - 1 + (j - 1) * lda], &lda, &a[j + (j - 1) * lda], &lda, &cs, &sn);
        if (wantq) {
            SLC_DROT(&nq, &q[iloq0 + (j - 1) * ldq], &ONE, &q[iloq0 + j * ldq], &ONE, &cs, &sn);
        }
    }
}
