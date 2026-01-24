// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb04ts(const char *trana, const char *tranb, i32 n, i32 ilo,
            f64 *a, i32 lda, f64 *b, i32 ldb, f64 *g, i32 ldg,
            f64 *q, i32 ldq, f64 *csl, f64 *csr, f64 *taul, f64 *taur,
            f64 *dwork, i32 ldwork, i32 *info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const i32 INT1 = 1;

    bool ltra = (*trana == 'T' || *trana == 't' || *trana == 'C' || *trana == 'c');
    bool ltrb = (*tranb == 'T' || *tranb == 't' || *tranb == 'C' || *tranb == 'c');

    *info = 0;

    if (!ltra && !(*trana == 'N' || *trana == 'n')) {
        *info = -1;
    } else if (!ltrb && !(*tranb == 'N' || *tranb == 'n')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ilo < 1 || (n > 0 && ilo > n) || (n == 0 && ilo != 1)) {
        *info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -8;
    } else if (ldg < (n > 1 ? n : 1)) {
        *info = -10;
    } else if (ldq < (n > 1 ? n : 1)) {
        *info = -12;
    } else if (ldwork < (n > 1 ? n : 1)) {
        dwork[0] = (f64)(n > 1 ? n : 1);
        *info = -18;
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        dwork[0] = ONE;
        return;
    }

    for (i32 i = ilo - 1; i < n; i++) {
        f64 alpha = q[i + i * ldq];
        f64 nu, temp, c, s;

        if (i < n - 1) {
            i32 len = n - i;
            SLC_DLARFG(&len, &alpha, &q[(i + 1) + i * ldq], &INT1, &nu);

            q[i + i * ldq] = ONE;
            i32 len2 = n - i - 1;
            SLC_DLARF("L", &len, &len2, &q[i + i * ldq], &INT1, &nu,
                      &q[i + (i + 1) * ldq], &ldq, dwork);

            if (ltra) {
                SLC_DLARF("R", &len, &len, &q[i + i * ldq], &INT1, &nu,
                          &a[i + i * lda], &lda, dwork);
            } else {
                SLC_DLARF("L", &len, &len, &q[i + i * ldq], &INT1, &nu,
                          &a[i + i * lda], &lda, dwork);
            }

            if (ltrb) {
                SLC_DLARF("R", &n, &len, &q[i + i * ldq], &INT1, &nu,
                          &b[i * ldb], &ldb, dwork);
            } else {
                SLC_DLARF("L", &len, &n, &q[i + i * ldq], &INT1, &nu,
                          &b[i], &ldb, dwork);
            }

            SLC_DLARF("L", &len, &n, &q[i + i * ldq], &INT1, &nu,
                      &g[i], &ldg, dwork);

            q[i + i * ldq] = nu;
        } else {
            q[i + i * ldq] = ZERO;
        }

        temp = a[i + i * lda];
        SLC_DLARTG(&temp, &alpha, &c, &s, &a[i + i * lda]);

        if (ltra) {
            i32 len = n - i - 1;
            SLC_DROT(&len, &a[(i + 1) + i * lda], &INT1, &q[i + (i + 1) * ldq], &ldq, &c, &s);
        } else {
            i32 len = n - i - 1;
            SLC_DROT(&len, &a[i + (i + 1) * lda], &lda, &q[i + (i + 1) * ldq], &ldq, &c, &s);
        }

        if (ltrb) {
            SLC_DROT(&n, &g[i], &ldg, &b[i * ldb], &INT1, &c, &s);
        } else {
            SLC_DROT(&n, &g[i], &ldg, &b[i], &ldb, &c, &s);
        }

        csl[2 * i] = c;
        csl[2 * i + 1] = s;

        if (i < n - 1) {
            if (ltra) {
                i32 len = n - i;
                SLC_DLARFG(&len, &a[i + i * lda], &a[i + (i + 1) * lda], &lda, &taul[i]);

                temp = a[i + i * lda];
                a[i + i * lda] = ONE;

                i32 len2 = n - i - 1;
                SLC_DLARF("R", &len2, &len, &a[i + i * lda], &lda, &taul[i],
                          &a[(i + 1) + i * lda], &lda, dwork);
                SLC_DLARF("L", &len, &len2, &a[i + i * lda], &lda, &taul[i],
                          &q[i + (i + 1) * ldq], &ldq, dwork);

                if (ltrb) {
                    SLC_DLARF("R", &n, &len, &a[i + i * lda], &lda, &taul[i],
                              &b[i * ldb], &ldb, dwork);
                } else {
                    SLC_DLARF("L", &len, &n, &a[i + i * lda], &lda, &taul[i],
                              &b[i], &ldb, dwork);
                }
                SLC_DLARF("L", &len, &n, &a[i + i * lda], &lda, &taul[i],
                          &g[i], &ldg, dwork);

                a[i + i * lda] = temp;
            } else {
                i32 len = n - i;
                SLC_DLARFG(&len, &a[i + i * lda], &a[(i + 1) + i * lda], &INT1, &taul[i]);

                temp = a[i + i * lda];
                a[i + i * lda] = ONE;

                i32 len2 = n - i - 1;
                SLC_DLARF("L", &len, &len2, &a[i + i * lda], &INT1, &taul[i],
                          &a[i + (i + 1) * lda], &lda, dwork);
                SLC_DLARF("L", &len, &len2, &a[i + i * lda], &INT1, &taul[i],
                          &q[i + (i + 1) * ldq], &ldq, dwork);

                if (ltrb) {
                    SLC_DLARF("R", &n, &len, &a[i + i * lda], &INT1, &taul[i],
                              &b[i * ldb], &ldb, dwork);
                } else {
                    SLC_DLARF("L", &len, &n, &a[i + i * lda], &INT1, &taul[i],
                              &b[i], &ldb, dwork);
                }
                SLC_DLARF("L", &len, &n, &a[i + i * lda], &INT1, &taul[i],
                          &g[i], &ldg, dwork);

                a[i + i * lda] = temp;
            }
        } else {
            taul[i] = ZERO;
        }

        if (i < n - 1) {
            alpha = q[i + (i + 1) * ldq];
        }

        if (i < n - 2) {
            i32 len = n - i - 1;
            SLC_DLARFG(&len, &alpha, &q[i + (i + 2) * ldq], &ldq, &nu);

            q[i + (i + 1) * ldq] = ONE;

            i32 len2 = n - i - 1;
            SLC_DLARF("R", &len2, &len, &q[i + (i + 1) * ldq], &ldq, &nu,
                      &q[(i + 1) + (i + 1) * ldq], &ldq, dwork);

            if (ltra) {
                SLC_DLARF("L", &len, &n, &q[i + (i + 1) * ldq], &ldq, &nu,
                          &a[(i + 1)], &lda, dwork);
            } else {
                SLC_DLARF("R", &n, &len, &q[i + (i + 1) * ldq], &ldq, &nu,
                          &a[(i + 1) * lda], &lda, dwork);
            }

            if (ltrb) {
                i32 len3 = n - i;
                SLC_DLARF("L", &len, &len3, &q[i + (i + 1) * ldq], &ldq, &nu,
                          &b[(i + 1) + i * ldb], &ldb, dwork);
            } else {
                i32 len3 = n - i;
                SLC_DLARF("R", &len3, &len, &q[i + (i + 1) * ldq], &ldq, &nu,
                          &b[i + (i + 1) * ldb], &ldb, dwork);
            }

            SLC_DLARF("R", &n, &len, &q[i + (i + 1) * ldq], &ldq, &nu,
                      &g[(i + 1) * ldg], &ldg, dwork);

            q[i + (i + 1) * ldq] = nu;
        } else if (i < n - 1) {
            q[i + (i + 1) * ldq] = ZERO;
        }

        if (i < n - 1) {
            if (ltrb) {
                temp = b[(i + 1) + i * ldb];
                SLC_DLARTG(&temp, &alpha, &c, &s, &b[(i + 1) + i * ldb]);
                s = -s;

                i32 len = n - i - 1;
                SLC_DROT(&len, &q[(i + 1) + (i + 1) * ldq], &INT1, &b[(i + 1) + (i + 1) * ldb], &ldb, &c, &s);
            } else {
                temp = b[i + (i + 1) * ldb];
                SLC_DLARTG(&temp, &alpha, &c, &s, &b[i + (i + 1) * ldb]);
                s = -s;

                i32 len = n - i - 1;
                SLC_DROT(&len, &q[(i + 1) + (i + 1) * ldq], &INT1, &b[(i + 1) + (i + 1) * ldb], &INT1, &c, &s);
            }

            if (ltra) {
                SLC_DROT(&n, &a[(i + 1)], &lda, &g[(i + 1) * ldg], &INT1, &c, &s);
            } else {
                SLC_DROT(&n, &a[(i + 1) * lda], &INT1, &g[(i + 1) * ldg], &INT1, &c, &s);
            }

            csr[2 * i] = c;
            csr[2 * i + 1] = s;
        }

        if (i < n - 2) {
            if (ltrb) {
                i32 len = n - i - 1;
                SLC_DLARFG(&len, &b[(i + 1) + i * ldb], &b[(i + 2) + i * ldb], &INT1, &taur[i]);

                temp = b[(i + 1) + i * ldb];
                b[(i + 1) + i * ldb] = ONE;

                i32 len2 = n - i - 1;
                SLC_DLARF("L", &len, &len2, &b[(i + 1) + i * ldb], &INT1, &taur[i],
                          &b[(i + 1) + (i + 1) * ldb], &ldb, dwork);
                SLC_DLARF("R", &len2, &len, &b[(i + 1) + i * ldb], &INT1, &taur[i],
                          &q[(i + 1) + (i + 1) * ldq], &ldq, dwork);

                if (ltra) {
                    SLC_DLARF("L", &len, &n, &b[(i + 1) + i * ldb], &INT1, &taur[i],
                              &a[(i + 1)], &lda, dwork);
                } else {
                    SLC_DLARF("R", &n, &len, &b[(i + 1) + i * ldb], &INT1, &taur[i],
                              &a[(i + 1) * lda], &lda, dwork);
                }
                SLC_DLARF("R", &n, &len, &b[(i + 1) + i * ldb], &INT1, &taur[i],
                          &g[(i + 1) * ldg], &ldg, dwork);

                b[(i + 1) + i * ldb] = temp;
            } else {
                i32 len = n - i - 1;
                SLC_DLARFG(&len, &b[i + (i + 1) * ldb], &b[i + (i + 2) * ldb], &ldb, &taur[i]);

                temp = b[i + (i + 1) * ldb];
                b[i + (i + 1) * ldb] = ONE;

                i32 len2 = n - i - 1;
                SLC_DLARF("R", &len2, &len, &b[i + (i + 1) * ldb], &ldb, &taur[i],
                          &b[(i + 1) + (i + 1) * ldb], &ldb, dwork);
                SLC_DLARF("R", &len2, &len, &b[i + (i + 1) * ldb], &ldb, &taur[i],
                          &q[(i + 1) + (i + 1) * ldq], &ldq, dwork);

                if (ltra) {
                    SLC_DLARF("L", &len, &n, &b[i + (i + 1) * ldb], &ldb, &taur[i],
                              &a[(i + 1)], &lda, dwork);
                } else {
                    SLC_DLARF("R", &n, &len, &b[i + (i + 1) * ldb], &ldb, &taur[i],
                              &a[(i + 1) * lda], &lda, dwork);
                }
                SLC_DLARF("R", &n, &len, &b[i + (i + 1) * ldb], &ldb, &taur[i],
                          &g[(i + 1) * ldg], &ldg, dwork);

                b[i + (i + 1) * ldb] = temp;
            }
        } else if (i < n - 1) {
            taur[i] = ZERO;
        }
    }

    dwork[0] = (f64)(n > 1 ? n : 1);
}
