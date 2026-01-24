/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <stdbool.h>

void mb02ud(const char* fact, const char* side, const char* trans,
            const char* jobp, const i32 m, const i32 n, const f64 alpha,
            const f64 rcond, i32* rank, f64* r, const i32 ldr,
            f64* q, const i32 ldq, f64* sv, f64* b, const i32 ldb,
            f64* rp, const i32 ldrp, f64* dwork, const i32 ldwork,
            i32* info)
{
    const f64 one = 1.0;
    const f64 zero = 0.0;

    char fact_u = (char)toupper((unsigned char)fact[0]);
    char side_u = (char)toupper((unsigned char)side[0]);
    char trans_u = (char)toupper((unsigned char)trans[0]);
    char jobp_u = (char)toupper((unsigned char)jobp[0]);

    bool nfct = (fact_u == 'N');
    bool left = (side_u == 'L');
    bool pinv = (jobp_u == 'P');
    bool tran = (trans_u == 'T' || trans_u == 'C');

    i32 l = left ? m : n;
    i32 mn = m * n;

    *info = 0;

    if (!nfct && fact_u != 'F') {
        *info = -1;
    } else if (!left && side_u != 'R') {
        *info = -2;
    } else if (!tran && trans_u != 'N') {
        *info = -3;
    } else if (!pinv && jobp_u != 'N') {
        *info = -4;
    } else if (m < 0) {
        *info = -5;
    } else if (n < 0) {
        *info = -6;
    } else if (nfct && rcond > one) {
        *info = -8;
    } else if (!nfct && (*rank < 0 || *rank > l)) {
        *info = -9;
    } else if (ldr < (l > 1 ? l : 1)) {
        *info = -11;
    } else if (ldq < (l > 1 ? l : 1)) {
        *info = -13;
    } else if (ldb < (m > 1 ? m : 1)) {
        *info = -16;
    } else if (ldrp < 1 || (pinv && ldrp < l)) {
        *info = -18;
    } else {
        bool lquery = (ldwork == -1);
        i32 minwrk = l > 1 ? l : 1;
        i32 maxwrk = (minwrk > mn) ? minwrk : mn;

        if (nfct) {
            i32 info_query = 0;
            i32 neg1 = -1;
            mb03ud('V', 'V', l, r, ldr, q, ldq, sv, dwork, neg1, &info_query);
            minwrk = (5 * l > 1) ? 5 * l : 1;
            i32 mb03ud_opt = (i32)dwork[0];
            maxwrk = (maxwrk > mb03ud_opt) ? maxwrk : mb03ud_opt;
            maxwrk = (maxwrk > minwrk) ? maxwrk : minwrk;
        }

        if (ldwork < minwrk && !lquery) {
            *info = -20;
        }

        if (lquery) {
            dwork[0] = (f64)maxwrk;
            return;
        }
    }

    if (*info != 0) {
        return;
    }

    if (l == 0) {
        if (nfct) {
            *rank = 0;
        }
        dwork[0] = one;
        return;
    }

    i32 maxwrk = (l > mn) ? l : mn;

    if (nfct) {
        i32 info_svd = 0;
        mb03ud('V', 'V', l, r, ldr, q, ldq, sv, dwork, ldwork, &info_svd);
        if (info_svd != 0) {
            *info = info_svd;
            return;
        }

        f64 toll = rcond;
        if (toll <= zero) {
            toll = SLC_DLAMCH("P");
        }
        f64 smin = SLC_DLAMCH("S");
        toll = (toll * sv[0] > smin) ? toll * sv[0] : smin;

        i32 i;
        for (i = 0; i < l; i++) {
            if (toll > sv[i]) {
                break;
            }
        }
        *rank = i;

        for (i = 0; i < *rank; i++) {
            sv[i] = one / sv[i];
        }

        if (pinv && *rank > 0) {
            mb01sd('R', *rank, l, r, ldr, sv, sv);

            i32 rnk = *rank;
            SLC_DGEMM("T", "T", &l, &l, &rnk, &one, r, &ldr, q, &ldq,
                      &zero, rp, &ldrp);
        }
    }

    i32 min_mn = (m < n) ? m : n;
    if (min_mn == 0 || *rank == 0) {
        dwork[0] = (f64)maxwrk;
        return;
    }

    if (alpha == zero) {
        SLC_DLASET("F", &m, &n, &zero, &zero, b, &ldb);
        dwork[0] = (f64)maxwrk;
        return;
    }

    if (pinv) {
        if (left) {
            if (ldwork >= mn) {
                const char* trans_str = tran ? "T" : "N";
                SLC_DGEMM(trans_str, "N", &m, &n, &m, &alpha, rp, &ldrp,
                          b, &ldb, &zero, dwork, &m);
                SLC_DLACPY("F", &m, &n, dwork, &m, b, &ldb);
            } else {
                const char* trans_str = tran ? "T" : "N";
                i32 int1 = 1;
                for (i32 j = 0; j < n; j++) {
                    SLC_DGEMV(trans_str, &m, &m, &alpha, rp, &ldrp, &b[j * ldb],
                              &int1, &zero, dwork, &int1);
                    SLC_DCOPY(&m, dwork, &int1, &b[j * ldb], &int1);
                }
            }
        } else {
            if (ldwork >= mn) {
                const char* trans_str = tran ? "T" : "N";
                SLC_DGEMM("N", trans_str, &m, &n, &n, &alpha, b, &ldb,
                          rp, &ldrp, &zero, dwork, &m);
                SLC_DLACPY("F", &m, &n, dwork, &m, b, &ldb);
            } else {
                const char* ntran = tran ? "N" : "T";
                for (i32 i = 0; i < m; i++) {
                    SLC_DGEMV(ntran, &n, &n, &alpha, rp, &ldrp, &b[i],
                              &ldb, &zero, dwork, &ldb);
                    SLC_DCOPY(&n, dwork, &ldb, &b[i], &ldb);
                }
            }
        }
    } else {
        i32 rnk = *rank;
        i32 int1 = 1;

        if (left) {
            if (ldwork >= mn) {
                if (tran) {
                    SLC_DGEMM("N", "N", &m, &n, &m, &alpha, r, &ldr,
                              b, &ldb, &zero, dwork, &m);
                    mb01sd('R', rnk, n, dwork, m, sv, sv);
                    SLC_DGEMM("N", "N", &m, &n, &rnk, &one, q, &ldq,
                              dwork, &m, &zero, b, &ldb);
                } else {
                    SLC_DGEMM("T", "N", &m, &n, &m, &alpha, q, &ldq,
                              b, &ldb, &zero, dwork, &m);
                    mb01sd('R', rnk, n, dwork, m, sv, sv);
                    SLC_DGEMM("T", "N", &m, &n, &rnk, &one, r, &ldr,
                              dwork, &m, &zero, b, &ldb);
                }
            } else {
                if (tran) {
                    for (i32 j = 0; j < n; j++) {
                        SLC_DGEMV("N", &m, &m, &alpha, r, &ldr, &b[j * ldb],
                                  &int1, &zero, dwork, &int1);
                        SLC_DCOPY(&m, dwork, &int1, &b[j * ldb], &int1);
                    }
                    mb01sd('R', rnk, n, b, ldb, sv, sv);
                    for (i32 j = 0; j < n; j++) {
                        SLC_DGEMV("N", &m, &rnk, &one, q, &ldq, &b[j * ldb],
                                  &int1, &zero, dwork, &int1);
                        SLC_DCOPY(&m, dwork, &int1, &b[j * ldb], &int1);
                    }
                } else {
                    for (i32 j = 0; j < n; j++) {
                        SLC_DGEMV("T", &m, &m, &alpha, q, &ldq, &b[j * ldb],
                                  &int1, &zero, dwork, &int1);
                        SLC_DCOPY(&m, dwork, &int1, &b[j * ldb], &int1);
                    }
                    mb01sd('R', rnk, n, b, ldb, sv, sv);
                    for (i32 j = 0; j < n; j++) {
                        SLC_DGEMV("T", &rnk, &m, &one, r, &ldr, &b[j * ldb],
                                  &int1, &zero, dwork, &int1);
                        SLC_DCOPY(&m, dwork, &int1, &b[j * ldb], &int1);
                    }
                }
            }
        } else {
            if (ldwork >= mn) {
                if (tran) {
                    SLC_DGEMM("N", "N", &m, &n, &n, &alpha, b, &ldb,
                              q, &ldq, &zero, dwork, &m);
                    mb01sd('C', m, rnk, dwork, m, sv, sv);
                    SLC_DGEMM("N", "N", &m, &n, &rnk, &one, dwork, &m,
                              r, &ldr, &zero, b, &ldb);
                } else {
                    SLC_DGEMM("N", "T", &m, &n, &n, &alpha, b, &ldb,
                              r, &ldr, &zero, dwork, &m);
                    mb01sd('C', m, rnk, dwork, m, sv, sv);
                    SLC_DGEMM("N", "T", &m, &n, &rnk, &one, dwork, &m,
                              q, &ldq, &zero, b, &ldb);
                }
            } else {
                if (tran) {
                    for (i32 i = 0; i < m; i++) {
                        SLC_DGEMV("T", &n, &n, &alpha, q, &ldq, &b[i],
                                  &ldb, &zero, dwork, &int1);
                        SLC_DCOPY(&n, dwork, &int1, &b[i], &ldb);
                    }
                    mb01sd('C', m, rnk, b, ldb, sv, sv);
                    for (i32 i = 0; i < m; i++) {
                        SLC_DGEMV("T", &rnk, &n, &one, r, &ldr, &b[i],
                                  &ldb, &zero, dwork, &int1);
                        SLC_DCOPY(&n, dwork, &int1, &b[i], &ldb);
                    }
                } else {
                    for (i32 i = 0; i < m; i++) {
                        SLC_DGEMV("N", &n, &n, &alpha, r, &ldr, &b[i],
                                  &ldb, &zero, dwork, &int1);
                        SLC_DCOPY(&n, dwork, &int1, &b[i], &ldb);
                    }
                    mb01sd('C', m, rnk, b, ldb, sv, sv);
                    for (i32 i = 0; i < m; i++) {
                        SLC_DGEMV("N", &n, &rnk, &one, q, &ldq, &b[i],
                                  &ldb, &zero, dwork, &int1);
                        SLC_DCOPY(&n, dwork, &int1, &b[i], &ldb);
                    }
                }
            }
        }
    }

    dwork[0] = (f64)maxwrk;
}
