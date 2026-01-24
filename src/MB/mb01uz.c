/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <stdbool.h>

void mb01uz(const char* side, const char* uplo, const char* trans,
            i32 m, i32 n, c128 alpha, c128* t, i32 ldt, const c128* a, i32 lda,
            c128* zwork, i32 lzwork, i32* info)
{
    const c128 zero = 0.0;
    const c128 one = 1.0;
    i32 int1 = 1;

    char side_c = (char)toupper((unsigned char)side[0]);
    char uplo_c = (char)toupper((unsigned char)uplo[0]);
    char trans_c = (char)toupper((unsigned char)trans[0]);

    bool lside = (side_c == 'L');
    bool luplo = (uplo_c == 'U');
    bool ttran = (trans_c == 'T');
    bool ltran = (trans_c == 'C' || ttran);

    i32 k = lside ? m : n;
    i32 l = lside ? n : m;
    i32 mn = m < n ? m : n;

    i32 wrkmin = 1;
    if (alpha != zero && mn > 0) {
        wrkmin = k > wrkmin ? k : wrkmin;
    }

    bool lquery = (lzwork == -1);

    *info = 0;
    if (!lside && side_c != 'R') {
        *info = -1;
    } else if (!luplo && uplo_c != 'L') {
        *info = -2;
    } else if (!ltran && trans_c != 'N') {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (ldt < 1 || ldt < m || (!lside && ldt < n)) {
        *info = -8;
    } else if (lda < 1 || lda < m) {
        *info = -10;
    } else if (lquery) {
        if (alpha != zero && mn > 0) {
            i32 lwork_query = -1;
            i32 qinfo;
            i32 max_mn = m > n ? m : n;
            SLC_ZGEQRF(&m, &max_mn, (c128*)a, &lda, zwork, zwork, &lwork_query, &qinfo);
            i32 wrkopt = wrkmin;
            i32 qopt = 2 * l;
            if (qopt > wrkopt) wrkopt = qopt;
            i32 zopt = (i32)creal(zwork[0]);
            if (zopt > wrkopt) wrkopt = zopt;
            zwork[0] = (f64)wrkopt;
        } else {
            zwork[0] = one;
        }
        return;
    } else if (lzwork < wrkmin) {
        zwork[0] = (f64)wrkmin;
        *info = -12;
    }

    if (*info != 0) {
        return;
    }

    if (mn == 0) {
        return;
    }

    if (alpha == zero) {
        SLC_ZLASET("Full", &m, &n, &zero, &zero, t, &ldt);
        return;
    }

    i32 nb = 1;
    if (l > 0) {
        nb = lzwork / l;
        if (nb > k) nb = k;
        if (nb < 1) nb = 1;
    }

    if (lzwork >= m * n) {
        SLC_ZLACPY("All", &m, &n, a, &lda, zwork, &m);
        SLC_ZTRMM(&side_c, &uplo_c, &trans_c, "NonUnit", &m, &n, &alpha, t, &ldt, zwork, &m);
        SLC_ZLACPY("All", &m, &n, zwork, &m, t, &ldt);
    } else if (nb > 1) {
        char uploc;
        char tranc;

        if (ltran) {
            char skew = ttran ? 'N' : 'G';
            ma02ez(uplo_c, trans_c, skew, k, t, ldt);
            uploc = luplo ? 'L' : 'U';
            tranc = 'N';
            luplo = !luplo;
            ltran = !ltran;
        } else {
            uploc = uplo_c;
            tranc = trans_c;
        }

        i32 bl = k / nb;
        if (bl < 1) bl = 1;
        i32 j_pos = nb * bl;
        if (j_pos > k) j_pos = k;

        if (lside) {
            if (luplo) {
                i32 nr, ii;
                if (j_pos == m) {
                    nr = nb;
                    ii = m - nb;
                    bl = bl - 1;
                } else {
                    nr = m - j_pos;
                    ii = j_pos;
                }

                SLC_ZLACPY("All", &nr, &n, &a[ii], &lda, zwork, &nr);
                SLC_ZTRMM(&side_c, &uploc, &tranc, "NonUnit", &nr, &n, &alpha, &t[ii + ii*ldt], &ldt, zwork, &nr);
                SLC_ZLACPY("All", &nr, &n, zwork, &nr, &t[ii], &ldt);

                for (i32 i = 0; i < bl; i++) {
                    i32 ij = ii;
                    ii = ii - nb;
                    SLC_ZLACPY("All", &nb, &n, &a[ii], &lda, zwork, &nb);
                    SLC_ZTRMM(&side_c, &uploc, &tranc, "NonUnit", &nb, &n, &alpha, &t[ii + ii*ldt], &ldt, zwork, &nb);
                    i32 ncols = m - ij;
                    SLC_ZGEMM(&tranc, "NoTrans", &nb, &n, &ncols, &alpha, &t[ii + ij*ldt], &ldt, &a[ij], &lda, &one, zwork, &nb);
                    SLC_ZLACPY("All", &nb, &n, zwork, &nb, &t[ii], &ldt);
                }
            } else {
                i32 nr;
                if (j_pos == m) {
                    nr = nb;
                    bl = bl - 1;
                } else {
                    nr = m - j_pos;
                }
                SLC_ZLACPY("All", &nr, &n, a, &lda, zwork, &nr);
                SLC_ZTRMM(&side_c, &uploc, &tranc, "NonUnit", &nr, &n, &alpha, t, &ldt, zwork, &nr);
                SLC_ZLACPY("All", &nr, &n, zwork, &nr, t, &ldt);
                i32 ii = nr;

                for (i32 i = 0; i < bl; i++) {
                    SLC_ZLACPY("All", &nb, &n, &a[ii], &lda, zwork, &nb);
                    SLC_ZTRMM(&side_c, &uploc, &tranc, "NonUnit", &nb, &n, &alpha, &t[ii + ii*ldt], &ldt, zwork, &nb);
                    i32 nrows = ii;
                    SLC_ZGEMM(&tranc, "NoTrans", &nb, &n, &nrows, &alpha, &t[ii], &ldt, a, &lda, &one, zwork, &nb);
                    SLC_ZLACPY("All", &nb, &n, zwork, &nb, &t[ii], &ldt);
                    ii = ii + nb;
                }
            }
        } else {
            if (luplo) {
                i32 nc;
                i32 ii = 0;
                if (j_pos == n) {
                    nc = nb;
                    bl = bl - 1;
                } else {
                    nc = n - j_pos;
                }
                SLC_ZLACPY("All", &m, &nc, a, &lda, zwork, &m);
                SLC_ZTRMM(&side_c, &uploc, &tranc, "NonUnit", &m, &nc, &alpha, t, &ldt, zwork, &m);
                SLC_ZLACPY("All", &m, &nc, zwork, &m, t, &ldt);
                ii = ii + nc;

                for (i32 i = 0; i < bl; i++) {
                    i32 ij = ii - 1;
                    SLC_ZLACPY("All", &m, &nb, &a[ii*lda], &lda, zwork, &m);
                    SLC_ZTRMM(&side_c, &uploc, &tranc, "NonUnit", &m, &nb, &alpha, &t[ii + ii*ldt], &ldt, zwork, &m);
                    i32 ncols = ij + 1;
                    SLC_ZGEMM(&tranc, "NoTrans", &m, &nb, &ncols, &alpha, a, &lda, &t[ii*ldt], &ldt, &one, zwork, &m);
                    SLC_ZLACPY("All", &m, &nb, zwork, &m, &t[ii*ldt], &ldt);
                    ii = ii + nb;
                }
            } else {
                i32 nc, ii;
                if (j_pos == n) {
                    nc = nb;
                    ii = n - nb;
                    bl = bl - 1;
                } else {
                    nc = n - j_pos;
                    ii = j_pos;
                }
                SLC_ZLACPY("All", &m, &nc, &a[ii*lda], &lda, zwork, &m);
                SLC_ZTRMM(&side_c, &uploc, &tranc, "NonUnit", &m, &nc, &alpha, &t[ii + ii*ldt], &ldt, zwork, &m);
                SLC_ZLACPY("All", &m, &nc, zwork, &m, &t[ii*ldt], &ldt);

                for (i32 i = 0; i < bl; i++) {
                    i32 ij = ii;
                    ii = ii - nb;
                    SLC_ZLACPY("All", &m, &nb, &a[ii*lda], &lda, zwork, &m);
                    SLC_ZTRMM(&side_c, &uploc, &tranc, "NonUnit", &m, &nb, &alpha, &t[ii + ii*ldt], &ldt, zwork, &m);
                    i32 ncols = nc;
                    SLC_ZGEMM(&tranc, "NoTrans", &m, &nb, &ncols, &alpha, &a[ij*lda], &lda, &t[ij + ii*ldt], &ldt, &one, zwork, &m);
                    SLC_ZLACPY("All", &m, &nb, zwork, &m, &t[ii*ldt], &ldt);
                    nc = nc + nb;
                }
            }
        }
    } else {
        bool fillin = ltran && ((lside && luplo) || (!lside && !luplo));
        if (fillin) {
            char skew = ttran ? 'N' : 'G';
            ma02ez(uplo_c, trans_c, skew, k, t, ldt);
        }

        c128 temp = conj(alpha);

        if (lside) {
            if (luplo) {
                if (!ltran) {
                    for (i32 i = 0; i < m; i++) {
                        i32 len = m - i;
                        SLC_ZCOPY(&len, &t[i + i*ldt], &ldt, zwork, &int1);
                        SLC_ZLACGV(&len, zwork, &int1);
                        SLC_ZGEMV("CTrans", &len, &n, &temp, &a[i], &lda, zwork, &int1, &zero, &t[i], &ldt);
                        SLC_ZLACGV(&n, &t[i], &ldt);
                    }
                } else if (ttran) {
                    for (i32 i = 0; i < m; i++) {
                        i32 len = i + 1;
                        SLC_ZCOPY(&len, &t[i], &ldt, zwork, &int1);
                        char trans_t = 'T';
                        SLC_ZGEMV(&trans_t, &len, &n, &alpha, a, &lda, zwork, &int1, &zero, &t[i], &ldt);
                    }
                } else {
                    for (i32 i = 0; i < m; i++) {
                        i32 len = i + 1;
                        SLC_ZCOPY(&len, &t[i], &ldt, zwork, &int1);
                        SLC_ZLACGV(&len, zwork, &int1);
                        char trans_c_op = 'C';
                        SLC_ZGEMV(&trans_c_op, &len, &n, &temp, a, &lda, zwork, &int1, &zero, &t[i], &ldt);
                        SLC_ZLACGV(&n, &t[i], &ldt);
                    }
                }
            } else {
                if (!ltran) {
                    for (i32 i = 0; i < m; i++) {
                        i32 len = i + 1;
                        SLC_ZCOPY(&len, &t[i], &ldt, zwork, &int1);
                        SLC_ZLACGV(&len, zwork, &int1);
                        SLC_ZGEMV("CTrans", &len, &n, &temp, a, &lda, zwork, &int1, &zero, &t[i], &ldt);
                        SLC_ZLACGV(&n, &t[i], &ldt);
                    }
                } else if (ttran) {
                    for (i32 i = 0; i < m; i++) {
                        i32 len = m - i;
                        SLC_ZCOPY(&len, &t[i + i*ldt], &int1, zwork, &int1);
                        char trans_t = 'T';
                        SLC_ZGEMV(&trans_t, &len, &n, &alpha, &a[i], &lda, zwork, &int1, &zero, &t[i], &ldt);
                    }
                } else {
                    for (i32 i = 0; i < m; i++) {
                        i32 len = m - i;
                        SLC_ZCOPY(&len, &t[i + i*ldt], &int1, zwork, &int1);
                        char trans_c_op = 'C';
                        SLC_ZGEMV(&trans_c_op, &len, &n, &temp, &a[i], &lda, zwork, &int1, &zero, &t[i], &ldt);
                        SLC_ZLACGV(&n, &t[i], &ldt);
                    }
                }
            }
        } else {
            if (luplo) {
                if (ttran) {
                    for (i32 i = 0; i < n; i++) {
                        i32 len = n - i;
                        SLC_ZCOPY(&len, &t[i + i*ldt], &ldt, zwork, &int1);
                        SLC_ZGEMV("NoTran", &m, &len, &alpha, &a[i*lda], &lda, zwork, &int1, &zero, &t[i*ldt], &int1);
                    }
                } else if (ltran) {
                    for (i32 i = 0; i < n; i++) {
                        i32 len = n - i;
                        SLC_ZCOPY(&len, &t[i + i*ldt], &ldt, zwork, &int1);
                        SLC_ZLACGV(&len, zwork, &int1);
                        SLC_ZGEMV("NoTran", &m, &len, &alpha, &a[i*lda], &lda, zwork, &int1, &zero, &t[i*ldt], &int1);
                    }
                } else {
                    for (i32 i = 0; i < n; i++) {
                        i32 len = i + 1;
                        SLC_ZCOPY(&len, &t[i*ldt], &int1, zwork, &int1);
                        SLC_ZGEMV("NoTran", &m, &len, &alpha, a, &lda, zwork, &int1, &zero, &t[i*ldt], &int1);
                    }
                }
            } else {
                if (ltran) {
                    for (i32 i = 0; i < n; i++) {
                        i32 len = i + 1;
                        SLC_ZCOPY(&len, &t[i*ldt], &int1, zwork, &int1);
                        SLC_ZGEMV("NoTran", &m, &len, &alpha, a, &lda, zwork, &int1, &zero, &t[i*ldt], &int1);
                    }
                } else {
                    for (i32 i = 0; i < n; i++) {
                        i32 len = n - i;
                        SLC_ZCOPY(&len, &t[i + i*ldt], &int1, zwork, &int1);
                        SLC_ZGEMV("NoTran", &m, &len, &alpha, &a[i*lda], &lda, zwork, &int1, &zero, &t[i*ldt], &int1);
                    }
                }
            }
        }
    }

    i32 wrkopt = wrkmin;
    zwork[0] = (f64)(wrkopt > wrkmin ? wrkopt : wrkmin);
}
