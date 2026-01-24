/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB04PD - Solution of continuous-time or discrete-time Sylvester equations
 *          using Bartels-Stewart method
 *
 * Solves for X either:
 *   Continuous-time: op(A)*X + ISGN*X*op(B) = scale*C
 *   Discrete-time:   op(A)*X*op(B) + ISGN*X = scale*C
 *
 * where op(M) = M or M**T, and ISGN = 1 or -1.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>

static int select_dummy(const f64* wr, const f64* wi) {
    (void)wr;
    (void)wi;
    return 0;
}

void sb04pd(
    const char dico,
    const char facta,
    const char factb,
    const char trana,
    const char tranb,
    const i32 isgn,
    const i32 m,
    const i32 n,
    f64* a,
    const i32 lda,
    f64* u,
    const i32 ldu,
    f64* b,
    const i32 ldb,
    f64* v,
    const i32 ldv,
    f64* c,
    const i32 ldc,
    f64* scale,
    f64* dwork,
    const i32 ldwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    bool cont, nofaca, nofacb, schura, schurb, notrna, notrnb;
    bool blas3a, blas3b, blocka, blockb;
    i32 ierr, j, jwork, maxwrk, minwrk, sdim;
    i32 ia, ib, availw, bl, chunka, chunkb, i;
    i32 int1 = 1;

    cont   = (dico == 'C' || dico == 'c');
    nofaca = (facta == 'N' || facta == 'n');
    nofacb = (factb == 'N' || factb == 'n');
    schura = (facta == 'S' || facta == 's');
    schurb = (factb == 'S' || factb == 's');
    notrna = (trana == 'N' || trana == 'n');
    notrnb = (tranb == 'N' || tranb == 'n');

    blas3a = false;
    blas3b = false;
    blocka = false;
    blockb = false;
    chunka = 0;
    chunkb = 0;

    *info = 0;

    if (!cont && !(dico == 'D' || dico == 'd')) {
        *info = -1;
    } else if (!nofaca && !(facta == 'F' || facta == 'f') && !schura) {
        *info = -2;
    } else if (!nofacb && !(factb == 'F' || factb == 'f') && !schurb) {
        *info = -3;
    } else if (!notrna && !(trana == 'T' || trana == 't') && !(trana == 'C' || trana == 'c')) {
        *info = -4;
    } else if (!notrnb && !(tranb == 'T' || tranb == 't') && !(tranb == 'C' || tranb == 'c')) {
        *info = -5;
    } else if (isgn != 1 && isgn != -1) {
        *info = -6;
    } else if (m < 0) {
        *info = -7;
    } else if (n < 0) {
        *info = -8;
    } else if (lda < 1 || (m > 0 && lda < m)) {
        *info = -10;
    } else if (ldu < 1 || (!schura && m > 0 && ldu < m)) {
        *info = -12;
    } else if (ldb < 1 || (n > 0 && ldb < n)) {
        *info = -14;
    } else if (ldv < 1 || (!schurb && n > 0 && ldv < n)) {
        *info = -16;
    } else if (ldc < 1 || (m > 0 && ldc < m)) {
        *info = -18;
    } else {
        if (nofaca) {
            ia = 1 + 2*m;
            minwrk = 3*m;
        } else {
            ia = 0;
            minwrk = 0;
        }
        if (schura) {
            minwrk = 0;
        } else if (!nofaca) {
            minwrk = m;
        }

        ib = 0;
        if (nofacb) {
            ib = 2*n;
            if (!nofaca)
                ib = ib + 1;
            if (ib + 3*n > minwrk)
                minwrk = ib + 3*n;
        } else if (!schurb) {
            if (n > minwrk)
                minwrk = n;
        }

        if (cont) {
            if (!schura) {
                if (ib + m > minwrk)
                    minwrk = ib + m;
            }
        } else {
            if (ib + 2*m > minwrk)
                minwrk = ib + 2*m;
        }

        if (ia + minwrk > 1)
            minwrk = ia + minwrk;
        else
            minwrk = 1;

        if (ldwork < minwrk)
            *info = -21;
    }

    if (*info != 0) {
        return;
    }

    if (m == 0 || n == 0) {
        *scale = one;
        dwork[0] = one;
        return;
    }

    maxwrk = minwrk;

    if (nofaca) {
        jwork = 2*m + 2;
        ia = jwork;
        availw = ldwork - jwork + 1;

        i32 mm = m;
        i32 ldaa = lda;
        i32 lduu = ldu;
        i32 bwork_unused = 0;

        SLC_DGEES("V", "N", select_dummy, &mm, a, &ldaa, &sdim,
                  &dwork[1], &dwork[m+1], u, &lduu, &dwork[jwork-1],
                  &availw, &bwork_unused, &ierr);

        if (ierr > 0) {
            *info = ierr;
            return;
        }
        i32 optwork = (i32)dwork[jwork-1] + jwork - 1;
        if (optwork > maxwrk)
            maxwrk = optwork;
    } else {
        jwork = 1;
        ia = 2;
        availw = ldwork;
    }

    if (!schura) {
        chunka = availw / m;
        blocka = (chunka > 1 && n > 1);
        blas3a = (chunka >= n && blocka);

        if (blas3a) {
            i32 mm = m;
            i32 nn = n;
            i32 ldcc = ldc;
            i32 lduu = ldu;
            SLC_DLACPY("F", &mm, &nn, c, &ldcc, &dwork[jwork-1], &mm);
            SLC_DGEMM("T", "N", &mm, &nn, &mm, &one, u, &lduu,
                      &dwork[jwork-1], &mm, &zero, c, &ldcc);
        } else if (blocka) {
            for (j = 1; j <= n; j += chunka) {
                bl = n - j + 1;
                if (bl > chunka) bl = chunka;
                i32 mm = m;
                i32 ldcc = ldc;
                i32 lduu = ldu;
                SLC_DLACPY("F", &mm, &bl, &c[(j-1)*ldc], &ldcc, &dwork[jwork-1], &mm);
                SLC_DGEMM("T", "N", &mm, &bl, &mm, &one, u, &lduu,
                          &dwork[jwork-1], &mm, &zero, &c[(j-1)*ldc], &ldcc);
            }
        } else {
            for (j = 1; j <= n; j++) {
                i32 mm = m;
                i32 lduu = ldu;
                SLC_DCOPY(&mm, &c[(j-1)*ldc], &int1, &dwork[jwork-1], &int1);
                SLC_DGEMV("T", &mm, &mm, &one, u, &lduu,
                          &dwork[jwork-1], &int1, &zero, &c[(j-1)*ldc], &int1);
            }
        }
        i32 optwork = jwork + m*n - 1;
        if (optwork > maxwrk)
            maxwrk = optwork;
    }

    if (nofacb) {
        jwork = ia + 2*n;
        availw = ldwork - jwork + 1;

        i32 nn = n;
        i32 ldbb = ldb;
        i32 ldvv = ldv;
        i32 bwork_unused = 0;

        SLC_DGEES("V", "N", select_dummy, &nn, b, &ldbb, &sdim,
                  &dwork[ia-1], &dwork[n+ia-1], v, &ldvv, &dwork[jwork-1],
                  &availw, &bwork_unused, &ierr);

        if (ierr > 0) {
            *info = ierr + m;
            return;
        }
        i32 optwork = (i32)dwork[jwork-1] + jwork - 1;
        if (optwork > maxwrk)
            maxwrk = optwork;

        if (!schura) {
            chunka = availw / m;
            blocka = (chunka > 1 && n > 1);
            blas3a = (chunka >= n && blocka);
        }
    }

    if (!schurb) {
        chunkb = availw / n;
        blockb = (chunkb > 1 && m > 1);
        blas3b = (chunkb >= m && blockb);

        if (blas3b) {
            i32 mm = m;
            i32 nn = n;
            i32 ldcc = ldc;
            i32 ldvv = ldv;
            SLC_DLACPY("F", &mm, &nn, c, &ldcc, &dwork[jwork-1], &mm);
            SLC_DGEMM("N", "N", &mm, &nn, &nn, &one,
                      &dwork[jwork-1], &mm, v, &ldvv, &zero, c, &ldcc);
        } else if (blockb) {
            for (i = 1; i <= m; i += chunkb) {
                bl = m - i + 1;
                if (bl > chunkb) bl = chunkb;
                i32 nn = n;
                i32 ldcc = ldc;
                i32 ldvv = ldv;
                SLC_DLACPY("F", &bl, &nn, &c[i-1], &ldcc, &dwork[jwork-1], &bl);
                SLC_DGEMM("N", "N", &bl, &nn, &nn, &one,
                          &dwork[jwork-1], &bl, v, &ldvv, &zero, &c[i-1], &ldcc);
            }
        } else {
            for (i = 1; i <= m; i++) {
                i32 nn = n;
                i32 ldvv = ldv;
                i32 ldc_val = ldc;
                SLC_DCOPY(&nn, &c[i-1], &ldc_val, &dwork[jwork-1], &int1);
                SLC_DGEMV("T", &nn, &nn, &one, v, &ldvv,
                          &dwork[jwork-1], &int1, &zero, &c[i-1], &ldc_val);
            }
        }
        i32 optwork = jwork + m*n - 1;
        if (optwork > maxwrk)
            maxwrk = optwork;
    }

    if (cont) {
        const char* tr_a = notrna ? "N" : "T";
        const char* tr_b = notrnb ? "N" : "T";
        i32 mm = m;
        i32 nn = n;
        i32 ldaa = lda;
        i32 ldbb = ldb;
        i32 ldcc = ldc;
        SLC_DTRSYL(tr_a, tr_b, &isgn, &mm, &nn, a, &ldaa, b, &ldbb,
                   c, &ldcc, scale, &ierr);
    } else {
        char tr_a = notrna ? 'N' : 'T';
        char tr_b = notrnb ? 'N' : 'T';
        i32 mm = m;
        i32 nn = n;
        i32 ldaa = lda;
        i32 ldbb = ldb;
        i32 ldcc = ldc;
        sb04py(tr_a, tr_b, isgn, mm, nn, a, ldaa, b, ldbb,
               c, ldcc, scale, &dwork[jwork-1], &ierr);
        i32 optwork = jwork + 2*m - 1;
        if (optwork > maxwrk)
            maxwrk = optwork;
    }

    if (ierr > 0)
        *info = m + n + 1;

    if (!schura) {
        if (blas3a) {
            i32 mm = m;
            i32 nn = n;
            i32 ldcc = ldc;
            i32 lduu = ldu;
            SLC_DLACPY("F", &mm, &nn, c, &ldcc, &dwork[jwork-1], &mm);
            SLC_DGEMM("N", "N", &mm, &nn, &mm, &one, u, &lduu,
                      &dwork[jwork-1], &mm, &zero, c, &ldcc);
        } else if (blocka) {
            for (j = 1; j <= n; j += chunka) {
                bl = n - j + 1;
                if (bl > chunka) bl = chunka;
                i32 mm = m;
                i32 ldcc = ldc;
                i32 lduu = ldu;
                SLC_DLACPY("F", &mm, &bl, &c[(j-1)*ldc], &ldcc, &dwork[jwork-1], &mm);
                SLC_DGEMM("N", "N", &mm, &bl, &mm, &one, u, &lduu,
                          &dwork[jwork-1], &mm, &zero, &c[(j-1)*ldc], &ldcc);
            }
        } else {
            for (j = 1; j <= n; j++) {
                i32 mm = m;
                i32 lduu = ldu;
                SLC_DCOPY(&mm, &c[(j-1)*ldc], &int1, &dwork[jwork-1], &int1);
                SLC_DGEMV("N", &mm, &mm, &one, u, &lduu,
                          &dwork[jwork-1], &int1, &zero, &c[(j-1)*ldc], &int1);
            }
        }
    }

    if (!schurb) {
        if (blas3b) {
            i32 mm = m;
            i32 nn = n;
            i32 ldcc = ldc;
            i32 ldvv = ldv;
            SLC_DLACPY("F", &mm, &nn, c, &ldcc, &dwork[jwork-1], &mm);
            SLC_DGEMM("N", "T", &mm, &nn, &nn, &one,
                      &dwork[jwork-1], &mm, v, &ldvv, &zero, c, &ldcc);
        } else if (blockb) {
            for (i = 1; i <= m; i += chunkb) {
                bl = m - i + 1;
                if (bl > chunkb) bl = chunkb;
                i32 nn = n;
                i32 ldcc = ldc;
                i32 ldvv = ldv;
                SLC_DLACPY("F", &bl, &nn, &c[i-1], &ldcc, &dwork[jwork-1], &bl);
                SLC_DGEMM("N", "T", &bl, &nn, &nn, &one,
                          &dwork[jwork-1], &bl, v, &ldvv, &zero, &c[i-1], &ldcc);
            }
        } else {
            for (i = 1; i <= m; i++) {
                i32 nn = n;
                i32 ldvv = ldv;
                i32 ldc_val = ldc;
                SLC_DCOPY(&nn, &c[i-1], &ldc_val, &dwork[jwork-1], &int1);
                SLC_DGEMV("N", &nn, &nn, &one, v, &ldvv,
                          &dwork[jwork-1], &int1, &zero, &c[i-1], &ldc_val);
            }
        }
    }

    dwork[0] = (f64)maxwrk;
}
