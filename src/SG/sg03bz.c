/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include <ctype.h>

static int delctg_dummy(const c128 *alpha_val, const c128 *beta_val) {
    (void)alpha_val;
    (void)beta_val;
    return 1;
}

void sg03bz(const char *dico, const char *fact, const char *trans,
            const i32 n, const i32 m, c128 *a, const i32 lda,
            c128 *e, const i32 lde, c128 *q, const i32 ldq,
            c128 *z, const i32 ldz, c128 *b, const i32 ldb,
            f64 *scale, c128 *alpha, c128 *beta,
            f64 *dwork, c128 *zwork, const i32 lzwork, i32 *info)
{
    const f64 mone = -1.0;
    const f64 one = 1.0;
    const f64 zero = 0.0;
    const c128 cone = 1.0 + 0.0*I;
    const c128 czero = 0.0 + 0.0*I;

    const i32 int0 = 0;
    const i32 int1 = 1;

    f64 bignms, bignum, eps, ma, mato, mb, mbto, me, meto, mn, mx, smlnum, t, tmp;
    i32 bl, i, info1, j, k, l, maxmn, minmn, minwrk, nc, nr, optwrk;
    bool isdisc, isfact, istran, lascl, lbscl, lescl, lquery, lscl, nunitq, nunitz, scalb;
    i32 bwork_dummy[1];

    char dico_upper = toupper((unsigned char)*dico);
    char fact_upper = toupper((unsigned char)*fact);
    char trans_upper = toupper((unsigned char)*trans);

    isdisc = (dico_upper == 'D');
    isfact = (fact_upper == 'F');
    istran = (trans_upper == 'C');
    lquery = (lzwork == -1);

    *info = 0;
    if (dico_upper != 'C' && dico_upper != 'D') {
        *info = -1;
    } else if (fact_upper != 'N' && fact_upper != 'F') {
        *info = -2;
    } else if (trans_upper != 'N' && trans_upper != 'C') {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (m < 0) {
        *info = -5;
    } else if (lda < (n > 0 ? n : 1)) {
        *info = -7;
    } else if (lde < (n > 0 ? n : 1)) {
        *info = -9;
    } else if (ldq < (n > 0 ? n : 1)) {
        *info = -11;
    } else if (ldz < (n > 0 ? n : 1)) {
        *info = -13;
    } else if ((istran && ldb < (n > 0 ? n : 1)) ||
               (!istran && ldb < ((m > n ? m : n) > 0 ? (m > n ? m : n) : 1))) {
        *info = -15;
    } else {
        minwrk = 1;
        if (2*n > minwrk) minwrk = 2*n;
        if (3*n - 3 > minwrk) minwrk = 3*n - 3;
        if (minwrk < 1) minwrk = 1;
        maxmn = (m > n) ? m : n;

        if (lquery) {
            optwrk = minwrk;
            if (!isfact) {
                i32 lwork_query = -1;
                SLC_ZGGES("V", "V", "N", delctg_dummy,
                          &n, a, &lda, e, &lde, &i, alpha, beta, q, &ldq, z, &ldz,
                          zwork, &lwork_query, dwork, bwork_dummy, &info1);
                i32 zgges_opt = (i32)creal(zwork[0]);
                if (zgges_opt > optwrk) optwrk = zgges_opt;
            }
            if (istran) {
                i32 lwork_query = -1;
                SLC_ZGERQF(&n, &maxmn, b, &ldb, zwork, zwork, &lwork_query, &info1);
                i32 rqf_opt = (i32)creal(zwork[0]) + n;
                if (rqf_opt > optwrk) optwrk = rqf_opt;
            } else {
                i32 lwork_query = -1;
                SLC_ZGEQRF(&maxmn, &n, b, &ldb, zwork, zwork, &lwork_query, &info1);
                i32 qrf_opt = (i32)creal(zwork[0]) + n;
                if (qrf_opt > optwrk) optwrk = qrf_opt;
            }
        } else if (lzwork < minwrk) {
            zwork[0] = (f64)minwrk + 0.0*I;
            *info = -21;
        }
    }

    if (*info != 0) {
        return;
    } else if (lquery) {
        zwork[0] = (f64)optwrk + 0.0*I;
        return;
    }

    *scale = one;

    if (istran) {
        k = n;
        l = m;
    } else {
        k = m;
        l = n;
    }
    mb = SLC_ZLANGE("M", &k, &l, b, &ldb, dwork);
    if (mb == zero) {
        if (n > 0) {
            SLC_ZLASET("F", &n, &n, &czero, &czero, b, &ldb);
        }
        zwork[0] = cone;
        return;
    }

    eps = SLC_DLAMCH("P");
    smlnum = SLC_DLAMCH("S");
    bignms = one / smlnum;
    SLC_DLABAD(&smlnum, &bignms);
    smlnum = sqrt(smlnum) / eps;
    bignum = one / smlnum;

    optwrk = minwrk;

    if (!isfact) {
        SLC_ZGGES("V", "V", "N", delctg_dummy,
                  &n, a, &lda, e, &lde, &i, alpha, beta, q, &ldq, z, &ldz,
                  zwork, &lzwork, dwork, bwork_dummy, &info1);
        if (info1 != 0) {
            *info = 4;
            return;
        }
        optwrk = (i32)creal(zwork[0]);
    } else {
        i32 ldap1 = lda + 1;
        i32 ldep1 = lde + 1;
        SLC_ZCOPY(&n, a, &ldap1, alpha, &int1);
        SLC_ZCOPY(&n, e, &ldep1, beta, &int1);
        optwrk = minwrk;
    }

    nunitq = !ma02hz("A", n, n, cone, q, ldq);
    nunitz = !ma02hz("A", n, n, cone, z, ldz);

    if (isdisc) {
        for (i = 0; i < n; i++) {
            if (cabs(alpha[i]) >= creal(beta[i])) {
                *info = 6;
                return;
            }
        }
    } else {
        for (i = 0; i < n; i++) {
            if (alpha[i] == czero || beta[i] == czero ||
                (copysign(one, creal(alpha[i])) * copysign(one, creal(beta[i])) >= zero)) {
                *info = 5;
                return;
            }
        }
    }

    f64 ma_tmp = SLC_ZLANTR("M", "U", "N", &n, &n, a, &lda, dwork);
    ma = (ma_tmp < bignms) ? ma_tmp : bignms;
    f64 me_tmp = SLC_ZLANTR("M", "U", "N", &n, &n, e, &lde, dwork);
    me = (me_tmp < bignms) ? me_tmp : bignms;

    mn = ma;
    if (me < mn) mn = me;
    if (mb < mn) mn = mb;

    mx = ma;
    if (me > mx) mx = me;
    if (mb > mx) mx = mb;

    lscl = (mn < mx * smlnum) || (mx < smlnum) || (mn > bignum);
    if (lscl) {
        mato = one;
        meto = one;
        mbto = one;
        lascl = true;
        lescl = true;
        lbscl = true;
    } else {
        if (ma > zero && ma < smlnum) {
            mato = smlnum;
            lascl = true;
        } else if (ma > bignum) {
            mato = bignum;
            lascl = true;
        } else {
            mato = ma;
            lascl = false;
        }

        if (me > zero && me < smlnum) {
            meto = smlnum;
            lescl = true;
        } else if (me > bignum) {
            meto = bignum;
            lescl = true;
        } else {
            meto = me;
            lescl = false;
        }

        if (mb > zero && mb < smlnum) {
            mbto = smlnum;
            lbscl = true;
        } else if (mb > bignum) {
            mbto = bignum;
            lbscl = true;
        } else {
            mbto = one;
            lbscl = false;
        }
    }

    if (isdisc && lascl && lescl) {
        if (mato / ma > meto / me) {
            me = ma;
            meto = mato;
        }
    }

    if (lascl) {
        SLC_ZLASCL("U", &int0, &int0, &ma, &mato, &n, &n, a, &lda, &info1);
    }
    if (lescl) {
        SLC_ZLASCL("U", &int0, &int0, &me, &meto, &n, &n, e, &lde, &info1);
    }
    scalb = (mb > bignms);
    mb = (mb < bignms) ? mb : bignms;
    if (lbscl && scalb) {
        SLC_ZLASCL("G", &int0, &int0, &mb, &mbto, &k, &l, b, &ldb, &info1);
    }

    if (istran) {
        if (nunitq) {
            nc = lzwork / n;
            for (j = 0; j < m; j += nc) {
                bl = m - j;
                if (bl > nc) bl = nc;
                SLC_ZGEMM("C", "N", &n, &bl, &n, &cone, q, &ldq,
                         &b[j * ldb], &ldb, &czero, zwork, &n);
                SLC_ZLACPY("A", &n, &bl, zwork, &n, &b[j * ldb], &ldb);
            }
        }
    } else {
        if (nunitz) {
            nr = lzwork / n;
            for (i = 0; i < m; i += nr) {
                bl = m - i;
                if (bl > nr) bl = nr;
                SLC_ZGEMM("N", "N", &bl, &n, &n, &cone, &b[i], &ldb,
                         z, &ldz, &czero, zwork, &bl);
                SLC_ZLACPY("A", &bl, &n, zwork, &bl, &b[i], &ldb);
            }
        }
    }

    minmn = (m < n) ? m : n;

    if (istran) {
        i32 lwork_rqf = lzwork - n;
        SLC_ZGERQF(&n, &m, b, &ldb, zwork, &zwork[n], &lwork_rqf, &info1);

        if (n >= m) {
            if (lbscl && !scalb) {
                i32 nm_minus = n - m;
                SLC_ZLASCL("G", &int0, &int0, &mb, &mbto, &nm_minus, &m, b, &ldb, &info1);
                i32 nm1_off = n - m;
                SLC_ZLASCL("U", &int0, &int0, &mb, &mbto, &m, &m, &b[nm1_off], &ldb, &info1);
            }
            if (n > m) {
                for (i = m - 1; i >= 0; i--) {
                    i32 len = i + 1 + n - m;
                    i32 dst_col = i + n - m;
                    SLC_ZCOPY(&len, &b[i * ldb], &int1, &b[dst_col * ldb], &int1);
                }
                i32 nm_minus = n - m;
                SLC_ZLASET("A", &n, &nm_minus, &czero, &czero, b, &ldb);
            }
            if (m > 1) {
                i32 m1 = m - 1;
                i32 off_row = n - m + 1;
                i32 off_col = n - m;
                SLC_ZLASET("L", &m1, &m1, &czero, &czero, &b[off_row + off_col * ldb], &ldb);
            }
        } else {
            for (i = 0; i < n; i++) {
                i32 len = i + 1;
                i32 src_col = m - n + i;
                SLC_ZCOPY(&len, &b[src_col * ldb], &int1, &b[i * ldb], &int1);
            }
            if (lbscl && !scalb) {
                SLC_ZLASCL("U", &int0, &int0, &mb, &mbto, &n, &n, b, &ldb, &info1);
            }
            if (n > 1) {
                i32 n1 = n - 1;
                SLC_ZLASET("L", &n1, &n1, &czero, &czero, &b[1], &ldb);
            }
        }

        for (i = n - minmn; i < n; i++) {
            if (creal(b[i + i * ldb]) < zero) {
                i32 len = i + 1;
                SLC_ZDSCAL(&len, &mone, &b[i * ldb], &int1);
            }
        }
    } else {
        i32 lwork_qrf = lzwork - n;
        SLC_ZGEQRF(&m, &n, b, &ldb, zwork, &zwork[n], &lwork_qrf, &info1);

        if (lbscl && !scalb) {
            SLC_ZLASCL("U", &int0, &int0, &mb, &mbto, &m, &n, b, &ldb, &info1);
        }
        if (maxmn > 1) {
            i32 maxmn1 = maxmn - 1;
            SLC_ZLASET("L", &maxmn1, &minmn, &czero, &czero, &b[1], &ldb);
        }
        if (n > m) {
            i32 nm_diff = n - m;
            SLC_ZLASET("A", &nm_diff, &n, &czero, &czero, &b[m], &ldb);
        }

        for (i = 0; i < minmn; i++) {
            if (creal(b[i + i * ldb]) < zero) {
                i32 len = n - i;
                SLC_ZDSCAL(&len, &mone, &b[i + i * ldb], &ldb);
            }
        }
    }

    if (isdisc) {
        sg03bs(trans, n, a, lda, e, lde, b, ldb, scale, dwork, zwork, &info1);
        if (info1 != 0) {
            if (info1 == 3) {
                *info = 6;
            }
            if (info1 == 4) {
                *info = 7;
            }
            return;
        }
    } else {
        sg03bt(trans, n, a, lda, e, lde, b, ldb, scale, dwork, zwork, &info1);
        if (info1 != 0) {
            if (info1 == 3) {
                *info = 5;
            }
            return;
        }
    }

    if (istran) {
        if (nunitz) {
            mb01uz("R", "U", "N", n, n, cone, b, ldb, z, ldz, zwork, lzwork, &info1);

            i32 lwork_rqf = lzwork - n;
            SLC_ZGERQF(&n, &n, b, &ldb, zwork, &zwork[n], &lwork_rqf, &info1);

            if (n > 1) {
                i32 n1 = n - 1;
                SLC_ZLASET("L", &n1, &n1, &czero, &czero, &b[1], &ldb);
            }

            for (i = 0; i < n; i++) {
                if (creal(b[i + i * ldb]) < zero) {
                    i32 len = i + 1;
                    SLC_ZDSCAL(&len, &mone, &b[i * ldb], &int1);
                }
            }
        }
    } else {
        if (nunitq) {
            mb01uz("R", "U", "C", n, n, cone, b, ldb, q, ldq, zwork, lzwork, &info1);

            for (i = 0; i < n; i++) {
                i32 len = i + 1;
                SLC_ZSWAP(&len, &b[i], &ldb, &b[i * ldb], &int1);
            }

            for (i = 0; i < n; i++) {
                SLC_ZLACGV(&n, &b[i * ldb], &int1);
            }

            i32 lwork_qrf = lzwork - n;
            SLC_ZGEQRF(&n, &n, b, &ldb, zwork, &zwork[n], &lwork_qrf, &info1);

            if (n > 1) {
                i32 n1 = n - 1;
                SLC_ZLASET("L", &n1, &n1, &czero, &czero, &b[1], &ldb);
            }

            for (i = 0; i < n; i++) {
                if (creal(b[i + i * ldb]) < zero) {
                    i32 len = n - i;
                    SLC_ZDSCAL(&len, &mone, &b[i + i * ldb], &ldb);
                }
            }
        }
    }

    tmp = one;
    if (lascl) {
        SLC_ZLASCL("U", &int0, &int0, &mato, &ma, &n, &n, a, &lda, &info1);
        tmp = sqrt(mato / ma);
    }
    if (lescl) {
        SLC_ZLASCL("U", &int0, &int0, &meto, &me, &n, &n, e, &lde, &info1);
        tmp = tmp * sqrt(meto / me);
    }
    if (lbscl) {
        mx = SLC_ZLANTR("M", "U", "N", &n, &n, b, &ldb, dwork);
        mn = (tmp < mb) ? tmp : mb;
        t = (tmp > mb) ? tmp : mb;
        if (t > one) {
            if (mn > bignms / t) {
                *scale = (*scale) / t;
                tmp = tmp / t;
            }
        }
        tmp = tmp * mb;
        if (tmp > one) {
            if (mx > bignms / tmp) {
                *scale = (*scale) / mx;
                tmp = tmp / mx;
            }
        }
    }
    SLC_ZLASCL("U", &int0, &int0, &mbto, &tmp, &n, &n, b, &ldb, &info1);

    i32 opt2 = (i32)creal(zwork[n]) + n;
    if (opt2 > optwrk) optwrk = opt2;

    i32 final_opt = (optwrk > minwrk) ? optwrk : minwrk;
    zwork[0] = (f64)final_opt + 0.0*I;
}
