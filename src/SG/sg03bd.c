/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool lsame(const char* ca, const char* cb) {
    return (ca[0] == cb[0]) || (ca[0] == cb[0] + 32) || (ca[0] == cb[0] - 32);
}

void sg03bd(
    const char* dico, const char* fact, const char* trans,
    const i32 n, const i32 m,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    f64* b, const i32 ldb,
    f64* scale,
    f64* alphar, f64* alphai, f64* beta,
    f64* dwork, const i32 ldwork,
    i32* info
)
{
    const f64 mone = -1.0, one = 1.0, zero = 0.0;
    const i32 int1 = 1;

    bool isdisc = lsame(dico, "D");
    bool isfact = lsame(fact, "F");
    bool istran = lsame(trans, "T");
    bool lquery = (ldwork == -1);

    i32 minwrk, mingg, maxmn;

    *info = 0;

    if (!isdisc && !lsame(dico, "C")) {
        *info = -1;
    } else if (!isfact && !lsame(fact, "N")) {
        *info = -2;
    } else if (!istran && !lsame(trans, "N")) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (m < 0) {
        *info = -5;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (lde < (n > 1 ? n : 1)) {
        *info = -9;
    } else if (ldq < (n > 1 ? n : 1)) {
        *info = -11;
    } else if (ldz < (n > 1 ? n : 1)) {
        *info = -13;
    } else if ((istran && ldb < (n > 1 ? n : 1)) ||
               (!istran && ldb < ((m > n ? m : n) > 1 ? (m > n ? m : n) : 1))) {
        *info = -15;
    } else {
        if (isfact) {
            minwrk = (n > 1 ? n : 1);
            if (minwrk < 2*n) minwrk = 2*n;
            if (minwrk < 6*n-6) minwrk = 6*n-6;
        } else {
            minwrk = (n > 1 ? n : 1);
            if (minwrk < 4*n) minwrk = 4*n;
            if (minwrk < 6*n-6) minwrk = 6*n-6;
        }

        mingg = minwrk > (8*n + 16) ? minwrk : (8*n + 16);
        maxmn = m > n ? m : n;

        if (lquery) {
            i32 optwrk = minwrk;
            if (!isfact) {
                i32 info1, sdim;
                i32 bwork_tmp;
                i32 lwork_query = -1;
                SLC_DGGES("V", "V", "N", delctg, &n, a, &lda, e, &lde,
                         &sdim, alphar, alphai, beta, q, &ldq, z, &ldz,
                         dwork, &lwork_query, &bwork_tmp, &info1);
                i32 dgges_opt = (i32)dwork[0];
                optwrk = optwrk > mingg ? optwrk : mingg;
                optwrk = optwrk > dgges_opt ? optwrk : dgges_opt;
            }
            i32 info1;
            i32 lwork_query = -1;
            if (istran) {
                SLC_DGERQF(&n, &maxmn, b, &ldb, dwork, dwork, &lwork_query, &info1);
            } else {
                SLC_DGEQRF(&maxmn, &n, b, &ldb, dwork, dwork, &lwork_query, &info1);
            }
            i32 qr_opt = (i32)dwork[0];
            optwrk = optwrk > (qr_opt + n) ? optwrk : (qr_opt + n);
            dwork[0] = (f64)optwrk;
            return;
        } else if (ldwork < minwrk) {
            dwork[0] = (f64)minwrk;
            *info = -21;
        }
    }

    if (*info != 0) {
        i32 abs_info = -*info;
        SLC_XERBLA("SG03BD", &abs_info);
        return;
    }

    *scale = one;

    i32 k, l;
    if (istran) {
        k = n;
        l = m;
    } else {
        k = m;
        l = n;
    }

    f64 mb = SLC_DLANGE("M", &k, &l, b, &ldb, dwork);
    if (mb == zero) {
        if (n > 0) {
            SLC_DLASET("F", &n, &n, &zero, &zero, b, &ldb);
        }
        dwork[0] = one;
        return;
    }

    f64 eps = SLC_DLAMCH("P");
    f64 safmin = SLC_DLAMCH("S");
    f64 smlnum = safmin;
    f64 bignms = one / smlnum;
    SLC_DLABAD(&smlnum, &bignms);
    smlnum = sqrt(smlnum) / eps;
    f64 bignum = one / smlnum;

    bool nunitq, nunitz;
    i32 optwrk = minwrk;

    if (!isfact) {
        i32 info1, sdim;
        i32* bwork = (i32*)malloc(n * sizeof(i32));
        if (bwork == NULL) {
            *info = -1;
            return;
        }

        SLC_DGGES("V", "V", "N", delctg, &n, a, &lda, e, &lde,
                 &sdim, alphar, alphai, beta, q, &ldq, z, &ldz,
                 dwork, &ldwork, bwork, &info1);

        free(bwork);

        if (info1 != 0) {
            *info = 4;
            return;
        }

        nunitq = true;
        nunitz = true;
        optwrk = (i32)dwork[0];
    } else {
        for (i32 i = 0; i < n - 2; i++) {
            if (a[(i+1) + i*lda] != zero && a[(i+2) + (i+1)*lda] != zero) {
                *info = 2;
                return;
            }
        }

        f64 e1[4];
        e1[2] = zero;

        i32 i = 0;
        while (i < n) {
            if (i < n - 1) {
                if (a[(i+1) + i*lda] == zero) {
                    alphar[i] = a[i + i*lda];
                    alphai[i] = zero;
                    beta[i] = e[i + i*lde];
                } else {
                    e1[0] = e[i + i*lde];
                    e1[1] = e[i + (i+1)*lde];
                    e1[3] = e[(i+1) + (i+1)*lde];

                    f64 s1, s2, wr1, wr2, wi;
                    i32 lde1 = 2;
                    SLC_DLAG2(&a[i + i*lda], &lda, e1, &lde1, &safmin,
                             &s1, &s2, &wr1, &wr2, &wi);

                    if (wi == zero) {
                        *info = 3;
                        return;
                    }

                    alphar[i] = wr1;
                    alphai[i] = wi;
                    beta[i] = s1;
                    i++;
                    alphar[i] = wr2;
                    alphai[i] = -wi;
                    beta[i] = s2;
                }
                i++;
            } else if (i == n - 1) {
                alphar[n-1] = a[(n-1) + (n-1)*lda];
                alphai[n-1] = zero;
                beta[n-1] = e[(n-1) + (n-1)*lde];
                i++;
            }
        }

        nunitq = false;
        for (i32 j = 0; j < n && !nunitq; j++) {
            for (i32 i = 0; i < n; i++) {
                f64 expected = (i == j) ? one : zero;
                if (fabs(q[i + j*ldq] - expected) > eps) {
                    nunitq = true;
                    break;
                }
            }
        }

        nunitz = false;
        for (i32 j = 0; j < n && !nunitz; j++) {
            for (i32 i = 0; i < n; i++) {
                f64 expected = (i == j) ? one : zero;
                if (fabs(z[i + j*ldz] - expected) > eps) {
                    nunitz = true;
                    break;
                }
            }
        }

        optwrk = minwrk;
    }

    for (i32 i = 0; i < n; i++) {
        if (isdisc) {
            f64 mag = SLC_DLAPY2(&alphar[i], &alphai[i]);
            if (mag >= beta[i]) {
                *info = 6;
                return;
            }
        } else {
            if (alphar[i] == zero || beta[i] == zero ||
                (alphar[i] > zero && beta[i] > zero) ||
                (alphar[i] < zero && beta[i] < zero)) {
                *info = 5;
                return;
            }
        }
    }

    f64 ma = SLC_DLANGE("M", &n, &n, a, &lda, dwork);
    if (ma > bignms) ma = bignms;

    f64 me = SLC_DLANGE("M", &n, &n, e, &lde, dwork);
    if (me > bignms) me = bignms;

    f64 mn = ma < me ? ma : me;
    mn = mn < mb ? mn : mb;
    f64 mx = ma > me ? ma : me;
    mx = mx > mb ? mx : mb;

    bool lascl, lescl, lbscl;
    f64 mato, meto, mbto;
    bool lscl = (mn < mx*smlnum) || (mx < smlnum) || (mn > bignum);

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
            lascl = false;
            mato = one;
        }

        if (me > zero && me < smlnum) {
            meto = smlnum;
            lescl = true;
        } else if (me > bignum) {
            meto = bignum;
            lescl = true;
        } else {
            lescl = false;
            meto = one;
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
        if (mato/ma > meto/me) {
            me = ma;
            meto = mato;
        }
    }

    i32 info_tmp = 0;
    i32 kl0 = 0, ku0 = 0;
    if (lascl) {
        SLC_DLASCL("H", &kl0, &ku0, &ma, &mato, &n, &n, a, &lda, &info_tmp);
    }
    if (lescl) {
        SLC_DLASCL("U", &kl0, &ku0, &me, &meto, &n, &n, e, &lde, &info_tmp);
    }

    bool scalb = (mb > bignms);
    mb = mb < bignms ? mb : bignms;
    if (lbscl && scalb) {
        SLC_DLASCL("G", &kl0, &ku0, &mb, &mbto, &k, &l, b, &ldb, &info_tmp);
    }

    if (istran) {
        if (nunitq) {
            i32 nc = ldwork / n;
            for (i32 j = 0; j < m; j += nc) {
                i32 bl = (m - j) < nc ? (m - j) : nc;
                SLC_DGEMM("T", "N", &n, &bl, &n, &one, q, &ldq,
                         &b[0 + j*ldb], &ldb, &zero, dwork, &n);
                SLC_DLACPY("A", &n, &bl, dwork, &n, &b[0 + j*ldb], &ldb);
            }
        }
    } else {
        if (nunitq) {
            i32 nr = ldwork / n;
            for (i32 i = 0; i < m; i += nr) {
                i32 bl = (m - i) < nr ? (m - i) : nr;
                SLC_DGEMM(trans, "N", &bl, &n, &n, &one, &b[i + 0*ldb],
                         &ldb, z, &ldz, &zero, dwork, &bl);
                SLC_DLACPY("A", &bl, &n, dwork, &bl, &b[i + 0*ldb], &ldb);
            }
        }
    }

    i32 minmn = m < n ? m : n;
    i32 info1;
    i32 lwork_qr = ldwork - n;

    if (istran) {
        SLC_DGERQF(&n, &m, b, &ldb, dwork, &dwork[n], &lwork_qr, &info1);

        if (n >= m) {
            if (lbscl && !scalb) {
                i32 nm = n - m;
                SLC_DLASCL("G", &kl0, &ku0, &mb, &mbto, &nm, &m, b, &ldb, &info_tmp);
                SLC_DLASCL("U", &kl0, &ku0, &mb, &mbto, &m, &m, &b[nm + 0*ldb], &ldb, &info_tmp);
            }

            if (n > m) {
                for (i32 i = m - 1; i >= 0; i--) {
                    i32 inm = i + n - m;
                    i32 inm1 = inm + 1;
                    SLC_DCOPY(&inm1, &b[0 + i*ldb], &int1, &b[0 + inm*ldb], &int1);
                }
                i32 nm = n - m;
                SLC_DLASET("A", &n, &nm, &zero, &zero, b, &ldb);
            }

            if (m > 1) {
                i32 m1 = m - 1;
                i32 nm1 = n - m + 1;
                SLC_DLASET("L", &m1, &m1, &zero, &zero, &b[nm1 + (n-m)*ldb], &ldb);
            }
        } else {
            if (lbscl && !scalb) {
                SLC_DLASCL("U", &kl0, &ku0, &mb, &mbto, &n, &m, b, &ldb, &info_tmp);
            }

            for (i32 i = 0; i < n; i++) {
                i32 i1 = i + 1;
                SLC_DCOPY(&i1, &b[0 + (m-n+i)*ldb], &int1, &b[0 + i*ldb], &int1);
            }

            if (n > 1) {
                i32 n1 = n - 1;
                SLC_DLASET("L", &n1, &n1, &zero, &zero, &b[1 + 0*ldb], &ldb);
            }
        }

        for (i32 i = n - minmn; i < n; i++) {
            if (b[i + i*ldb] < zero) {
                i32 ip1 = i + 1;
                SLC_DSCAL(&ip1, &mone, &b[0 + i*ldb], &int1);
            }
        }
    } else {
        SLC_DGEQRF(&m, &n, b, &ldb, dwork, &dwork[n], &lwork_qr, &info1);

        if (lbscl && !scalb) {
            SLC_DLASCL("U", &kl0, &ku0, &mb, &mbto, &m, &n, b, &ldb, &info_tmp);
        }

        if (maxmn > 1) {
            i32 maxmn1 = maxmn - 1;
            SLC_DLASET("L", &maxmn1, &minmn, &zero, &zero, &b[1 + 0*ldb], &ldb);
        }

        if (n > m) {
            i32 nm = n - m;
            SLC_DLASET("A", &nm, &n, &zero, &zero, &b[m + 0*ldb], &ldb);
        }

        for (i32 i = 0; i < minmn; i++) {
            if (b[i + i*ldb] < zero) {
                i32 ni1 = n - i;
                SLC_DSCAL(&ni1, &mone, &b[i + i*ldb], &ldb);
            }
        }
    }

    if (isdisc) {
        sg03bu(trans, n, a, lda, e, lde, b, ldb, scale, dwork, &info1);
        if (info1 != 0) {
            if (info1 == 1) *info = 1;
            if (info1 == 2) *info = 3;
            if (info1 == 3) *info = 6;
            if (info1 == 4) *info = 7;
            if (*info != 1) return;
        }
    } else {
        sg03bv(trans, n, a, lda, e, lde, b, ldb, scale, dwork, &info1);
        if (info1 != 0) {
            if (info1 == 1) *info = 1;
            if (info1 >= 2) *info = 3;
            if (info1 == 3) *info = 5;
            if (*info != 1) return;
        }
    }

    if (istran) {
        if (nunitz) {
            mb01uy("R", "U", "N", n, n, one, b, ldb, z, ldz, dwork, ldwork, info);

            SLC_DGERQF(&n, &n, b, &ldb, dwork, &dwork[n], &lwork_qr, &info1);

            if (n > 1) {
                i32 n1 = n - 1;
                SLC_DLASET("L", &n1, &n1, &zero, &zero, &b[1 + 0*ldb], &ldb);
            }

            for (i32 i = 0; i < n; i++) {
                if (b[i + i*ldb] < zero) {
                    i32 ip1 = i + 1;
                    SLC_DSCAL(&ip1, &mone, &b[0 + i*ldb], &int1);
                }
            }
        }
    } else {
        if (nunitq) {
            mb01uy("R", "U", "T", n, n, one, b, ldb, q, ldq, dwork, ldwork, info);

            for (i32 i = 0; i < n; i++) {
                i32 ip1 = i + 1;
                SLC_DSWAP(&ip1, &b[i + 0*ldb], &ldb, &b[0 + i*ldb], &int1);
            }

            SLC_DGEQRF(&n, &n, b, &ldb, dwork, &dwork[n], &lwork_qr, &info1);

            if (n > 1) {
                i32 n1 = n - 1;
                SLC_DLASET("L", &n1, &n1, &zero, &zero, &b[1 + 0*ldb], &ldb);
            }

            for (i32 i = 0; i < n; i++) {
                if (b[i + i*ldb] < zero) {
                    i32 ni1 = n - i;
                    SLC_DSCAL(&ni1, &mone, &b[i + i*ldb], &ldb);
                }
            }
        }
    }

    f64 tmp = one;
    if (lascl) {
        SLC_DLASCL("H", &kl0, &ku0, &mato, &ma, &n, &n, a, &lda, &info_tmp);
        tmp = sqrt(mato / ma);
    }
    if (lescl) {
        SLC_DLASCL("U", &kl0, &ku0, &meto, &me, &n, &n, e, &lde, &info_tmp);
        tmp = tmp * sqrt(meto / me);
    }
    if (lbscl) {
        mx = SLC_DLANGE("M", &n, &n, b, &ldb, dwork);
        mn = tmp < mb ? tmp : mb;
        f64 t = tmp > mb ? tmp : mb;

        if (t > one) {
            if (mn > bignms / t) {
                *scale = *scale / t;
                tmp = tmp / t;
            }
        }

        tmp = tmp * mb;
        if (tmp > one) {
            if (mx > bignms / tmp) {
                *scale = *scale / mx;
                tmp = tmp / mx;
            }
        }
    }

    if (lascl || lescl || lbscl) {
        SLC_DLASCL("U", &kl0, &ku0, &mbto, &tmp, &n, &n, b, &ldb, &info_tmp);
    }

    optwrk = optwrk > ((i32)dwork[n] + n) ? optwrk : ((i32)dwork[n] + n);
    optwrk = optwrk > minwrk ? optwrk : minwrk;
    dwork[0] = (f64)optwrk;
}
