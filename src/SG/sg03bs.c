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

void sg03bs(const char *trans, const i32 n, c128 *a, const i32 lda,
            c128 *e, const i32 lde, c128 *b, const i32 ldb,
            f64 *scale, f64 *dwork, c128 *zwork, i32 *info)
{
    const f64 mone = -1.0;
    const f64 one = 1.0;
    const f64 zero = 0.0;
    const c128 cone = 1.0 + 0.0*I;

    c128 m3[4], m3c[2];
    f64 d[2], es[2], w[2];
    i32 iwork[7];

    c128 m1, r, s, x, z;
    f64 bignum, c_rot, delta1, eps, m2, scale1, smlnum, t, uii;
    i32 apt, i, j, kl, kl1, upt, wpt;
    bool notrns;

    const i32 int0 = 0;
    const i32 int1 = 1;
    const i32 int2 = 2;

    char trans_upper = toupper((unsigned char)*trans);
    notrns = (trans_upper == 'N');

    if (!notrns && trans_upper != 'C') {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 0 ? n : 1)) {
        *info = -4;
    } else if (lde < (n > 0 ? n : 1)) {
        *info = -6;
    } else if (ldb < (n > 0 ? n : 1)) {
        *info = -8;
    } else {
        *info = 0;
    }

    if (*info != 0) {
        return;
    }

    *scale = one;

    if (n == 0) {
        return;
    }

    eps = SLC_DLAMCH("P");
    smlnum = SLC_DLAMCH("S") / eps;
    bignum = one / smlnum;
    SLC_DLABAD(&smlnum, &bignum);

    upt = 0;
    wpt = n - 1;
    apt = 2*n - 2;

    iwork[1] = 1;
    iwork[2] = 0;
    iwork[3] = 2;
    iwork[4] = 0;
    w[0] = one;
    w[1] = zero;

    if (notrns) {
        if (n > 1) {
            i32 nm1 = n - 1;
            i32 ldap1 = lda + 1;
            SLC_ZCOPY(&nm1, &a[1 + lda], &ldap1, &zwork[apt], &int1);
        }
        ma02ez('U', 'C', 'N', n, e, lde);

        for (kl = 0; kl < n; kl++) {
            delta1 = creal(e[kl + kl*lde]);
            t = cabs(a[kl + kl*lda]);
            m2 = (delta1 > t) ? delta1 : t;
            delta1 = delta1 / m2;
            t = t / m2;
            if (delta1 <= t) {
                *info = 3;
                return;
            }
            delta1 = sqrt(one - t) * sqrt(one + t) * m2;
            t = creal(b[kl + kl*ldb]) * smlnum;
            if (t > delta1) {
                scale1 = delta1 / t;
                *scale = scale1 * (*scale);
                SLC_ZLASCL("Upper", &int0, &int0, &one, &scale1, &n, &n, b, &ldb, info);
            }

            uii = creal(b[kl + kl*ldb]) / delta1;

            if (kl < n - 1) {
                m1 = a[kl + kl*lda] / creal(e[kl + kl*lde]);
                m2 = delta1 / creal(e[kl + kl*lde]);

                kl1 = kl + 1;
                i32 nkl = n - kl - 1;

                ma02ez('U', 'C', 'G', n - kl, &a[kl + kl*lda], lda);

                SLC_ZCOPY(&nkl, &e[kl1 + kl*lde], &int1, &zwork[upt], &int1);
                c128 neg_m1 = -m1;
                SLC_ZAXPY(&nkl, &neg_m1, &a[kl1 + kl*lda], &int1, &zwork[upt], &int1);

                for (i = 0; i < nkl; i++) {
                    zwork[upt + i] = uii * zwork[upt + i] - m2 * conj(b[kl + (kl1 + i)*ldb]);
                }

                for (j = kl1; j < n; j++) {
                    for (i = j; i < n; i++) {
                        a[i + j*lda] = m1 * a[i + j*lda] - e[i + j*lde];
                    }
                }

                SLC_ZLATRS("Lower", "N", "N", "N", &nkl, &a[kl1 + kl1*lda], &lda,
                          &zwork[upt], &scale1, dwork, info);
                if (scale1 != one) {
                    *scale = scale1 * (*scale);
                    uii = scale1 * uii;
                    SLC_ZLASCL("Upper", &int0, &int0, &one, &scale1, &n, &n, b, &ldb, info);
                }

                a[kl + kl*lda] = conj(a[kl + kl*lda]);
                for (i = 0; i < nkl; i++) {
                    a[kl1 + i + (kl1 + i)*lda] = zwork[apt + kl + i];
                }

                m3[1] = -m2 * conj(m1);
                x = m3[1];
                c128 tau_z;
                SLC_ZLARFG(&int1, &x, &m3[1], &int1, &tau_z);
                d[0] = cabs(m1) * cabs(m1);
                d[1] = m2 * m2;
                es[0] = creal(x);

                SLC_ZSTEIN(&int2, d, es, &int1, w, &iwork[1], &iwork[3], m3c, &int2,
                          dwork, &iwork[5], iwork, info);
                if (*info != 0) {
                    *info = 4;
                    return;
                }
                m3c[0] = (cone - tau_z) * m3c[0];

                SLC_ZCOPY(&nkl, &zwork[upt], &int1, &zwork[wpt], &int1);
                SLC_ZTRMV("Upper", "C", "N", &nkl, &a[kl1 + kl1*lda], &lda, &zwork[wpt], &int1);
                c128 uii_c = uii + 0.0*I;
                SLC_ZAXPY(&nkl, &uii_c, &a[kl1 + kl*lda], &int1, &zwork[wpt], &int1);
                SLC_ZSCAL(&nkl, &m3c[1], &zwork[wpt], &int1);
                SLC_ZLACGV(&nkl, &zwork[wpt], &int1);
                c128 m3c0_conj = conj(m3c[0]);
                SLC_ZAXPY(&nkl, &m3c0_conj, &b[kl + kl1*ldb], &ldb, &zwork[wpt], &int1);

                for (i = 0; i < nkl; i++) {
                    x = b[kl1 + i + (kl1 + i)*ldb];
                    z = zwork[wpt + i];
                    SLC_ZLARTG(&x, &z, &c_rot, &s, &r);
                    b[kl1 + i + (kl1 + i)*ldb] = r;
                    if (i < nkl - 1) {
                        i32 rem = nkl - i - 1;
                        SLC_ZROT(&rem, &b[kl1 + i + (kl1 + i + 1)*ldb], &ldb,
                                &zwork[wpt + i + 1], &int1, &c_rot, &s);
                    }
                }

                for (i = kl1; i < n; i++) {
                    if (creal(b[i + i*ldb]) < zero) {
                        i32 cnt = n - i;
                        SLC_ZDSCAL(&cnt, &mone, &b[i + i*ldb], &ldb);
                    }
                }

                SLC_ZLACGV(&nkl, &zwork[upt], &int1);
                SLC_ZCOPY(&nkl, &zwork[upt], &int1, &b[kl + kl1*ldb], &ldb);
            }

            b[kl + kl*ldb] = uii + 0.0*I;
        }

    } else {
        if (n > 1) {
            i32 nm1 = n - 1;
            i32 ldap1 = lda + 1;
            SLC_ZCOPY(&nm1, a, &ldap1, &zwork[apt], &int1);
        }

        for (kl = n - 1; kl >= 0; kl--) {
            delta1 = creal(e[kl + kl*lde]);
            t = cabs(a[kl + kl*lda]);
            m2 = (delta1 > t) ? delta1 : t;
            delta1 = delta1 / m2;
            t = t / m2;
            if (delta1 <= t) {
                *info = 3;
                return;
            }
            delta1 = sqrt(one - t) * sqrt(one + t) * m2;
            t = creal(b[kl + kl*ldb]) * smlnum;
            if (t > delta1) {
                scale1 = delta1 / t;
                *scale = scale1 * (*scale);
                SLC_ZLASCL("Upper", &int0, &int0, &one, &scale1, &n, &n, b, &ldb, info);
            }

            uii = creal(b[kl + kl*ldb]) / delta1;

            if (kl > 0) {
                m1 = conj(a[kl + kl*lda]) / creal(e[kl + kl*lde]);
                m2 = delta1 / creal(e[kl + kl*lde]);

                kl1 = kl;

                ma02ez('U', 'T', 'G', kl, a, lda);

                SLC_ZCOPY(&kl, &e[kl*lde], &int1, &zwork[upt], &int1);
                c128 neg_m1 = -m1;
                SLC_ZAXPY(&kl, &neg_m1, &a[kl*lda], &int1, &zwork[upt], &int1);
                SLC_ZDSCAL(&kl, &uii, &zwork[upt], &int1);
                c128 neg_m2 = -m2 + 0.0*I;
                SLC_ZAXPY(&kl, &neg_m2, &b[kl*ldb], &int1, &zwork[upt], &int1);

                for (j = 0; j < kl; j++) {
                    for (i = 0; i <= j; i++) {
                        a[i + j*lda] = m1 * a[i + j*lda] - e[i + j*lde];
                    }
                }

                SLC_ZLATRS("Upper", "N", "N", "N", &kl, a, &lda,
                          &zwork[upt], &scale1, dwork, info);
                if (scale1 != one) {
                    *scale = scale1 * (*scale);
                    uii = scale1 * uii;
                    SLC_ZLASCL("Upper", &int0, &int0, &one, &scale1, &n, &n, b, &ldb, info);
                }

                ma02ez('L', 'T', 'G', kl, a, lda);
                for (i = 0; i < kl; i++) {
                    a[i + i*lda] = zwork[apt + i];
                }

                m3[1] = -m2 * conj(m1);
                x = m3[1];
                c128 tau_z;
                SLC_ZLARFG(&int1, &x, &m3[1], &int1, &tau_z);
                d[0] = cabs(m1) * cabs(m1);
                d[1] = m2 * m2;
                es[0] = creal(x);

                SLC_ZSTEIN(&int2, d, es, &int1, w, &iwork[1], &iwork[3], m3c, &int2,
                          dwork, &iwork[5], iwork, info);
                if (*info != 0) {
                    *info = 4;
                    return;
                }
                m3c[0] = (cone - tau_z) * m3c[0];

                SLC_ZSCAL(&kl, &m3c[0], &b[kl*ldb], &int1);
                SLC_ZCOPY(&kl, &zwork[upt], &int1, &zwork[wpt], &int1);
                SLC_ZTRMV("Upper", "N", "N", &kl, a, &lda, &zwork[wpt], &int1);
                c128 uii_c = uii + 0.0*I;
                SLC_ZAXPY(&kl, &uii_c, &a[kl*lda], &int1, &zwork[wpt], &int1);
                SLC_ZAXPY(&kl, &m3c[1], &zwork[wpt], &int1, &b[kl*ldb], &int1);

                for (i = kl - 1; i >= 0; i--) {
                    x = b[i + i*ldb];
                    z = b[i + kl*ldb];
                    SLC_ZLARTG(&x, &z, &c_rot, &s, &r);
                    b[i + i*ldb] = r;
                    if (i > 0) {
                        SLC_ZROT(&i, &b[i*ldb], &int1, &b[kl*ldb], &int1, &c_rot, &s);
                    }
                }

                for (i = 0; i < kl; i++) {
                    if (creal(b[i + i*ldb]) < zero) {
                        i32 cnt = i + 1;
                        SLC_ZDSCAL(&cnt, &mone, &b[i*ldb], &int1);
                    }
                }

                SLC_ZCOPY(&kl, &zwork[upt], &int1, &b[kl*ldb], &int1);
            }

            b[kl + kl*ldb] = uii + 0.0*I;
        }
    }
}
