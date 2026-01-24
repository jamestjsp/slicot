/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB03OS - Complex triangular Lyapunov equation solver for Cholesky factor
 *
 * Solves for X = op(U)^H * op(U) either the stable continuous-time Lyapunov equation:
 *     op(S)^H * X + X * op(S) = -scale^2 * op(R)^H * op(R)
 * or the convergent discrete-time Lyapunov equation:
 *     op(S)^H * X * op(S) - X = -scale^2 * op(R)^H * op(R)
 *
 * where S and R are complex N-by-N upper triangular matrices.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void sb03os(
    const bool discr,
    const bool ltrans,
    const i32 n,
    c128* s,
    const i32 lds,
    c128* r,
    const i32 ldr,
    f64* scale,
    f64* dwork,
    c128* zwork,
    i32* info)
{
    const c128 czero = 0.0 + 0.0*I;
    const c128 cone = 1.0 + 0.0*I;
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 two = 2.0;
    const f64 neg_one = -1.0;

    c128 alpha, sn, tmp, x, z;
    f64 absskk, bignum, c, dr, eps, scaloc, smlnum, sqtwo, temp;
    i32 i, j, k, k1, kount, kp1, ksz;
    bool slv;

    *info = 0;

    if (n < 0) {
        *info = -3;
        return;
    }
    if (lds < (n > 1 ? n : 1)) {
        *info = -5;
        return;
    }
    if (ldr < (n > 1 ? n : 1)) {
        *info = -7;
        return;
    }

    *scale = one;

    if (n == 0) {
        return;
    }

    eps = SLC_DLAMCH("P");
    smlnum = SLC_DLAMCH("S");
    bignum = one / smlnum;
    SLC_DLABAD(&smlnum, &bignum);
    smlnum = smlnum * (f64)(n*n) / eps;
    bignum = one / smlnum;

    sqtwo = sqrt(two);

    i32 int1 = 1;

    if (!ltrans) {
        kount = 1;

        while (kount <= n) {
            k = kount - 1;
            kp1 = k + 1;
            kount = kp1 + 1;

            if (discr) {
                absskk = cabs(s[k + k*lds]);
                if (absskk >= one) {
                    *info = 3;
                    return;
                }
                temp = sqrt(one - absskk) * sqrt(one + absskk);
            } else {
                temp = creal(s[k + k*lds]);
                if (temp >= zero) {
                    *info = 3;
                    return;
                }
                temp = sqtwo * sqrt(-temp);
            }

            scaloc = one;
            dr = creal(r[k + k*ldr]);
            if (temp < one && dr > one) {
                if (dr > bignum * temp)
                    scaloc = one / dr;
            }
            alpha = temp + 0.0*I;
            r[k + k*ldr] = r[k + k*ldr] / alpha;

            if (scaloc != one) {
                *scale *= scaloc;
                for (j = 0; j < n; j++) {
                    i32 jp1 = j + 1;
                    SLC_ZLASCL("Upper", &(i32){0}, &(i32){0}, &one, &scaloc, &jp1, &int1, &r[j*ldr], &ldr, info);
                }
            }

            if (kount <= n) {
                ksz = n - k - 1;
                k1 = ksz;

                z = conj(s[k + k*lds]);
                SLC_ZCOPY(&ksz, &r[k + kp1*ldr], &ldr, zwork, &int1);
                c128 neg_alpha = -alpha;
                SLC_ZSCAL(&ksz, &neg_alpha, zwork, &int1);
                if (discr) {
                    c128 coef = -z * r[k + k*ldr];
                    SLC_ZAXPY(&ksz, &coef, &s[k + kp1*lds], &lds, zwork, &int1);
                } else {
                    c128 coef = -r[k + k*ldr];
                    SLC_ZAXPY(&ksz, &coef, &s[k + kp1*lds], &lds, zwork, &int1);
                }
                SLC_ZLACGV(&ksz, zwork, &int1);

                slv = true;
                if (discr) {
                    if (z == czero) {
                        slv = false;
                        c128 neg_cone = -cone;
                        SLC_ZSCAL(&ksz, &neg_cone, zwork, &int1);
                    } else {
                        i = 0;
                        for (j = kp1; j < n; j++) {
                            i++;
                            SLC_ZSCAL(&i, &z, &s[kp1 + j*lds], &int1);
                            s[j + j*lds] = s[j + j*lds] - cone;
                        }
                    }
                } else {
                    for (j = kp1; j < n; j++) {
                        s[j + j*lds] = s[j + j*lds] + z;
                    }
                }

                if (slv) {
                    SLC_ZLATRS("Upper", "CTran", "NoDiag", "NoNorm",
                              &ksz, &s[kp1 + kp1*lds], &lds, zwork, &scaloc,
                              dwork, info);
                    if (scaloc != one) {
                        *scale *= scaloc;
                        for (j = 0; j < n; j++) {
                            i32 jp1 = j + 1;
                            SLC_ZLASCL("Upper", &(i32){0}, &(i32){0}, &one, &scaloc, &jp1, &int1, &r[j*ldr], &ldr, info);
                        }
                    }
                }

                if (discr) {
                    if (z != czero) {
                        z = cone / z;
                        i = 0;
                        for (j = kp1; j < n; j++) {
                            s[j + j*lds] = s[j + j*lds] + cone;
                            i++;
                            SLC_ZSCAL(&i, &z, &s[kp1 + j*lds], &int1);
                        }
                    }
                    SLC_ZCOPY(&ksz, zwork, &int1, &zwork[k1], &int1);
                    SLC_ZLACGV(&ksz, zwork, &int1);
                } else {
                    for (j = kp1; j < n; j++) {
                        s[j + j*lds] = s[j + j*lds] - z;
                    }
                    SLC_ZLACGV(&ksz, zwork, &int1);
                    SLC_ZCOPY(&ksz, zwork, &int1, &zwork[k1], &int1);
                }

                SLC_ZSWAP(&ksz, zwork, &int1, &r[k + kp1*ldr], &ldr);

                if (discr) {
                    c128 neg_skk = -s[k + k*lds];
                    SLC_ZSCAL(&ksz, &neg_skk, zwork, &int1);
                    c128 coef = alpha * r[k + k*ldr];
                    SLC_ZAXPY(&ksz, &coef, &s[k + kp1*lds], &lds, zwork, &int1);
                    SLC_ZLACGV(&ksz, zwork, &int1);

                    SLC_ZTRMV("Upper", "CTrans", "NoUnit", &ksz,
                             &s[kp1 + kp1*lds], &lds, &zwork[k1], &int1);
                    SLC_ZAXPY(&ksz, &alpha, &zwork[k1], &int1, zwork, &int1);
                    SLC_ZLACGV(&ksz, zwork, &int1);
                } else {
                    c128 neg_alpha = -alpha;
                    SLC_ZAXPY(&ksz, &neg_alpha, &zwork[k1], &int1, zwork, &int1);
                }

                for (i = 0; i < ksz; i++) {
                    x = r[(k+1+i) + (k+1+i)*ldr];
                    z = zwork[i];
                    SLC_ZLARTG(&x, &z, &c, &sn, &tmp);
                    r[(k+1+i) + (k+1+i)*ldr] = tmp;
                    if (i < ksz - 1) {
                        i32 rem = ksz - i - 1;
                        SLC_ZROT(&rem, &r[(k+1+i) + (k+2+i)*ldr], &ldr, &zwork[i+1], &int1, &c, &sn);
                    }
                }

                for (i = kp1; i < n; i++) {
                    if (creal(r[i + i*ldr]) < zero) {
                        i32 cnt = n - i;
                        SLC_ZDSCAL(&cnt, &neg_one, &r[i + i*ldr], &ldr);
                    }
                }
            }
        }
    } else {
        kount = n;

        while (kount >= 1) {
            k = kount - 1;
            kount = kount - 1;

            if (discr) {
                absskk = cabs(s[k + k*lds]);
                if (absskk >= one) {
                    *info = 3;
                    return;
                }
                temp = sqrt(one - absskk) * sqrt(one + absskk);
            } else {
                temp = creal(s[k + k*lds]);
                if (temp >= zero) {
                    *info = 3;
                    return;
                }
                temp = sqtwo * sqrt(-temp);
            }

            scaloc = one;
            dr = creal(r[k + k*ldr]);
            if (temp < one && dr > one) {
                if (dr > bignum * temp)
                    scaloc = one / dr;
            }
            alpha = temp + 0.0*I;
            r[k + k*ldr] = r[k + k*ldr] / alpha;

            if (scaloc != one) {
                *scale *= scaloc;
                for (j = 0; j < n; j++) {
                    i32 jp1 = j + 1;
                    SLC_ZLASCL("Upper", &(i32){0}, &(i32){0}, &one, &scaloc, &jp1, &int1, &r[j*ldr], &ldr, info);
                }
            }

            if (kount > 0) {
                ksz = k;
                k1 = ksz;

                z = conj(s[k + k*lds]);
                SLC_ZCOPY(&ksz, &r[k*ldr], &int1, zwork, &int1);
                c128 neg_alpha = -alpha;
                SLC_ZSCAL(&ksz, &neg_alpha, zwork, &int1);
                if (discr) {
                    c128 coef = -z * r[k + k*ldr];
                    SLC_ZAXPY(&ksz, &coef, &s[k*lds], &int1, zwork, &int1);
                } else {
                    c128 coef = -r[k + k*ldr];
                    SLC_ZAXPY(&ksz, &coef, &s[k*lds], &int1, zwork, &int1);
                }

                slv = true;
                if (discr) {
                    if (z == czero) {
                        slv = false;
                        c128 neg_cone = -cone;
                        SLC_ZSCAL(&ksz, &neg_cone, zwork, &int1);
                    } else {
                        for (j = 0; j < k1; j++) {
                            i32 jp1 = j + 1;
                            SLC_ZSCAL(&jp1, &z, &s[j*lds], &int1);
                            s[j + j*lds] = s[j + j*lds] - cone;
                        }
                    }
                } else {
                    for (j = 0; j < k1; j++) {
                        s[j + j*lds] = s[j + j*lds] + z;
                    }
                }

                if (slv) {
                    SLC_ZLATRS("Upper", "NoTran", "NoDiag", "NoNorm",
                              &ksz, s, &lds, zwork, &scaloc,
                              dwork, info);
                    if (scaloc != one) {
                        *scale *= scaloc;
                        for (j = 0; j < n; j++) {
                            i32 jp1 = j + 1;
                            SLC_ZLASCL("Upper", &(i32){0}, &(i32){0}, &one, &scaloc, &jp1, &int1, &r[j*ldr], &ldr, info);
                        }
                    }
                }

                if (discr) {
                    if (z != czero) {
                        z = cone / z;
                        for (j = 0; j < k1; j++) {
                            s[j + j*lds] = s[j + j*lds] + cone;
                            i32 jp1 = j + 1;
                            SLC_ZSCAL(&jp1, &z, &s[j*lds], &int1);
                        }
                    }
                } else {
                    for (j = 0; j < k1; j++) {
                        s[j + j*lds] = s[j + j*lds] - z;
                    }
                }

                SLC_ZCOPY(&ksz, zwork, &int1, &zwork[k], &int1);
                SLC_ZSWAP(&ksz, zwork, &int1, &r[k*ldr], &int1);

                if (discr) {
                    c128 neg_skk = -s[k + k*lds];
                    SLC_ZSCAL(&ksz, &neg_skk, zwork, &int1);
                    c128 coef = alpha * r[k + k*ldr];
                    SLC_ZAXPY(&ksz, &coef, &s[k*lds], &int1, zwork, &int1);

                    SLC_ZTRMV("Upper", "NoTran", "NoUnit", &ksz, s, &lds, &zwork[k], &int1);
                    SLC_ZAXPY(&ksz, &alpha, &zwork[k], &int1, zwork, &int1);
                } else {
                    c128 neg_alpha = -alpha;
                    SLC_ZAXPY(&ksz, &neg_alpha, &zwork[k], &int1, zwork, &int1);
                }

                for (i = ksz - 1; i >= 0; i--) {
                    x = r[i + i*ldr];
                    z = conj(zwork[i]);
                    SLC_ZLARTG(&x, &z, &c, &sn, &tmp);
                    r[i + i*ldr] = tmp;
                    if (i > 0) {
                        c128 sn_conj = conj(sn);
                        SLC_ZROT(&i, &r[i*ldr], &int1, zwork, &int1, &c, &sn_conj);
                    }
                }

                for (i = 0; i < ksz; i++) {
                    if (creal(r[i + i*ldr]) < zero) {
                        i32 cnt = i + 1;
                        SLC_ZDSCAL(&cnt, &neg_one, &r[i*ldr], &int1);
                    }
                }
            }
        }
    }
}
