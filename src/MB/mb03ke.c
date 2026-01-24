/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MB03KE - Solve small periodic Sylvester-like equations
 *
 * Solves periodic Sylvester-like equations (PSLE):
 *   op(A(i))*X(i)   + isgn*X(i+1)*op(B(i)) = -scale*C(i), S(i) =  1
 *   op(A(i))*X(i+1) + isgn*X(i)  *op(B(i)) = -scale*C(i), S(i) = -1
 *
 * for i = 1, ..., K, where A, B, C are K-periodic matrix sequences,
 * A(i) are M-by-M, B(i) are N-by-N, with 1 <= M, N <= 2.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void mb03ke(const bool trana, const bool tranb, const i32 isgn,
            const i32 k, const i32 m, const i32 n,
            const f64 prec, const f64 smin, const i32 *s,
            const f64 *a, const f64 *b, f64 *c,
            f64 *scale, f64 *dwork, const i32 ldwork, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    i32 cb, i, ia1, ia3, ib1, ib3, ic1, ii, im1, ixa, ixb, ixc;
    i32 iz, j, km2, km3, kmn, l, ldw, len, minwrk, mm, mn, mn6, mn7;
    i32 nn, zc, zd, zi, zi2, zis;
    f64 ac, ad, beta, bignum, dmin, elem, scaloc, sgn, spiv, tau, temp;
    bool doscal, lquery;

    i32 int1 = 1;

    *info = 0;
    lquery = (ldwork == -1);

    mn = m * n;
    kmn = k * mn;

    minwrk = (4 * k - 3) * mn * mn + kmn;
    if (!lquery && ldwork < minwrk) {
        *info = -21;
    }

    dwork[0] = (f64)minwrk;
    if (lquery) {
        return;
    } else if (*info != 0) {
        i32 neg_info = -(*info);
        SLC_XERBLA("MB03KE", &neg_info);
        return;
    }

    bignum = prec / smin;

    mm = m * m;
    nn = n * n;
    sgn = (f64)isgn;
    ldw = 3 * mn;
    if (m == 2 && n == 2) {
        mn6 = ldw + ldw;
        mn7 = mn6 + ldw;
        km2 = kmn + kmn;
        km3 = km2 + kmn;
    } else {
        mn6 = 0;
        mn7 = 0;
        km2 = 0;
        km3 = 0;
    }

    zd = 0;
    zc = zd + ldw * mn * (k - 1);

    cb = zc + mn * kmn;

    for (j = 0; j < cb; j++) {
        dwork[j] = ZERO;
    }

    ixa = 0;
    ixb = 0;
    ixc = 0;
    im1 = k - 1;
    zi = zd + mn;

    for (i = 0; i < k - 1; i++) {
        if (s[im1] == -1) {
            ia1 = im1 * mm;
            dwork[zi] = a[ia1];
            if (m == 2) {
                ia3 = ia1 + 2;
                if (!trana) {
                    dwork[zi + 1] = a[ia1 + 1];
                    dwork[zi + ldw] = a[ia3];
                } else {
                    dwork[zi + 1] = a[ia3];
                    dwork[zi + ldw] = a[ia1 + 1];
                }
                dwork[zi + ldw + 1] = a[ia3 + 1];
            }
            if (n == 2) {
                zi2 = zi + (ldw + 1) * m;
                dwork[zi2] = dwork[zi];
                if (m == 2) {
                    dwork[zi2 + 1] = dwork[zi + 1];
                    dwork[zi2 + ldw] = dwork[zi + ldw];
                    dwork[zi2 + ldw + 1] = dwork[zi + ldw + 1];
                }
            }
        } else {
            ib1 = im1 * nn;
            dwork[zi] = sgn * b[ib1];
            if (!tranb) {
                if (m == 2) {
                    dwork[zi + ldw + 1] = dwork[zi];
                    if (n == 2) {
                        ib3 = ib1 + 2;
                        dwork[zi + 2] = sgn * b[ib3];
                        dwork[zi + ldw + 3] = dwork[zi + 2];
                        dwork[zi + mn6] = sgn * b[ib1 + 1];
                        dwork[zi + mn6 + 2] = sgn * b[ib3 + 1];
                        dwork[zi + mn7 + 1] = dwork[zi + mn6];
                        dwork[zi + mn7 + 3] = dwork[zi + mn6 + 2];
                    }
                } else if (n == 2) {
                    ib3 = ib1 + 2;
                    dwork[zi + 1] = sgn * b[ib3];
                    dwork[zi + ldw] = sgn * b[ib1 + 1];
                    dwork[zi + ldw + 1] = sgn * b[ib3 + 1];
                }
            } else {
                if (m == 2) {
                    dwork[zi + ldw + 1] = dwork[zi];
                    if (n == 2) {
                        ib3 = ib1 + 2;
                        dwork[zi + 2] = sgn * b[ib1 + 1];
                        dwork[zi + ldw + 3] = dwork[zi + 2];
                        dwork[zi + mn6] = sgn * b[ib3];
                        dwork[zi + mn6 + 2] = sgn * b[ib3 + 1];
                        dwork[zi + mn7 + 1] = dwork[zi + mn6];
                        dwork[zi + mn7 + 3] = dwork[zi + mn6 + 2];
                    }
                } else if (n == 2) {
                    ib3 = ib1 + 2;
                    dwork[zi + 1] = sgn * b[ib1 + 1];
                    dwork[zi + ldw] = sgn * b[ib3];
                    dwork[zi + ldw + 1] = sgn * b[ib3 + 1];
                }
            }
        }

        zi = zi + mn;
        if (s[i] == 1) {
            ia1 = ixa;
            dwork[zi] = a[ia1];
            if (m == 2) {
                ia3 = ia1 + 2;
                if (!trana) {
                    dwork[zi + 1] = a[ia1 + 1];
                    dwork[zi + ldw] = a[ia3];
                } else {
                    dwork[zi + 1] = a[ia3];
                    dwork[zi + ldw] = a[ia1 + 1];
                }
                dwork[zi + ldw + 1] = a[ia3 + 1];
            }
            if (n == 2) {
                zi2 = zi + (ldw + 1) * m;
                dwork[zi2] = dwork[zi];
                if (m == 2) {
                    dwork[zi2 + 1] = dwork[zi + 1];
                    dwork[zi2 + ldw] = dwork[zi + ldw];
                    dwork[zi2 + ldw + 1] = dwork[zi + ldw + 1];
                }
            }
        } else {
            ib1 = ixb;
            dwork[zi] = sgn * b[ib1];
            if (!tranb) {
                if (m == 2) {
                    dwork[zi + ldw + 1] = dwork[zi];
                    if (n == 2) {
                        ib3 = ib1 + 2;
                        dwork[zi + 2] = sgn * b[ib3];
                        dwork[zi + ldw + 3] = dwork[zi + 2];
                        dwork[zi + mn6] = sgn * b[ib1 + 1];
                        dwork[zi + mn6 + 2] = sgn * b[ib3 + 1];
                        dwork[zi + mn7 + 1] = dwork[zi + mn6];
                        dwork[zi + mn7 + 3] = dwork[zi + mn6 + 2];
                    }
                } else if (n == 2) {
                    ib3 = ib1 + 2;
                    dwork[zi + 1] = sgn * b[ib3];
                    dwork[zi + ldw] = sgn * b[ib1 + 1];
                    dwork[zi + ldw + 1] = sgn * b[ib3 + 1];
                }
            } else {
                if (m == 2) {
                    dwork[zi + ldw + 1] = dwork[zi];
                    if (n == 2) {
                        ib3 = ib1 + 2;
                        dwork[zi + 2] = sgn * b[ib1 + 1];
                        dwork[zi + ldw + 3] = dwork[zi + 2];
                        dwork[zi + mn6] = sgn * b[ib3];
                        dwork[zi + mn6 + 2] = sgn * b[ib3 + 1];
                        dwork[zi + mn7 + 1] = dwork[zi + mn6];
                        dwork[zi + mn7 + 3] = dwork[zi + mn6 + 2];
                    }
                } else if (n == 2) {
                    ib3 = ib1 + 2;
                    dwork[zi + 1] = sgn * b[ib1 + 1];
                    dwork[zi + ldw] = sgn * b[ib3];
                    dwork[zi + ldw + 1] = sgn * b[ib3 + 1];
                }
            }
        }

        ixa = ixa + mm;
        ixb = ixb + nn;
        im1 = i;
        zi = zi + mn * (ldw - 1);
    }

    ixa = ixa - mm;
    ixb = ixb - nn;
    zi = zc + kmn - mn;
    if (s[k - 2] == -1) {
        ia1 = ixa;
        dwork[zi] = a[ia1];
        if (m == 2) {
            ia3 = ia1 + 2;
            if (!trana) {
                dwork[zi + 1] = a[ia1 + 1];
                dwork[zi + kmn] = a[ia3];
            } else {
                dwork[zi + 1] = a[ia3];
                dwork[zi + kmn] = a[ia1 + 1];
            }
            dwork[zi + kmn + 1] = a[ia3 + 1];
        }
        if (n == 2) {
            zi2 = zi + (kmn + 1) * m;
            dwork[zi2] = dwork[zi];
            if (m == 2) {
                dwork[zi2 + 1] = dwork[zi + 1];
                dwork[zi2 + kmn] = dwork[zi + kmn];
                dwork[zi2 + kmn + 1] = dwork[zi + kmn + 1];
            }
        }
    } else {
        ib1 = ixb;
        dwork[zi] = sgn * b[ib1];
        if (!tranb) {
            if (m == 2) {
                dwork[zi + kmn + 1] = dwork[zi];
                if (n == 2) {
                    ib3 = ib1 + 2;
                    dwork[zi + 2] = sgn * b[ib3];
                    dwork[zi + kmn + 3] = dwork[zi + 2];
                    dwork[zi + km2] = sgn * b[ib1 + 1];
                    dwork[zi + km2 + 2] = sgn * b[ib3 + 1];
                    dwork[zi + km3 + 1] = dwork[zi + km2];
                    dwork[zi + km3 + 3] = dwork[zi + km2 + 2];
                }
            } else if (n == 2) {
                ib3 = ib1 + 2;
                dwork[zi + 1] = sgn * b[ib3];
                dwork[zi + kmn] = sgn * b[ib1 + 1];
                dwork[zi + kmn + 1] = sgn * b[ib3 + 1];
            }
        } else {
            if (m == 2) {
                dwork[zi + kmn + 1] = dwork[zi];
                if (n == 2) {
                    ib3 = ib1 + 2;
                    dwork[zi + 2] = sgn * b[ib1 + 1];
                    dwork[zi + kmn + 3] = dwork[zi + 2];
                    dwork[zi + km2] = sgn * b[ib3];
                    dwork[zi + km2 + 2] = sgn * b[ib3 + 1];
                    dwork[zi + km3 + 1] = dwork[zi + km2];
                    dwork[zi + km3 + 3] = dwork[zi + km2 + 2];
                }
            } else if (n == 2) {
                ib3 = ib1 + 2;
                dwork[zi + 1] = sgn * b[ib1 + 1];
                dwork[zi + kmn] = sgn * b[ib3];
                dwork[zi + kmn + 1] = sgn * b[ib3 + 1];
            }
        }
    }

    if (s[k - 1] == 1) {
        ia1 = ixa + mm;
        dwork[zc] = a[ia1];
        if (m == 2) {
            ia3 = ia1 + 2;
            if (!trana) {
                dwork[zc + 1] = a[ia1 + 1];
                dwork[zc + kmn] = a[ia3];
            } else {
                dwork[zc + 1] = a[ia3];
                dwork[zc + kmn] = a[ia1 + 1];
            }
            dwork[zc + kmn + 1] = a[ia3 + 1];
        }
        if (n == 2) {
            zi2 = zc + (kmn + 1) * m;
            dwork[zi2] = dwork[zc];
            if (m == 2) {
                dwork[zi2 + 1] = dwork[zc + 1];
                dwork[zi2 + kmn] = dwork[zc + kmn];
                dwork[zi2 + kmn + 1] = dwork[zc + kmn + 1];
            }
        }
    } else {
        ib1 = ixb + nn;
        dwork[zc] = sgn * b[ib1];
        if (!tranb) {
            if (m == 2) {
                dwork[zc + kmn + 1] = dwork[zc];
                if (n == 2) {
                    ib3 = ib1 + 2;
                    dwork[zc + 2] = sgn * b[ib3];
                    dwork[zc + kmn + 3] = dwork[zc + 2];
                    dwork[zc + km2] = sgn * b[ib1 + 1];
                    dwork[zc + km2 + 2] = sgn * b[ib3 + 1];
                    dwork[zc + km3 + 1] = dwork[zc + km2];
                    dwork[zc + km3 + 3] = dwork[zc + km2 + 2];
                }
            } else if (n == 2) {
                ib3 = ib1 + 2;
                dwork[zc + 1] = sgn * b[ib3];
                dwork[zc + kmn] = sgn * b[ib1 + 1];
                dwork[zc + kmn + 1] = sgn * b[ib3 + 1];
            }
        } else {
            if (m == 2) {
                dwork[zc + kmn + 1] = dwork[zc];
                if (n == 2) {
                    ib3 = ib1 + 2;
                    dwork[zc + 2] = sgn * b[ib1 + 1];
                    dwork[zc + kmn + 3] = dwork[zc + 2];
                    dwork[zc + km2] = sgn * b[ib3];
                    dwork[zc + km2 + 2] = sgn * b[ib3 + 1];
                    dwork[zc + km3 + 1] = dwork[zc + km2];
                    dwork[zc + km3 + 3] = dwork[zc + km2 + 2];
                }
            } else if (n == 2) {
                ib3 = ib1 + 2;
                dwork[zc + 1] = sgn * b[ib1 + 1];
                dwork[zc + kmn] = sgn * b[ib3];
                dwork[zc + kmn + 1] = sgn * b[ib3 + 1];
            }
        }
    }

    zi = cb + mn;
    for (l = 0; l < k - 1; l++) {
        ic1 = ixc;
        dwork[zi] = -c[ic1];
        if (m == 1) {
            if (n == 2) {
                dwork[zi + 1] = -c[ic1 + 1];
            }
        } else {
            dwork[zi + 1] = -c[ic1 + 1];
            if (n == 2) {
                dwork[zi + 2] = -c[ic1 + 2];
                dwork[zi + 3] = -c[ic1 + 3];
            }
        }
        ixc = ixc + mn;
        zi = zi + mn;
    }

    zi = cb;
    ic1 = ixc;
    dwork[zi] = -c[ic1];
    if (m == 1) {
        if (n == 2) {
            dwork[zi + 1] = -c[ic1 + 1];
        }
    } else {
        dwork[zi + 1] = -c[ic1 + 1];
        if (n == 2) {
            dwork[zi + 2] = -c[ic1 + 2];
            dwork[zi + 3] = -c[ic1 + 3];
        }
    }

    i = 0;
    ii = 0;
    zis = zd + mn;
    zi2 = zd + mn * ldw;

    dmin = bignum;

    for (l = 0; l < kmn - mn; l++) {
        ii = ii + 1;
        zi = zis + 2 * mn;
        len = 2 * mn - ii + 1;

        do {
            zi = zi - 1;
            elem = dwork[zi];
            if (elem == ZERO) {
                len = len - 1;
            }
        } while (elem == ZERO && len > 0);

        if (len > 1) {
            zi = zi - len + 1;
            SLC_DLARFG(&len, &dwork[zi], &dwork[zi + 1], &int1, &tau);
            beta = dwork[zi];
            dwork[zi] = ONE;

            i32 ncols = mn - ii;
            SLC_DLARFX("L", &len, &ncols, &dwork[zi], &tau, &dwork[zi + ldw], &ldw, dwork);

            if (i < k - 2) {
                SLC_DLARFX("L", &len, &mn, &dwork[zi], &tau, &dwork[zi2], &ldw, dwork);
            }

            SLC_DLARFX("L", &len, &mn, &dwork[zi], &tau, &dwork[zc + l], &kmn, dwork);

            SLC_DLARFX("L", &len, &int1, &dwork[zi], &tau, &dwork[cb + l], &kmn, dwork);

            dwork[zi] = beta;
            dmin = fmin(dmin, fabs(beta));
        }

        zis = zis + ldw;
        zi2 = zi2 + 1;
        if ((l + 1) % mn == 0) {
            i = i + 1;
            ii = 0;
            zi2 = zd + (i + 1) * mn * ldw;
        }
    }

    ii = 0;
    zi = zc + kmn - mn;

    for (l = kmn - mn; l < kmn; l++) {
        ii = ii + 1;
        len = mn - ii + 1;
        if (len > 1) {
            SLC_DLARFG(&len, &dwork[zi], &dwork[zi + 1], &int1, &tau);
            beta = dwork[zi];
            dwork[zi] = ONE;

            i32 ncols = mn - ii;
            SLC_DLARFX("L", &len, &ncols, &dwork[zi], &tau, &dwork[zi + kmn], &kmn, dwork);

            SLC_DLARFX("L", &len, &int1, &dwork[zi], &tau, &dwork[cb + l], &kmn, dwork);

            dwork[zi] = beta;
            dmin = fmin(dmin, fabs(beta));
        }
        zi = zi + kmn + 1;
    }

    *scale = ONE;
    doscal = false;
    dmin = fmax(dmin, smin);
    spiv = fmax(prec * dmin, smin);

    i = SLC_IDAMAX(&kmn, &dwork[cb], &int1);
    ac = fabs(dwork[cb + i - 1]);
    if (TWO * smin * ac > dmin) {
        temp = (ONE / TWO) / ac;
        SLC_DSCAL(&kmn, &temp, &dwork[cb], &int1);
        *scale = *scale * temp;
    }

    zi = cb - 1;

    for (i = kmn - 1; i >= kmn - mn; i--) {
        ad = fabs(dwork[zi]);
        ac = fabs(dwork[cb + i]);
        if (ad < spiv) {
            ad = spiv;
            dwork[zi] = spiv;
        }
        scaloc = ONE;
        if (ad < ONE && ac > ONE) {
            if (ac > bignum * ad) {
                *info = 1;
                scaloc = bignum * ad / ac;
                doscal = true;
                *scale = *scale * scaloc;
            }
        }
        temp = (dwork[cb + i] * scaloc) / dwork[zi];
        if (doscal) {
            doscal = false;
            SLC_DSCAL(&kmn, &scaloc, &dwork[cb], &int1);
        }
        dwork[cb + i] = temp;

        i32 len_ax = i;
        f64 neg_temp = -temp;
        SLC_DAXPY(&len_ax, &neg_temp, &dwork[zi - i], &int1, &dwork[cb], &int1);

        zi = zi - kmn - 1;
    }

    zis = zc - ldw;
    zi = zis + 2 * mn - 1;
    iz = 0;

    for (i = kmn - mn - 1; i >= 0; i--) {
        ad = fabs(dwork[zi]);
        ac = fabs(dwork[cb + i]);
        if (ad < spiv) {
            ad = spiv;
            dwork[zi] = spiv;
        }
        scaloc = ONE;
        if (ad < ONE && ac > ONE) {
            if (ac > bignum * ad) {
                *info = 1;
                scaloc = bignum * ad / ac;
                doscal = true;
                *scale = *scale * scaloc;
            }
        }
        temp = (dwork[cb + i] * scaloc) / dwork[zi];
        if (doscal) {
            doscal = false;
            SLC_DSCAL(&kmn, &scaloc, &dwork[cb], &int1);
        }
        dwork[cb + i] = temp;
        len = mn + (i % mn) + 1;
        zi2 = zis;
        while (dwork[zi2] == ZERO) {
            len = len - 1;
            zi2 = zi2 + 1;
        }

        j = (i - len + 1 > 0) ? (i - len + 1) : 0;
        i32 len_ax = i - j;
        f64 neg_temp = -temp;
        SLC_DAXPY(&len_ax, &neg_temp, &dwork[zi - i + j], &int1, &dwork[cb + j], &int1);

        if (mn > 1) {
            if ((i + 1) % mn == 1) {
                iz = 1 - mn;
            } else {
                iz = 1;
            }
        }
        zi = zi - ldw - iz;
        zis = zis - ldw;
    }

    ic1 = 0;
    zi = cb;

    for (l = 0; l < k; l++) {
        c[ic1] = dwork[zi];
        if (m == 1) {
            if (n == 2) {
                c[ic1 + 1] = dwork[zi + 1];
            }
        } else {
            c[ic1 + 1] = dwork[zi + 1];
            if (n == 2) {
                c[ic1 + 2] = dwork[zi + 2];
                c[ic1 + 3] = dwork[zi + 3];
            }
        }
        ic1 = ic1 + mn;
        zi = zi + mn;
    }

    dwork[0] = (f64)minwrk;
}
