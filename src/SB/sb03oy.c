/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB03OY - Solve 2x2 Lyapunov equation for Cholesky factor
 *
 * Solves for the Cholesky factor U of X, where op(U)'*op(U) = X, either
 * the continuous-time Lyapunov equation:
 *    op(S)'*X + X*op(S) = -ISGN*scale^2*op(R)'*op(R)
 * or the discrete-time Lyapunov equation:
 *    op(S)'*X*op(S) - X = -ISGN*scale^2*op(R)'*op(R)
 *
 * where S is 2x2 with complex conjugate eigenvalues, R is 2x2 upper triangular.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

void sb03oy(
    const bool discr,
    const bool ltrans,
    const i32 isgn,
    f64* s, const i32 lds,
    f64* r, const i32 ldr,
    f64* a, const i32 lda,
    f64* scale,
    i32* info
)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 two = 2.0;
    const f64 four = 4.0;

    f64 absb, absg, abst, alpha, bignum, e1, e2, eps;
    f64 eta, p1, p3, p3i, p3r, s11, s12, s21, s22;
    f64 scaloc, sgn, smin, smlnum, snp, snq, snt, tempi;
    f64 tempr, v1, v3;

    f64 csp[2], csq[2], cst[2], delta[2], dp[2], dt[2];
    f64 g[2], gamma_[2], p2[2], t[2], temp[2], v2[2];
    f64 x11[2], x12[2], x21[2], x22[2], y[2];

    *info = 0;
    sgn = (f64)isgn;
    s11 = s[0 + 0*lds];
    s12 = s[0 + 1*lds];
    s21 = s[1 + 0*lds];
    s22 = s[1 + 1*lds];

    eps = SLC_DLAMCH("P");
    smlnum = SLC_DLAMCH("S");
    bignum = one / smlnum;
    SLC_DLABAD(&smlnum, &bignum);
    smlnum = smlnum * four / eps;
    bignum = one / smlnum;

    smin = smlnum;
    *scale = one;

    SLC_DLANV2(&s11, &s12, &s21, &s22, &tempr, &tempi, &e1, &e2, csp, csq);
    if (tempi == zero) {
        *info = 4;
        return;
    }
    absb = SLC_DLAPY2(&e1, &e2);
    if (discr) {
        if (sgn * (absb - one) >= zero) {
            *info = 2;
            return;
        }
    } else {
        if (sgn * e1 >= zero) {
            *info = 2;
            return;
        }
    }

    temp[0] = s[0 + 0*lds] - e1;
    temp[1] = e2;
    if (ltrans) temp[1] = -e2;
    sb03ov(temp, s[1 + 0*lds], smlnum, csq, &snq);

    temp[0] = csq[0]*s[0 + 1*lds] - snq*s[0 + 0*lds];
    temp[1] = csq[1]*s[0 + 1*lds];
    tempr = csq[0]*s[1 + 1*lds] - snq*s[1 + 0*lds];
    tempi = csq[1]*s[1 + 1*lds];
    t[0] = csq[0]*temp[0] - csq[1]*temp[1] + snq*tempr;
    t[1] = csq[0]*temp[1] + csq[1]*temp[0] + snq*tempi;

    if (ltrans) {
        temp[0] = csq[0]*r[1 + 1*ldr] - snq*r[0 + 1*ldr];
        temp[1] = -csq[1]*r[1 + 1*ldr];
        sb03ov(temp, -snq*r[0 + 0*ldr], smlnum, csp, &snp);

        p1 = temp[0];
        temp[0] = csq[0]*r[0 + 1*ldr] + snq*r[1 + 1*ldr];
        temp[1] = -csq[1]*r[0 + 1*ldr];
        tempr = csq[0]*r[0 + 0*ldr];
        tempi = -csq[1]*r[0 + 0*ldr];
        p2[0] = csp[0]*temp[0] - csp[1]*temp[1] + snp*tempr;
        p2[1] = -csp[0]*temp[1] - csp[1]*temp[0] - snp*tempi;
        p3r = csp[0]*tempr + csp[1]*tempi - snp*temp[0];
        p3i = csp[0]*tempi - csp[1]*tempr - snp*temp[1];
    } else {
        temp[0] = csq[0]*r[0 + 0*ldr] + snq*r[0 + 1*ldr];
        temp[1] = csq[1]*r[0 + 0*ldr];
        sb03ov(temp, snq*r[1 + 1*ldr], smlnum, csp, &snp);

        p1 = temp[0];
        temp[0] = csq[0]*r[0 + 1*ldr] - snq*r[0 + 0*ldr];
        temp[1] = csq[1]*r[0 + 1*ldr];
        tempr = csq[0]*r[1 + 1*ldr];
        tempi = csq[1]*r[1 + 1*ldr];
        p2[0] = csp[0]*temp[0] - csp[1]*temp[1] + snp*tempr;
        p2[1] = csp[0]*temp[1] + csp[1]*temp[0] + snp*tempi;
        p3r = csp[0]*tempr + csp[1]*tempi - snp*temp[0];
        p3i = csp[1]*tempr - csp[0]*tempi + snp*temp[1];
    }

    if (p3i == zero) {
        p3 = fabs(p3r);
        dp[0] = (p3r >= zero) ? one : -one;
        dp[1] = zero;
    } else {
        p3 = SLC_DLAPY2(&p3r, &p3i);
        dp[0] = p3r / p3;
        dp[1] = -p3i / p3;
    }

    if (discr) {
        alpha = sqrt(fabs(one - absb) * (one + absb));
    } else {
        alpha = sqrt(fabs(two * e1));
    }

    scaloc = one;
    if (alpha < smin) {
        alpha = smin;
        *info = 1;
    }
    abst = fabs(p1);
    if (alpha < one && abst > one) {
        if (abst > bignum * alpha)
            scaloc = one / abst;
    }
    if (scaloc != one) {
        p1 = scaloc * p1;
        p2[0] = scaloc * p2[0];
        p2[1] = scaloc * p2[1];
        p3 = scaloc * p3;
        *scale = scaloc * (*scale);
    }
    v1 = p1 / alpha;

    if (discr) {
        g[0] = (one - e1) * (one + e1) + e2 * e2;
        g[1] = -two * e1 * e2;
        absg = SLC_DLAPY2(&g[0], &g[1]);
        scaloc = one;
        if (absg < smin) {
            absg = smin;
            *info = 1;
        }
        temp[0] = sgn * alpha * p2[0] + v1 * (e1 * t[0] - e2 * t[1]);
        temp[1] = sgn * alpha * p2[1] + v1 * (e1 * t[1] + e2 * t[0]);
        abst = fmax(fabs(temp[0]), fabs(temp[1]));
        if (absg < one && abst > one) {
            if (abst > bignum * absg)
                scaloc = one / abst;
        }
        if (scaloc != one) {
            v1 = scaloc * v1;
            temp[0] = scaloc * temp[0];
            temp[1] = scaloc * temp[1];
            p1 = scaloc * p1;
            p2[0] = scaloc * p2[0];
            p2[1] = scaloc * p2[1];
            p3 = scaloc * p3;
            *scale = scaloc * (*scale);
        }
        temp[0] = temp[0] / absg;
        temp[1] = temp[1] / absg;

        scaloc = one;
        v2[0] = g[0] * temp[0] + g[1] * temp[1];
        v2[1] = g[0] * temp[1] - g[1] * temp[0];
        abst = fmax(fabs(v2[0]), fabs(v2[1]));
        if (absg < one && abst > one) {
            if (abst > bignum * absg)
                scaloc = one / abst;
        }
        if (scaloc != one) {
            v1 = scaloc * v1;
            v2[0] = scaloc * v2[0];
            v2[1] = scaloc * v2[1];
            p1 = scaloc * p1;
            p2[0] = scaloc * p2[0];
            p2[1] = scaloc * p2[1];
            p3 = scaloc * p3;
            *scale = scaloc * (*scale);
        }
        v2[0] = v2[0] / absg;
        v2[1] = v2[1] / absg;

        scaloc = one;
        temp[0] = p1 * t[0] - two * e2 * p2[1];
        temp[1] = p1 * t[1] + two * e2 * p2[0];
        abst = fmax(fabs(temp[0]), fabs(temp[1]));
        if (absg < one && abst > one) {
            if (abst > bignum * absg)
                scaloc = one / abst;
        }
        if (scaloc != one) {
            temp[0] = scaloc * temp[0];
            temp[1] = scaloc * temp[1];
            v1 = scaloc * v1;
            v2[0] = scaloc * v2[0];
            v2[1] = scaloc * v2[1];
            p3 = scaloc * p3;
            *scale = scaloc * (*scale);
        }
        temp[0] = temp[0] / absg;
        temp[1] = temp[1] / absg;

        scaloc = one;
        y[0] = -(g[0] * temp[0] + g[1] * temp[1]);
        y[1] = -(g[0] * temp[1] - g[1] * temp[0]);
        abst = fmax(fabs(y[0]), fabs(y[1]));
        if (absg < one && abst > one) {
            if (abst > bignum * absg)
                scaloc = one / abst;
        }
        if (scaloc != one) {
            y[0] = scaloc * y[0];
            y[1] = scaloc * y[1];
            v1 = scaloc * v1;
            v2[0] = scaloc * v2[0];
            v2[1] = scaloc * v2[1];
            p3 = scaloc * p3;
            *scale = scaloc * (*scale);
        }
        y[0] = y[0] / absg;
        y[1] = y[1] / absg;
    } else {
        scaloc = one;
        if (absb < smin) {
            absb = smin;
            *info = 1;
        }
        temp[0] = sgn * alpha * p2[0] + v1 * t[0];
        temp[1] = sgn * alpha * p2[1] + v1 * t[1];
        abst = fmax(fabs(temp[0]), fabs(temp[1]));
        if (absb < one && abst > one) {
            if (abst > bignum * absb)
                scaloc = one / abst;
        }
        if (scaloc != one) {
            v1 = scaloc * v1;
            temp[0] = scaloc * temp[0];
            temp[1] = scaloc * temp[1];
            p2[0] = scaloc * p2[0];
            p2[1] = scaloc * p2[1];
            p3 = scaloc * p3;
            *scale = scaloc * (*scale);
        }
        temp[0] = temp[0] / (two * absb);
        temp[1] = temp[1] / (two * absb);
        scaloc = one;
        v2[0] = -(e1 * temp[0] + e2 * temp[1]);
        v2[1] = -(e1 * temp[1] - e2 * temp[0]);
        abst = fmax(fabs(v2[0]), fabs(v2[1]));
        if (absb < one && abst > one) {
            if (abst > bignum * absb)
                scaloc = one / abst;
        }
        if (scaloc != one) {
            v1 = scaloc * v1;
            v2[0] = scaloc * v2[0];
            v2[1] = scaloc * v2[1];
            p2[0] = scaloc * p2[0];
            p2[1] = scaloc * p2[1];
            p3 = scaloc * p3;
            *scale = scaloc * (*scale);
        }
        v2[0] = v2[0] / absb;
        v2[1] = v2[1] / absb;
        y[0] = p2[0] - alpha * v2[0];
        y[1] = p2[1] - alpha * v2[1];
    }

    scaloc = one;
    v3 = SLC_DLAPY3(&p3, &y[0], &y[1]);
    if (alpha < one && v3 > one) {
        if (v3 > bignum * alpha)
            scaloc = one / v3;
    }
    if (scaloc != one) {
        v1 = scaloc * v1;
        v2[0] = scaloc * v2[0];
        v2[1] = scaloc * v2[1];
        v3 = scaloc * v3;
        p3 = scaloc * p3;
        *scale = scaloc * (*scale);
    }
    v3 = v3 / alpha;

    if (ltrans) {
        x11[0] = csq[0] * v3;
        x11[1] = csq[1] * v3;
        x21[0] = snq * v3;
        x12[0] = csq[0] * v2[0] + csq[1] * v2[1] - snq * v1;
        x12[1] = -csq[0] * v2[1] + csq[1] * v2[0];
        x22[0] = csq[0] * v1 + snq * v2[0];
        x22[1] = -csq[1] * v1 - snq * v2[1];

        x22[1] = -x22[1];
        sb03ov(x22, x21[0], smlnum, cst, &snt);
        r[1 + 1*ldr] = x22[0];
        r[0 + 1*ldr] = cst[0] * x12[0] - cst[1] * x12[1] + snt * x11[0];
        tempr = cst[0] * x11[0] + cst[1] * x11[1] - snt * x12[0];
        tempi = cst[0] * x11[1] - cst[1] * x11[0] - snt * x12[1];
        if (tempi == zero) {
            r[0 + 0*ldr] = fabs(tempr);
            dt[0] = (tempr >= zero) ? one : -one;
            dt[1] = zero;
        } else {
            r[0 + 0*ldr] = SLC_DLAPY2(&tempr, &tempi);
            dt[0] = tempr / r[0 + 0*ldr];
            dt[1] = -tempi / r[0 + 0*ldr];
        }
    } else {
        x11[0] = csq[0] * v1 - snq * v2[0];
        x11[1] = -csq[1] * v1 + snq * v2[1];
        x21[0] = -snq * v3;
        x12[0] = csq[0] * v2[0] + csq[1] * v2[1] + snq * v1;
        x12[1] = -csq[0] * v2[1] + csq[1] * v2[0];
        x22[0] = csq[0] * v3;
        x22[1] = csq[1] * v3;

        sb03ov(x11, x21[0], smlnum, cst, &snt);
        r[0 + 0*ldr] = x11[0];
        r[0 + 1*ldr] = cst[0] * x12[0] + cst[1] * x12[1] + snt * x22[0];
        tempr = cst[0] * x22[0] - cst[1] * x22[1] - snt * x12[0];
        tempi = cst[0] * x22[1] + cst[1] * x22[0] - snt * x12[1];
        if (tempi == zero) {
            r[1 + 1*ldr] = fabs(tempr);
            dt[0] = (tempr >= zero) ? one : -one;
            dt[1] = zero;
        } else {
            r[1 + 1*ldr] = SLC_DLAPY2(&tempr, &tempi);
            dt[0] = tempr / r[1 + 1*ldr];
            dt[1] = -tempi / r[1 + 1*ldr];
        }
    }

    if ((fabs(y[0]) < smlnum) && (fabs(y[1]) <= smlnum)) {
        delta[0] = zero;
        delta[1] = zero;
        gamma_[0] = zero;
        gamma_[1] = zero;
        eta = alpha;
    } else {
        delta[0] = y[0] / v3;
        delta[1] = y[1] / v3;
        gamma_[0] = -alpha * delta[0];
        gamma_[1] = -alpha * delta[1];
        eta = p3 / v3;
        if (discr) {
            tempr = e1 * delta[0] - e2 * delta[1];
            delta[1] = e1 * delta[1] + e2 * delta[0];
            delta[0] = tempr;
        }
    }

    if (ltrans) {
        x11[0] = cst[0] * e1 + cst[1] * e2;
        x11[1] = -cst[0] * e2 + cst[1] * e1;
        x21[0] = snt * e1;
        x21[1] = -snt * e2;
        x12[0] = sgn * (cst[0] * gamma_[0] + cst[1] * gamma_[1]) - snt * e1;
        x12[1] = sgn * (-cst[0] * gamma_[1] + cst[1] * gamma_[0]) - snt * e2;
        x22[0] = cst[0] * e1 + cst[1] * e2 + sgn * snt * gamma_[0];
        x22[1] = cst[0] * e2 - cst[1] * e1 - sgn * snt * gamma_[1];

        s[0 + 0*lds] = cst[0] * x11[0] + cst[1] * x11[1] - snt * x12[0];
        tempr = cst[0] * x21[0] + cst[1] * x21[1] - snt * x22[0];
        tempi = cst[0] * x21[1] - cst[1] * x21[0] - snt * x22[1];
        s[1 + 0*lds] = dt[0] * tempr - dt[1] * tempi;
        tempr = cst[0] * x12[0] - cst[1] * x12[1] + snt * x11[0];
        tempi = cst[0] * x12[1] + cst[1] * x12[0] + snt * x11[1];
        s[0 + 1*lds] = dt[0] * tempr + dt[1] * tempi;
        s[1 + 1*lds] = cst[0] * x22[0] - cst[1] * x22[1] + snt * x21[0];

        tempr = dp[0] * eta;
        tempi = -dp[1] * eta;
        x11[0] = csp[0] * tempr - csp[1] * tempi + snp * delta[0];
        x11[1] = csp[0] * tempi + csp[1] * tempr - snp * delta[1];
        x21[0] = snp * alpha;
        x12[0] = -snp * tempr + csp[0] * delta[0] - csp[1] * delta[1];
        x12[1] = -snp * tempi - csp[0] * delta[1] - csp[1] * delta[0];
        x22[0] = csp[0] * alpha;
        x22[1] = -csp[1] * alpha;

        tempr = cst[0] * x11[0] - cst[1] * x11[1] - snt * x21[0];
        tempi = cst[0] * x22[1] + cst[1] * x22[0];
        a[0 + 0*lda] = dt[0] * tempr + dt[1] * tempi;
        tempr = cst[0] * x12[0] - cst[1] * x12[1] - snt * x22[0];
        tempi = cst[0] * x12[1] + cst[1] * x12[0] - snt * x22[1];
        a[0 + 1*lda] = dt[0] * tempr + dt[1] * tempi;
        a[1 + 0*lda] = zero;
        a[1 + 1*lda] = cst[0] * x22[0] + cst[1] * x22[1] + snt * x12[0];
    } else {
        x11[0] = cst[0] * e1 + cst[1] * e2;
        x11[1] = cst[0] * e2 - cst[1] * e1;
        x21[0] = -snt * e1;
        x21[1] = -snt * e2;
        x12[0] = sgn * (cst[0] * gamma_[0] - cst[1] * gamma_[1]) + snt * e1;
        x12[1] = sgn * (-cst[0] * gamma_[1] - cst[1] * gamma_[0]) - snt * e2;
        x22[0] = cst[0] * e1 + cst[1] * e2 - sgn * snt * gamma_[0];
        x22[1] = -cst[0] * e2 + cst[1] * e1 + sgn * snt * gamma_[1];

        s[0 + 0*lds] = cst[0] * x11[0] - cst[1] * x11[1] + snt * x12[0];
        tempr = cst[0] * x21[0] - cst[1] * x21[1] + snt * x22[0];
        tempi = cst[0] * x21[1] + cst[1] * x21[0] + snt * x22[1];
        s[1 + 0*lds] = dt[0] * tempr - dt[1] * tempi;
        tempr = cst[0] * x12[0] + cst[1] * x12[1] - snt * x11[0];
        tempi = cst[0] * x12[1] - cst[1] * x12[0] - snt * x11[1];
        s[0 + 1*lds] = dt[0] * tempr + dt[1] * tempi;
        s[1 + 1*lds] = cst[0] * x22[0] + cst[1] * x22[1] - snt * x21[0];

        tempr = dp[0] * eta;
        tempi = -dp[1] * eta;
        x11[0] = csp[0] * alpha;
        x11[1] = csp[1] * alpha;
        x21[0] = snp * alpha;
        x12[0] = csp[0] * delta[0] + csp[1] * delta[1] - snp * tempr;
        x12[1] = -csp[0] * delta[1] + csp[1] * delta[0] - snp * tempi;
        x22[0] = csp[0] * tempr + csp[1] * tempi + snp * delta[0];
        x22[1] = csp[0] * tempi - csp[1] * tempr - snp * delta[1];

        a[0 + 0*lda] = cst[0] * x11[0] - cst[1] * x11[1] + snt * x12[0];
        a[1 + 0*lda] = zero;
        a[0 + 1*lda] = cst[0] * x12[0] + cst[1] * x12[1] - snt * x11[0];
        tempr = cst[0] * x22[0] + cst[1] * x22[1] - snt * x21[0];
        tempi = cst[0] * x22[1] - cst[1] * x22[0];
        a[1 + 1*lda] = dt[0] * tempr + dt[1] * tempi;
    }

    if (*scale != one) {
        a[0 + 0*lda] = (*scale) * a[0 + 0*lda];
        a[0 + 1*lda] = (*scale) * a[0 + 1*lda];
        a[1 + 1*lda] = (*scale) * a[1 + 1*lda];
    }

    r[1 + 0*ldr] = zero;
}
