/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include <math.h>

void mb03bb(f64 base, f64 lgbas, f64 ulp, i32 k, const i32 *amap,
            const i32 *s, i32 sinv, f64 *a, i32 lda1, i32 lda2,
            f64 *alphar, f64 *alphai, f64 *beta, i32 *scal,
            f64 *dwork, i32 *info) {
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    *info = 0;
    i32 ldas = lda1 * lda2;
    i32 pdw = 0;

    for (i32 i = 0; i < k; i++) {
        i32 ai = amap[i] - 1;
        f64 *a_slice = a + ai * ldas;
        dwork[pdw] = a_slice[0];
        dwork[pdw + 1] = ZERO;
        dwork[pdw + 2] = a_slice[1];
        dwork[pdw + 3] = ZERO;
        dwork[pdw + 4] = a_slice[lda1];
        dwork[pdw + 5] = ZERO;
        dwork[pdw + 6] = a_slice[1 + lda1];
        dwork[pdw + 7] = ZERO;
        pdw += 8;
    }

    i32 pdm = pdw;

    f64 cs, cst;
    f64 snr[2], sntr[2], sni[2], snti[2];
    f64 tr[3][3], ti[3][3], zr[3][3], zi[3][3];

    for (i32 iiter = 0; iiter < 80; iiter++) {
        f64 lhs = sqrt(dwork[2] * dwork[2] + dwork[3] * dwork[3]);
        f64 rhs = fmax(sqrt(dwork[0] * dwork[0] + dwork[1] * dwork[1]),
                       sqrt(dwork[6] * dwork[6] + dwork[7] * dwork[7]));
        if (rhs == ZERO) {
            rhs = sqrt(dwork[4] * dwork[4] + dwork[5] * dwork[5]);
        }
        if (lhs <= ulp * rhs) {
            goto converged;
        }

        if (iiter == 0) {
            f64 ar = ONE, ai_v = -TWO;
            f64 br = TWO, bi = TWO;
            f64 r = sqrt(ar * ar + ai_v * ai_v + br * br + bi * bi);
            cs = sqrt(ar * ar + ai_v * ai_v) / r;
            snr[0] = (ar * br + ai_v * bi) / (r * sqrt(ar * ar + ai_v * ai_v));
            sni[0] = (ai_v * br - ar * bi) / (r * sqrt(ar * ar + ai_v * ai_v));
            if (cs == ZERO) {
                snr[0] = ONE;
                sni[0] = ZERO;
            }
        } else if ((iiter + 1) % 40 == 0) {
            f64 ar = ONE, ai_v = ONE;  // Fortran uses I=1 from previous loop
            f64 br = ONE, bi = -TWO;
            f64 r = sqrt(ar * ar + ai_v * ai_v + br * br + bi * bi);
            cs = sqrt(ar * ar + ai_v * ai_v) / r;
            snr[0] = (ar * br + ai_v * bi) / (r * sqrt(ar * ar + ai_v * ai_v));
            sni[0] = (ai_v * br - ar * bi) / (r * sqrt(ar * ar + ai_v * ai_v));
            if (cs == ZERO) {
                snr[0] = ONE;
                sni[0] = ZERO;
            }
        } else {
            cs = ONE;
            snr[0] = ZERO;
            sni[0] = ZERO;
            f64 temp_r = ONE / sqrt(TWO);
            cst = temp_r;
            sntr[0] = temp_r;
            snti[0] = ZERO;
            pdw = pdm;

            for (i32 i = k - 1; i >= 1; i--) {
                pdw -= 8;
                f64 temp_real = dwork[pdw];
                f64 temp_imag = dwork[pdw + 1];

                for (i32 r = 0; r < 3; r++) {
                    for (i32 c = 0; c < 3; c++) {
                        zr[r][c] = ZERO;
                        zi[r][c] = ZERO;
                    }
                }
                zr[0][0] = temp_real;
                zi[0][0] = temp_imag;
                zr[1][1] = temp_real;
                zi[1][1] = temp_imag;
                zr[2][1] = dwork[pdw + 2];
                zi[2][1] = dwork[pdw + 3];
                zr[1][2] = dwork[pdw + 4];
                zi[1][2] = dwork[pdw + 5];
                zr[2][2] = dwork[pdw + 6];
                zi[2][2] = dwork[pdw + 7];

                i32 ai = amap[i] - 1;
                if (s[ai] == sinv) {
                    for (i32 r = 0; r < 3; r++) {
                        f64 x1r = zr[r][0], x1i = zi[r][0];
                        f64 x3r = zr[r][2], x3i = zi[r][2];
                        zr[r][0] = cst * x1r + sntr[0] * x3r + snti[0] * x3i;
                        zi[r][0] = cst * x1i + sntr[0] * x3i - snti[0] * x3r;
                        zr[r][2] = cst * x3r - sntr[0] * x1r + snti[0] * x1i;
                        zi[r][2] = cst * x3i - sntr[0] * x1i - snti[0] * x1r;
                    }
                    for (i32 r = 0; r < 3; r++) {
                        f64 x1r = zr[r][0], x1i = zi[r][0];
                        f64 x2r = zr[r][1], x2i = zi[r][1];
                        zr[r][0] = cs * x1r + snr[0] * x2r + sni[0] * x2i;
                        zi[r][0] = cs * x1i + snr[0] * x2i - sni[0] * x2r;
                        zr[r][1] = cs * x2r - snr[0] * x1r + sni[0] * x1i;
                        zi[r][1] = cs * x2i - snr[0] * x1i - sni[0] * x1r;
                    }
                    f64 x1r = zr[0][0], x1i = zi[0][0];
                    f64 x3r = zr[2][0], x3i = zi[2][0];
                    f64 scale = sqrt(x1r * x1r + x1i * x1i + x3r * x3r + x3i * x3i);
                    if (scale > ZERO) {
                        cst = sqrt(x1r * x1r + x1i * x1i) / scale;
                        if (cst > ZERO) {
                            sntr[0] = (x1r * x3r + x1i * x3i) / (scale * sqrt(x1r * x1r + x1i * x1i));
                            snti[0] = (x1i * x3r - x1r * x3i) / (scale * sqrt(x1r * x1r + x1i * x1i));
                        } else {
                            sntr[0] = ONE;
                            snti[0] = ZERO;
                        }
                    }
                    f64 tr_val = cst * x1r + sntr[0] * x3r - snti[0] * x3i;
                    f64 ti_val = cst * x1i + sntr[0] * x3i + snti[0] * x3r;
                    f64 x2r = zr[1][0], x2i = zi[1][0];
                    scale = sqrt(tr_val * tr_val + ti_val * ti_val + x2r * x2r + x2i * x2i);
                    if (scale > ZERO) {
                        cs = sqrt(tr_val * tr_val + ti_val * ti_val) / scale;
                        if (cs > ZERO) {
                            snr[0] = (tr_val * x2r + ti_val * x2i) / (scale * sqrt(tr_val * tr_val + ti_val * ti_val));
                            sni[0] = (ti_val * x2r - tr_val * x2i) / (scale * sqrt(tr_val * tr_val + ti_val * ti_val));
                        } else {
                            snr[0] = ONE;
                            sni[0] = ZERO;
                        }
                    }
                } else {
                    for (i32 c = 0; c < 3; c++) {
                        f64 x1r = zr[0][c], x1i = zi[0][c];
                        f64 x3r = zr[2][c], x3i = zi[2][c];
                        zr[0][c] = cst * x1r + sntr[0] * x3r - snti[0] * x3i;
                        zi[0][c] = cst * x1i + sntr[0] * x3i + snti[0] * x3r;
                        zr[2][c] = cst * x3r - sntr[0] * x1r - snti[0] * x1i;
                        zi[2][c] = cst * x3i - sntr[0] * x1i + snti[0] * x1r;
                    }
                    for (i32 c = 0; c < 3; c++) {
                        f64 x1r = zr[0][c], x1i = zi[0][c];
                        f64 x2r = zr[1][c], x2i = zi[1][c];
                        zr[0][c] = cs * x1r + snr[0] * x2r - sni[0] * x2i;
                        zi[0][c] = cs * x1i + snr[0] * x2i + sni[0] * x2r;
                        zr[1][c] = cs * x2r - snr[0] * x1r - sni[0] * x1i;
                        zi[1][c] = cs * x2i - snr[0] * x1i + sni[0] * x1r;
                    }
                    f64 x3r = zr[2][2], x3i = zi[2][2];
                    f64 x31r = zr[2][0], x31i = zi[2][0];
                    f64 scale = sqrt(x3r * x3r + x3i * x3i + x31r * x31r + x31i * x31i);
                    if (scale > ZERO) {
                        cst = sqrt(x3r * x3r + x3i * x3i) / scale;
                        if (cst > ZERO) {
                            sntr[0] = -(x3r * x31r + x3i * x31i) / (scale * sqrt(x3r * x3r + x3i * x3i));
                            snti[0] = -(x3i * x31r - x3r * x31i) / (scale * sqrt(x3r * x3r + x3i * x3i));
                        } else {
                            sntr[0] = ONE;
                            snti[0] = ZERO;
                        }
                    }
                    for (i32 r = 0; r < 2; r++) {
                        f64 xr1 = zr[r][0], xi1 = zi[r][0];
                        f64 xr3 = zr[r][2], xi3 = zi[r][2];
                        zr[r][0] = cst * xr1 + sntr[0] * xr3 + snti[0] * xi3;
                        zi[r][0] = cst * xi1 + sntr[0] * xi3 - snti[0] * xr3;
                        zr[r][2] = cst * xr3 - sntr[0] * xr1 + snti[0] * xi1;
                        zi[r][2] = cst * xi3 - sntr[0] * xi1 - snti[0] * xr1;
                    }
                    f64 x2r = zr[1][1], x2i = zi[1][1];
                    f64 x21r = zr[1][0], x21i = zi[1][0];
                    scale = sqrt(x2r * x2r + x2i * x2i + x21r * x21r + x21i * x21i);
                    if (scale > ZERO) {
                        cs = sqrt(x2r * x2r + x2i * x2i) / scale;
                        if (cs > ZERO) {
                            snr[0] = -(x2r * x21r + x2i * x21i) / (scale * sqrt(x2r * x2r + x2i * x2i));
                            sni[0] = -(x2i * x21r - x2r * x21i) / (scale * sqrt(x2r * x2r + x2i * x2i));
                        } else {
                            snr[0] = ONE;
                            sni[0] = ZERO;
                        }
                    }
                }
            }

            pdw = 0;
            zr[0][0] = dwork[pdw];
            zi[0][0] = dwork[pdw + 1];
            zr[1][0] = dwork[pdw + 2];
            zi[1][0] = dwork[pdw + 3];
            zr[0][1] = -dwork[pdw + 2];
            zi[0][1] = -dwork[pdw + 3];
            zr[1][1] = ZERO;
            zi[1][1] = ZERO;
            zr[0][2] = -dwork[pdw + 6];
            zi[0][2] = -dwork[pdw + 7];
            zr[1][2] = ZERO;
            zi[1][2] = ZERO;

            for (i32 r = 0; r < 2; r++) {
                f64 x1r = zr[r][0], x1i = zi[r][0];
                f64 x3r = zr[r][2], x3i = zi[r][2];
                zr[r][0] = cst * x1r + sntr[0] * x3r + snti[0] * x3i;
                zi[r][0] = cst * x1i + sntr[0] * x3i - snti[0] * x3r;
                zr[r][2] = cst * x3r - sntr[0] * x1r + snti[0] * x1i;
                zi[r][2] = cst * x3i - sntr[0] * x1i - snti[0] * x1r;
            }
            for (i32 r = 0; r < 2; r++) {
                f64 x1r = zr[r][0], x1i = zi[r][0];
                f64 x2r = zr[r][1], x2i = zi[r][1];
                zr[r][0] = cs * x1r + snr[0] * x2r + sni[0] * x2i;
                zi[r][0] = cs * x1i + snr[0] * x2i - sni[0] * x2r;
                zr[r][1] = cs * x2r - snr[0] * x1r + sni[0] * x1i;
                zi[r][1] = cs * x2i - snr[0] * x1i - sni[0] * x1r;
            }
            f64 x1r = zr[0][0], x1i = zi[0][0];
            f64 x2r = zr[1][0], x2i = zi[1][0];
            f64 scale = sqrt(x1r * x1r + x1i * x1i + x2r * x2r + x2i * x2i);
            if (scale > ZERO) {
                cs = sqrt(x1r * x1r + x1i * x1i) / scale;
                if (cs > ZERO) {
                    snr[0] = (x1r * x2r + x1i * x2i) / (scale * sqrt(x1r * x1r + x1i * x1i));
                    sni[0] = (x1i * x2r - x1r * x2i) / (scale * sqrt(x1r * x1r + x1i * x1i));
                } else {
                    snr[0] = ONE;
                    sni[0] = ZERO;
                }
            }
        }

        cst = cs;
        sntr[0] = snr[0];
        snti[0] = sni[0];
        pdw = pdm;

        for (i32 i = k - 1; i >= 1; i--) {
            pdw -= 8;
            tr[0][0] = dwork[pdw];
            ti[0][0] = dwork[pdw + 1];
            tr[1][0] = dwork[pdw + 2];
            ti[1][0] = dwork[pdw + 3];
            tr[0][1] = dwork[pdw + 4];
            ti[0][1] = dwork[pdw + 5];
            tr[1][1] = dwork[pdw + 6];
            ti[1][1] = dwork[pdw + 7];

            i32 ai = amap[i] - 1;
            if (s[ai] == sinv) {
                for (i32 r = 0; r < 2; r++) {
                    f64 x1r = tr[r][0], x1i = ti[r][0];
                    f64 x2r = tr[r][1], x2i = ti[r][1];
                    tr[r][0] = cs * x1r + snr[0] * x2r + sni[0] * x2i;
                    ti[r][0] = cs * x1i + snr[0] * x2i - sni[0] * x2r;
                    tr[r][1] = cs * x2r - snr[0] * x1r + sni[0] * x1i;
                    ti[r][1] = cs * x2i - snr[0] * x1i - sni[0] * x1r;
                }
                f64 x1r = tr[0][0], x1i = ti[0][0];
                f64 x2r = tr[1][0], x2i = ti[1][0];
                f64 scale = sqrt(x1r * x1r + x1i * x1i + x2r * x2r + x2i * x2i);
                if (scale > ZERO) {
                    cs = sqrt(x1r * x1r + x1i * x1i) / scale;
                    if (cs > ZERO) {
                        snr[0] = (x1r * x2r + x1i * x2i) / (scale * sqrt(x1r * x1r + x1i * x1i));
                        sni[0] = (x1i * x2r - x1r * x2i) / (scale * sqrt(x1r * x1r + x1i * x1i));
                    } else {
                        snr[0] = ONE;
                        sni[0] = ZERO;
                    }
                }
                tr[0][0] = cs * x1r + snr[0] * x2r - sni[0] * x2i;
                ti[0][0] = cs * x1i + snr[0] * x2i + sni[0] * x2r;
                tr[1][0] = ZERO;
                ti[1][0] = ZERO;
                {
                    f64 y1r = tr[0][1], y1i = ti[0][1];
                    f64 y2r = tr[1][1], y2i = ti[1][1];
                    tr[0][1] = cs * y1r + snr[0] * y2r - sni[0] * y2i;
                    ti[0][1] = cs * y1i + snr[0] * y2i + sni[0] * y2r;
                    tr[1][1] = cs * y2r - snr[0] * y1r - sni[0] * y1i;
                    ti[1][1] = cs * y2i - snr[0] * y1i + sni[0] * y1r;
                }
            } else {
                for (i32 c = 0; c < 2; c++) {
                    f64 x1r = tr[0][c], x1i = ti[0][c];
                    f64 x2r = tr[1][c], x2i = ti[1][c];
                    tr[0][c] = cs * x1r + snr[0] * x2r - sni[0] * x2i;
                    ti[0][c] = cs * x1i + snr[0] * x2i + sni[0] * x2r;
                    tr[1][c] = cs * x2r - snr[0] * x1r - sni[0] * x1i;
                    ti[1][c] = cs * x2i - snr[0] * x1i + sni[0] * x1r;
                }
                f64 x1r = tr[1][1], x1i = ti[1][1];
                f64 x2r = tr[1][0], x2i = ti[1][0];
                f64 scale = sqrt(x1r * x1r + x1i * x1i + x2r * x2r + x2i * x2i);
                if (scale > ZERO) {
                    cs = sqrt(x1r * x1r + x1i * x1i) / scale;
                    if (cs > ZERO) {
                        snr[0] = (x1r * x2r + x1i * x2i) / (scale * sqrt(x1r * x1r + x1i * x1i));
                        sni[0] = (x1i * x2r - x1r * x2i) / (scale * sqrt(x1r * x1r + x1i * x1i));
                    } else {
                        snr[0] = ONE;
                        sni[0] = ZERO;
                    }
                }
                tr[1][1] = cs * x1r + snr[0] * x2r - sni[0] * x2i;
                ti[1][1] = cs * x1i + snr[0] * x2i + sni[0] * x2r;
                tr[1][0] = ZERO;
                ti[1][0] = ZERO;
                snr[0] = -snr[0];
                sni[0] = -sni[0];
                {
                    f64 y1r = tr[0][0], y1i = ti[0][0];
                    f64 y2r = tr[0][1], y2i = ti[0][1];
                    tr[0][0] = cs * y1r + snr[0] * y2r + sni[0] * y2i;
                    ti[0][0] = cs * y1i + snr[0] * y2i - sni[0] * y2r;
                    tr[0][1] = cs * y2r - snr[0] * y1r + sni[0] * y1i;
                    ti[0][1] = cs * y2i - snr[0] * y1i - sni[0] * y1r;
                }
            }
            dwork[pdw] = tr[0][0];
            dwork[pdw + 1] = ti[0][0];
            dwork[pdw + 2] = tr[1][0];
            dwork[pdw + 3] = ti[1][0];
            dwork[pdw + 4] = tr[0][1];
            dwork[pdw + 5] = ti[0][1];
            dwork[pdw + 6] = tr[1][1];
            dwork[pdw + 7] = ti[1][1];
        }

        pdw = 0;
        tr[0][0] = dwork[pdw];
        ti[0][0] = dwork[pdw + 1];
        tr[1][0] = dwork[pdw + 2];
        ti[1][0] = dwork[pdw + 3];
        tr[0][1] = dwork[pdw + 4];
        ti[0][1] = dwork[pdw + 5];
        tr[1][1] = dwork[pdw + 6];
        ti[1][1] = dwork[pdw + 7];

        for (i32 c = 0; c < 2; c++) {
            f64 x1r = tr[0][c], x1i = ti[0][c];
            f64 x2r = tr[1][c], x2i = ti[1][c];
            tr[0][c] = cst * x1r + sntr[0] * x2r - snti[0] * x2i;
            ti[0][c] = cst * x1i + sntr[0] * x2i + snti[0] * x2r;
            tr[1][c] = cst * x2r - sntr[0] * x1r - snti[0] * x1i;
            ti[1][c] = cst * x2i - sntr[0] * x1i + snti[0] * x1r;
        }
        for (i32 r = 0; r < 2; r++) {
            f64 x1r = tr[r][0], x1i = ti[r][0];
            f64 x2r = tr[r][1], x2i = ti[r][1];
            tr[r][0] = cs * x1r + snr[0] * x2r + sni[0] * x2i;
            ti[r][0] = cs * x1i + snr[0] * x2i - sni[0] * x2r;
            tr[r][1] = cs * x2r - snr[0] * x1r + sni[0] * x1i;
            ti[r][1] = cs * x2i - snr[0] * x1i - sni[0] * x1r;
        }

        dwork[pdw] = tr[0][0];
        dwork[pdw + 1] = ti[0][0];
        dwork[pdw + 2] = tr[1][0];
        dwork[pdw + 3] = ti[1][0];
        dwork[pdw + 4] = tr[0][1];
        dwork[pdw + 5] = ti[0][1];
        dwork[pdw + 6] = tr[1][1];
        dwork[pdw + 7] = ti[1][1];
    }

    *info = 1;

converged:
    for (i32 j = 0; j < 2; j++) {
        pdw = (j == 1) ? 6 : 0;
        f64 tempi = ZERO;
        f64 tempr = ONE;
        beta[j] = ONE;
        scal[j] = 0;

        for (i32 i = 0; i < k; i++) {
            f64 rhs = sqrt(dwork[pdw] * dwork[pdw] + dwork[pdw + 1] * dwork[pdw + 1]);
            i32 sl;
            if (rhs != ZERO) {
                sl = (i32)(log(rhs) / lgbas);
                f64 scale = pow(base, (f64)sl);
                dwork[pdw] /= scale;
                dwork[pdw + 1] /= scale;
            } else {
                sl = 0;
            }
            i32 ai = amap[i] - 1;
            if (s[ai] == 1) {
                f64 lhs = tempi;
                tempi = tempr * dwork[pdw + 1] + tempi * dwork[pdw];
                tempr = tempr * dwork[pdw] - lhs * dwork[pdw + 1];
                scal[j] += sl;
            } else if (rhs == ZERO) {
                beta[j] = ZERO;
            } else {
                f64 d = dwork[pdw] * dwork[pdw] + dwork[pdw + 1] * dwork[pdw + 1];
                f64 old_r = tempr;
                tempr = (tempr * dwork[pdw] + tempi * dwork[pdw + 1]) / d;
                tempi = (tempi * dwork[pdw] - old_r * dwork[pdw + 1]) / d;
                scal[j] -= sl;
            }
            if ((i + 1) % 10 == 0 || i == k - 1) {
                rhs = sqrt(tempr * tempr + tempi * tempi);
                if (rhs == ZERO) {
                    scal[j] = 0;
                } else {
                    sl = (i32)(log(rhs) / lgbas);
                    f64 scale = pow(base, (f64)sl);
                    tempr /= scale;
                    tempi /= scale;
                    scal[j] += sl;
                }
            }
            pdw += 8;
        }
        alphar[j] = tempr;
        alphai[j] = tempi;
    }

    if (alphai[1] > ZERO) {
        f64 tmp;
        tmp = alphar[1]; alphar[1] = alphar[0]; alphar[0] = tmp;
        tmp = alphai[1]; alphai[1] = alphai[0]; alphai[0] = tmp;
        tmp = beta[1]; beta[1] = beta[0]; beta[0] = tmp;
        i32 tmp_scal = scal[1]; scal[1] = scal[0]; scal[0] = tmp_scal;
    }

    if (alphai[0] != ZERO || alphai[1] != ZERO) {
        i32 sl;
        f64 tempr, tempi, lhs_v, rhs_v, cst_v;
        if (scal[0] >= scal[1]) {
            sl = scal[0] - scal[1];
            f64 scale = pow(base, (f64)sl);
            tempr = alphar[1] / scale;
            tempi = alphai[1] / scale;
            lhs_v = alphar[0] - tempr;
            rhs_v = alphai[0] + tempi;
            cst_v = alphai[0];
        } else {
            sl = scal[1] - scal[0];
            f64 scale = pow(base, (f64)sl);
            tempr = alphar[0] / scale;
            tempi = alphai[0] / scale;
            lhs_v = alphar[1] - tempr;
            rhs_v = alphai[1] + tempi;
            cst_v = alphai[1];
        }

        f64 misr = sqrt(cst_v * cst_v + tempi * tempi);
        f64 misc = sqrt(lhs_v * lhs_v + rhs_v * rhs_v) / TWO;

        f64 cs_val = fmax(fmax(sqrt(alphar[0] * alphar[0] + alphai[0] * alphai[0]), ONE),
                          sqrt(alphar[1] * alphar[1] + alphai[1] * alphai[1]));
        if (fmin(misr, misc) > cs_val * sqrt(ulp)) {
            *info = 2;
        }

        if (misr > misc) {
            if (scal[0] >= scal[1]) {
                alphar[0] = (alphar[0] + tempr) / TWO;
                alphai[0] = fabs(alphai[0] - tempi) / TWO;
                scal[1] = scal[0];
            } else {
                alphar[0] = (alphar[1] + tempr) / TWO;
                alphai[0] = fabs(alphai[1] - tempi) / TWO;
                scal[0] = scal[1];
            }
            alphar[1] = alphar[0];
            alphai[1] = -alphai[0];
        } else {
            alphai[0] = ZERO;
            alphai[1] = ZERO;
        }
    }
}
