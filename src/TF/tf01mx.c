/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>

void tf01mx(const i32 n, const i32 m, const i32 p, const i32 ny,
            const f64* s, const i32 lds, const f64* u, const i32 ldu,
            f64* x, f64* y, const i32 ldy, f64* dwork, const i32 ldwork,
            i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const i32 int1 = 1;
    i32 i, ic, iu, iw, iy, j, jw, k, n2m, n2p, nb, nf, nm, np, ns;

    *info = 0;

    np = n + p;
    nm = n + m;
    iw = nm + np;

    // Validate parameters
    if (n < 0) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (p < 0) {
        *info = -3;
    } else if (ny < 0) {
        *info = -4;
    } else if (lds < (np > 1 ? np : 1)) {
        *info = -6;
    } else if (ldu < (ny > 1 ? ny : 1)) {
        *info = -8;
    } else if (ldy < (ny > 1 ? ny : 1)) {
        *info = -11;
    } else {
        i32 jw;
        if (n == 0 || p == 0 || ny == 0) {
            jw = 0;
        } else if (m == 0) {
            jw = np;
        } else {
            jw = iw;
        }
        if (ldwork < jw) {
            *info = -13;
        }
    }

    if (*info != 0) {
        return;
    }

    // Quick return if possible
    if (ny == 0 || p == 0) {
        return;
    }

    if (n == 0) {
        // Non-dynamic system: compute output vectors
        if (m == 0) {
            SLC_DLASET("Full", &ny, &p, &zero, &zero, y, &ldy);
        } else {
            // Y = U * S^T (since S is stored column-major and we want Y(k,:) = U(k,:) * D^T)
            SLC_DGEMM("No transpose", "Transpose", &ny, &p, &m, &one,
                      u, &ldu, s, &lds, &zero, y, &ldy);
        }
        return;
    }

    // Determine block size (use fixed value instead of ILAENV)
    nb = 64;

    // Find number of state vectors that can be accommodated in workspace
    i32 jw_val = (m == 0) ? np : iw;
    i32 nb_sq = nb * nb;
    i32 ldwork_div_jw = ldwork / jw_val;
    i32 nb_sq_div_jw = nb_sq / jw_val;
    ns = (ldwork_div_jw < nb_sq_div_jw) ? ldwork_div_jw : nb_sq_div_jw;
    ns = (ns < ny) ? ns : ny;
    n2p = n + np;

    if (m == 0) {
        // System with no inputs
        if (ns <= 1 || ny * p <= nb_sq) {
            // Small problem or limited workspace
            iy = n;

            for (i = 0; i < ny; i++) {
                // Compute [x(i+1); y(i)] = [A; C] * x(i)
                SLC_DGEMV("NoTranspose", &np, &n, &one, s, &lds, x, &int1,
                          &zero, dwork, &int1);
                SLC_DCOPY(&n, dwork, &int1, x, &int1);
                SLC_DCOPY(&p, &dwork[iy], &int1, &y[i], &ldy);
            }
        } else {
            // Large problem with sufficient workspace
            nf = (ny / ns) * ns;
            SLC_DCOPY(&n, x, &int1, dwork, &int1);

            for (i = 0; i < nf; i += ns) {
                // Compute current ns extended state vectors
                for (ic = 0; ic < (ns - 1) * np; ic += np) {
                    SLC_DGEMV("No transpose", &np, &n, &one, s, &lds,
                              &dwork[ic], &int1, &zero, &dwork[ic + np], &int1);
                }

                // Prepare next iteration
                SLC_DGEMV("No transpose", &np, &n, &one, s, &lds,
                          &dwork[(ns - 1) * np], &int1, &zero, dwork, &int1);

                // Transpose ns output vectors
                for (j = 0; j < p; j++) {
                    SLC_DCOPY(&ns, &dwork[n2p + j], &np, &y[i + j * ldy], &int1);
                }
            }

            ns = ny - nf;

            if (ns > 1) {
                // Compute last ns output vectors
                for (ic = 0; ic < (ns - 1) * np; ic += np) {
                    SLC_DGEMV("No transpose", &np, &n, &one, s, &lds,
                              &dwork[ic], &int1, &zero, &dwork[ic + np], &int1);
                }

                SLC_DGEMV("No transpose", &np, &n, &one, s, &lds,
                          &dwork[(ns - 1) * np], &int1, &zero, dwork, &int1);

                for (j = 0; j < p; j++) {
                    SLC_DCOPY(&ns, &dwork[n2p + j], &np, &y[nf + j * ldy], &int1);
                }
            } else if (ns == 1) {
                SLC_DCOPY(&n, dwork, &int1, &dwork[np], &int1);
                SLC_DGEMV("No transpose", &np, &n, &one, s, &lds,
                          &dwork[np], &int1, &zero, dwork, &int1);
                SLC_DCOPY(&p, &dwork[n], &int1, &y[nf], &ldy);
            }

            SLC_DCOPY(&n, dwork, &int1, x, &int1);
        }
    } else {
        // General case (M > 0)
        SLC_DCOPY(&n, x, &int1, dwork, &int1);

        if (ns <= 1 || ny * (m + p) <= nb_sq) {
            // Small problem or limited workspace
            iu = n;
            jw = iu + m;
            iy = jw + n;

            for (i = 0; i < ny; i++) {
                // Compute [x(i+1); y(i)] = [A,B; C,D] * [x(i); u(i)]
                SLC_DCOPY(&m, &u[i], &ldu, &dwork[iu], &int1);
                SLC_DGEMV("NoTranspose", &np, &nm, &one, s, &lds, dwork, &int1,
                          &zero, &dwork[jw], &int1);
                SLC_DCOPY(&n, &dwork[jw], &int1, dwork, &int1);
                SLC_DCOPY(&p, &dwork[iy], &int1, &y[i], &ldy);
            }
        } else {
            // Large problem with sufficient workspace
            nf = (ny / ns) * ns;
            n2m = n + nm;

            for (i = 0; i < nf; i += ns) {
                jw = 0;

                // Copy ns input vectors to workspace
                for (j = 0; j < m; j++) {
                    SLC_DCOPY(&ns, &u[i + j * ldu], &int1, &dwork[n + j], &iw);
                }

                // Compute current ns extended state vectors
                for (k = 0; k < ns - 1; k++) {
                    SLC_DGEMV("No transpose", &np, &nm, &one, s, &lds,
                              &dwork[jw], &int1, &zero, &dwork[jw + nm], &int1);
                    jw += nm;
                    SLC_DCOPY(&n, &dwork[jw], &int1, &dwork[jw + np], &int1);
                    jw += np;
                }

                // Prepare next iteration
                SLC_DGEMV("No transpose", &np, &nm, &one, s, &lds,
                          &dwork[jw], &int1, &zero, &dwork[jw + nm], &int1);
                SLC_DCOPY(&n, &dwork[jw + nm], &int1, dwork, &int1);

                // Transpose ns output vectors
                for (j = 0; j < p; j++) {
                    SLC_DCOPY(&ns, &dwork[n2m + j], &iw, &y[i + j * ldy], &int1);
                }
            }

            ns = ny - nf;

            if (ns > 1) {
                jw = 0;

                // Copy ns input vectors
                for (j = 0; j < m; j++) {
                    SLC_DCOPY(&ns, &u[nf + j * ldu], &int1, &dwork[n + j], &iw);
                }

                // Compute last ns extended state vectors
                for (k = 0; k < ns - 1; k++) {
                    SLC_DGEMV("No transpose", &np, &nm, &one, s, &lds,
                              &dwork[jw], &int1, &zero, &dwork[jw + nm], &int1);
                    jw += nm;
                    SLC_DCOPY(&n, &dwork[jw], &int1, &dwork[jw + np], &int1);
                    jw += np;
                }

                SLC_DGEMV("No transpose", &np, &nm, &one, s, &lds,
                          &dwork[jw], &int1, &zero, &dwork[jw + nm], &int1);
                SLC_DCOPY(&n, &dwork[jw + nm], &int1, dwork, &int1);

                for (j = 0; j < p; j++) {
                    SLC_DCOPY(&ns, &dwork[n2m + j], &iw, &y[nf + j * ldy], &int1);
                }
            } else if (ns == 1) {
                SLC_DCOPY(&n, dwork, &int1, &dwork[np], &int1);
                SLC_DCOPY(&m, &u[nf], &ldu, &dwork[n2p], &int1);
                SLC_DGEMV("No transpose", &np, &nm, &one, s, &lds,
                          &dwork[np], &int1, &zero, dwork, &int1);
                SLC_DCOPY(&p, &dwork[n], &int1, &y[nf], &ldy);
            }
        }

        SLC_DCOPY(&n, dwork, &int1, x, &int1);
    }
}
