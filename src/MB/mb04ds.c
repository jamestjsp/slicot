/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 *
 * MB04DS - Balance a real skew-Hamiltonian matrix
 *
 * Purpose:
 *   Balances a real 2N-by-2N skew-Hamiltonian matrix:
 *       S = [  A   G  ]
 *           [  Q  A^T ]
 *   where A is N-by-N and G, Q are N-by-N skew-symmetric matrices.
 *
 *   Balancing involves:
 *   1. Permuting S to isolate eigenvalues in first 1:ILO-1 diagonal elements
 *   2. Diagonal similarity transformation on rows/columns ILO:N, N+ILO:2*N
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <math.h>

void mb04ds(const char *job, i32 n, f64 *a, i32 lda,
            f64 *qg, i32 ldqg, i32 *ilo, f64 *scale, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 FACTOR = 0.95;

    char job_upper = (char)toupper((unsigned char)job[0]);
    bool lperm = (job_upper == 'P') || (job_upper == 'B');
    bool lscal = (job_upper == 'S') || (job_upper == 'B');

    *info = 0;

    if (!lperm && !lscal && job_upper != 'N') {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -4;
    } else if (ldqg < (1 > n ? 1 : n)) {
        *info = -6;
    }

    if (*info != 0) {
        return;
    }

    *ilo = 1;

    if (n == 0) {
        return;
    }

    if (!lperm && !lscal) {
        for (i32 i = 0; i < n; i++) {
            scale[i] = ONE;
        }
        return;
    }

    i32 int1 = 1;
    f64 neg_one = -ONE;

    if (lperm) {
        i32 iloold = 0;

        while (*ilo != iloold) {
            iloold = *ilo;

            i32 i = *ilo;
            while (i <= n && *ilo == iloold) {
                bool found_nonzero = false;

                for (i32 j = *ilo; j < i && !found_nonzero; j++) {
                    if (a[(j - 1) + (i - 1) * lda] != ZERO) {
                        found_nonzero = true;
                    }
                }
                for (i32 j = i + 1; j <= n && !found_nonzero; j++) {
                    if (a[(j - 1) + (i - 1) * lda] != ZERO) {
                        found_nonzero = true;
                    }
                }
                for (i32 j = *ilo; j < i && !found_nonzero; j++) {
                    if (qg[(i - 1) + (j - 1) * ldqg] != ZERO) {
                        found_nonzero = true;
                    }
                }
                for (i32 j = i + 1; j <= n && !found_nonzero; j++) {
                    if (qg[(j - 1) + (i - 1) * ldqg] != ZERO) {
                        found_nonzero = true;
                    }
                }

                if (found_nonzero) {
                    i++;
                    continue;
                }

                scale[*ilo - 1] = (f64)i;
                if (*ilo != i) {
                    SLC_DSWAP(&n, &a[((*ilo) - 1) * lda], &int1, &a[(i - 1) * lda], &int1);
                    i32 len1 = n - *ilo + 1;
                    SLC_DSWAP(&len1, &a[(*ilo - 1) + (*ilo - 1) * lda], &lda, &a[(i - 1) + (*ilo - 1) * lda], &lda);

                    if (i < n) {
                        i32 len2 = n - i;
                        SLC_DSWAP(&len2, &qg[i + (i - 1) * ldqg], &int1, &qg[i + (*ilo - 1) * ldqg], &int1);
                    }
                    if (i > *ilo + 1) {
                        i32 len3 = i - *ilo - 1;
                        SLC_DSCAL(&len3, &neg_one, &qg[*ilo + (*ilo - 1) * ldqg], &int1);
                        SLC_DSWAP(&len3, &qg[*ilo + (*ilo - 1) * ldqg], &int1, &qg[(i - 1) + *ilo * ldqg], &ldqg);
                    }

                    i32 len4 = *ilo - 1;
                    SLC_DSWAP(&len4, &qg[i * ldqg], &int1, &qg[*ilo * ldqg], &int1);
                    if (n > i) {
                        i32 len5 = n - i;
                        SLC_DSWAP(&len5, &qg[(i - 1) + (i + 1) * ldqg], &ldqg, &qg[(*ilo - 1) + (i + 1) * ldqg], &ldqg);
                    }
                    if (i > *ilo + 1) {
                        i32 len6 = i - *ilo - 1;
                        SLC_DSCAL(&len6, &neg_one, &qg[*ilo + i * ldqg], &int1);
                        SLC_DSWAP(&len6, &qg[(*ilo - 1) + (*ilo + 1) * ldqg], &ldqg, &qg[*ilo + i * ldqg], &int1);
                    }
                    i32 len7 = i - *ilo;
                    SLC_DSCAL(&len7, &neg_one, &qg[(*ilo - 1) + i * ldqg], &int1);
                }
                (*ilo)++;
                break;
            }

            if (*ilo == iloold) {
                i = *ilo;
                while (i <= n && *ilo == iloold) {
                    bool found_nonzero = false;

                    for (i32 j = *ilo; j < i && !found_nonzero; j++) {
                        if (a[(i - 1) + (j - 1) * lda] != ZERO) {
                            found_nonzero = true;
                        }
                    }
                    for (i32 j = i + 1; j <= n && !found_nonzero; j++) {
                        if (a[(i - 1) + (j - 1) * lda] != ZERO) {
                            found_nonzero = true;
                        }
                    }
                    for (i32 j = *ilo; j < i && !found_nonzero; j++) {
                        if (qg[(j - 1) + i * ldqg] != ZERO) {
                            found_nonzero = true;
                        }
                    }
                    for (i32 j = i + 1; j <= n && !found_nonzero; j++) {
                        if (qg[(i - 1) + (j + 1) * ldqg] != ZERO) {
                            found_nonzero = true;
                        }
                    }

                    if (found_nonzero) {
                        i++;
                        continue;
                    }

                    scale[*ilo - 1] = (f64)(n + i);

                    i32 len_swap1 = i - *ilo;
                    SLC_DSWAP(&len_swap1, &a[(i - 1) + (*ilo - 1) * lda], &lda, &qg[(i - 1) + (*ilo - 1) * ldqg], &ldqg);
                    SLC_DSCAL(&len_swap1, &neg_one, &a[(i - 1) + (*ilo - 1) * lda], &lda);
                    i32 len_swap2 = n - i;
                    if (len_swap2 > 0) {
                        SLC_DSWAP(&len_swap2, &a[(i - 1) + i * lda], &lda, &qg[i + (i - 1) * ldqg], &int1);
                        SLC_DSCAL(&len_swap2, &neg_one, &qg[i + (i - 1) * ldqg], &int1);
                    }
                    i32 len_swap3 = i - 1;
                    SLC_DSWAP(&len_swap3, &a[(i - 1) * lda], &int1, &qg[i * ldqg], &int1);
                    SLC_DSCAL(&len_swap3, &neg_one, &a[(i - 1) * lda], &int1);
                    if (len_swap2 > 0) {
                        SLC_DSCAL(&len_swap2, &neg_one, &a[i + (i - 1) * lda], &int1);
                        SLC_DSWAP(&len_swap2, &a[i + (i - 1) * lda], &int1, &qg[(i - 1) + (i + 1) * ldqg], &ldqg);
                    }

                    if (*ilo != i) {
                        SLC_DSWAP(&n, &a[((*ilo) - 1) * lda], &int1, &a[(i - 1) * lda], &int1);
                        i32 len1 = n - *ilo + 1;
                        SLC_DSWAP(&len1, &a[(*ilo - 1) + (*ilo - 1) * lda], &lda, &a[(i - 1) + (*ilo - 1) * lda], &lda);

                        if (i < n) {
                            i32 len2 = n - i;
                            SLC_DSWAP(&len2, &qg[i + (i - 1) * ldqg], &int1, &qg[i + (*ilo - 1) * ldqg], &int1);
                        }
                        if (i > *ilo + 1) {
                            i32 len3 = i - *ilo - 1;
                            SLC_DSCAL(&len3, &neg_one, &qg[*ilo + (*ilo - 1) * ldqg], &int1);
                            SLC_DSWAP(&len3, &qg[*ilo + (*ilo - 1) * ldqg], &int1, &qg[(i - 1) + *ilo * ldqg], &ldqg);
                        }

                        i32 len4 = *ilo - 1;
                        SLC_DSWAP(&len4, &qg[i * ldqg], &int1, &qg[*ilo * ldqg], &int1);
                        if (n > i) {
                            i32 len5 = n - i;
                            SLC_DSWAP(&len5, &qg[(i - 1) + (i + 1) * ldqg], &ldqg, &qg[(*ilo - 1) + (i + 1) * ldqg], &ldqg);
                        }
                        if (i > *ilo + 1) {
                            i32 len6 = i - *ilo - 1;
                            SLC_DSCAL(&len6, &neg_one, &qg[*ilo + i * ldqg], &int1);
                            SLC_DSWAP(&len6, &qg[(*ilo - 1) + (*ilo + 1) * ldqg], &ldqg, &qg[*ilo + i * ldqg], &int1);
                        }
                        i32 len7 = i - *ilo;
                        SLC_DSCAL(&len7, &neg_one, &qg[(*ilo - 1) + i * ldqg], &int1);
                    }
                    (*ilo)++;
                    break;
                }
            }
        }
    }

    for (i32 i = *ilo; i <= n; i++) {
        scale[i - 1] = ONE;
    }

    if (lscal) {
        f64 sclfac = SLC_DLAMCH("B");
        f64 sfmin1 = SLC_DLAMCH("S") / SLC_DLAMCH("P");
        f64 sfmax1 = ONE / sfmin1;
        f64 sfmin2 = sfmin1 * sclfac;
        f64 sfmax2 = ONE / sfmin2;

        bool conv = false;
        while (!conv) {
            conv = true;

            for (i32 i = *ilo; i <= n; i++) {
                f64 r = ZERO;
                f64 c = ZERO;

                i32 len_r1 = i - *ilo;
                i32 len_r2 = n - i;
                if (len_r1 > 0) {
                    r += SLC_DASUM(&len_r1, &a[(i - 1) + (*ilo - 1) * lda], &lda);
                }
                if (len_r2 > 0) {
                    r += SLC_DASUM(&len_r2, &a[(i - 1) + i * lda], &lda);
                }
                if (len_r1 > 0) {
                    r += SLC_DASUM(&len_r1, &qg[(*ilo - 1) + i * ldqg], &int1);
                }
                if (len_r2 > 0) {
                    r += SLC_DASUM(&len_r2, &qg[(i - 1) + (i + 1) * ldqg], &ldqg);
                }

                if (len_r1 > 0) {
                    c += SLC_DASUM(&len_r1, &a[(*ilo - 1) + (i - 1) * lda], &int1);
                }
                if (len_r2 > 0) {
                    c += SLC_DASUM(&len_r2, &a[i + (i - 1) * lda], &int1);
                }
                i32 len_c1 = i - *ilo;
                if (len_c1 > 0) {
                    c += SLC_DASUM(&len_c1, &qg[(i - 1) + (*ilo - 1) * ldqg], &ldqg);
                }
                if (len_r2 > 0) {
                    c += SLC_DASUM(&len_r2, &qg[i + (i - 1) * ldqg], &int1);
                }

                i32 ic;
                f64 maxr, maxc;

                i32 len_r_full = n - *ilo + 1;
                ic = SLC_IDAMAX(&len_r_full, &a[(i - 1) + (*ilo - 1) * lda], &lda);
                maxr = fabs(a[(i - 1) + (ic + *ilo - 2) * lda]);

                if (i > 1) {
                    i32 len_i1 = i - 1;
                    ic = SLC_IDAMAX(&len_i1, &qg[i * ldqg], &int1);
                    f64 tmp = fabs(qg[(ic - 1) + i * ldqg]);
                    if (tmp > maxr) maxr = tmp;
                }
                if (n > i) {
                    i32 len_ni = n - i;
                    ic = SLC_IDAMAX(&len_ni, &qg[(i - 1) + (i + 1) * ldqg], &ldqg);
                    f64 tmp = fabs(qg[(i - 1) + (ic + i) * ldqg]);
                    if (tmp > maxr) maxr = tmp;
                }

                ic = SLC_IDAMAX(&n, &a[(i - 1) * lda], &int1);
                maxc = fabs(a[(ic - 1) + (i - 1) * lda]);

                if (i > *ilo) {
                    i32 len_iilo = i - *ilo;
                    ic = SLC_IDAMAX(&len_iilo, &qg[(i - 1) + (*ilo - 1) * ldqg], &ldqg);
                    f64 tmp = fabs(qg[(i - 1) + (ic + *ilo - 2) * ldqg]);
                    if (tmp > maxc) maxc = tmp;
                }
                if (n > i) {
                    i32 len_ni = n - i;
                    ic = SLC_IDAMAX(&len_ni, &qg[i + (i - 1) * ldqg], &int1);
                    f64 tmp = fabs(qg[(ic + i - 1) + (i - 1) * ldqg]);
                    if (tmp > maxc) maxc = tmp;
                }

                if (c == ZERO || r == ZERO) {
                    continue;
                }

                f64 g = r / sclfac;
                f64 f = ONE;
                f64 s = c + r;

                while (c < g && (f < sfmax2 && c < sfmax2 && maxc < sfmax2) &&
                       (r > sfmin2 && g > sfmin2 && maxr > sfmin2)) {
                    f = f * sclfac;
                    g = g / sclfac;
                    c = c * sclfac;
                    r = r / sclfac;
                    maxc = maxc * sclfac;
                    maxr = maxr / sclfac;
                }

                g = c / sclfac;
                while (g < r && (r < sfmax2 && maxr < sfmax2) &&
                       (f > sfmin2 && c > sfmin2 && g > sfmin2 && maxc > sfmin2)) {
                    f = f / sclfac;
                    g = g / sclfac;
                    c = c / sclfac;
                    r = r * sclfac;
                    maxc = maxc / sclfac;
                    maxr = maxr * sclfac;
                }

                if ((c + r) >= FACTOR * s) {
                    continue;
                }
                if (f < ONE && scale[i - 1] < ONE) {
                    if (f * scale[i - 1] <= sfmin1) {
                        continue;
                    }
                }
                if (f > ONE && scale[i - 1] > ONE) {
                    if (scale[i - 1] >= sfmax1 / f) {
                        continue;
                    }
                }

                conv = false;
                scale[i - 1] = scale[i - 1] * f;

                if (len_r1 > 0) {
                    SLC_DRSCL(&len_r1, &f, &a[(i - 1) + (*ilo - 1) * lda], &lda);
                }
                if (len_r2 > 0) {
                    SLC_DRSCL(&len_r2, &f, &a[(i - 1) + i * lda], &lda);
                }
                i32 len_i1 = i - 1;
                if (len_i1 > 0) {
                    SLC_DSCAL(&len_i1, &f, &a[(i - 1) * lda], &int1);
                }
                if (len_r2 > 0) {
                    SLC_DSCAL(&len_r2, &f, &a[i + (i - 1) * lda], &int1);
                }
                if (len_i1 > 0) {
                    SLC_DRSCL(&len_i1, &f, &qg[i * ldqg], &int1);
                }
                if (len_r2 > 0) {
                    SLC_DRSCL(&len_r2, &f, &qg[(i - 1) + (i + 1) * ldqg], &ldqg);
                }
                if (len_c1 > 0) {
                    SLC_DSCAL(&len_c1, &f, &qg[(i - 1) + (*ilo - 1) * ldqg], &ldqg);
                }
                if (len_r2 > 0) {
                    SLC_DSCAL(&len_r2, &f, &qg[i + (i - 1) * ldqg], &int1);
                }
            }
        }
    }
}
