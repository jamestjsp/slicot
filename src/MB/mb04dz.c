// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <complex.h>
#include <math.h>

static inline f64 cabs1(c128 z) {
    return fabs(creal(z)) + fabs(cimag(z));
}

void mb04dz(const char *job, i32 n, c128 *a, i32 lda,
            c128 *qg, i32 ldqg, i32 *ilo, f64 *scale, i32 *info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 NEG_ONE = -1.0;
    const i32 INT1 = 1;

    bool lperm = (*job == 'P' || *job == 'p' || *job == 'B' || *job == 'b');
    bool lscal = (*job == 'S' || *job == 's' || *job == 'B' || *job == 'b');

    *info = 0;

    if (!lperm && !lscal && !(*job == 'N' || *job == 'n')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    } else if (ldqg < (n > 1 ? n : 1)) {
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

    // Permutations to isolate eigenvalues if possible
    if (lperm) {
        i32 iloold = 0;
        while (*ilo != iloold) {
            iloold = *ilo;

            // Scan columns ILO .. N
            i32 i = *ilo - 1;
            while (i < n && *ilo == iloold) {
                bool found_nonzero = false;

                for (i32 j = *ilo - 1; j < i && !found_nonzero; j++) {
                    if (creal(a[j + i * lda]) != ZERO || cimag(a[j + i * lda]) != ZERO)
                        found_nonzero = true;
                }
                for (i32 j = i + 1; j < n && !found_nonzero; j++) {
                    if (creal(a[j + i * lda]) != ZERO || cimag(a[j + i * lda]) != ZERO)
                        found_nonzero = true;
                }
                for (i32 j = *ilo - 1; j <= i && !found_nonzero; j++) {
                    if (creal(qg[i + j * ldqg]) != ZERO || cimag(qg[i + j * ldqg]) != ZERO)
                        found_nonzero = true;
                }
                for (i32 j = i + 1; j < n && !found_nonzero; j++) {
                    if (creal(qg[j + i * ldqg]) != ZERO || cimag(qg[j + i * ldqg]) != ZERO)
                        found_nonzero = true;
                }

                if (found_nonzero) {
                    i++;
                    continue;
                }

                // Exchange columns/rows ILO <-> I
                scale[*ilo - 1] = (f64)(i + 1);
                if (*ilo - 1 != i) {
                    i32 ilo0 = *ilo - 1;
                    SLC_ZSWAP(&n, &a[ilo0 * lda], &INT1, &a[i * lda], &INT1);
                    i32 len = n - ilo0;
                    SLC_ZSWAP(&len, &a[ilo0 + ilo0 * lda], &lda, &a[i + ilo0 * lda], &lda);

                    SLC_ZSWAP(&INT1, &qg[i + ilo0 * ldqg], &ldqg, &qg[ilo0 + ilo0 * ldqg], &ldqg);
                    i32 len2 = n - i;
                    SLC_ZSWAP(&len2, &qg[i + i * ldqg], &INT1, &qg[i + ilo0 * ldqg], &INT1);
                    i32 len3 = i - ilo0;
                    SLC_ZSWAP(&len3, &qg[ilo0 + ilo0 * ldqg], &INT1, &qg[i + ilo0 * ldqg], &ldqg);

                    i32 ilo1 = ilo0 + 1;
                    SLC_ZSWAP(&ilo1, &qg[(i + 1) * ldqg], &INT1, &qg[ilo1 * ldqg], &INT1);
                    SLC_ZSWAP(&len2, &qg[i + (i + 1) * ldqg], &ldqg, &qg[ilo0 + (i + 1) * ldqg], &ldqg);
                    SLC_ZSWAP(&len3, &qg[ilo0 + ilo1 * ldqg], &ldqg, &qg[ilo0 + (i + 1) * ldqg], &INT1);
                }
                (*ilo)++;
                break;
            }

            if (*ilo == iloold) {
                // Scan columns N+ILO .. 2*N
                i = *ilo - 1;
                while (i < n && *ilo == iloold) {
                    bool found_nonzero = false;

                    for (i32 j = *ilo - 1; j < i && !found_nonzero; j++) {
                        if (creal(a[i + j * lda]) != ZERO || cimag(a[i + j * lda]) != ZERO)
                            found_nonzero = true;
                    }
                    for (i32 j = i + 1; j < n && !found_nonzero; j++) {
                        if (creal(a[i + j * lda]) != ZERO || cimag(a[i + j * lda]) != ZERO)
                            found_nonzero = true;
                    }
                    for (i32 j = *ilo - 1; j <= i && !found_nonzero; j++) {
                        if (creal(qg[j + (i + 1) * ldqg]) != ZERO || cimag(qg[j + (i + 1) * ldqg]) != ZERO)
                            found_nonzero = true;
                    }
                    for (i32 j = i + 1; j < n && !found_nonzero; j++) {
                        if (creal(qg[i + (j + 1) * ldqg]) != ZERO || cimag(qg[i + (j + 1) * ldqg]) != ZERO)
                            found_nonzero = true;
                    }

                    if (found_nonzero) {
                        i++;
                        continue;
                    }

                    scale[*ilo - 1] = (f64)(n + i + 1);
                    i32 ilo0 = *ilo - 1;

                    // Exchange columns/rows I <-> I+N with symplectic permutation
                    i32 len1 = i - ilo0;
                    if (len1 > 0) {
                        SLC_ZSWAP(&len1, &a[i + ilo0 * lda], &lda, &qg[i + ilo0 * ldqg], &ldqg);
                        SLC_ZDSCAL(&len1, &NEG_ONE, &a[i + ilo0 * lda], &lda);
                    }

                    i32 len2 = n - i - 1;
                    if (len2 > 0) {
                        SLC_ZSWAP(&len2, &a[i + (i + 1) * lda], &lda, &qg[i + 1 + i * ldqg], &INT1);
                        SLC_ZDSCAL(&len2, &NEG_ONE, &a[i + (i + 1) * lda], &lda);
                    }

                    if (i > 0) {
                        SLC_ZSWAP(&i, &a[i * lda], &INT1, &qg[(i + 1) * ldqg], &INT1);
                        SLC_ZDSCAL(&i, &NEG_ONE, &a[i * lda], &INT1);
                    }

                    if (len2 > 0) {
                        SLC_ZSWAP(&len2, &a[i + 1 + i * lda], &INT1, &qg[i + (i + 2) * ldqg], &ldqg);
                        SLC_ZDSCAL(&len2, &NEG_ONE, &a[i + 1 + i * lda], &INT1);
                    }

                    a[i + i * lda] = -a[i + i * lda];
                    c128 temp = qg[i + i * ldqg];
                    qg[i + i * ldqg] = -qg[i + (i + 1) * ldqg];
                    qg[i + (i + 1) * ldqg] = -temp;

                    // Exchange columns/rows ILO <-> I
                    if (ilo0 != i) {
                        SLC_ZSWAP(&n, &a[ilo0 * lda], &INT1, &a[i * lda], &INT1);
                        i32 len = n - ilo0;
                        SLC_ZSWAP(&len, &a[ilo0 + ilo0 * lda], &lda, &a[i + ilo0 * lda], &lda);

                        SLC_ZSWAP(&INT1, &qg[i + ilo0 * ldqg], &ldqg, &qg[ilo0 + ilo0 * ldqg], &ldqg);
                        i32 len3 = n - i;
                        SLC_ZSWAP(&len3, &qg[i + i * ldqg], &INT1, &qg[i + ilo0 * ldqg], &INT1);
                        i32 len4 = i - ilo0;
                        SLC_ZSWAP(&len4, &qg[ilo0 + ilo0 * ldqg], &INT1, &qg[i + ilo0 * ldqg], &ldqg);

                        i32 ilo1 = ilo0 + 1;
                        SLC_ZSWAP(&ilo1, &qg[(i + 1) * ldqg], &INT1, &qg[ilo1 * ldqg], &INT1);
                        SLC_ZSWAP(&len3, &qg[i + (i + 1) * ldqg], &ldqg, &qg[ilo0 + (i + 1) * ldqg], &ldqg);
                        SLC_ZSWAP(&len4, &qg[ilo0 + ilo1 * ldqg], &ldqg, &qg[ilo0 + (i + 1) * ldqg], &INT1);
                    }
                    (*ilo)++;
                    break;
                }
            }
        }
    }

    for (i32 i = *ilo - 1; i < n; i++) {
        scale[i] = ONE;
    }

    // Scale to reduce the 1-norm
    if (lscal) {
        f64 sclfac = SLC_DLAMCH("B");
        f64 sfmin1 = SLC_DLAMCH("S") / SLC_DLAMCH("P");
        f64 sfmax1 = ONE / sfmin1;
        f64 sfmin2 = sfmin1 * sclfac;
        f64 sfmax2 = ONE / sfmin2;

        bool conv = false;
        while (!conv) {
            conv = true;

            for (i32 i = *ilo - 1; i < n; i++) {
                // Compute 1-norm of row and column I
                f64 r = 0.0, c = 0.0;

                i32 len1 = i - (*ilo - 1);
                i32 len2 = n - i - 1;

                if (len1 > 0) {
                    r += SLC_DZASUM(&len1, &a[i + (*ilo - 1) * lda], &lda);
                    c += SLC_DZASUM(&len1, &a[*ilo - 1 + i * lda], &INT1);
                }
                if (len2 > 0) {
                    r += SLC_DZASUM(&len2, &a[i + (i + 1) * lda], &lda);
                    c += SLC_DZASUM(&len2, &a[i + 1 + i * lda], &INT1);
                }

                i32 len3 = i - (*ilo - 1);
                if (len3 > 0) {
                    r += SLC_DZASUM(&len3, &qg[*ilo - 1 + (i + 1) * ldqg], &INT1);
                    c += SLC_DZASUM(&len3, &qg[i + (*ilo - 1) * ldqg], &ldqg);
                }
                if (len2 > 0) {
                    r += SLC_DZASUM(&len2, &qg[i + (i + 2) * ldqg], &ldqg);
                    c += SLC_DZASUM(&len2, &qg[i + 1 + i * ldqg], &INT1);
                }

                f64 qii = cabs1(qg[i + i * ldqg]);
                f64 gii = cabs1(qg[i + (i + 1) * ldqg]);

                // Compute inf-norms
                i32 len_full = n - (*ilo - 1);
                i32 ic = SLC_IZAMAX(&len_full, &a[i + (*ilo - 1) * lda], &lda);
                f64 maxr = cabs1(a[i + (ic + *ilo - 2) * lda]);

                if (i > 0) {
                    ic = SLC_IZAMAX(&i, &qg[(i + 1) * ldqg], &INT1);
                    f64 tmp = cabs1(qg[ic - 1 + (i + 1) * ldqg]);
                    if (tmp > maxr) maxr = tmp;
                }
                if (n > i + 1) {
                    ic = SLC_IZAMAX(&len2, &qg[i + (i + 2) * ldqg], &ldqg);
                    f64 tmp = cabs1(qg[i + (ic + i + 1) * ldqg]);
                    if (tmp > maxr) maxr = tmp;
                }

                ic = SLC_IZAMAX(&n, &a[i * lda], &INT1);
                f64 maxc = cabs1(a[ic - 1 + i * lda]);

                if (i > *ilo - 1) {
                    ic = SLC_IZAMAX(&len1, &qg[i + (*ilo - 1) * ldqg], &ldqg);
                    f64 tmp = cabs1(qg[i + (ic + *ilo - 2) * ldqg]);
                    if (tmp > maxc) maxc = tmp;
                }
                if (n > i + 1) {
                    ic = SLC_IZAMAX(&len2, &qg[i + 1 + i * ldqg], &INT1);
                    f64 tmp = cabs1(qg[ic + i + i * ldqg]);
                    if (tmp > maxc) maxc = tmp;
                }

                if ((c + qii) == ZERO || (r + gii) == ZERO) continue;

                f64 f = ONE;

                // Scale up
                while (((r + gii / sclfac) / sclfac) >= ((c + qii * sclfac) * sclfac) &&
                       fmax(f * sclfac, fmax(c * sclfac, fmax(maxc * sclfac, qii * sclfac * sclfac))) < sfmax2 &&
                       fmin((r + gii / sclfac) / sclfac, fmax(maxr / sclfac, gii / sclfac / sclfac)) > sfmin2) {
                    f = f * sclfac;
                    c = c * sclfac;
                    qii = qii * sclfac * sclfac;
                    r = r / sclfac;
                    gii = gii / sclfac / sclfac;
                    maxc = maxc * sclfac;
                    maxr = maxr / sclfac;
                }

                // Scale down
                while (((r + gii * sclfac) * sclfac) <= ((c + qii / sclfac) / sclfac) &&
                       fmax(r * sclfac, fmax(maxr * sclfac, gii * sclfac * sclfac)) < sfmax2 &&
                       fmin(f / sclfac, fmin((c + qii / sclfac) / sclfac, fmax(maxc / sclfac, qii / sclfac / sclfac))) > sfmin2) {
                    f = f / sclfac;
                    c = c / sclfac;
                    qii = qii / sclfac / sclfac;
                    r = r * sclfac;
                    gii = gii * sclfac * sclfac;
                    maxc = maxc / sclfac;
                    maxr = maxr * sclfac;
                }

                // Apply balancing
                if (f != ONE) {
                    if (f < ONE && scale[i] < ONE) {
                        if (f * scale[i] <= sfmin1) continue;
                    }
                    if (f > ONE && scale[i] > ONE) {
                        if (scale[i] >= sfmax1 / f) continue;
                    }
                    conv = false;
                    scale[i] = scale[i] * f;

                    if (len1 > 0) {
                        SLC_ZDRSCL(&len1, &f, &a[i + (*ilo - 1) * lda], &lda);
                    }
                    if (len2 > 0) {
                        SLC_ZDRSCL(&len2, &f, &a[i + (i + 1) * lda], &lda);
                    }
                    if (i > 0) {
                        SLC_ZDSCAL(&i, &f, &a[i * lda], &INT1);
                    }
                    if (len2 > 0) {
                        SLC_ZDSCAL(&len2, &f, &a[i + 1 + i * lda], &INT1);
                    }
                    if (i > 0) {
                        SLC_ZDRSCL(&i, &f, &qg[(i + 1) * ldqg], &INT1);
                    }
                    qg[i + (i + 1) * ldqg] = qg[i + (i + 1) * ldqg] / f / f;
                    if (len2 > 0) {
                        SLC_ZDRSCL(&len2, &f, &qg[i + (i + 2) * ldqg], &ldqg);
                    }
                    if (len1 > 0) {
                        SLC_ZDSCAL(&len1, &f, &qg[i + (*ilo - 1) * ldqg], &ldqg);
                    }
                    qg[i + i * ldqg] = qg[i + i * ldqg] * f * f;
                    if (len2 > 0) {
                        SLC_ZDSCAL(&len2, &f, &qg[i + 1 + i * ldqg], &INT1);
                    }
                }
            }
        }
    }
}
