// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include <math.h>

void mb03td(const char *typ, const char *compu, bool *select, bool *lower,
            i32 n, f64 *a, i32 lda, f64 *g, i32 ldg,
            f64 *u1, i32 ldu1, f64 *u2, i32 ldu2,
            f64 *wr, f64 *wi, i32 *m, f64 *dwork, i32 ldwork, i32 *info) {
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool isham = (typ[0] == 'H' || typ[0] == 'h');
    bool wantu = (compu[0] == 'U' || compu[0] == 'u');

    i32 wrkmin = (n > 1) ? n : 1;

    *info = 0;
    *m = 0;

    if (!isham && !(typ[0] == 'S' || typ[0] == 's')) {
        *info = -1;
    } else if (!wantu && !(compu[0] == 'N' || compu[0] == 'n')) {
        *info = -2;
    } else if (n < 0) {
        *info = -5;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldg < (n > 1 ? n : 1)) {
        *info = -9;
    } else if (ldu1 < 1 || (wantu && ldu1 < n)) {
        *info = -11;
    } else if (ldu2 < 1 || (wantu && ldu2 < n)) {
        *info = -13;
    } else if (ldwork < wrkmin) {
        *info = -18;
        dwork[0] = (f64)wrkmin;
    }

    if (*info != 0) {
        return;
    }

    bool pair = false;
    for (i32 k = 1; k <= n; k++) {
        if (pair) {
            pair = false;
        } else {
            if (k < n) {
                if (a[k + (k - 1) * lda] == ZERO) {
                    if (select[k - 1]) {
                        (*m)++;
                    }
                } else {
                    pair = true;
                    if (select[k - 1] || select[k]) {
                        (*m) += 2;
                    }
                }
            } else {
                if (select[n - 1]) {
                    (*m)++;
                }
            }
        }
    }

    if (n == 0) {
        dwork[0] = ONE;
        return;
    }

    i32 ks = 0;
    pair = false;

    for (i32 k = 1; k <= n; k++) {
        if (pair) {
            pair = false;
            continue;
        }

        bool swap = select[k - 1];
        bool flow = lower[k - 1];

        if (k < n) {
            if (a[k + (k - 1) * lda] != ZERO) {
                pair = true;
                swap = swap || select[k];
                flow = flow || lower[k];
            }
        }

        i32 nbf = pair ? 2 : 1;

        if (swap) {
            ks++;
            i32 ifst, ilst, nbl;

            if (flow) {
                ifst = k;
                ilst = n;
                nbl = 1;

                if (ilst > 1) {
                    if (a[(ilst - 1) + (ilst - 2) * lda] != ZERO) {
                        ilst--;
                        nbl = 2;
                    }
                }

                if (nbf == 2 && nbl == 1) ilst--;
                if (nbf == 1 && nbl == 2) ilst++;

                if (ilst != ifst) {
                    i32 here = ifst;

                    while (here < ilst) {
                        if (nbf == 1 || nbf == 2) {
                            i32 nbnext = 1;
                            if (here + nbf + 1 <= n) {
                                if (a[(here + nbf) + (here + nbf - 1) * lda] != ZERO) {
                                    nbnext = 2;
                                }
                            }

                            i32 ierr = 0;
                            mb03ts(isham, wantu, n, a, lda, g, ldg,
                                   u1, ldu1, u2, ldu2, here, nbf, nbnext,
                                   dwork, &ierr);

                            if (ierr != 0) {
                                *info = 1;
                                goto store_eigenvalues;
                            }

                            here += nbnext;

                            if (nbf == 2) {
                                if (a[here + (here - 1) * lda] == ZERO) {
                                    nbf = 3;
                                }
                            }
                        } else {
                            i32 nbnext = 1;
                            if (here + 3 <= n) {
                                if (a[(here + 2) + (here + 1) * lda] != ZERO) {
                                    nbnext = 2;
                                }
                            }

                            i32 ierr = 0;
                            mb03ts(isham, wantu, n, a, lda, g, ldg,
                                   u1, ldu1, u2, ldu2, here + 1, 1, nbnext,
                                   dwork, &ierr);

                            if (ierr != 0) {
                                *info = 1;
                                goto store_eigenvalues;
                            }

                            if (nbnext == 1) {
                                mb03ts(isham, wantu, n, a, lda, g, ldg,
                                       u1, ldu1, u2, ldu2, here, 1, nbnext,
                                       dwork, &ierr);
                                here++;
                            } else {
                                if (a[(here + 1) + here * lda] == ZERO) {
                                    nbnext = 1;
                                }
                                if (nbnext == 2) {
                                    i32 ierr2 = 0;
                                    mb03ts(isham, wantu, n, a, lda, g, ldg,
                                           u1, ldu1, u2, ldu2, here, 1, nbnext,
                                           dwork, &ierr2);
                                    if (ierr2 != 0) {
                                        *info = 1;
                                        goto store_eigenvalues;
                                    }
                                    here += 2;
                                } else {
                                    mb03ts(isham, wantu, n, a, lda, g, ldg,
                                           u1, ldu1, u2, ldu2, here, 1, 1,
                                           dwork, &ierr);
                                    mb03ts(isham, wantu, n, a, lda, g, ldg,
                                           u1, ldu1, u2, ldu2, here + 1, 1, 1,
                                           dwork, &ierr);
                                    here += 2;
                                }
                            }
                        }
                    }
                }

                i32 ierr = 0;
                if (nbf == 1) {
                    mb03ts(isham, wantu, n, a, lda, g, ldg,
                           u1, ldu1, u2, ldu2, n, 1, 1, dwork, &ierr);
                } else if (nbf == 2) {
                    mb03ts(isham, wantu, n, a, lda, g, ldg,
                           u1, ldu1, u2, ldu2, n - 1, 2, 2, dwork, &ierr);
                    if (ierr != 0) {
                        *info = 1;
                        goto store_eigenvalues;
                    }
                    if (a[(n - 2) + (n - 1) * lda] == ZERO) {
                        nbf = 3;
                    }
                } else {
                    mb03ts(isham, wantu, n, a, lda, g, ldg,
                           u1, ldu1, u2, ldu2, n, 1, 1, dwork, &ierr);
                    mb03ts(isham, wantu, n, a, lda, g, ldg,
                           u1, ldu1, u2, ldu2, n - 1, 1, 1, dwork, &ierr);
                    mb03ts(isham, wantu, n, a, lda, g, ldg,
                           u1, ldu1, u2, ldu2, n, 1, 1, dwork, &ierr);
                }

                ifst = n;
                if (pair) ifst = n - 1;
            } else {
                ifst = k;
            }

            ilst = ks;
            nbl = 1;

            if (ilst > 1) {
                if (a[(ilst - 1) + (ilst - 2) * lda] != ZERO) {
                    ilst--;
                    nbl = 2;
                }
            }

            if (ilst != ifst) {
                i32 here = ifst;

                while (here > ilst) {
                    if (nbf == 1 || nbf == 2) {
                        i32 nbnext = 1;
                        if (here >= 3) {
                            if (a[(here - 2) + (here - 3) * lda] != ZERO) {
                                nbnext = 2;
                            }
                        }

                        i32 ierr = 0;
                        mb03ts(isham, wantu, n, a, lda, g, ldg,
                               u1, ldu1, u2, ldu2, here - nbnext, nbnext, nbf,
                               dwork, &ierr);

                        if (ierr != 0) {
                            *info = 1;
                            goto store_eigenvalues;
                        }

                        here -= nbnext;

                        if (nbf == 2) {
                            if (a[here + (here - 1) * lda] == ZERO) {
                                nbf = 3;
                            }
                        }
                    } else {
                        i32 nbnext = 1;
                        if (here >= 3) {
                            if (a[(here - 2) + (here - 3) * lda] != ZERO) {
                                nbnext = 2;
                            }
                        }

                        i32 ierr = 0;
                        mb03ts(isham, wantu, n, a, lda, g, ldg,
                               u1, ldu1, u2, ldu2, here - nbnext, nbnext, 1,
                               dwork, &ierr);

                        if (ierr != 0) {
                            *info = 1;
                            goto store_eigenvalues;
                        }

                        if (nbnext == 1) {
                            mb03ts(isham, wantu, n, a, lda, g, ldg,
                                   u1, ldu1, u2, ldu2, here, nbnext, 1,
                                   dwork, &ierr);
                            here--;
                        } else {
                            if (a[(here - 1) + (here - 2) * lda] == ZERO) {
                                nbnext = 1;
                            }
                            if (nbnext == 2) {
                                i32 ierr2 = 0;
                                mb03ts(isham, wantu, n, a, lda, g, ldg,
                                       u1, ldu1, u2, ldu2, here - 1, 2, 1,
                                       dwork, &ierr2);
                                if (ierr2 != 0) {
                                    *info = 1;
                                    goto store_eigenvalues;
                                }
                                here -= 2;
                            } else {
                                mb03ts(isham, wantu, n, a, lda, g, ldg,
                                       u1, ldu1, u2, ldu2, here, 1, 1,
                                       dwork, &ierr);
                                mb03ts(isham, wantu, n, a, lda, g, ldg,
                                       u1, ldu1, u2, ldu2, here - 1, 1, 1,
                                       dwork, &ierr);
                                here -= 2;
                            }
                        }
                    }
                }
            }

            if (pair) ks++;
        }
    }

store_eigenvalues:
    for (i32 k = 1; k <= n; k++) {
        wr[k - 1] = a[(k - 1) + (k - 1) * lda];
        wi[k - 1] = ZERO;
    }

    for (i32 k = 1; k <= n - 1; k++) {
        if (a[k + (k - 1) * lda] != ZERO) {
            wi[k - 1] = sqrt(fabs(a[(k - 1) + k * lda])) * sqrt(fabs(a[k + (k - 1) * lda]));
            wi[k] = -wi[k - 1];
        }
    }

    dwork[0] = (f64)wrkmin;
}
