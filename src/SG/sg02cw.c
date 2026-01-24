// SPDX-License-Identifier: BSD-3-Clause
// SG02CW - Riccati equation residual computation

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <stdbool.h>

void sg02cw(
    const char* dico, const char* job, const char* jobe, const char* flag,
    const char* jobg, const char* uplo, const char* trans,
    const i32 n, const i32 m,
    f64* a, const i32 lda,
    const f64* e, const i32 lde,
    f64* g, const i32 ldg,
    f64* x, const i32 ldx,
    const f64* f, const i32 ldf,
    const f64* k, const i32 ldk,
    const f64* xe, const i32 ldxe,
    f64* r, const i32 ldr,
    f64* c, const i32 ldc,
    f64* norms,
    f64* dwork, const i32 ldwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 two = 2.0;
    i32 int1 = 1;

    char dico_c = (char)toupper((unsigned char)dico[0]);
    char job_c = (char)toupper((unsigned char)job[0]);
    char jobe_c = (char)toupper((unsigned char)jobe[0]);
    char flag_c = (char)toupper((unsigned char)flag[0]);
    char jobg_c = (char)toupper((unsigned char)jobg[0]);
    char uplo_c = (char)toupper((unsigned char)uplo[0]);
    char trans_c = (char)toupper((unsigned char)trans[0]);

    bool discr = (dico_c == 'D');
    bool ljoba = (job_c == 'A');
    bool ljobc = (job_c == 'C');
    bool ljobn = (job_c == 'N');
    bool ljobb = (job_c == 'B');
    bool ljobr = (job_c == 'R');
    bool ljobe = (jobe_c == 'G');
    bool lflag = (flag_c == 'M');
    bool ljobg = (jobg_c == 'G');
    bool ljobf = (jobg_c == 'F');
    bool ljobh = (jobg_c == 'H');
    bool luplo = (uplo_c == 'U');
    bool ltrans = (trans_c == 'T' || trans_c == 'C');
    bool ljobl = ljobf || ljobh;
    bool nljobc = !ljobc;
    bool nljobr = !ljobr && !ljobb;
    bool withd = !ljobl || nljobr;
    bool unite = !ljobe;
    bool withe = ljobe && (nljobc || !(discr || ljobl));

    *info = 0;

    if (!discr && dico_c != 'C') {
        *info = -1;
    } else if (!ljoba && nljobc && !ljobn && nljobr) {
        *info = -2;
    } else if (unite && jobe_c != 'I') {
        *info = -3;
    } else if (!lflag && flag_c != 'P') {
        *info = -4;
    } else if (!ljobg && jobg_c != 'D' && !ljobl) {
        *info = -5;
    } else if (!luplo && uplo_c != 'L') {
        *info = -6;
    } else if (!ltrans && trans_c != 'N') {
        *info = -7;
    } else if (n < 0) {
        *info = -8;
    } else if (!ljobg && m < 0) {
        *info = -9;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -11;
    } else if (lde < 1 || (withe && lde < n)) {
        *info = -13;
    } else if (ldg < 1 || (withd && ldg < n)) {
        *info = -15;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -17;
    } else if (ldf < 1 || (ljobl && ldf < n)) {
        *info = -19;
    } else if (ldk < 1 || (ljobh && ldk < m)) {
        *info = -21;
    } else if (ldxe < 1 || (ljobl && nljobc && ldxe < n &&
              (discr || (!discr && ljobe)))) {
        *info = -23;
    } else if (ldr < 1 || (nljobc && ldr < n)) {
        *info = -25;
    } else if (ldc < 1 || (nljobr && ldc < n)) {
        *info = -27;
    }

    if (*info != 0) {
        return;
    }

    i32 nn = n * n;
    i32 nm = !ljobg ? n * m : 0;
    i32 ia = (ljobe || discr) ? 1 : 0;

    i32 minwrk = 0;
    i32 optwrk = 0;
    bool use1 = false;
    bool lcond = false;
    bool lcnds = false;
    bool lcndt = false;

    if (ljobn || ljobb) {
        i32 ib = ljobb ? 1 : 0;
        if (ljobl) {
            if (ljobb && discr) {
                minwrk = (ia + 1) * nn;
            } else if (ljobe || ljobb) {
                minwrk = nn;
            } else {
                minwrk = 0;
            }
            optwrk = minwrk;
            use1 = false;
        } else if (ljobg) {
            minwrk = (ia + ib + 1) * nn;
            optwrk = (ib + 2) * nn;
            use1 = false;
        } else {
            if (ljobn) {
                lcnds = 4 * m <= 3 * n;
            } else {
                lcnds = 2 * m <= 3 * n;
            }
            if (discr) {
                i32 maxnm = nn > nm ? nn : nm;
                if (ljobn) {
                    minwrk = (maxnm < 2 * nn) ? maxnm : 2 * nn;
                    if (minwrk < maxnm) minwrk = maxnm > 2*nn ? 2*nn : maxnm;
                } else {
                    i32 tmp = nn + ((maxnm < 2*nn) ? maxnm : 2*nn);
                    minwrk = tmp;
                }
                if (lcnds) {
                    if (ljobn) {
                        optwrk = maxnm;
                    } else {
                        optwrk = nn + maxnm;
                    }
                } else {
                    optwrk = (ib + 2) * nn;
                }
                if (ljobb && optwrk < minwrk) optwrk = minwrk;
                use1 = lcnds || ldwork < (ib + 2) * nn;
            } else {
                i32 t1 = nn + (ia * ib * nn > nm ? ia * ib * nn : nm);
                i32 t2 = nn + (ia + ib) * nn;
                minwrk = (t1 < t2) ? t1 : t2;
                if (lcnds) {
                    optwrk = t1;
                } else {
                    optwrk = (ia + ib + 1) * nn;
                }
                use1 = lcnds && ldwork >= t1;
                if (ljobn) {
                    use1 = (ljobe && (lcnds || ldwork < optwrk)) ||
                           (unite && use1);
                } else {
                    use1 = use1 || ldwork < (ia + ib + 1) * nn;
                }
            }
        }
    } else {
        lcond = (3 + 2 * ia) * m <= (2 + 2 * ia) * n;

        if (discr) {
            if (ljobl) {
                if (ljobe && ljobr) {
                    minwrk = nn;
                } else {
                    minwrk = 0;
                }
                optwrk = minwrk;
            } else {
                lcnds = 4 * m <= 3 * n;
                lcndt = 2 * m <= 3 * n;
                if (ljobg) {
                    if (ljobr) {
                        minwrk = 2 * nn;
                    } else {
                        minwrk = nn;
                    }
                    optwrk = minwrk;
                    use1 = false;
                } else {
                    if (ljobc) {
                        minwrk = (nn < nm) ? nn : nm;
                        if (lcond) {
                            optwrk = nm;
                        } else {
                            optwrk = 2 * nn;
                        }
                        use1 = lcond || ldwork < nn;
                    } else if (ljobr) {
                        i32 t1 = nn + nm;
                        i32 t2 = 3 * nn;
                        minwrk = (t1 < t2) ? t1 : t2;
                        if (lcndt) {
                            optwrk = nn + nm;
                        } else {
                            optwrk = 3 * nn;
                        }
                        use1 = lcndt || ldwork < optwrk;
                    } else {
                        i32 maxnm = (nn > nm) ? nn : nm;
                        minwrk = (maxnm < 2 * nn) ? maxnm : 2 * nn;
                        if (lcnds) {
                            optwrk = minwrk;
                        } else {
                            optwrk = 2 * nn;
                        }
                        use1 = lcnds || ldwork < optwrk;
                    }
                }
            }
        } else {
            if (ljobl) {
                minwrk = 0;
                if (unite && ljobr) {
                    optwrk = nn;
                } else {
                    optwrk = 0;
                }
                use1 = false;
            } else {
                lcnds = 3 * m <= 2 * n;
                lcndt = m <= n;
                if (ljobg) {
                    if (ljobr) {
                        minwrk = (ia + 1) * nn;
                        optwrk = 2 * nn;
                    } else {
                        minwrk = ia * nn;
                        if (ljobc) {
                            optwrk = nn;
                        } else {
                            optwrk = 2 * nn;
                        }
                    }
                    use1 = false;
                } else {
                    if (ljobc) {
                        minwrk = (nn < nm) ? nn : nm;
                        if (lcond) {
                            optwrk = nm;
                        } else {
                            optwrk = nn;
                        }
                        use1 = lcond || ldwork < optwrk;
                    } else if (ljobr) {
                        i32 t1 = ia * nn + nm;
                        i32 t2 = (ia + 2) * nn;
                        minwrk = (t1 < t2) ? t1 : t2;
                        if (lcndt) {
                            optwrk = nn + ia * nm;
                        } else {
                            optwrk = 3 * nn;
                        }
                        use1 = lcndt || ldwork < (ia + 2) * nn;
                    } else {
                        i32 t1 = nm;
                        i32 t2 = (ia + 1) * nn;
                        minwrk = (t1 < t2) ? t1 : t2;
                        optwrk = 2 * nn;
                        use1 = minwrk == nm && ldwork < optwrk;
                    }
                }
            }
        }
    }

    bool useatw = false;
    if (!discr && nljobc) {
        useatw = use1 || m == 0;
        if (ljoba) {
            useatw = useatw || (ljobg && ldwork < optwrk);
        }
    }

    if (ldwork == -2) {
        dwork[0] = (f64)((minwrk > 1) ? minwrk : 1);
        return;
    } else if (ldwork == -1) {
        dwork[0] = (f64)((optwrk > 1) ? optwrk : 1);
        return;
    }

    if (ldwork < minwrk) {
        *info = -30;
        dwork[0] = (f64)((minwrk > 1) ? minwrk : 1);
        return;
    }

    if (n == 0) {
        if (ljobn || ljobb) {
            norms[0] = zero;
            norms[1] = zero;
            if (discr && ljobe) {
                norms[2] = zero;
            }
        }
        return;
    }

    i32 wp = 1;
    i32 yp = 1;
    bool keepx = false;
    bool useopt = false;

    if (ljobn || ljobb) {
        if (discr || ljobe || ljobg || use1) {
            wp = nn + 1;
        } else {
            wp = 1;
        }
        if (!(discr || ljobe) && ljobg) {
            if (ljobn) {
                keepx = ldwork >= 2 * nn;
            } else {
                keepx = ldwork >= 3 * nn;
            }
        } else {
            keepx = false;
        }
    } else if (discr) {
        wp = nn + 1;
        if (ljobc && !ljobg && !lcond) {
            useopt = ldwork >= 2 * nn;
        } else {
            useopt = false;
        }
    } else if (ljobl || ljobc) {
        wp = 1;
        keepx = unite && ldwork >= optwrk;
    } else {
        if (ljobg) {
            keepx = unite && (ldwork >= optwrk || (ljoba && ldwork >= nn));
            if (ldwork < 2 * nn) {
                wp = 1;
            } else {
                wp = nn + 1;
            }
        } else {
            if (ldwork < optwrk) {
                keepx = false;
                if (ljobr) {
                    wp = ia * nm + 1;
                    yp = wp + nn;
                } else {
                    wp = nn + 1;
                }
            } else {
                keepx = true;
                if (ljoba) {
                    wp = nn + 1;
                } else {
                    if (use1) {
                        wp = ia * nm + 1;
                    } else {
                        wp = nn + 1;
                        yp = wp + nn;
                    }
                }
            }
        }
    }

    f64 beta = lflag ? -one : one;

    const char* nt = "N";
    const char* tr = "T";
    const char* side = ltrans ? "R" : "L";
    const char* nside = ltrans ? "L" : "R";
    const char* ntrans = ltrans ? "N" : "T";

    if (ljobn) {
        if (discr) {
            if (ljobe) {
                i32 tmpinfo = 0;
                mb01ru(&uplo_c, ntrans, n, n, zero, one, c, ldc, e, lde, x, ldx, dwork, nn, &tmpinfo);

                norms[2] = SLC_DLANSY("F", &uplo_c, &n, c, &ldc, dwork);

                if (luplo) {
                    for (i32 j = 0; j < n; j++) {
                        f64 mone = -one;
                        i32 jj = j + 1;
                        SLC_DAXPY(&jj, &mone, &c[j * ldc], &int1, &r[j * ldr], &int1);
                    }
                } else {
                    for (i32 j = 0; j < n; j++) {
                        f64 mone = -one;
                        i32 len = n - j;
                        SLC_DAXPY(&len, &mone, &c[j + j * ldc], &int1, &r[j + j * ldr], &int1);
                    }
                }
            } else {
                if (luplo) {
                    for (i32 j = 0; j < n; j++) {
                        f64 mone = -one;
                        i32 jj = j + 1;
                        SLC_DAXPY(&jj, &mone, &x[j * ldx], &int1, &r[j * ldr], &int1);
                    }
                } else {
                    for (i32 j = 0; j < n; j++) {
                        f64 mone = -one;
                        i32 len = n - j;
                        SLC_DAXPY(&len, &mone, &x[j + j * ldx], &int1, &r[j + j * ldr], &int1);
                    }
                }
            }

            if (ljobl) {
                i32 tmpinfo = 0;
                mb01rb(side, &uplo_c, tr, n, n, zero, one, c, ldc, a, lda, xe, ldxe, &tmpinfo);
            } else {
                SLC_DSYMM(side, &uplo_c, &n, &n, &one, x, &ldx, a, &lda, &zero, dwork, &n);
                i32 tmpinfo = 0;
                mb01rb(side, &uplo_c, tr, n, n, zero, one, c, ldc, a, lda, dwork, n, &tmpinfo);
            }

            norms[0] = SLC_DLANSY("F", &uplo_c, &n, c, &ldc, dwork);

            if (luplo) {
                for (i32 j = 0; j < n; j++) {
                    i32 jj = j + 1;
                    SLC_DAXPY(&jj, &one, &c[j * ldc], &int1, &r[j * ldr], &int1);
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    i32 len = n - j;
                    SLC_DAXPY(&len, &one, &c[j + j * ldc], &int1, &r[j + j * ldr], &int1);
                }
            }

            if (ljobl) {
                if (ljobf) {
                    SLC_DSYRK(&uplo_c, nt, &n, &m, &one, f, &ldf, &zero, c, &ldc);
                } else {
                    i32 tmpinfo = 0;
                    mb01rb("L", &uplo_c, nt, n, m, zero, one, c, ldc, f, ldf, k, ldk, &tmpinfo);
                }

                norms[1] = SLC_DLANSY("F", &uplo_c, &n, c, &ldc, dwork);

                if (luplo) {
                    for (i32 j = 0; j < n; j++) {
                        i32 jj = j + 1;
                        SLC_DAXPY(&jj, &beta, &c[j * ldc], &int1, &r[j * ldr], &int1);
                    }
                } else {
                    for (i32 j = 0; j < n; j++) {
                        i32 len = n - j;
                        SLC_DAXPY(&len, &beta, &c[j + j * ldc], &int1, &r[j + j * ldr], &int1);
                    }
                }

                SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);

                if (ljobf) {
                    if (ltrans) {
                        SLC_DGEMM(nt, tr, &n, &n, &m, &beta, f, &ldf, g, &ldg, &one, c, &ldc);
                    } else {
                        SLC_DGEMM(nt, tr, &n, &n, &m, &beta, g, &ldg, f, &ldf, &one, c, &ldc);
                    }
                } else {
                    if (ltrans) {
                        SLC_DGEMM(tr, tr, &n, &n, &m, &beta, k, &ldk, g, &ldg, &one, c, &ldc);
                    } else {
                        SLC_DGEMM(nt, nt, &n, &n, &m, &beta, g, &ldg, k, &ldk, &one, c, &ldc);
                    }
                }
            } else if (ljobg) {
                SLC_DSYMM(side, &uplo_c, &n, &n, &beta, g, &ldg, dwork, &n, &zero, c, &ldc);
                i32 tmpinfo = 0;
                mb01rb(side, &uplo_c, tr, n, n, zero, one, &dwork[wp - 1], n, dwork, n, c, ldc, &tmpinfo);

                norms[1] = SLC_DLANSY("F", &uplo_c, &n, &dwork[wp - 1], &n, dwork);

                i32 idx = wp - 1;
                if (luplo) {
                    for (i32 j = 0; j < n; j++) {
                        i32 jj = j + 1;
                        SLC_DAXPY(&jj, &one, &dwork[idx], &int1, &r[j * ldr], &int1);
                        idx += n;
                    }
                } else {
                    for (i32 j = 0; j < n; j++) {
                        i32 len = n - j;
                        SLC_DAXPY(&len, &one, &dwork[idx], &int1, &r[j + j * ldr], &int1);
                        idx += n + 1;
                    }
                }

                for (i32 j = 0; j < n; j++) {
                    SLC_DAXPY(&n, &one, &a[j * lda], &int1, &c[j * ldc], &int1);
                }
            } else if (m > 0) {
                if (use1) {
                    SLC_DLACPY("A", &n, &n, dwork, &n, c, &ldc);
                    if (ltrans) {
                        SLC_DGEMM(nt, nt, &n, &m, &n, &one, c, &ldc, g, &ldg, &zero, dwork, &n);
                        SLC_DSYRK(&uplo_c, nt, &n, &m, &one, dwork, &n, &zero, c, &ldc);
                    } else {
                        SLC_DGEMM(tr, nt, &m, &n, &n, &one, g, &ldg, c, &ldc, &zero, dwork, &m);
                        SLC_DSYRK(&uplo_c, tr, &n, &m, &one, dwork, &m, &zero, c, &ldc);
                    }

                    norms[1] = SLC_DLANSY("F", &uplo_c, &n, c, &ldc, dwork);

                    if (luplo) {
                        for (i32 j = 0; j < n; j++) {
                            i32 jj = j + 1;
                            SLC_DAXPY(&jj, &beta, &c[j * ldc], &int1, &r[j * ldr], &int1);
                        }
                    } else {
                        for (i32 j = 0; j < n; j++) {
                            i32 len = n - j;
                            SLC_DAXPY(&len, &beta, &c[j + j * ldc], &int1, &r[j + j * ldr], &int1);
                        }
                    }

                    SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);
                    if (ltrans) {
                        SLC_DGEMM(nt, tr, &n, &n, &m, &beta, dwork, &n, g, &ldg, &one, c, &ldc);
                    } else {
                        SLC_DGEMM(nt, nt, &n, &n, &m, &beta, g, &ldg, dwork, &m, &one, c, &ldc);
                    }
                } else {
                    SLC_DSYRK(&uplo_c, nt, &n, &m, &one, g, &ldg, &zero, &dwork[wp - 1], &n);
                    SLC_DSYMM(side, &uplo_c, &n, &n, &beta, &dwork[wp - 1], &n, dwork, &n, &zero, c, &ldc);
                    i32 tmpinfo = 0;
                    mb01rb(side, &uplo_c, tr, n, n, zero, one, &dwork[wp - 1], n, dwork, n, c, ldc, &tmpinfo);

                    norms[1] = SLC_DLANSY("F", &uplo_c, &n, &dwork[wp - 1], &n, dwork);

                    i32 idx = wp - 1;
                    if (luplo) {
                        for (i32 j = 0; j < n; j++) {
                            i32 jj = j + 1;
                            SLC_DAXPY(&jj, &one, &dwork[idx], &int1, &r[j * ldr], &int1);
                            idx += n;
                        }
                    } else {
                        for (i32 j = 0; j < n; j++) {
                            i32 len = n - j;
                            SLC_DAXPY(&len, &one, &dwork[idx], &int1, &r[j + j * ldr], &int1);
                            idx += n + 1;
                        }
                    }

                    for (i32 j = 0; j < n; j++) {
                        SLC_DAXPY(&n, &one, &a[j * lda], &int1, &c[j * ldc], &int1);
                    }
                }
            } else {
                norms[1] = zero;
                SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);
            }
        } else {
            // Continuous-time case for JOBN
            if (ljobe) {
                if (ljobl) {
                    if (ltrans) {
                        SLC_DGEMM(nt, tr, &n, &n, &n, &one, xe, &ldxe, a, &lda, &zero, c, &ldc);
                    } else {
                        SLC_DGEMM(tr, nt, &n, &n, &n, &one, a, &lda, xe, &ldxe, &zero, c, &ldc);
                    }
                } else {
                    SLC_DSYMM(side, &uplo_c, &n, &n, &one, x, &ldx, e, &lde, &zero, dwork, &n);
                    if (ltrans) {
                        SLC_DGEMM(nt, tr, &n, &n, &n, &one, dwork, &n, a, &lda, &zero, c, &ldc);
                    } else {
                        SLC_DGEMM(tr, nt, &n, &n, &n, &one, a, &lda, dwork, &n, &zero, c, &ldc);
                    }
                }
            } else {
                SLC_DSYMM(side, &uplo_c, &n, &n, &one, x, &ldx, a, &lda, &zero, c, &ldc);
            }

            norms[0] = SLC_DLANGE("F", &n, &n, c, &ldc, dwork);

            if (luplo) {
                for (i32 j = 0; j < n; j++) {
                    i32 jj = j + 1;
                    SLC_DAXPY(&jj, &one, &c[j * ldc], &int1, &r[j * ldr], &int1);
                    SLC_DAXPY(&jj, &one, &c[j], &ldc, &r[j * ldr], &int1);
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    i32 len = n - j;
                    SLC_DAXPY(&len, &one, &c[j + j * ldc], &int1, &r[j + j * ldr], &int1);
                    SLC_DAXPY(&len, &one, &c[j + j * ldc], &ldc, &r[j + j * ldr], &int1);
                }
            }

            if (ljobl) {
                if (ljobf) {
                    SLC_DSYRK(&uplo_c, nt, &n, &m, &one, f, &ldf, &zero, c, &ldc);
                } else {
                    i32 tmpinfo = 0;
                    mb01rb("L", &uplo_c, nt, n, m, zero, one, c, ldc, f, ldf, k, ldk, &tmpinfo);
                }

                norms[1] = SLC_DLANSY("F", &uplo_c, &n, c, &ldc, dwork);

                if (luplo) {
                    for (i32 j = 0; j < n; j++) {
                        i32 jj = j + 1;
                        SLC_DAXPY(&jj, &beta, &c[j * ldc], &int1, &r[j * ldr], &int1);
                    }
                } else {
                    for (i32 j = 0; j < n; j++) {
                        i32 len = n - j;
                        SLC_DAXPY(&len, &beta, &c[j + j * ldc], &int1, &r[j + j * ldr], &int1);
                    }
                }

                SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);

                if (ljobf) {
                    if (ltrans) {
                        SLC_DGEMM(nt, tr, &n, &n, &m, &beta, f, &ldf, g, &ldg, &one, c, &ldc);
                    } else {
                        SLC_DGEMM(nt, tr, &n, &n, &m, &beta, g, &ldg, f, &ldf, &one, c, &ldc);
                    }
                } else {
                    if (ltrans) {
                        SLC_DGEMM(tr, tr, &n, &n, &m, &beta, k, &ldk, g, &ldg, &one, c, &ldc);
                    } else {
                        SLC_DGEMM(nt, nt, &n, &n, &m, &beta, g, &ldg, k, &ldk, &one, c, &ldc);
                    }
                }
            } else if (ljobg) {
                i32 idx;
                if (ljobe) {
                    SLC_DSYMM(side, &uplo_c, &n, &n, &one, g, &ldg, dwork, &n, &zero, &dwork[wp - 1], &n);
                    i32 tmpinfo = 0;
                    mb01rb(side, &uplo_c, tr, n, n, zero, one, c, ldc, dwork, n, &dwork[wp - 1], n, &tmpinfo);
                    idx = wp - 1;
                } else {
                    if (keepx) {
                        SLC_DLACPY(&uplo_c, &n, &n, x, &ldx, &dwork[wp - 1], &n);
                        ma02ed(uplo_c, n, &dwork[wp - 1], n);
                        SLC_DSYMM(side, &uplo_c, &n, &n, &one, g, &ldg, &dwork[wp - 1], &n, &zero, dwork, &n);
                        i32 tmpinfo = 0;
                        mb01rb(side, &uplo_c, tr, n, n, zero, one, c, ldc, &dwork[wp - 1], n, dwork, n, &tmpinfo);
                    } else {
                        SLC_DSYMM(side, &uplo_c, &n, &n, &one, g, &ldg, x, &ldx, &zero, dwork, &n);
                        i32 tmpinfo = 0;
                        mb01rb(side, &uplo_c, tr, n, n, zero, one, c, ldc, x, ldx, dwork, n, &tmpinfo);
                    }
                    idx = 0;
                }

                norms[1] = SLC_DLANSY("F", &uplo_c, &n, c, &ldc, dwork);

                if (luplo) {
                    for (i32 j = 0; j < n; j++) {
                        i32 jj = j + 1;
                        SLC_DAXPY(&jj, &beta, &c[j * ldc], &int1, &r[j * ldr], &int1);
                    }
                } else {
                    for (i32 j = 0; j < n; j++) {
                        i32 len = n - j;
                        SLC_DAXPY(&len, &beta, &c[j + j * ldc], &int1, &r[j + j * ldr], &int1);
                    }
                }

                for (i32 j = 0; j < n; j++) {
                    SLC_DCOPY(&n, &a[j * lda], &int1, &c[j * ldc], &int1);
                    SLC_DAXPY(&n, &beta, &dwork[idx], &int1, &c[j * ldc], &int1);
                    idx += n;
                }
            } else if (m > 0) {
                if (use1) {
                    i32 idx;
                    if (ljobe) {
                        if (ltrans) {
                            SLC_DGEMM(nt, nt, &n, &m, &n, &one, dwork, &n, g, &ldg, &zero, &dwork[wp - 1], &n);
                        } else {
                            SLC_DGEMM(tr, nt, &m, &n, &n, &one, g, &ldg, dwork, &n, &zero, &dwork[wp - 1], &m);
                        }
                    } else {
                        SLC_DSYMM("L", &uplo_c, &n, &m, &one, x, &ldx, g, &ldg, &zero, &dwork[wp - 1], &n);
                    }

                    SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);
                    if (unite || ltrans) {
                        SLC_DSYRK(&uplo_c, nt, &n, &m, &one, &dwork[wp - 1], &n, &zero, dwork, &n);
                        if (ltrans) {
                            SLC_DGEMM(nt, tr, &n, &n, &m, &beta, &dwork[wp - 1], &n, g, &ldg, &one, c, &ldc);
                        } else {
                            SLC_DGEMM(nt, tr, &n, &n, &m, &beta, g, &ldg, &dwork[wp - 1], &n, &one, c, &ldc);
                        }
                    } else {
                        SLC_DSYRK(&uplo_c, tr, &n, &m, &one, &dwork[wp - 1], &m, &zero, dwork, &n);
                        SLC_DGEMM(nt, nt, &n, &n, &m, &beta, g, &ldg, &dwork[wp - 1], &m, &one, c, &ldc);
                    }

                    norms[1] = SLC_DLANSY("F", &uplo_c, &n, dwork, &n, dwork);

                    idx = 0;
                    if (luplo) {
                        for (i32 j = 0; j < n; j++) {
                            i32 jj = j + 1;
                            SLC_DAXPY(&jj, &beta, &dwork[idx], &int1, &r[j * ldr], &int1);
                            idx += n;
                        }
                    } else {
                        for (i32 j = 0; j < n; j++) {
                            i32 len = n - j;
                            SLC_DAXPY(&len, &beta, &dwork[idx], &int1, &r[j + j * ldr], &int1);
                            idx += n + 1;
                        }
                    }
                } else {
                    SLC_DSYRK(&uplo_c, nt, &n, &m, &one, g, &ldg, &zero, &dwork[wp - 1], &n);

                    if (ljobe) {
                        SLC_DSYMM(side, &uplo_c, &n, &n, &beta, &dwork[wp - 1], &n, dwork, &n, &zero, c, &ldc);
                        if (ltrans) {
                            SLC_DGEMM(nt, tr, &n, &n, &n, &one, c, &ldc, dwork, &n, &zero, &dwork[wp - 1], &n);
                        } else {
                            SLC_DGEMM(tr, nt, &n, &n, &n, &one, dwork, &n, c, &ldc, &zero, &dwork[wp - 1], &n);
                        }
                    } else {
                        ma02ed(uplo_c, n, dwork, n);
                        SLC_DSYMM(nside, &uplo_c, &n, &n, &beta, x, &ldx, dwork, &n, &zero, c, &ldc);
                        SLC_DSYMM(side, &uplo_c, &n, &n, &one, x, &ldx, c, &ldc, &zero, dwork, &n);
                    }

                    norms[1] = SLC_DLANSY("F", &uplo_c, &n, &dwork[wp - 1], &n, dwork);
                    i32 idx = wp - 1;

                    if (luplo) {
                        for (i32 j = 0; j < n; j++) {
                            i32 jj = j + 1;
                            SLC_DAXPY(&jj, &one, &dwork[idx], &int1, &r[j * ldr], &int1);
                            idx += n;
                        }
                    } else {
                        for (i32 j = 0; j < n; j++) {
                            i32 len = n - j;
                            SLC_DAXPY(&len, &one, &dwork[idx], &int1, &r[j + j * ldr], &int1);
                            idx += n + 1;
                        }
                    }

                    for (i32 j = 0; j < n; j++) {
                        SLC_DAXPY(&n, &one, &a[j * lda], &int1, &c[j * ldc], &int1);
                    }
                }
            } else {
                norms[1] = zero;
                SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);
            }
        }
    } else if (ljobb) {
        // JOB = 'B' case - similar structure, residual and norms only
        if (discr) {
            if (ljobe) {
                i32 tmpinfo = 0;
                mb01ru(&uplo_c, ntrans, n, n, zero, one, dwork, n, e, lde, x, ldx, &dwork[nn], nn, &tmpinfo);

                norms[2] = SLC_DLANSY("F", &uplo_c, &n, dwork, &n, dwork);

                i32 idx = 0;
                if (luplo) {
                    for (i32 j = 0; j < n; j++) {
                        f64 mone = -one;
                        i32 jj = j + 1;
                        SLC_DAXPY(&jj, &mone, &dwork[idx], &int1, &r[j * ldr], &int1);
                        idx += n;
                    }
                } else {
                    for (i32 j = 0; j < n; j++) {
                        f64 mone = -one;
                        i32 len = n - j;
                        SLC_DAXPY(&len, &mone, &dwork[idx], &int1, &r[j + j * ldr], &int1);
                        idx += n + 1;
                    }
                }
            } else {
                if (luplo) {
                    for (i32 j = 0; j < n; j++) {
                        f64 mone = -one;
                        i32 jj = j + 1;
                        SLC_DAXPY(&jj, &mone, &x[j * ldx], &int1, &r[j * ldr], &int1);
                    }
                } else {
                    for (i32 j = 0; j < n; j++) {
                        f64 mone = -one;
                        i32 len = n - j;
                        SLC_DAXPY(&len, &mone, &x[j + j * ldx], &int1, &r[j + j * ldr], &int1);
                    }
                }
            }

            i32 idx, iw;
            if (ljobl) {
                idx = 0;
                i32 tmpinfo = 0;
                mb01rb(side, &uplo_c, tr, n, n, zero, one, dwork, n, a, lda, xe, ldxe, &tmpinfo);
            } else {
                if (use1) {
                    iw = 0;
                    idx = wp - 1;
                } else {
                    iw = wp - 1;
                    idx = 0;
                }

                SLC_DSYMM(side, &uplo_c, &n, &n, &one, x, &ldx, a, &lda, &zero, &dwork[iw], &n);
                i32 tmpinfo = 0;
                mb01rb(side, &uplo_c, tr, n, n, zero, one, &dwork[idx], n, a, lda, &dwork[iw], n, &tmpinfo);
            }

            norms[0] = SLC_DLANSY("F", &uplo_c, &n, &dwork[idx], &n, dwork);

            if (luplo) {
                for (i32 j = 0; j < n; j++) {
                    i32 jj = j + 1;
                    SLC_DAXPY(&jj, &one, &dwork[idx], &int1, &r[j * ldr], &int1);
                    idx += n;
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    i32 len = n - j;
                    SLC_DAXPY(&len, &one, &dwork[idx], &int1, &r[j + j * ldr], &int1);
                    idx += n + 1;
                }
            }

            if (ljobl) {
                if (ljobf) {
                    SLC_DSYRK(&uplo_c, nt, &n, &m, &one, f, &ldf, &zero, dwork, &n);
                } else {
                    i32 tmpinfo = 0;
                    mb01rb("L", &uplo_c, nt, n, m, zero, one, dwork, n, f, ldf, k, ldk, &tmpinfo);
                }

                norms[1] = SLC_DLANSY("F", &uplo_c, &n, dwork, &n, dwork);

                idx = 0;
                if (luplo) {
                    for (i32 j = 0; j < n; j++) {
                        i32 jj = j + 1;
                        SLC_DAXPY(&jj, &beta, &dwork[idx], &int1, &r[j * ldr], &int1);
                        idx += n;
                    }
                } else {
                    for (i32 j = 0; j < n; j++) {
                        i32 len = n - j;
                        SLC_DAXPY(&len, &beta, &dwork[idx], &int1, &r[j + j * ldr], &int1);
                        idx += n + 1;
                    }
                }
            } else if (ljobg) {
                i32 tmpinfo = 0;
                mb01ru(&uplo_c, ntrans, n, n, zero, one, dwork, n, &dwork[iw], n, g, ldg, &dwork[2*nn], nn, &tmpinfo);

                norms[1] = SLC_DLANSY("F", &uplo_c, &n, dwork, &n, dwork);

                idx = 0;
                if (luplo) {
                    for (i32 j = 0; j < n; j++) {
                        i32 jj = j + 1;
                        SLC_DAXPY(&jj, &beta, &dwork[idx], &int1, &r[j * ldr], &int1);
                        idx += n;
                    }
                } else {
                    for (i32 j = 0; j < n; j++) {
                        i32 len = n - j;
                        SLC_DAXPY(&len, &beta, &dwork[idx], &int1, &r[j + j * ldr], &int1);
                        idx += n + 1;
                    }
                }
            } else if (m > 0) {
                if (use1) {
                    if (ltrans) {
                        SLC_DGEMM(nt, nt, &n, &m, &n, &one, dwork, &n, g, &ldg, &zero, &dwork[wp - 1], &n);
                        SLC_DSYRK(&uplo_c, nt, &n, &m, &one, &dwork[wp - 1], &n, &zero, dwork, &n);
                    } else {
                        SLC_DGEMM(tr, nt, &m, &n, &n, &one, g, &ldg, dwork, &n, &zero, &dwork[wp - 1], &m);
                        SLC_DSYRK(&uplo_c, tr, &n, &m, &one, &dwork[wp - 1], &m, &zero, dwork, &n);
                    }

                    norms[1] = SLC_DLANSY("F", &uplo_c, &n, dwork, &n, dwork);

                    idx = 0;
                    if (luplo) {
                        for (i32 j = 0; j < n; j++) {
                            i32 jj = j + 1;
                            SLC_DAXPY(&jj, &beta, &dwork[idx], &int1, &r[j * ldr], &int1);
                            idx += n;
                        }
                    } else {
                        for (i32 j = 0; j < n; j++) {
                            i32 len = n - j;
                            SLC_DAXPY(&len, &beta, &dwork[idx], &int1, &r[j + j * ldr], &int1);
                            idx += n + 1;
                        }
                    }
                } else {
                    SLC_DSYRK(&uplo_c, nt, &n, &m, &one, g, &ldg, &zero, dwork, &n);
                    i32 tmpinfo = 0;
                    mb01ru(&uplo_c, ntrans, n, n, zero, one, dwork, n, &dwork[iw], n, dwork, n, &dwork[2*nn], nn, &tmpinfo);
                    i32 ldx1 = n + 1;
                    f64 half = one / two;
                    SLC_DSCAL(&n, &half, dwork, &ldx1);

                    norms[1] = SLC_DLANSY("F", &uplo_c, &n, dwork, &n, dwork);

                    idx = 0;
                    if (luplo) {
                        for (i32 j = 0; j < n; j++) {
                            i32 jj = j + 1;
                            SLC_DAXPY(&jj, &beta, &dwork[idx], &int1, &r[j * ldr], &int1);
                            idx += n;
                        }
                    } else {
                        for (i32 j = 0; j < n; j++) {
                            i32 len = n - j;
                            SLC_DAXPY(&len, &beta, &dwork[idx], &int1, &r[j + j * ldr], &int1);
                            idx += n + 1;
                        }
                    }
                }
            } else {
                norms[1] = zero;
            }
        } else {
            // Continuous-time JOB='B'
            i32 idx, iw;

            if (ljobe) {
                if (ljobl) {
                    idx = 0;
                    if (ltrans) {
                        SLC_DGEMM(nt, tr, &n, &n, &n, &one, xe, &ldxe, a, &lda, &zero, dwork, &n);
                    } else {
                        SLC_DGEMM(tr, nt, &n, &n, &n, &one, a, &lda, xe, &ldxe, &zero, dwork, &n);
                    }
                } else {
                    if (use1) {
                        iw = 0;
                        idx = wp - 1;
                    } else {
                        iw = wp - 1;
                        idx = 0;
                    }

                    SLC_DSYMM(side, &uplo_c, &n, &n, &one, x, &ldx, e, &lde, &zero, &dwork[iw], &n);

                    if (ltrans) {
                        SLC_DGEMM(nt, tr, &n, &n, &n, &one, &dwork[iw], &n, a, &lda, &zero, &dwork[idx], &n);
                    } else {
                        SLC_DGEMM(tr, nt, &n, &n, &n, &one, a, &lda, &dwork[iw], &n, &zero, &dwork[idx], &n);
                    }
                }
            } else {
                idx = 0;
                SLC_DSYMM(side, &uplo_c, &n, &n, &one, x, &ldx, a, &lda, &zero, dwork, &n);
            }

            norms[0] = SLC_DLANGE("F", &n, &n, &dwork[idx], &n, dwork);

            if (luplo) {
                i32 l = idx;
                for (i32 j = 0; j < n; j++) {
                    i32 jj = j + 1;
                    SLC_DAXPY(&jj, &one, &dwork[idx], &int1, &r[j * ldr], &int1);
                    SLC_DAXPY(&jj, &one, &dwork[l], &n, &r[j * ldr], &int1);
                    idx += n;
                    l++;
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    i32 len = n - j;
                    SLC_DAXPY(&len, &one, &dwork[idx], &int1, &r[j + j * ldr], &int1);
                    SLC_DAXPY(&len, &one, &dwork[idx], &n, &r[j + j * ldr], &int1);
                    idx += n + 1;
                }
            }

            if (ljobl) {
                if (ljobf) {
                    SLC_DSYRK(&uplo_c, nt, &n, &m, &one, f, &ldf, &zero, dwork, &n);
                } else {
                    i32 tmpinfo = 0;
                    mb01rb("L", &uplo_c, nt, n, m, zero, one, dwork, n, f, ldf, k, ldk, &tmpinfo);
                }

                norms[1] = SLC_DLANSY("F", &uplo_c, &n, dwork, &n, dwork);

                idx = 0;
                if (luplo) {
                    for (i32 j = 0; j < n; j++) {
                        i32 jj = j + 1;
                        SLC_DAXPY(&jj, &beta, &dwork[idx], &int1, &r[j * ldr], &int1);
                        idx += n;
                    }
                } else {
                    for (i32 j = 0; j < n; j++) {
                        i32 len = n - j;
                        SLC_DAXPY(&len, &beta, &dwork[idx], &int1, &r[j + j * ldr], &int1);
                        idx += n + 1;
                    }
                }
            } else if (ljobg) {
                if (ljobe) {
                    i32 tmpinfo = 0;
                    mb01ru(&uplo_c, ntrans, n, n, zero, one, dwork, n, &dwork[iw], n, g, ldg, &dwork[2*nn], nn, &tmpinfo);
                } else {
                    if (keepx) {
                        SLC_DLACPY(&uplo_c, &n, &n, x, &ldx, &dwork[wp - 1], &n);
                        ma02ed(uplo_c, n, &dwork[wp - 1], n);
                        i32 tmpinfo = 0;
                        mb01ru(&uplo_c, ntrans, n, n, zero, one, dwork, n, &dwork[wp - 1], n, g, ldg, &dwork[2*nn], nn, &tmpinfo);
                    } else {
                        i32 tmpinfo = 0;
                        mb01ru(&uplo_c, ntrans, n, n, zero, one, dwork, n, x, ldx, g, ldg, &dwork[nn], nn, &tmpinfo);
                    }
                }

                norms[1] = SLC_DLANSY("F", &uplo_c, &n, dwork, &n, dwork);

                idx = 0;
                if (luplo) {
                    for (i32 j = 0; j < n; j++) {
                        i32 jj = j + 1;
                        SLC_DAXPY(&jj, &beta, &dwork[idx], &int1, &r[j * ldr], &int1);
                        idx += n;
                    }
                } else {
                    for (i32 j = 0; j < n; j++) {
                        i32 len = n - j;
                        SLC_DAXPY(&len, &beta, &dwork[idx], &int1, &r[j + j * ldr], &int1);
                        idx += n + 1;
                    }
                }
            } else if (m > 0) {
                if (use1) {
                    if (ljobe) {
                        if (ltrans) {
                            SLC_DGEMM(nt, nt, &n, &m, &n, &one, dwork, &n, g, &ldg, &zero, &dwork[wp - 1], &n);
                        } else {
                            SLC_DGEMM(tr, nt, &m, &n, &n, &one, g, &ldg, dwork, &n, &zero, &dwork[wp - 1], &m);
                        }
                    } else {
                        SLC_DSYMM("L", &uplo_c, &n, &m, &one, x, &ldx, g, &ldg, &zero, &dwork[wp - 1], &n);
                    }

                    if (unite || ltrans) {
                        SLC_DSYRK(&uplo_c, nt, &n, &m, &one, &dwork[wp - 1], &n, &zero, dwork, &n);
                    } else {
                        SLC_DSYRK(&uplo_c, tr, &n, &m, &one, &dwork[wp - 1], &m, &zero, dwork, &n);
                    }

                    norms[1] = SLC_DLANSY("F", &uplo_c, &n, dwork, &n, dwork);

                    idx = 0;
                    if (luplo) {
                        for (i32 j = 0; j < n; j++) {
                            i32 jj = j + 1;
                            SLC_DAXPY(&jj, &beta, &dwork[idx], &int1, &r[j * ldr], &int1);
                            idx += n;
                        }
                    } else {
                        for (i32 j = 0; j < n; j++) {
                            i32 len = n - j;
                            SLC_DAXPY(&len, &beta, &dwork[idx], &int1, &r[j + j * ldr], &int1);
                            idx += n + 1;
                        }
                    }
                } else {
                    SLC_DSYRK(&uplo_c, nt, &n, &m, &one, g, &ldg, &zero, dwork, &n);

                    if (ljobe) {
                        i32 tmpinfo = 0;
                        mb01ru(&uplo_c, ntrans, n, n, zero, beta, dwork, n, &dwork[iw], n, dwork, n, &dwork[2*nn], nn, &tmpinfo);
                        i32 ldx1 = n + 1;
                        f64 half = one / two;
                        SLC_DSCAL(&n, &half, dwork, &ldx1);
                    } else if (ldwork >= 3 * nn) {
                        SLC_DLACPY(&uplo_c, &n, &n, x, &ldx, &dwork[nn], &n);
                        ma02ed(uplo_c, n, &dwork[nn], n);
                        i32 tmpinfo = 0;
                        mb01ru(&uplo_c, ntrans, n, n, zero, beta, dwork, n, &dwork[nn], n, dwork, n, &dwork[2*nn], nn, &tmpinfo);
                        i32 ldx1 = n + 1;
                        f64 half = one / two;
                        SLC_DSCAL(&n, &half, dwork, &ldx1);
                    } else {
                        ma02ed(uplo_c, n, dwork, n);
                        SLC_DSYMM(nside, &uplo_c, &n, &n, &beta, x, &ldx, dwork, &n, &zero, &dwork[nn], &n);
                        SLC_DSYMM(side, &uplo_c, &n, &n, &one, x, &ldx, &dwork[nn], &n, &zero, dwork, &n);
                    }

                    norms[1] = SLC_DLANSY("F", &uplo_c, &n, dwork, &n, dwork);
                    idx = 0;

                    if (luplo) {
                        for (i32 j = 0; j < n; j++) {
                            i32 jj = j + 1;
                            SLC_DAXPY(&jj, &one, &dwork[idx], &int1, &r[j * ldr], &int1);
                            idx += n;
                        }
                    } else {
                        for (i32 j = 0; j < n; j++) {
                            i32 len = n - j;
                            SLC_DAXPY(&len, &one, &dwork[idx], &int1, &r[j + j * ldr], &int1);
                            idx += n + 1;
                        }
                    }
                }
            } else {
                norms[1] = zero;
            }
        }
    } else {
        // JOB != 'N' and JOB != 'B' (i.e., JOB = 'A', 'R', or 'C')
        if (discr) {
            // Discrete-time case
            if (ljobl) {
                if (nljobc) {
                    i32 tmpinfo = 0;
                    mb01rb(side, &uplo_c, tr, n, n, one, one, r, ldr, a, lda, xe, ldxe, &tmpinfo);

                    if (ljobe) {
                        if (ljobr) {
                            tmpinfo = 0;
                            mb01ru(&uplo_c, ntrans, n, n, one, -one, r, ldr, e, lde, x, ldx, dwork, nn, &tmpinfo);
                        } else {
                            tmpinfo = 0;
                            mb01ru(&uplo_c, ntrans, n, n, one, -one, r, ldr, e, lde, x, ldx, c, nn, &tmpinfo);
                        }
                    }

                    if (ljobf) {
                        SLC_DSYRK(&uplo_c, nt, &n, &m, &beta, f, &ldf, &one, r, &ldr);
                    } else {
                        tmpinfo = 0;
                        mb01rb("L", &uplo_c, nt, n, m, one, beta, r, ldr, f, ldf, k, ldk, &tmpinfo);
                    }
                }

                if (nljobr) {
                    SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);

                    if (ljobf) {
                        if (ltrans) {
                            SLC_DGEMM(nt, tr, &n, &n, &m, &beta, f, &ldf, g, &ldg, &one, c, &ldc);
                        } else {
                            SLC_DGEMM(nt, tr, &n, &n, &m, &beta, g, &ldg, f, &ldf, &one, c, &ldc);
                        }
                    } else {
                        if (ltrans) {
                            SLC_DGEMM(tr, tr, &n, &n, &m, &beta, k, &ldk, g, &ldg, &one, c, &ldc);
                        } else {
                            SLC_DGEMM(nt, nt, &n, &n, &m, &beta, g, &ldg, k, &ldk, &one, c, &ldc);
                        }
                    }
                }
            } else {
                // Usual case (JOBG = 'G' or JOBG = 'D')
                bool usec = ljoba && use1;

                if (usec) {
                    SLC_DSYMM(side, &uplo_c, &n, &n, &one, x, &ldx, a, &lda, &zero, c, &ldc);
                } else if (nljobc || ljobg || useopt) {
                    SLC_DSYMM(side, &uplo_c, &n, &n, &one, x, &ldx, a, &lda, &zero, dwork, &n);
                }

                if (ljobc) {
                    if (ljobg) {
                        SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);
                        SLC_DSYMM(side, &uplo_c, &n, &n, &beta, g, &ldg, dwork, &n, &one, c, &ldc);
                    } else if (m == 0) {
                        SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);
                    } else if (use1) {
                        SLC_DSYMM("L", &uplo_c, &n, &m, &one, x, &ldx, g, &ldg, &zero, c, &ldc);

                        if (ltrans) {
                            SLC_DGEMM(nt, nt, &n, &m, &n, &one, a, &lda, c, &ldc, &zero, dwork, &n);
                            SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);
                            SLC_DGEMM(nt, tr, &n, &n, &m, &beta, dwork, &n, g, &ldg, &one, c, &ldc);
                        } else {
                            SLC_DGEMM(tr, nt, &m, &n, &n, &one, c, &ldc, a, &lda, &zero, dwork, &m);
                            SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);
                            SLC_DGEMM(nt, nt, &n, &n, &m, &beta, g, &ldg, dwork, &m, &one, c, &ldc);
                        }
                    } else if (useopt) {
                        SLC_DSYRK(&uplo_c, nt, &n, &m, &one, g, &ldg, &zero, &dwork[wp - 1], &n);
                        SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);
                        SLC_DSYMM(side, &uplo_c, &n, &n, &beta, &dwork[wp - 1], &n, dwork, &n, &one, c, &ldc);
                    } else {
                        SLC_DSYRK(&uplo_c, nt, &n, &m, &one, g, &ldg, &zero, c, &ldc);
                        ma02ed(uplo_c, n, c, ldc);
                        SLC_DSYMM(nside, &uplo_c, &n, &n, &one, x, &ldx, c, &ldc, &zero, dwork, &n);

                        SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);
                        if (ltrans) {
                            SLC_DGEMM(nt, nt, &n, &n, &n, &beta, a, &lda, dwork, &n, &one, c, &ldc);
                        } else {
                            SLC_DGEMM(nt, nt, &n, &n, &n, &beta, dwork, &n, a, &lda, &one, c, &ldc);
                        }
                    }
                } else {
                    if (usec) {
                        i32 tmpinfo = 0;
                        mb01rb(side, &uplo_c, tr, n, n, one, one, r, ldr, a, lda, c, ldc, &tmpinfo);
                    } else {
                        i32 tmpinfo = 0;
                        mb01rb(side, &uplo_c, tr, n, n, one, one, r, ldr, a, lda, dwork, n, &tmpinfo);
                    }

                    if (ljobg) {
                        if (ljobr) {
                            i32 tmpinfo = 0;
                            mb01ru(&uplo_c, ntrans, n, n, one, beta, r, ldr, dwork, n, g, ldg, &dwork[wp - 1], nn, &tmpinfo);
                        } else {
                            SLC_DSYMM(side, &uplo_c, &n, &n, &beta, g, &ldg, dwork, &n, &zero, c, &ldc);
                            i32 tmpinfo = 0;
                            mb01rb(side, &uplo_c, tr, n, n, one, one, r, ldr, dwork, n, c, ldc, &tmpinfo);

                            for (i32 j = 0; j < n; j++) {
                                SLC_DAXPY(&n, &one, &a[j * lda], &int1, &c[j * ldc], &int1);
                            }
                        }
                    } else if (m > 0) {
                        if (ljobr) {
                            if (use1) {
                                if (ltrans) {
                                    SLC_DGEMM(nt, nt, &n, &m, &n, &one, dwork, &n, g, &ldg, &zero, &dwork[wp - 1], &n);
                                    SLC_DSYRK(&uplo_c, nt, &n, &m, &beta, &dwork[wp - 1], &n, &one, r, &ldr);
                                } else {
                                    SLC_DGEMM(tr, nt, &m, &n, &n, &one, g, &ldg, dwork, &n, &zero, &dwork[wp - 1], &m);
                                    SLC_DSYRK(&uplo_c, tr, &n, &m, &beta, &dwork[wp - 1], &m, &one, r, &ldr);
                                }
                            } else {
                                SLC_DSYRK(&uplo_c, nt, &n, &m, &one, g, &ldg, &zero, &dwork[wp - 1], &n);
                                i32 tmpinfo = 0;
                                mb01ru(&uplo_c, ntrans, n, n, one, beta, r, ldr, dwork, n, &dwork[wp - 1], n, &dwork[wp - 1 + nn], nn, &tmpinfo);
                            }
                        } else {
                            if (usec) {
                                if (ltrans) {
                                    SLC_DGEMM(nt, nt, &n, &m, &n, &one, c, &ldc, g, &ldg, &zero, dwork, &n);
                                    SLC_DSYRK(&uplo_c, nt, &n, &m, &beta, dwork, &n, &one, r, &ldr);
                                    SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);
                                    SLC_DGEMM(nt, tr, &n, &n, &m, &beta, dwork, &n, g, &ldg, &one, c, &ldc);
                                } else {
                                    SLC_DGEMM(tr, nt, &m, &n, &n, &one, g, &ldg, c, &ldc, &zero, dwork, &m);
                                    SLC_DSYRK(&uplo_c, tr, &n, &m, &beta, dwork, &m, &one, r, &ldr);
                                    SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);
                                    SLC_DGEMM(nt, nt, &n, &n, &m, &beta, g, &ldg, dwork, &m, &one, c, &ldc);
                                }
                            } else {
                                SLC_DSYRK(&uplo_c, nt, &n, &m, &one, g, &ldg, &zero, &dwork[wp - 1], &n);
                                SLC_DSYMM(side, &uplo_c, &n, &n, &beta, &dwork[wp - 1], &n, dwork, &n, &zero, c, &ldc);
                                i32 tmpinfo = 0;
                                mb01rb(side, &uplo_c, tr, n, n, one, one, r, ldr, dwork, n, c, ldc, &tmpinfo);

                                for (i32 j = 0; j < n; j++) {
                                    SLC_DAXPY(&n, &one, &a[j * lda], &int1, &c[j * ldc], &int1);
                                }
                            }
                        }
                    } else if (nljobr) {
                        SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);
                    }
                }
            }

            if (nljobc) {
                if (unite) {
                    if (luplo) {
                        for (i32 j = 0; j < n; j++) {
                            f64 mone = -one;
                            i32 jj = j + 1;
                            SLC_DAXPY(&jj, &mone, &x[j * ldx], &int1, &r[j * ldr], &int1);
                        }
                    } else {
                        for (i32 j = 0; j < n; j++) {
                            f64 mone = -one;
                            i32 len = n - j;
                            SLC_DAXPY(&len, &mone, &x[j + j * ldx], &int1, &r[j + j * ldr], &int1);
                        }
                    }
                } else if (!ljobl) {
                    i32 tmpinfo = 0;
                    mb01ru(&uplo_c, ntrans, n, n, one, -one, r, ldr, e, lde, x, ldx, dwork, nn, &tmpinfo);
                }
            }
        } else {
            // Continuous-time case (JOB != 'N' and JOB != 'B')
            bool usec = ljoba && use1;
            bool fullx = unite && !keepx;

            if (ljobe && !ljobl) {
                if (usec) {
                    SLC_DSYMM(side, &uplo_c, &n, &n, &one, x, &ldx, e, &lde, &zero, c, &ldc);
                } else if (ljobg || nljobc) {
                    SLC_DSYMM(side, &uplo_c, &n, &n, &one, x, &ldx, e, &lde, &zero, &dwork[wp - 1], &n);
                }
            } else if (usec) {
                if (keepx) {
                    SLC_DLACPY(&uplo_c, &n, &n, x, &ldx, c, &ldc);
                    ma02ed(uplo_c, n, c, ldc);
                }
            }

            if (ljobl) {
                if (nljobc) {
                    if (fullx) {
                        SLC_DSYR2K(&uplo_c, ntrans, &n, &n, &one, a, &lda, x, &ldx, &one, r, &ldr);
                    } else if (unite) {
                        if (ljobr) {
                            SLC_DLACPY(&uplo_c, &n, &n, x, &ldx, dwork, &n);
                            ma02ed(uplo_c, n, dwork, n);
                            SLC_DSYR2K(&uplo_c, ntrans, &n, &n, &one, a, &lda, dwork, &n, &one, r, &ldr);
                        } else {
                            SLC_DLACPY(&uplo_c, &n, &n, x, &ldx, c, &ldc);
                            ma02ed(uplo_c, n, c, ldc);
                            SLC_DSYR2K(&uplo_c, ntrans, &n, &n, &one, a, &lda, c, &ldc, &one, r, &ldr);
                        }
                    } else {
                        SLC_DSYR2K(&uplo_c, ntrans, &n, &n, &one, a, &lda, xe, &ldxe, &one, r, &ldr);
                    }

                    if (ljobf) {
                        SLC_DSYRK(&uplo_c, nt, &n, &m, &beta, f, &ldf, &one, r, &ldr);
                    } else {
                        i32 tmpinfo = 0;
                        mb01rb("L", &uplo_c, nt, n, m, one, beta, r, ldr, f, ldf, k, ldk, &tmpinfo);
                    }
                }

                if (nljobr) {
                    SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);

                    if (ljobf) {
                        if (ltrans) {
                            SLC_DGEMM(nt, tr, &n, &n, &m, &beta, f, &ldf, g, &ldg, &one, c, &ldc);
                        } else {
                            SLC_DGEMM(nt, tr, &n, &n, &m, &beta, g, &ldg, f, &ldf, &one, c, &ldc);
                        }
                    } else {
                        if (ltrans) {
                            SLC_DGEMM(tr, tr, &n, &n, &m, &beta, k, &ldk, g, &ldg, &one, c, &ldc);
                        } else {
                            SLC_DGEMM(nt, nt, &n, &n, &m, &beta, g, &ldg, k, &ldk, &one, c, &ldc);
                        }
                    }
                }
            } else {
                if (ljobc) {
                    if (ljobg) {
                        SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);

                        if (fullx) {
                            SLC_DSYMM(side, &uplo_c, &n, &n, &beta, g, &ldg, x, &ldx, &one, c, &ldc);
                        } else {
                            if (unite) {
                                SLC_DLACPY(&uplo_c, &n, &n, x, &ldx, dwork, &n);
                                ma02ed(uplo_c, n, dwork, n);
                            }
                            SLC_DSYMM(side, &uplo_c, &n, &n, &beta, g, &ldg, dwork, &n, &one, c, &ldc);
                        }
                    } else if (m == 0) {
                        SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);
                    } else if (ljobe) {
                        if (use1) {
                            SLC_DSYMM("L", &uplo_c, &n, &m, &one, x, &ldx, g, &ldg, &zero, c, &ldc);

                            if (ltrans) {
                                SLC_DGEMM(nt, nt, &n, &m, &n, &one, e, &lde, c, &ldc, &zero, dwork, &n);
                                SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);
                                SLC_DGEMM(nt, tr, &n, &n, &m, &beta, dwork, &n, g, &ldg, &one, c, &ldc);
                            } else {
                                SLC_DGEMM(tr, nt, &m, &n, &n, &one, c, &ldc, e, &lde, &zero, dwork, &m);
                                SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);
                                SLC_DGEMM(nt, nt, &n, &n, &m, &beta, g, &ldg, dwork, &m, &one, c, &ldc);
                            }
                        } else {
                            SLC_DSYRK(&uplo_c, nt, &n, &m, &one, g, &ldg, &zero, c, &ldc);
                            ma02ed(uplo_c, n, c, ldc);
                            SLC_DSYMM(nside, &uplo_c, &n, &n, &one, x, &ldx, c, &ldc, &zero, dwork, &n);
                            SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);
                            if (ltrans) {
                                SLC_DGEMM(nt, nt, &n, &n, &n, &beta, e, &lde, dwork, &n, &one, c, &ldc);
                            } else {
                                SLC_DGEMM(nt, nt, &n, &n, &n, &beta, dwork, &n, e, &lde, &one, c, &ldc);
                            }
                        }
                    } else {
                        SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);

                        if (use1) {
                            SLC_DSYMM("L", &uplo_c, &n, &m, &one, x, &ldx, g, &ldg, &zero, dwork, &n);

                            if (ltrans) {
                                SLC_DGEMM(nt, tr, &n, &n, &m, &beta, dwork, &n, g, &ldg, &one, c, &ldc);
                            } else {
                                SLC_DGEMM(nt, tr, &n, &n, &m, &beta, g, &ldg, dwork, &n, &one, c, &ldc);
                            }
                        } else {
                            SLC_DSYRK(&uplo_c, nt, &n, &m, &one, g, &ldg, &zero, dwork, &n);
                            ma02ed(uplo_c, n, dwork, n);
                            SLC_DSYMM(nside, &uplo_c, &n, &n, &beta, x, &ldx, dwork, &n, &one, c, &ldc);
                        }
                    }
                } else {
                    // Compute R (and op(C))
                    if (useatw) {
                        if (fullx) {
                            SLC_DSYR2K(&uplo_c, ntrans, &n, &n, &one, a, &lda, x, &ldx, &one, r, &ldr);
                        } else if (usec) {
                            SLC_DSYR2K(&uplo_c, ntrans, &n, &n, &one, a, &lda, c, &ldc, &one, r, &ldr);
                        } else {
                            if (unite) {
                                SLC_DLACPY(&uplo_c, &n, &n, x, &ldx, &dwork[wp - 1], &n);
                                ma02ed(uplo_c, n, &dwork[wp - 1], n);
                            }
                            SLC_DSYR2K(&uplo_c, ntrans, &n, &n, &one, a, &lda, &dwork[wp - 1], &n, &one, r, &ldr);
                        }
                    }

                    f64 alpha = beta / two;

                    if (ljobg) {
                        if (ljobr) {
                            SLC_DLACPY("A", &n, &n, a, &lda, dwork, &n);

                            if (fullx) {
                                SLC_DSYMM(side, &uplo_c, &n, &n, &alpha, g, &ldg, x, &ldx, &one, dwork, &n);
                                SLC_DSYR2K(&uplo_c, ntrans, &n, &n, &one, dwork, &n, x, &ldx, &one, r, &ldr);
                            } else {
                                if (unite) {
                                    SLC_DLACPY(&uplo_c, &n, &n, x, &ldx, &dwork[wp - 1], &n);
                                    ma02ed(uplo_c, n, &dwork[wp - 1], n);
                                }
                                SLC_DSYMM(side, &uplo_c, &n, &n, &alpha, g, &ldg, &dwork[wp - 1], &n, &one, dwork, &n);
                                SLC_DSYR2K(&uplo_c, ntrans, &n, &n, &one, dwork, &n, &dwork[wp - 1], &n, &one, r, &ldr);
                            }
                        } else {
                            if (ldwork < optwrk) {
                                if (fullx) {
                                    SLC_DSYMM(side, &uplo_c, &n, &n, &beta, g, &ldg, x, &ldx, &zero, c, &ldc);
                                    i32 tmpinfo = 0;
                                    mb01rb(side, &uplo_c, nt, n, n, one, one, r, ldr, x, ldx, c, ldc, &tmpinfo);
                                } else {
                                    if (unite) {
                                        SLC_DLACPY(&uplo_c, &n, &n, x, &ldx, &dwork[wp - 1], &n);
                                        ma02ed(uplo_c, n, &dwork[wp - 1], n);
                                    }
                                    SLC_DSYMM(side, &uplo_c, &n, &n, &beta, g, &ldg, &dwork[wp - 1], &n, &zero, c, &ldc);
                                    i32 tmpinfo = 0;
                                    mb01rb(side, &uplo_c, tr, n, n, one, one, r, ldr, &dwork[wp - 1], n, c, ldc, &tmpinfo);
                                }

                                for (i32 j = 0; j < n; j++) {
                                    SLC_DAXPY(&n, &one, &a[j * lda], &int1, &c[j * ldc], &int1);
                                }
                            } else {
                                i32 idx = 0;
                                if (fullx) {
                                    SLC_DSYMM(side, &uplo_c, &n, &n, &alpha, g, &ldg, x, &ldx, &zero, dwork, &n);

                                    for (i32 j = 0; j < n; j++) {
                                        SLC_DCOPY(&n, &a[j * lda], &int1, &c[j * ldc], &int1);
                                        SLC_DAXPY(&n, &two, &dwork[idx], &int1, &c[j * ldc], &int1);
                                        SLC_DAXPY(&n, &one, &a[j * lda], &int1, &dwork[idx], &int1);
                                        idx += n;
                                    }

                                    SLC_DSYR2K(&uplo_c, ntrans, &n, &n, &one, dwork, &n, x, &ldx, &one, r, &ldr);
                                } else {
                                    if (unite) {
                                        SLC_DLACPY(&uplo_c, &n, &n, x, &ldx, &dwork[wp - 1], &n);
                                        ma02ed(uplo_c, n, &dwork[wp - 1], n);
                                    }

                                    SLC_DSYMM(side, &uplo_c, &n, &n, &alpha, g, &ldg, &dwork[wp - 1], &n, &zero, dwork, &n);

                                    for (i32 j = 0; j < n; j++) {
                                        SLC_DCOPY(&n, &a[j * lda], &int1, &c[j * ldc], &int1);
                                        SLC_DAXPY(&n, &two, &dwork[idx], &int1, &c[j * ldc], &int1);
                                        SLC_DAXPY(&n, &one, &a[j * lda], &int1, &dwork[idx], &int1);
                                        idx += n;
                                    }

                                    SLC_DSYR2K(&uplo_c, ntrans, &n, &n, &one, dwork, &n, &dwork[wp - 1], &n, &one, r, &ldr);
                                }
                            }
                        }
                    } else if (m == 0) {
                        if (nljobr) {
                            SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);
                        }
                    } else {
                        // Use D
                        if (ljobr) {
                            if (useatw) {
                                if (ljobe) {
                                    i32 jj;
                                    if (ltrans) {
                                        SLC_DGEMM(nt, nt, &n, &m, &n, &one, &dwork[wp - 1], &n, g, &ldg, &zero, dwork, &n);
                                        jj = n;
                                    } else {
                                        SLC_DGEMM(tr, nt, &m, &n, &n, &one, g, &ldg, &dwork[wp - 1], &n, &zero, dwork, &m);
                                        jj = m;
                                    }
                                    SLC_DSYRK(&uplo_c, ntrans, &n, &m, &beta, dwork, &jj, &one, r, &ldr);
                                } else {
                                    SLC_DSYMM("L", &uplo_c, &n, &m, &one, x, &ldx, g, &ldg, &zero, dwork, &n);
                                    SLC_DSYRK(&uplo_c, nt, &n, &m, &beta, dwork, &n, &one, r, &ldr);
                                }
                            } else {
                                SLC_DSYRK(&uplo_c, nt, &n, &m, &one, g, &ldg, &zero, &dwork[yp - 1], &n);
                                SLC_DLACPY("A", &n, &n, a, &lda, dwork, &n);

                                if (fullx) {
                                    SLC_DSYMM(side, &uplo_c, &n, &n, &alpha, &dwork[yp - 1], &n, x, &ldx, &one, dwork, &n);
                                    SLC_DSYR2K(&uplo_c, ntrans, &n, &n, &one, dwork, &n, x, &ldx, &one, r, &ldr);
                                } else {
                                    if (unite) {
                                        SLC_DLACPY(&uplo_c, &n, &n, x, &ldx, &dwork[wp - 1], &n);
                                        ma02ed(uplo_c, n, &dwork[wp - 1], n);
                                    }
                                    SLC_DSYMM(side, &uplo_c, &n, &n, &alpha, &dwork[yp - 1], &n, &dwork[wp - 1], &n, &one, dwork, &n);
                                    SLC_DSYR2K(&uplo_c, ntrans, &n, &n, &one, dwork, &n, &dwork[wp - 1], &n, &one, r, &ldr);
                                }
                            }
                        } else {
                            if (use1) {
                                if (ljobe) {
                                    if (ltrans) {
                                        SLC_DGEMM(nt, nt, &n, &m, &n, &one, c, &ldc, g, &ldg, &zero, dwork, &n);
                                        SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);
                                        SLC_DGEMM(nt, tr, &n, &n, &m, &beta, dwork, &n, g, &ldg, &one, c, &ldc);
                                        SLC_DSYRK(&uplo_c, nt, &n, &m, &beta, dwork, &n, &one, r, &ldr);
                                    } else {
                                        SLC_DGEMM(tr, nt, &m, &n, &n, &one, g, &ldg, c, &ldc, &zero, dwork, &m);
                                        SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);
                                        SLC_DGEMM(nt, nt, &n, &n, &m, &beta, g, &ldg, dwork, &m, &one, c, &ldc);
                                        SLC_DSYRK(&uplo_c, tr, &n, &m, &beta, dwork, &m, &one, r, &ldr);
                                    }
                                } else {
                                    SLC_DSYMM("L", &uplo_c, &n, &m, &one, x, &ldx, g, &ldg, &zero, dwork, &n);
                                    SLC_DLACPY("A", &n, &n, a, &lda, c, &ldc);
                                    if (ltrans) {
                                        SLC_DGEMM(nt, tr, &n, &n, &m, &beta, dwork, &n, g, &ldg, &one, c, &ldc);
                                    } else {
                                        SLC_DGEMM(nt, tr, &n, &n, &m, &beta, g, &ldg, dwork, &n, &one, c, &ldc);
                                    }
                                    SLC_DSYRK(&uplo_c, nt, &n, &m, &beta, dwork, &n, &one, r, &ldr);
                                }
                            } else if (lcnds) {
                                if (ljobe) {
                                    if (ltrans) {
                                        SLC_DGEMM(nt, nt, &n, &m, &n, &one, &dwork[wp - 1], &n, g, &ldg, &zero, c, &ldc);
                                        SLC_DGEMM(nt, tr, &n, &n, &m, &alpha, c, &ldc, g, &ldg, &zero, dwork, &n);
                                    } else {
                                        SLC_DGEMM(tr, nt, &m, &n, &n, &one, g, &ldg, &dwork[wp - 1], &n, &zero, c, &ldc);
                                        SLC_DGEMM(nt, nt, &n, &n, &m, &alpha, g, &ldg, c, &ldc, &zero, dwork, &n);
                                    }
                                } else {
                                    SLC_DSYMM("L", &uplo_c, &n, &m, &one, x, &ldx, g, &ldg, &zero, c, &ldc);
                                    if (ltrans) {
                                        SLC_DGEMM(nt, tr, &n, &n, &m, &alpha, c, &ldc, g, &ldg, &zero, dwork, &n);
                                    } else {
                                        SLC_DGEMM(nt, tr, &n, &n, &m, &alpha, g, &ldg, c, &ldc, &zero, dwork, &n);
                                    }
                                }

                                i32 idx = 0;
                                for (i32 j = 0; j < n; j++) {
                                    SLC_DCOPY(&n, &a[j * lda], &int1, &c[j * ldc], &int1);
                                    SLC_DAXPY(&n, &two, &dwork[idx], &int1, &c[j * ldc], &int1);
                                    SLC_DAXPY(&n, &one, &a[j * lda], &int1, &dwork[idx], &int1);
                                    idx += n;
                                }

                                if (fullx) {
                                    SLC_DSYR2K(&uplo_c, ntrans, &n, &n, &one, dwork, &n, x, &ldx, &one, r, &ldr);
                                } else {
                                    if (unite) {
                                        SLC_DLACPY(&uplo_c, &n, &n, x, &ldx, &dwork[wp - 1], &n);
                                        ma02ed(uplo_c, n, &dwork[wp - 1], n);
                                    }
                                    SLC_DSYR2K(&uplo_c, ntrans, &n, &n, &one, dwork, &n, &dwork[wp - 1], &n, &one, r, &ldr);
                                }
                            } else {
                                SLC_DSYRK(&uplo_c, nt, &n, &m, &one, g, &ldg, &zero, c, &ldc);

                                if (fullx) {
                                    SLC_DSYMM(side, &uplo_c, &n, &n, &alpha, c, &ldc, x, &ldx, &zero, dwork, &n);
                                } else {
                                    if (unite) {
                                        SLC_DLACPY(&uplo_c, &n, &n, x, &ldx, &dwork[wp - 1], &n);
                                        ma02ed(uplo_c, n, &dwork[wp - 1], n);
                                    }
                                    SLC_DSYMM(side, &uplo_c, &n, &n, &alpha, c, &ldc, &dwork[wp - 1], &n, &zero, dwork, &n);
                                }

                                i32 idx = 0;
                                for (i32 j = 0; j < n; j++) {
                                    SLC_DCOPY(&n, &a[j * lda], &int1, &c[j * ldc], &int1);
                                    SLC_DAXPY(&n, &two, &dwork[idx], &int1, &c[j * ldc], &int1);
                                    SLC_DAXPY(&n, &one, &a[j * lda], &int1, &dwork[idx], &int1);
                                    idx += n;
                                }

                                if (fullx) {
                                    SLC_DSYR2K(&uplo_c, ntrans, &n, &n, &one, dwork, &n, x, &ldx, &one, r, &ldr);
                                } else {
                                    SLC_DSYR2K(&uplo_c, ntrans, &n, &n, &one, dwork, &n, &dwork[wp - 1], &n, &one, r, &ldr);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    dwork[0] = (f64)((optwrk > 1) ? optwrk : 1);
}
