/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB04OD - Solve generalized Sylvester equations with separation estimation
 *
 * Solves for R and L one of the generalized Sylvester equations:
 *   A * R - L * B = scale * C
 *   D * R - L * E = scale * F    (equation 1)
 * or
 *   A' * R + D' * L = scale * C
 *   R * B' + L * E' = scale * (-F)   (equation 2)
 *
 * where A and D are M-by-M, B and E are N-by-N, and C, F, R, L are M-by-N.
 * The solution (R, L) overwrites (C, F).
 *
 * Optionally computes Dif estimate measuring separation of (A,D) from (B,E).
 */

#include "slicot.h"
#include "slicot_blas.h"

static int dummy_select(const f64* ar, const f64* ai, const f64* b)
{
    (void)ar;
    (void)ai;
    (void)b;
    return 0;
}

void sb04od(
    const char* reduce,
    const char* trans,
    const char* jobd,
    const i32 m,
    const i32 n,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* c,
    const i32 ldc,
    f64* d,
    const i32 ldd,
    f64* e,
    const i32 lde,
    f64* f,
    const i32 ldf,
    f64* scale,
    f64* dif,
    f64* p,
    const i32 ldp,
    f64* q,
    const i32 ldq,
    f64* u,
    const i32 ldu,
    f64* v,
    const i32 ldv,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    *info = 0;

    i32 mn = (m > n) ? m : n;

    bool lredur = (*reduce == 'R' || *reduce == 'r');
    bool lredua = (*reduce == 'A' || *reduce == 'a');
    bool lredub = (*reduce == 'B' || *reduce == 'b');
    bool lredra = lredur || lredua;
    bool lredrb = lredur || lredub;
    bool lreduc = lredra || lredub;
    bool lquery = (ldwork == -1);

    i32 minwrk;
    if (lredur) {
        i32 t1 = 11 * mn;
        i32 t2 = 10 * mn + 23;
        minwrk = (t1 > t2) ? t1 : t2;
        if (minwrk < 1) minwrk = 1;
    } else if (lredua) {
        i32 t1 = 11 * m;
        i32 t2 = 10 * m + 23;
        minwrk = (t1 > t2) ? t1 : t2;
        if (minwrk < 1) minwrk = 1;
    } else if (lredub) {
        i32 t1 = 11 * n;
        i32 t2 = 10 * n + 23;
        minwrk = (t1 > t2) ? t1 : t2;
        if (minwrk < 1) minwrk = 1;
    } else {
        minwrk = 1;
    }

    bool ltrann = (*trans == 'N' || *trans == 'n');
    bool ljob1 = false, ljob2 = false, ljobd = false, ljobf = false, ljobdf = false;

    if (ltrann) {
        ljob1 = (*jobd == '1');
        ljob2 = (*jobd == '2');
        ljobd = (*jobd == 'D' || *jobd == 'd');
        ljobf = (*jobd == 'F' || *jobd == 'f');
        ljobdf = ljob1 || ljob2 || ljobd || ljobf;
        if (ljobd || ljobf) {
            i32 t = 2 * m * n;
            if (minwrk < t) minwrk = t;
        }
    }

    if (!lreduc && !(*reduce == 'N' || *reduce == 'n')) {
        *info = -1;
    } else if (!ltrann && !(*trans == 'T' || *trans == 't')) {
        *info = -2;
    } else if (ltrann) {
        if (!ljobdf && !(*jobd == 'N' || *jobd == 'n')) {
            *info = -3;
        }
    }

    if (m < 0) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (lda < 1 || lda < m) {
        *info = -7;
    } else if (ldb < 1 || ldb < n) {
        *info = -9;
    } else if (ldc < 1 || ldc < m) {
        *info = -11;
    } else if (ldd < 1 || ldd < m) {
        *info = -13;
    } else if (lde < 1 || lde < n) {
        *info = -15;
    } else if (ldf < 1 || ldf < m) {
        *info = -17;
    } else if (ldp < 1 || (lredra && ldp < m)) {
        *info = -21;
    } else if (ldq < 1 || (lredra && ldq < m)) {
        *info = -23;
    } else if (ldu < 1 || (lredrb && ldu < n)) {
        *info = -25;
    } else if (ldv < 1 || (lredrb && ldv < n)) {
        *info = -27;
    }

    if (*info == 0) {
        if (lquery) {
            i32 wrkopt = (m * n > minwrk) ? m * n : minwrk;
            if (lreduc) {
                if (!lredub) {
                    i32 sdim;
                    i32 bwork_dummy;
                    i32 lwork_query = -1;
                    SLC_DGGES("V", "V", "N", dummy_select, &m, a, &lda, d, &ldd,
                              &sdim, dwork, dwork, dwork, p, &ldp, q, &ldq,
                              dwork, &lwork_query, &bwork_dummy, info);
                    i32 opt_dgges = (i32)dwork[0] + 3 * m;
                    if (wrkopt < opt_dgges) wrkopt = opt_dgges;
                }
                if (!lredua) {
                    i32 sdim;
                    i32 bwork_dummy;
                    i32 lwork_query = -1;
                    SLC_DGGES("V", "V", "N", dummy_select, &n, b, &ldb, e, &lde,
                              &sdim, dwork, dwork, dwork, u, &ldu, v, &ldv,
                              dwork, &lwork_query, &bwork_dummy, info);
                    i32 opt_dgges = (i32)dwork[0] + 3 * n;
                    if (wrkopt < opt_dgges) wrkopt = opt_dgges;
                }
            }
            dwork[0] = (f64)wrkopt;
            return;
        }
        if (ldwork < minwrk) {
            *info = -30;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || m == 0) {
        *scale = one;
        dwork[0] = one;
        if (ltrann && ljobdf) {
            *dif = one;
        }
        return;
    }

    i32 wrkopt = minwrk;
    bool sufwrk = (ldwork >= m * n);

    i32 bwork_dummy;

    if (lreduc) {
        if (!lredub) {
            i32 sdim;
            i32 lwork_dgges = ldwork - 3 * m;
            SLC_DGGES("V", "V", "N", dummy_select, &m, a, &lda, d, &ldd,
                      &sdim, dwork, &dwork[m], &dwork[2*m], p, &ldp, q, &ldq,
                      &dwork[3*m], &lwork_dgges, &bwork_dummy, info);
            if (*info != 0) {
                *info = 1;
                return;
            }
            i32 opt = (i32)dwork[3*m] + 3 * m;
            if (wrkopt < opt) wrkopt = opt;
        }
        if (!lredua) {
            i32 sdim;
            i32 lwork_dgges = ldwork - 3 * n;
            SLC_DGGES("V", "V", "N", dummy_select, &n, b, &ldb, e, &lde,
                      &sdim, dwork, &dwork[n], &dwork[2*n], u, &ldu, v, &ldv,
                      &dwork[3*n], &lwork_dgges, &bwork_dummy, info);
            if (*info != 0) {
                *info = 1;
                return;
            }
            i32 opt = (i32)dwork[3*n] + 3 * n;
            if (wrkopt < opt) wrkopt = opt;
        }
    }

    if (!lredur) {
        if (!lredua) {
            i32 i = 0;
            while (i <= m - 3) {
                if (a[(i + 1) + i * lda] != zero) {
                    if (a[(i + 2) + (i + 1) * lda] != zero) {
                        *info = 2;
                        return;
                    } else {
                        i++;
                    }
                }
                i++;
            }
        }
        if (!lredub) {
            i32 i = 0;
            while (i <= n - 3) {
                if (b[(i + 1) + i * ldb] != zero) {
                    if (b[(i + 2) + (i + 1) * ldb] != zero) {
                        *info = 2;
                        return;
                    } else {
                        i++;
                    }
                }
                i++;
            }
        }
    }

    if (lreduc) {
        if (wrkopt < m * n) wrkopt = m * n;
        i32 nbc = 0, nbr = 0;

        if (sufwrk) {
            if (ltrann) {
                if (!lredub) {
                    SLC_DGEMM("T", "N", &m, &n, &m, &one, p, &ldp, c, &ldc, &zero, dwork, &m);
                } else {
                    SLC_DLACPY("A", &m, &n, c, &ldc, dwork, &m);
                }
                if (!lredua) {
                    SLC_DGEMM("N", "N", &m, &n, &n, &one, dwork, &m, v, &ldv, &zero, c, &ldc);
                } else {
                    SLC_DLACPY("A", &m, &n, dwork, &m, c, &ldc);
                }
                if (!lredub) {
                    SLC_DGEMM("T", "N", &m, &n, &m, &one, p, &ldp, f, &ldf, &zero, dwork, &m);
                } else {
                    SLC_DLACPY("A", &m, &n, f, &ldf, dwork, &m);
                }
                if (!lredua) {
                    SLC_DGEMM("N", "N", &m, &n, &n, &one, dwork, &m, v, &ldv, &zero, f, &ldf);
                } else {
                    SLC_DLACPY("A", &m, &n, dwork, &m, f, &ldf);
                }
            } else {
                if (!lredub) {
                    SLC_DGEMM("T", "N", &m, &n, &m, &one, q, &ldq, c, &ldc, &zero, dwork, &m);
                } else {
                    SLC_DLACPY("A", &m, &n, c, &ldc, dwork, &m);
                }
                if (!lredua) {
                    SLC_DGEMM("N", "N", &m, &n, &n, &one, dwork, &m, v, &ldv, &zero, c, &ldc);
                } else {
                    SLC_DLACPY("A", &m, &n, dwork, &m, c, &ldc);
                }
                if (!lredub) {
                    SLC_DGEMM("T", "N", &m, &n, &m, &one, p, &ldp, f, &ldf, &zero, dwork, &m);
                } else {
                    SLC_DLACPY("A", &m, &n, f, &ldf, dwork, &m);
                }
                if (!lredua) {
                    SLC_DGEMM("N", "N", &m, &n, &n, &one, dwork, &m, u, &ldu, &zero, f, &ldf);
                } else {
                    SLC_DLACPY("A", &m, &n, dwork, &m, f, &ldf);
                }
            }
        } else {
            if (ltrann) {
                if (!lredub) {
                    nbc = ldwork / m;
                    for (i32 i = 0; i < n; i += nbc) {
                        i32 nc = (nbc < n - i) ? nbc : (n - i);
                        SLC_DGEMM("T", "N", &m, &nc, &m, &one, p, &ldp, &c[i * ldc], &ldc, &zero, dwork, &m);
                        SLC_DLACPY("A", &m, &nc, dwork, &m, &c[i * ldc], &ldc);
                    }
                }
                if (!lredua) {
                    nbr = ldwork / n;
                    for (i32 i = 0; i < m; i += nbr) {
                        i32 nr = (nbr < m - i) ? nbr : (m - i);
                        SLC_DGEMM("N", "N", &nr, &n, &n, &one, &c[i], &ldc, v, &ldv, &zero, dwork, &nr);
                        SLC_DLACPY("A", &nr, &n, dwork, &nr, &c[i], &ldc);
                    }
                }
                if (!lredub) {
                    for (i32 i = 0; i < n; i += nbc) {
                        i32 nc = (nbc < n - i) ? nbc : (n - i);
                        SLC_DGEMM("T", "N", &m, &nc, &m, &one, p, &ldp, &f[i * ldf], &ldf, &zero, dwork, &m);
                        SLC_DLACPY("A", &m, &nc, dwork, &m, &f[i * ldf], &ldf);
                    }
                }
                if (!lredua) {
                    for (i32 i = 0; i < m; i += nbr) {
                        i32 nr = (nbr < m - i) ? nbr : (m - i);
                        SLC_DGEMM("N", "N", &nr, &n, &n, &one, &f[i], &ldf, v, &ldv, &zero, dwork, &nr);
                        SLC_DLACPY("A", &nr, &n, dwork, &nr, &f[i], &ldf);
                    }
                }
            } else {
                if (!lredub) {
                    nbc = ldwork / m;
                    for (i32 i = 0; i < n; i += nbc) {
                        i32 nc = (nbc < n - i) ? nbc : (n - i);
                        SLC_DGEMM("T", "N", &m, &nc, &m, &one, q, &ldq, &c[i * ldc], &ldc, &zero, dwork, &m);
                        SLC_DLACPY("A", &m, &nc, dwork, &m, &c[i * ldc], &ldc);
                    }
                }
                if (!lredua) {
                    nbr = ldwork / n;
                    for (i32 i = 0; i < m; i += nbr) {
                        i32 nr = (nbr < m - i) ? nbr : (m - i);
                        SLC_DGEMM("N", "N", &nr, &n, &n, &one, &c[i], &ldc, v, &ldv, &zero, dwork, &nr);
                        SLC_DLACPY("A", &nr, &n, dwork, &nr, &c[i], &ldc);
                    }
                }
                if (!lredub) {
                    for (i32 i = 0; i < n; i += nbc) {
                        i32 nc = (nbc < n - i) ? nbc : (n - i);
                        SLC_DGEMM("T", "N", &m, &nc, &m, &one, p, &ldp, &f[i * ldf], &ldf, &zero, dwork, &m);
                        SLC_DLACPY("A", &m, &nc, dwork, &m, &f[i * ldf], &ldf);
                    }
                }
                if (!lredua) {
                    for (i32 i = 0; i < m; i += nbr) {
                        i32 nr = (nbr < m - i) ? nbr : (m - i);
                        SLC_DGEMM("N", "N", &nr, &n, &n, &one, &f[i], &ldf, u, &ldu, &zero, dwork, &nr);
                        SLC_DLACPY("A", &nr, &n, dwork, &nr, &f[i], &ldf);
                    }
                }
            }
        }
    }

    i32 ijob;
    if (ltrann) {
        if (ljobd) {
            ijob = 1;
        } else if (ljobf) {
            ijob = 2;
        } else if (ljob1) {
            ijob = 3;
        } else if (ljob2) {
            ijob = 4;
        } else {
            ijob = 0;
        }
    } else {
        ijob = 0;
    }

    const char* trans_str = ltrann ? "N" : "T";
    SLC_DTGSYL(trans_str, &ijob, &m, &n, a, &lda, b, &ldb, c, &ldc,
               d, &ldd, e, &lde, f, &ldf, scale, dif, dwork, &ldwork,
               iwork, info);
    if (*info != 0) {
        *info = 3;
        return;
    }

    i32 nbc = 0, nbr = 0;

    if (lreduc) {
        if (sufwrk) {
            if (ltrann) {
                if (!lredub) {
                    SLC_DGEMM("N", "N", &m, &n, &m, &one, q, &ldq, c, &ldc, &zero, dwork, &m);
                } else {
                    SLC_DLACPY("A", &m, &n, c, &ldc, dwork, &m);
                }
                if (!lredua) {
                    SLC_DGEMM("N", "T", &m, &n, &n, &one, dwork, &m, v, &ldv, &zero, c, &ldc);
                } else {
                    SLC_DLACPY("A", &m, &n, dwork, &m, c, &ldc);
                }
                if (!lredub) {
                    SLC_DGEMM("N", "N", &m, &n, &m, &one, p, &ldp, f, &ldf, &zero, dwork, &m);
                } else {
                    SLC_DLACPY("A", &m, &n, f, &ldf, dwork, &m);
                }
                if (!lredua) {
                    SLC_DGEMM("N", "T", &m, &n, &n, &one, dwork, &m, u, &ldu, &zero, f, &ldf);
                } else {
                    SLC_DLACPY("A", &m, &n, dwork, &m, f, &ldf);
                }
            } else {
                if (!lredub) {
                    SLC_DGEMM("N", "N", &m, &n, &m, &one, p, &ldp, c, &ldc, &zero, dwork, &m);
                } else {
                    SLC_DLACPY("A", &m, &n, c, &ldc, dwork, &m);
                }
                if (!lredua) {
                    SLC_DGEMM("N", "T", &m, &n, &n, &one, dwork, &m, v, &ldv, &zero, c, &ldc);
                } else {
                    SLC_DLACPY("A", &m, &n, dwork, &m, c, &ldc);
                }
                if (!lredub) {
                    SLC_DGEMM("N", "N", &m, &n, &m, &one, p, &ldp, f, &ldf, &zero, dwork, &m);
                } else {
                    SLC_DLACPY("A", &m, &n, f, &ldf, dwork, &m);
                }
                if (!lredua) {
                    SLC_DGEMM("N", "T", &m, &n, &n, &one, dwork, &m, v, &ldv, &zero, f, &ldf);
                } else {
                    SLC_DLACPY("A", &m, &n, dwork, &m, f, &ldf);
                }
            }
        } else {
            if (ltrann) {
                if (!lredub) {
                    nbc = ldwork / m;
                    for (i32 i = 0; i < n; i += nbc) {
                        i32 nc = (nbc < n - i) ? nbc : (n - i);
                        SLC_DGEMM("N", "N", &m, &nc, &m, &one, q, &ldq, &c[i * ldc], &ldc, &zero, dwork, &m);
                        SLC_DLACPY("A", &m, &nc, dwork, &m, &c[i * ldc], &ldc);
                    }
                }
                if (!lredua) {
                    nbr = ldwork / n;
                    for (i32 i = 0; i < m; i += nbr) {
                        i32 nr = (nbr < m - i) ? nbr : (m - i);
                        SLC_DGEMM("N", "T", &nr, &n, &n, &one, &c[i], &ldc, v, &ldv, &zero, dwork, &nr);
                        SLC_DLACPY("A", &nr, &n, dwork, &nr, &c[i], &ldc);
                    }
                }
                if (!lredub) {
                    for (i32 i = 0; i < n; i += nbc) {
                        i32 nc = (nbc < n - i) ? nbc : (n - i);
                        SLC_DGEMM("N", "N", &m, &nc, &m, &one, p, &ldp, &f[i * ldf], &ldf, &zero, dwork, &m);
                        SLC_DLACPY("A", &m, &nc, dwork, &m, &f[i * ldf], &ldf);
                    }
                }
                if (!lredua) {
                    for (i32 i = 0; i < m; i += nbr) {
                        i32 nr = (nbr < m - i) ? nbr : (m - i);
                        SLC_DGEMM("N", "T", &nr, &n, &n, &one, &f[i], &ldf, u, &ldu, &zero, dwork, &nr);
                        SLC_DLACPY("A", &nr, &n, dwork, &nr, &f[i], &ldf);
                    }
                }
            } else {
                if (!lredub) {
                    nbc = ldwork / m;
                    for (i32 i = 0; i < n; i += nbc) {
                        i32 nc = (nbc < n - i) ? nbc : (n - i);
                        SLC_DGEMM("N", "N", &m, &nc, &m, &one, p, &ldp, &c[i * ldc], &ldc, &zero, dwork, &m);
                        SLC_DLACPY("A", &m, &nc, dwork, &m, &c[i * ldc], &ldc);
                    }
                }
                if (!lredua) {
                    nbr = ldwork / n;
                    for (i32 i = 0; i < m; i += nbr) {
                        i32 nr = (nbr < m - i) ? nbr : (m - i);
                        SLC_DGEMM("N", "T", &nr, &n, &n, &one, &c[i], &ldc, v, &ldv, &zero, dwork, &nr);
                        SLC_DLACPY("A", &nr, &n, dwork, &nr, &c[i], &ldc);
                    }
                }
                if (!lredub) {
                    for (i32 i = 0; i < n; i += nbc) {
                        i32 nc = (nbc < n - i) ? nbc : (n - i);
                        SLC_DGEMM("N", "N", &m, &nc, &m, &one, p, &ldp, &f[i * ldf], &ldf, &zero, dwork, &m);
                        SLC_DLACPY("A", &m, &nc, dwork, &m, &f[i * ldf], &ldf);
                    }
                }
                if (!lredua) {
                    for (i32 i = 0; i < m; i += nbr) {
                        i32 nr = (nbr < m - i) ? nbr : (m - i);
                        SLC_DGEMM("N", "T", &nr, &n, &n, &one, &f[i], &ldf, v, &ldv, &zero, dwork, &nr);
                        SLC_DLACPY("A", &nr, &n, dwork, &nr, &f[i], &ldf);
                    }
                }
            }
        }
    }

    dwork[0] = (f64)wrkopt;
}
