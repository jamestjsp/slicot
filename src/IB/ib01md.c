#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void ib01md(const char *meth_str, const char *alg_str, const char *batch_str,
            const char *conct_str, i32 nobr, i32 m, i32 l, i32 nsmp,
            const f64 *u, i32 ldu, const f64 *y, i32 ldy,
            f64 *r, i32 ldr, i32 *iwork, f64 *dwork, i32 ldwork,
            i32 *iwarn, i32 *info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const i32 MAXCYC = 100;

    char meth = meth_str[0];
    char alg = alg_str[0];
    char batch = batch_str[0];
    char conct = conct_str[0];

    bool moesp = (meth == 'M' || meth == 'm');
    bool n4sid = (meth == 'N' || meth == 'n');
    bool fqralg = (alg == 'F' || alg == 'f');
    bool qralg = (alg == 'Q' || alg == 'q');
    bool chalg = (alg == 'C' || alg == 'c');
    bool onebch = (batch == 'O' || batch == 'o');
    bool first = (batch == 'F' || batch == 'f') || onebch;
    bool interm = (batch == 'I' || batch == 'i');
    bool last = (batch == 'L' || batch == 'l') || onebch;
    bool connec = false;
    if (!onebch) {
        connec = (conct == 'C' || conct == 'c');
    }

    i32 mnobr = m * nobr;
    i32 lnobr = l * nobr;
    i32 lmnobr = lnobr + mnobr;
    i32 mmnobr = mnobr + mnobr;
    i32 nobrm1 = nobr - 1;
    i32 nobr21 = nobr + nobrm1;
    i32 nobr2 = nobr21 + 1;
    i32 nr = lmnobr + lmnobr;

    *iwarn = 0;
    *info = 0;
    i32 ierr = 0;

    i32 icycle, maxwrk, nsmpsm;
    if (first) {
        icycle = 1;
        maxwrk = 1;
        nsmpsm = 0;
    } else if (!onebch) {
        icycle = iwork[0];
        maxwrk = iwork[1];
        nsmpsm = iwork[2];
    } else {
        icycle = 1;
        maxwrk = 1;
        nsmpsm = 0;
    }
    nsmpsm = nsmpsm + nsmp;

    if (!(moesp || n4sid)) {
        *info = -1;
    } else if (!(fqralg || qralg || chalg)) {
        *info = -2;
    } else if (!(first || interm || last)) {
        *info = -3;
    } else if (!onebch) {
        bool connec_n = (conct == 'N' || conct == 'n');
        if (!(connec || connec_n)) {
            *info = -4;
        }
    }

    if (*info == 0) {
        if (nobr <= 0) {
            *info = -5;
        } else if (m < 0) {
            *info = -6;
        } else if (l <= 0) {
            *info = -7;
        } else if (nsmp < nobr2 || (last && nsmpsm < nr + nobr21)) {
            *info = -8;
        } else if (ldu < 1 || (m > 0 && ldu < nsmp)) {
            *info = -10;
        } else if (ldy < nsmp) {
            *info = -12;
        } else if (ldr < nr) {
            *info = -14;
        } else {
            bool lquery = (ldwork == -1);
            i32 ns = nsmp - nobr21;
            i32 minwrk;

            if (chalg) {
                if (!onebch && connec) {
                    minwrk = 2 * (nr - m - l);
                } else {
                    minwrk = 1;
                }
            } else if (fqralg) {
                if (!onebch && connec) {
                    minwrk = nr * (m + l + 3);
                } else if (first || interm) {
                    minwrk = nr * (m + l + 1);
                } else {
                    minwrk = 2 * nr * (m + l + 1) + nr;
                }
                if (lquery) {
                    maxwrk = minwrk;
                }
            } else {
                minwrk = 2 * nr;
                f64 work_query;
                i32 qr_info;
                SLC_DGEQRF(&ns, &nr, dwork, &ns, dwork, &work_query, &(i32){-1}, &qr_info);
                maxwrk = nr + (i32)work_query;
                if (first) {
                    if (ldr < ns) {
                        minwrk = minwrk + nr;
                        maxwrk = ns * nr + maxwrk;
                    }
                } else {
                    if (connec) {
                        minwrk = minwrk * (nobr + 1);
                    } else {
                        minwrk = minwrk + nr;
                    }
                    maxwrk = ns * nr + maxwrk;
                }
            }
            maxwrk = (minwrk > maxwrk) ? minwrk : maxwrk;

            if (ldwork < minwrk && !lquery) {
                *info = -17;
                dwork[0] = (f64)minwrk;
            }
        }
    }

    if (*info != 0) {
        if (!onebch) {
            iwork[0] = 1;
            iwork[1] = maxwrk;
            iwork[2] = 0;
        }
        if (*info == -17) {
            dwork[0] = (f64)maxwrk;
        }
        return;
    }

    bool lquery = (ldwork == -1);
    if (lquery) {
        dwork[0] = (f64)maxwrk;
        return;
    }

    i32 ns = nsmp - nobr21;

    if (chalg) {
        i32 ldrwrk = 2 * nobr2 - 2;
        f64 upd = first ? ZERO : ONE;
        i32 jd;

        if (!first && connec) {
            i32 irev = nr - m - l - nobr21;
            i32 icol = 2 * (nr - m - l) - ldrwrk;

            for (i32 j = 1; j < m + l; j++) {
                for (i32 i = nobr21 - 1; i >= 0; i--) {
                    dwork[icol + i] = dwork[irev + i];
                }
                irev -= nobr21;
                icol -= ldrwrk;
            }

            if (m > 0) {
                SLC_DLACPY("Full", &nobr21, &m, u, &ldu, &dwork[nobr2 - 1], &ldrwrk);
            }
            SLC_DLACPY("Full", &nobr21, &l, y, &ldy, &dwork[ldrwrk * m + nobr2 - 1], &ldrwrk);
        }

        if (m > 0) {
            if (!first) {
                for (i32 i = nobr21 * m - 1; i >= 0; i--) {
                    i32 col = i + 1;
                    SLC_DAXPY(&col, &(f64){-ONE}, &r[i * ldr], &(i32){1},
                              &r[m + (m + i) * ldr], &(i32){1});
                }
            }

            if (!first && connec) {
                SLC_DSYRK("Upper", "Transpose", &m, &nobr21, &ONE, dwork, &ldrwrk,
                          &upd, r, &ldr);
            }
            SLC_DSYRK("Upper", "Transpose", &m, &ns, &ONE, u, &ldu, &upd, r, &ldr);

            i32 jd = 0;

            if (first || !connec) {
                for (i32 j = 1; j < nobr2; j++) {
                    jd += m;
                    i32 id = m;

                    SLC_DGEMM("T", "N", &m, &m, &ns, &ONE,
                              u, &ldu, &u[j], &ldu, &upd, &r[jd * ldr], &ldr);

                    if (first) {
                        for (i32 i = jd - m; i < jd; i++) {
                            i32 len = i + 1;
                            SLC_DCOPY(&len, &r[i * ldr], &(i32){1},
                                      &r[m + (m + i) * ldr], &(i32){1});
                        }
                    } else {
                        for (i32 i = jd - m; i < jd; i++) {
                            i32 len = i + 1;
                            SLC_DAXPY(&len, &ONE, &r[i * ldr], &(i32){1},
                                      &r[m + (m + i) * ldr], &(i32){1});
                        }
                    }

                    for (i32 i = 1; i < j; i++) {
                        SLC_DGER(&m, &m, &ONE, &u[(ns + i - 1) * 1], &ldu,
                                 &u[(ns + j - 1) * 1], &ldu, &r[id + jd * ldr], &ldr);
                        SLC_DGER(&m, &m, &(f64){-ONE}, &u[(i - 1) * 1], &ldu,
                                 &u[(j - 1) * 1], &ldu, &r[id + jd * ldr], &ldr);
                        id += m;
                    }

                    for (i32 i = 0; i < m; i++) {
                        i32 len = i + 1;
                        f64 val1 = u[(ns + j - 1) + i * ldu];
                        f64 val2 = -u[(j - 1) + i * ldu];
                        SLC_DAXPY(&len, &val1, &u[ns + j - 1], &ldu, &r[jd + (jd + i) * ldr], &(i32){1});
                        SLC_DAXPY(&len, &val2, &u[j - 1], &ldu, &r[jd + (jd + i) * ldr], &(i32){1});
                    }
                }
            } else {
                for (i32 j = 1; j < nobr2; j++) {
                    jd += m;
                    i32 id = m;

                    SLC_DGEMM("T", "N", &m, &m, &nobr21, &ONE,
                              dwork, &ldrwrk, &dwork[j - 1], &ldrwrk, &upd, &r[jd * ldr], &ldr);
                    SLC_DGEMM("T", "N", &m, &m, &ns, &ONE,
                              u, &ldu, &u[j], &ldu, &ONE, &r[jd * ldr], &ldr);

                    if (first) {
                        for (i32 i = jd - m; i < jd; i++) {
                            i32 len = i + 1;
                            SLC_DCOPY(&len, &r[i * ldr], &(i32){1},
                                      &r[m + (m + i) * ldr], &(i32){1});
                        }
                    } else {
                        for (i32 i = jd - m; i < jd; i++) {
                            i32 len = i + 1;
                            SLC_DAXPY(&len, &ONE, &r[i * ldr], &(i32){1},
                                      &r[m + (m + i) * ldr], &(i32){1});
                        }
                    }

                    for (i32 i = 1; i < j; i++) {
                        SLC_DGER(&m, &m, &ONE, &u[(ns + i - 1) * 1], &ldu,
                                 &u[(ns + j - 1) * 1], &ldu, &r[id + jd * ldr], &ldr);
                        SLC_DGER(&m, &m, &(f64){-ONE}, &dwork[i - 1], &ldrwrk,
                                 &dwork[j - 1], &ldrwrk, &r[id + jd * ldr], &ldr);
                        id += m;
                    }

                    for (i32 i = 0; i < m; i++) {
                        i32 len = i + 1;
                        f64 val1 = u[(ns + j - 1) + i * ldu];
                        f64 val2 = -dwork[(i) * ldrwrk + j - 1];
                        SLC_DAXPY(&len, &val1, &u[ns + j - 1], &ldu, &r[jd + (jd + i) * ldr], &(i32){1});
                        SLC_DAXPY(&len, &val2, &dwork[j - 1], &ldrwrk, &r[jd + (jd + i) * ldr], &(i32){1});
                    }
                }
            }

            if (last && moesp) {
                f64 temp = r[0];
                r[0] = r[mnobr + mnobr * ldr];
                r[mnobr + mnobr * ldr] = temp;

                for (i32 j = 1; j < mnobr; j++) {
                    i32 len = j + 1;
                    SLC_DSWAP(&len, &r[j * ldr], &(i32){1}, &r[(mnobr + j) * ldr + mnobr], &(i32){1});
                    i32 len2 = j;
                    SLC_DSWAP(&len2, &r[(mnobr + j) * ldr], &(i32){1}, &r[j + mnobr * ldr], &ldr);
                }
            }

            i32 ii = mmnobr - m;
            if (!first) {
                for (i32 i = nr - l - 1; i >= mmnobr; i--) {
                    SLC_DAXPY(&ii, &(f64){-ONE}, &r[i * ldr], &(i32){1},
                              &r[m + (l + i) * ldr], &(i32){1});
                }
            }

            if (first || !connec) {
                for (i32 i = 0; i < nobr2; i++) {
                    SLC_DGEMM("T", "N", &m, &l, &ns, &ONE,
                              &u[i], &ldu, y, &ldy, &upd,
                              &r[(i) * m + (mmnobr) * ldr], &ldr);
                }
            } else {
                for (i32 i = 0; i < nobr2; i++) {
                    SLC_DGEMM("T", "N", &m, &l, &nobr21, &ONE,
                              &dwork[i], &ldrwrk, &dwork[ldrwrk * m], &ldrwrk, &upd,
                              &r[(i) * m + (mmnobr) * ldr], &ldr);
                    SLC_DGEMM("T", "N", &m, &l, &ns, &ONE,
                              &u[i], &ldu, y, &ldy, &ONE,
                              &r[(i) * m + (mmnobr) * ldr], &ldr);
                }
            }

            jd = mmnobr;

            if (first || !connec) {
                for (i32 j = 1; j < nobr2; j++) {
                    jd += l;
                    i32 id = m;

                    SLC_DGEMM("T", "N", &m, &l, &ns, &ONE,
                              u, &ldu, &y[j], &ldy, &upd, &r[jd * ldr], &ldr);

                    if (first) {
                        for (i32 i = jd - l; i < jd; i++) {
                            SLC_DCOPY(&ii, &r[i * ldr], &(i32){1},
                                      &r[m + (l + i) * ldr], &(i32){1});
                        }
                    } else {
                        for (i32 i = jd - l; i < jd; i++) {
                            SLC_DAXPY(&ii, &ONE, &r[i * ldr], &(i32){1},
                                      &r[m + (l + i) * ldr], &(i32){1});
                        }
                    }

                    for (i32 i = 1; i < nobr2; i++) {
                        SLC_DGER(&m, &l, &ONE, &u[(ns + i - 1) * 1], &ldu,
                                 &y[(ns + j - 1) * 1], &ldy, &r[id + jd * ldr], &ldr);
                        SLC_DGER(&m, &l, &(f64){-ONE}, &u[(i - 1) * 1], &ldu,
                                 &y[(j - 1) * 1], &ldy, &r[id + jd * ldr], &ldr);
                        id += m;
                    }
                }
            } else {
                for (i32 j = 1; j < nobr2; j++) {
                    jd += l;
                    i32 id = m;

                    SLC_DGEMM("T", "N", &m, &l, &nobr21, &ONE,
                              dwork, &ldrwrk, &dwork[ldrwrk * m + j - 1], &ldrwrk, &upd, &r[jd * ldr], &ldr);
                    SLC_DGEMM("T", "N", &m, &l, &ns, &ONE,
                              u, &ldu, &y[j], &ldy, &ONE, &r[jd * ldr], &ldr);

                    if (first) {
                        for (i32 i = jd - l; i < jd; i++) {
                            SLC_DCOPY(&ii, &r[i * ldr], &(i32){1},
                                      &r[m + (l + i) * ldr], &(i32){1});
                        }
                    } else {
                        for (i32 i = jd - l; i < jd; i++) {
                            SLC_DAXPY(&ii, &ONE, &r[i * ldr], &(i32){1},
                                      &r[m + (l + i) * ldr], &(i32){1});
                        }
                    }

                    for (i32 i = 1; i < nobr2; i++) {
                        SLC_DGER(&m, &l, &ONE, &u[(ns + i - 1) * 1], &ldu,
                                 &y[(ns + j - 1) * 1], &ldy, &r[id + jd * ldr], &ldr);
                        SLC_DGER(&m, &l, &(f64){-ONE}, &dwork[i - 1], &ldrwrk,
                                 &dwork[ldrwrk * m + j - 1], &ldrwrk, &r[id + jd * ldr], &ldr);
                        id += m;
                    }
                }
            }

            if (last && moesp) {
                for (i32 j = mmnobr; j < nr; j++) {
                    SLC_DSWAP(&mnobr, &r[j * ldr], &(i32){1}, &r[mnobr + j * ldr], &(i32){1});
                }
            }
        }

        jd = mmnobr;

        if (!first) {
            for (i32 i = nr - l - 1; i >= mmnobr; i--) {
                i32 len = i - mmnobr + 1;
                SLC_DAXPY(&len, &(f64){-ONE}, &r[jd + i * ldr], &(i32){1},
                          &r[jd + l + (l + i) * ldr], &(i32){1});
            }
        }

        if (!first && connec) {
            SLC_DSYRK("Upper", "Transpose", &l, &nobr21, &ONE,
                      &dwork[ldrwrk * m], &ldrwrk, &upd, &r[jd + jd * ldr], &ldr);
        }
        SLC_DSYRK("Upper", "Transpose", &l, &ns, &ONE, y, &ldy, &upd, &r[jd + jd * ldr], &ldr);

        if (first || !connec) {
            for (i32 j = 1; j < nobr2; j++) {
                jd += l;
                i32 id = mmnobr + l;

                SLC_DGEMM("T", "N", &l, &l, &ns, &ONE, y, &ldy, &y[j], &ldy, &upd,
                          &r[mmnobr + jd * ldr], &ldr);

                if (first) {
                    for (i32 i = jd - l; i < jd; i++) {
                        i32 len = i - mmnobr + 1;
                        SLC_DCOPY(&len, &r[mmnobr + i * ldr], &(i32){1},
                                  &r[mmnobr + l + (l + i) * ldr], &(i32){1});
                    }
                } else {
                    for (i32 i = jd - l; i < jd; i++) {
                        i32 len = i - mmnobr + 1;
                        SLC_DAXPY(&len, &ONE, &r[mmnobr + i * ldr], &(i32){1},
                                  &r[mmnobr + l + (l + i) * ldr], &(i32){1});
                    }
                }

                for (i32 i = 1; i < j; i++) {
                    SLC_DGER(&l, &l, &ONE, &y[(ns + i - 1) * 1], &ldy,
                             &y[(ns + j - 1) * 1], &ldy, &r[id + jd * ldr], &ldr);
                    SLC_DGER(&l, &l, &(f64){-ONE}, &y[(i - 1) * 1], &ldy,
                             &y[(j - 1) * 1], &ldy, &r[id + jd * ldr], &ldr);
                    id += l;
                }

                for (i32 i = 0; i < l; i++) {
                    i32 len = i + 1;
                    f64 val1 = y[(ns + j - 1) + i * ldy];
                    f64 val2 = -y[(j - 1) + i * ldy];
                    SLC_DAXPY(&len, &val1, &y[ns + j - 1], &ldy, &r[jd + (jd + i) * ldr], &(i32){1});
                    SLC_DAXPY(&len, &val2, &y[j - 1], &ldy, &r[jd + (jd + i) * ldr], &(i32){1});
                }
            }
        } else {
            for (i32 j = 1; j < nobr2; j++) {
                jd += l;
                i32 id = mmnobr + l;

                SLC_DGEMM("T", "N", &l, &l, &nobr21, &ONE,
                          &dwork[ldrwrk * m], &ldrwrk, &dwork[ldrwrk * m + j - 1], &ldrwrk, &upd,
                          &r[mmnobr + jd * ldr], &ldr);
                SLC_DGEMM("T", "N", &l, &l, &ns, &ONE, y, &ldy, &y[j], &ldy, &ONE,
                          &r[mmnobr + jd * ldr], &ldr);

                if (first) {
                    for (i32 i = jd - l; i < jd; i++) {
                        i32 len = i - mmnobr + 1;
                        SLC_DCOPY(&len, &r[mmnobr + i * ldr], &(i32){1},
                                  &r[mmnobr + l + (l + i) * ldr], &(i32){1});
                    }
                } else {
                    for (i32 i = jd - l; i < jd; i++) {
                        i32 len = i - mmnobr + 1;
                        SLC_DAXPY(&len, &ONE, &r[mmnobr + i * ldr], &(i32){1},
                                  &r[mmnobr + l + (l + i) * ldr], &(i32){1});
                    }
                }

                for (i32 i = 1; i < j; i++) {
                    SLC_DGER(&l, &l, &ONE, &y[(ns + i - 1) * 1], &ldy,
                             &y[(ns + j - 1) * 1], &ldy, &r[id + jd * ldr], &ldr);
                    SLC_DGER(&l, &l, &(f64){-ONE}, &dwork[ldrwrk * m + i - 1], &ldrwrk,
                             &dwork[ldrwrk * m + j - 1], &ldrwrk, &r[id + jd * ldr], &ldr);
                    id += l;
                }

                for (i32 i = 0; i < l; i++) {
                    i32 len = i + 1;
                    f64 val1 = y[(ns + j - 1) + i * ldy];
                    f64 val2 = -dwork[ldrwrk * (m + i) + j - 1];
                    SLC_DAXPY(&len, &val1, &y[ns + j - 1], &ldy, &r[jd + (jd + i) * ldr], &(i32){1});
                    SLC_DAXPY(&len, &val2, &dwork[ldrwrk * m + j - 1], &ldrwrk, &r[jd + (jd + i) * ldr], &(i32){1});
                }
            }
        }

        if (!last) {
            if (connec) {
                if (m > 0) {
                    SLC_DLACPY("Full", &nobr21, &m, &u[ns], &ldu, dwork, &nobr21);
                }
                SLC_DLACPY("Full", &nobr21, &l, &y[ns], &ldy, &dwork[mmnobr - m], &nobr21);
            }

            icycle++;
            iwork[0] = icycle;
            iwork[1] = maxwrk;
            iwork[2] = nsmpsm;
            if (icycle > MAXCYC) {
                *iwarn = 1;
            }
            return;
        } else {
            SLC_DPOTRF("Upper", &nr, r, &ldr, &ierr);
            goto label_370;
        }
    } else if (fqralg) {
        ib01my(meth_str, batch_str, conct_str, nobr, m, l, nsmp,
               u, ldu, y, ldy, r, ldr, iwork, dwork, ldwork, iwarn, info);
        return;
    }

label_370:
    if (ierr != 0) {
        if (onebch) {
            qralg = true;
            *iwarn = 2;
            i32 minwrk = 2 * nr;
            f64 work_query;
            i32 qr_info;
            SLC_DGEQRF(&ns, &nr, dwork, &ns, dwork, &work_query, &(i32){-1}, &qr_info);
            maxwrk = nr + (i32)work_query;
            if (ldr < ns) {
                minwrk = minwrk + nr;
                maxwrk = ns * nr + maxwrk;
            }
            maxwrk = (minwrk > maxwrk) ? minwrk : maxwrk;

            if (ldwork < minwrk) {
                *info = -17;
                dwork[0] = (f64)minwrk;
                return;
            }
        } else {
            *info = 1;
            return;
        }
    }

    if (qralg) {
        i32 initi = 0;
        i32 ldrwmx = ldwork / nr - 2;
        i32 ncycle = 1;
        i32 nslast = nsmp;
        bool linr = false;
        i32 ldrwrk;

        if (first) {
            linr = (ldr >= ns);
            ldrwrk = ns;
        } else if (connec) {
            ldrwrk = nsmp;
        } else {
            ldrwrk = ns;
        }

        i32 inicyc = 1;
        i32 mldrw, lldrw, inu, iny;

        if (!linr) {
            if (ldrwmx < ldrwrk) {
                ncycle = ldrwrk / ldrwmx;
                nslast = ldrwrk % ldrwmx;
                if (nslast != 0) {
                    ncycle++;
                } else {
                    nslast = ldrwmx;
                }
                ldrwrk = ldrwmx;
                ns = ldrwrk;
                if (first) inicyc = 2;
            }
            mldrw = m * ldrwrk;
            lldrw = l * ldrwrk;
            inu = mldrw * nobr;
            iny = mldrw * nobr2;
        }

        if (!first && connec) {
            i32 irev = nr - m - l - nobr21;
            i32 icol = iny + lldrw - ldrwrk;

            for (i32 j = 0; j < l; j++) {
                for (i32 i = nobr21 - 1; i >= 0; i--) {
                    dwork[icol + i] = dwork[irev + i];
                }
                irev -= nobr21;
                icol -= ldrwrk;
            }

            if (moesp) {
                icol = inu + mldrw - ldrwrk;
            } else {
                icol = mldrw - ldrwrk;
            }

            for (i32 j = 0; j < m; j++) {
                for (i32 i = nobr21 - 1; i >= 0; i--) {
                    dwork[icol + i] = dwork[irev + i];
                }
                irev -= nobr21;
                icol -= ldrwrk;
            }

            if (moesp) {
                SLC_DLACPY("Full", &nobrm1, &m, &dwork[inu + nobr], &ldrwrk, dwork, &ldrwrk);
            }
        }

        if (first) {
            if (linr) {
                if (m > 0) {
                    if (moesp) {
                        for (i32 i = 0; i < nobr; i++) {
                            SLC_DLACPY("Full", &ns, &m, &u[nobr + i], &ldu,
                                       &r[m * i * ldr], &ldr);
                        }
                        for (i32 i = 0; i < nobr; i++) {
                            SLC_DLACPY("Full", &ns, &m, &u[i], &ldu,
                                       &r[(mnobr + m * i) * ldr], &ldr);
                        }
                    } else {
                        for (i32 i = 0; i < nobr2; i++) {
                            SLC_DLACPY("Full", &ns, &m, &u[i], &ldu,
                                       &r[m * i * ldr], &ldr);
                        }
                    }
                }

                for (i32 i = 0; i < nobr2; i++) {
                    SLC_DLACPY("Full", &ns, &l, &y[i], &ldy,
                               &r[(mmnobr + l * i) * ldr], &ldr);
                }

                i32 itau = 0;
                i32 jwork = itau + nr;
                i32 work_size = ldwork - jwork;
                SLC_DGEQRF(&ns, &nr, r, &ldr, &dwork[itau], &dwork[jwork], &work_size, &ierr);
            } else {
                if (m > 0) {
                    i32 ishftu = 0;
                    if (moesp) {
                        i32 ishft2 = inu;
                        for (i32 i = 0; i < nobr; i++) {
                            SLC_DLACPY("Full", &ns, &m, &u[nobr + i], &ldu,
                                       &dwork[ishftu], &ldrwrk);
                            ishftu += mldrw;
                        }
                        for (i32 i = 0; i < nobr; i++) {
                            SLC_DLACPY("Full", &ns, &m, &u[i], &ldu,
                                       &dwork[ishft2], &ldrwrk);
                            ishft2 += mldrw;
                        }
                    } else {
                        for (i32 i = 0; i < nobr2; i++) {
                            SLC_DLACPY("Full", &ns, &m, &u[i], &ldu,
                                       &dwork[ishftu], &ldrwrk);
                            ishftu += mldrw;
                        }
                    }
                }

                i32 ishfty = iny;
                for (i32 i = 0; i < nobr2; i++) {
                    SLC_DLACPY("Full", &ns, &l, &y[i], &ldy,
                               &dwork[ishfty], &ldrwrk);
                    ishfty += lldrw;
                }

                i32 itau = ldrwrk * nr;
                i32 jwork = itau + nr;
                i32 work_size = ldwork - jwork;
                SLC_DGEQRF(&ns, &nr, dwork, &ldrwrk, &dwork[itau], &dwork[jwork], &work_size, &ierr);

                i32 minns = (ns < nr) ? ns : nr;
                SLC_DLACPY("Upper", &minns, &nr, dwork, &ldrwrk, r, &ldr);
            }

            if (ns < nr) {
                i32 diff = nr - ns;
                SLC_DLASET("Upper", &diff, &diff, &ZERO, &ZERO, &r[ns + ns * ldr], &ldr);
            }
            initi = initi + ns;
        }

        if (ncycle > 1 || !first) {
            i32 nsl = ldrwrk;
            if (!connec) nsl = ns;
            i32 itau = ldrwrk * nr;
            i32 jwork = itau + nr;

            for (i32 nicycl = inicyc; nicycl <= ncycle; nicycl++) {
                i32 init;
                if (connec && nicycl == 1) {
                    init = nobr2 - 1;
                } else {
                    init = 0;
                }

                if (ncycle > 1 && nicycl == ncycle) {
                    ns = nslast;
                    nsl = nslast;
                }

                i32 nsf = ns;
                if (init > 0 && ncycle > 1) nsf = nsf - nobr21;

                if (m > 0) {
                    i32 ishftu = init;
                    if (moesp) {
                        i32 ishft2 = init + inu;
                        for (i32 i = 0; i < nobr; i++) {
                            SLC_DLACPY("Full", &nsf, &m, &u[initi + nobr + i], &ldu,
                                       &dwork[ishftu], &ldrwrk);
                            ishftu += mldrw;
                        }
                        for (i32 i = 0; i < nobr; i++) {
                            SLC_DLACPY("Full", &nsf, &m, &u[initi + i], &ldu,
                                       &dwork[ishft2], &ldrwrk);
                            ishft2 += mldrw;
                        }
                    } else {
                        for (i32 i = 0; i < nobr2; i++) {
                            SLC_DLACPY("Full", &nsf, &m, &u[initi + i], &ldu,
                                       &dwork[ishftu], &ldrwrk);
                            ishftu += mldrw;
                        }
                    }
                }

                i32 ishfty = init + iny;
                for (i32 i = 0; i < nobr2; i++) {
                    SLC_DLACPY("Full", &nsf, &l, &y[initi + i], &ldy,
                               &dwork[ishfty], &ldrwrk);
                    ishfty += lldrw;
                }

                if (init > 0) {
                    if (moesp && m > 0) {
                        SLC_DLACPY("Full", &nobr, &m, u, &ldu, &dwork[nobr - 1], &ldrwrk);
                    }

                    if (m > 0) {
                        i32 ishftu_inner = mldrw;
                        if (moesp) {
                            i32 ishft2_inner = mldrw + inu;
                            for (i32 i = 0; i < nobrm1; i++) {
                                SLC_DLACPY("Full", &nobr21, &m, &dwork[ishftu_inner - mldrw], &ldrwrk,
                                           &dwork[ishftu_inner], &ldrwrk);
                                ishftu_inner += mldrw;
                            }
                            for (i32 i = 0; i < nobrm1; i++) {
                                SLC_DLACPY("Full", &nobr21, &m, &dwork[ishft2_inner - mldrw], &ldrwrk,
                                           &dwork[ishft2_inner], &ldrwrk);
                                ishft2_inner += mldrw;
                            }
                        } else {
                            for (i32 i = 0; i < nobr21; i++) {
                                SLC_DLACPY("Full", &nobr21, &m, &dwork[ishftu_inner - mldrw], &ldrwrk,
                                           &dwork[ishftu_inner], &ldrwrk);
                                ishftu_inner += mldrw;
                            }
                        }
                    }

                    i32 ishfty_inner = lldrw + iny;
                    for (i32 i = 0; i < nobr21; i++) {
                        SLC_DLACPY("Full", &nobr21, &l, &dwork[ishfty_inner - lldrw], &ldrwrk,
                                   &dwork[ishfty_inner], &ldrwrk);
                        ishfty_inner += lldrw;
                    }
                }

                f64 dum = 0.0;
                i32 zero_m = 0;
                mb04od("Full", nr, zero_m, nsl, r, ldr, dwork, ldrwrk, &dum, nr, &dum, nr,
                       &dwork[itau], &dwork[jwork]);
                initi = initi + nsf;
            }
        }

        if (!last) {
            if (connec) {
                if (m > 0) {
                    SLC_DLACPY("Full", &nobr21, &m, &u[initi], &ldu, dwork, &nobr21);
                }
                SLC_DLACPY("Full", &nobr21, &l, &y[initi], &ldy, &dwork[mmnobr - m], &nobr21);
            }

            icycle++;
            iwork[0] = icycle;
            iwork[1] = maxwrk;
            iwork[2] = nsmpsm;
            if (icycle <= MAXCYC) {
                return;
            }
            *iwarn = 1;
            icycle = 1;
        }
    }

    dwork[0] = (f64)maxwrk;
    if (!onebch) {
        iwork[0] = icycle;
        iwork[1] = maxwrk;
        iwork[2] = nsmpsm;
    }
}
