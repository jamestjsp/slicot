// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

void mb03zd(const char *which, const char *meth, const char *stab,
            const char *balanc, const char *ortbal, const i32 *select,
            i32 n, i32 mm, i32 ilo, const f64 *scale,
            f64 *s, i32 lds, f64 *t, i32 ldt, f64 *g, i32 ldg,
            f64 *u1, i32 ldu1, f64 *u2, i32 ldu2,
            f64 *v1, i32 ldv1, f64 *v2, i32 ldv2,
            i32 *m, f64 *wr, f64 *wi,
            f64 *us, i32 ldus, f64 *uu, i32 lduu,
            bool *lwork, i32 *iwork, f64 *dwork, i32 ldwork, i32 *info) {
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 HUNDRD = 100.0;
    const f64 NEG_ONE = -1.0;

    i32 int1 = 1, int0 = 0;
    i32 ierr = 0;
    i32 wrkopt = 1;

    bool lall = (which[0] == 'A' || which[0] == 'a');
    bool lext = false;
    if (lall) {
        lext = (meth[0] == 'L' || meth[0] == 'l' ||
                meth[0] == 'R' || meth[0] == 'r');
    }
    bool lric = (meth[0] == 'Q' || meth[0] == 'q' ||
                 meth[0] == 'R' || meth[0] == 'r');
    bool l_us = (stab[0] == 'S' || stab[0] == 's' ||
                 stab[0] == 'B' || stab[0] == 'b');
    bool l_uu = (stab[0] == 'U' || stab[0] == 'u' ||
                 stab[0] == 'B' || stab[0] == 'b');
    bool lbal = (balanc[0] == 'P' || balanc[0] == 'p' ||
                 balanc[0] == 'S' || balanc[0] == 's' ||
                 balanc[0] == 'B' || balanc[0] == 'b');
    bool lbef = false;
    if (lbal) {
        lbef = (ortbal[0] == 'B' || ortbal[0] == 'b');
    }

    *info = 0;

    if (!lall && !(which[0] == 'S' || which[0] == 's')) {
        *info = -1;
    } else if (lall && (!lext && !lric && !(meth[0] == 'S' || meth[0] == 's'))) {
        *info = -2;
    } else if (!l_us && !l_uu) {
        *info = -3;
    } else if (!lbal && !(balanc[0] == 'N' || balanc[0] == 'n')) {
        *info = -4;
    } else if (lbal && (!lbef && !(ortbal[0] == 'A' || ortbal[0] == 'a'))) {
        *info = -5;
    }

    if (*info != 0) {
        return;
    }

    if (lall) {
        *m = n;
    } else {
        *m = 0;
        bool pair = false;
        for (i32 k = 0; k < n; k++) {
            if (pair) {
                pair = false;
            } else {
                if (k < n - 1) {
                    if (s[(k + 1) + k * lds] == ZERO) {
                        if (select[k])
                            (*m)++;
                    } else {
                        pair = true;
                        if (select[k] || select[k + 1])
                            *m += 2;
                    }
                } else {
                    if (select[n - 1])
                        (*m)++;
                }
            }
        }
    }

    i32 wrkmin;
    i32 mm_m = (mm < *m) ? mm : *m;
    if (mm_m == 0) {
        wrkmin = 1;
    } else if (!lext) {
        i32 tmp1 = 8 * (*m);
        i32 tmp2 = 4 * n;
        i32 maxval = (tmp1 > tmp2) ? tmp1 : tmp2;
        wrkmin = 4 * (*m) * (*m) + maxval;
        if (wrkmin < 1) wrkmin = 1;
    } else {
        if (l_us && l_uu) {
            wrkmin = 8 * n + 1;
            if (wrkmin < 1) wrkmin = 1;
        } else {
            i32 tmp1 = 2 * n * n + 2 * n;
            i32 tmp2 = 8 * n;
            wrkmin = (tmp1 > tmp2) ? tmp1 : tmp2;
            if (wrkmin < 1) wrkmin = 1;
        }
    }
    wrkopt = wrkmin;
    bool lquery = (ldwork == -1);

    if (n < 0) {
        *info = -7;
    } else if (mm < *m || (lext && mm < 2 * n)) {
        *info = -8;
    } else if (lbal && (ilo < 1 || ilo > n + 1)) {
        *info = -9;
    } else if (lds < (n > 1 ? n : 1)) {
        *info = -12;
    } else if (ldt < (n > 1 ? n : 1)) {
        *info = -14;
    } else if (ldg < 1 || (lext && ldg < n)) {
        *info = -16;
    } else if (ldu1 < (n > 1 ? n : 1)) {
        *info = -18;
    } else if (ldu2 < (n > 1 ? n : 1)) {
        *info = -20;
    } else if (ldv1 < (n > 1 ? n : 1)) {
        *info = -22;
    } else if (ldv2 < (n > 1 ? n : 1)) {
        *info = -24;
    } else if (ldus < 1 || (l_us && ldus < 2 * n)) {
        *info = -29;
    } else if (lduu < 1 || (l_uu && lduu < 2 * n)) {
        *info = -31;
    } else if (lquery) {
        i32 n_mm = (n < mm) ? n : mm;
        if (n_mm == 0) {
            dwork[0] = ONE;
            return;
        } else if (!lext) {
            mb01ux("Right", "Upper", "No Transpose", n, *m, ONE,
                   dwork, *m, v1, ldv1, dwork, -1, &ierr);
            i32 opt1 = (i32)dwork[0] + 2 * (*m) * (*m);
            wrkopt = (wrkopt > opt1) ? wrkopt : opt1;
            if (l_us || l_uu) {
                i32 n2 = 2 * n;
                SLC_DGEQRF(&n2, m, dwork, &n2, dwork, dwork, &int1, &ierr);
                i32 opt2 = (i32)dwork[0] + (*m);
                wrkopt = (wrkopt > opt2) ? wrkopt : opt2;
                SLC_DORGQR(&n2, m, m, dwork, &n2, dwork, dwork, &int1, &ierr);
                i32 opt3 = (i32)dwork[0] + (*m);
                wrkopt = (wrkopt > opt3) ? wrkopt : opt3;
            }
        } else {
            i32 n2 = 2 * n;
            mb01ux("Right", "Upper", "No Transpose", n2, n, ONE,
                   dwork, n, dwork, n2, dwork, -1, &ierr);
            i32 extra = 0;
            if ((l_us && !l_uu) || (l_uu && !l_us)) {
                extra = 2 * n * n;
            }
            i32 opt1 = (i32)dwork[0] + extra;
            wrkopt = (wrkopt > opt1) ? wrkopt : opt1;
            if (l_us || l_uu) {
                SLC_DGEQP3(&n2, &n2, dwork, &n2, iwork, dwork, dwork, &int1, &ierr);
                i32 opt2 = (i32)dwork[0] + 2 * n;
                wrkopt = (wrkopt > opt2) ? wrkopt : opt2;
                SLC_DORGQR(&n2, &n2, &n, dwork, &n2, dwork, dwork, &int1, &ierr);
                i32 opt3 = (i32)dwork[0] + 2 * n;
                wrkopt = (wrkopt > opt3) ? wrkopt : opt3;
            }
        }
        dwork[0] = (f64)wrkopt;
        return;
    } else if (ldwork < wrkmin) {
        *info = -35;
        dwork[0] = (f64)wrkmin;
    }

    if (*info != 0) {
        return;
    }

    i32 n_mm = (n < mm) ? n : mm;
    if (n_mm == 0) {
        dwork[0] = ONE;
        return;
    }
    wrkopt = wrkmin;

    if (!lext) {
        i32 pw = 0;
        i32 pdw = pw + 4 * (*m) * (*m);
        i32 ldw = 2 * (*m);
        i32 ldwork_sub = ldwork - pdw;

        mb03za("No Update", "Update", "Update", "Init", which,
               select, n, s, lds, t, ldt, g, ldg, u1, ldu1, u2, ldu2,
               v1, ldv1, v2, ldv2, &dwork[pw], ldw, wr, wi, m,
               &dwork[pdw], ldwork_sub, &ierr);
        if (ierr != 0)
            goto error_handling;

        pdw = pw + 2 * (*m) * (*m);
        ldwork_sub = ldwork - pdw;
        mb01ux("Right", "Upper", "No Transpose", n, *m, ONE,
               &dwork[pw], ldw, v1, ldv1, &dwork[pdw], ldwork_sub, &ierr);
        i32 opt1 = (i32)dwork[pdw] + pdw;
        wrkopt = (wrkopt > opt1) ? wrkopt : opt1;

        if (l_us) {
            SLC_DLACPY("All", &n, m, v1, &ldv1, us, &ldus);
        }
        if (l_uu) {
            SLC_DLACPY("All", &n, m, v1, &ldv1, uu, &lduu);
        }

        mb01ux("Right", "Upper", "No Transpose", n, *m, ONE,
               &dwork[pw + (*m)], ldw, u1, ldu1, &dwork[pdw], ldwork_sub, &ierr);

        if (l_us) {
            for (i32 j = 0; j < *m; j++) {
                SLC_DAXPY(&n, &NEG_ONE, &u1[j * ldu1], &int1, &us[j * ldus], &int1);
            }
        }
        if (l_uu) {
            for (i32 j = 0; j < *m; j++) {
                SLC_DAXPY(&n, &ONE, &u1[j * ldu1], &int1, &uu[j * lduu], &int1);
            }
        }

        mb01ux("Right", "Upper", "No Transpose", n, *m, NEG_ONE,
               &dwork[pw], ldw, v2, ldv2, &dwork[pdw], ldwork_sub, &ierr);

        if (l_us) {
            SLC_DLACPY("All", &n, m, v2, &ldv2, &us[n], &ldus);
        }
        if (l_uu) {
            SLC_DLACPY("All", &n, m, v2, &ldv2, &uu[n], &lduu);
        }

        mb01ux("Right", "Upper", "No Transpose", n, *m, ONE,
               &dwork[pw + (*m)], ldw, u2, ldu2, &dwork[pdw], ldwork_sub, &ierr);

        f64 nrmin = SLC_DLAMCH("Overflow");
        f64 tol = SLC_DLAMCH("Precision") * HUNDRD;

        if (l_us) {
            for (i32 j = 0; j < *m; j++) {
                SLC_DAXPY(&n, &ONE, &u2[j * ldu2], &int1, &us[n + j * ldus], &int1);
                i32 len = 2 * (*m);
                f64 temp = SLC_DASUM(&len, &us[j * ldus], &int1);
                if (temp < nrmin)
                    nrmin = temp;
            }
            i32 rows = 2 * (*m);
            f64 norm = SLC_DLANGE("1", &rows, m, us, &ldus, dwork);
            f64 maxnorm = (norm > ONE) ? norm : ONE;
            if (nrmin <= maxnorm * tol)
                *info = 5;
        }

        if (l_uu) {
            for (i32 j = 0; j < *m; j++) {
                SLC_DAXPY(&n, &NEG_ONE, &u2[j * ldu2], &int1, &uu[n + j * lduu], &int1);
                i32 len = 2 * (*m);
                f64 temp = SLC_DASUM(&len, &uu[j * lduu], &int1);
                if (temp < nrmin)
                    nrmin = temp;
            }
            i32 rows = 2 * (*m);
            f64 norm = SLC_DLANGE("1", &rows, m, uu, &lduu, dwork);
            f64 maxnorm = (norm > ONE) ? norm : ONE;
            if (nrmin <= maxnorm * tol)
                *info = 6;
        }

        if (lric) {
            if (lbal) {
                if (l_us) {
                    mb04di(balanc, "Positive", n, ilo, scale, *m, us, ldus,
                           &us[n], ldus, &ierr);
                }
                if (l_uu) {
                    mb04di(balanc, "Positive", n, ilo, scale, *m, uu, lduu,
                           &uu[n], lduu, &ierr);
                }
            }
            return;
        }

        if (lbal && lbef) {
            if (l_us) {
                mb04di(balanc, "Positive", n, ilo, scale, *m, us, ldus,
                       &us[n], ldus, &ierr);
            }
            if (l_uu) {
                mb04di(balanc, "Positive", n, ilo, scale, *m, uu, lduu,
                       &uu[n], lduu, &ierr);
            }
        }

        if (l_us) {
            i32 n2 = 2 * n;
            i32 ldwork_qr = ldwork - (*m);
            SLC_DGEQRF(&n2, m, us, &ldus, dwork, &dwork[*m], &ldwork_qr, &ierr);
            i32 opt2 = (i32)dwork[*m] + (*m);
            wrkopt = (wrkopt > opt2) ? wrkopt : opt2;

            f64 rcond;
            SLC_DTRCON("1", "Upper", "NotUnit", m, us, &ldus, &rcond,
                       &dwork[*m], iwork, &ierr);
            if (rcond <= tol)
                *info = 5;

            SLC_DORGQR(&n2, m, m, us, &ldus, dwork, &dwork[*m], &ldwork_qr, &ierr);
            i32 opt3 = (i32)dwork[*m] + (*m);
            wrkopt = (wrkopt > opt3) ? wrkopt : opt3;
        }

        if (l_uu) {
            i32 n2 = 2 * n;
            i32 ldwork_qr = ldwork - (*m);
            SLC_DGEQRF(&n2, m, uu, &lduu, dwork, &dwork[*m], &ldwork_qr, &ierr);
            i32 opt2 = (i32)dwork[*m] + (*m);
            wrkopt = (wrkopt > opt2) ? wrkopt : opt2;

            f64 rcond;
            SLC_DTRCON("1", "Upper", "NotUnit", m, uu, &lduu, &rcond,
                       &dwork[*m], iwork, &ierr);
            if (rcond <= tol)
                *info = 6;

            SLC_DORGQR(&n2, m, m, uu, &lduu, dwork, &dwork[*m], &ldwork_qr, &ierr);
            i32 opt3 = (i32)dwork[*m] + (*m);
            wrkopt = (wrkopt > opt3) ? wrkopt : opt3;
        }

        if (lbal && !lbef) {
            if (l_us) {
                mb04di(balanc, "Positive", n, ilo, scale, *m, us, ldus,
                       &us[n], ldus, &ierr);
            }
            if (l_uu) {
                mb04di(balanc, "Positive", n, ilo, scale, *m, uu, lduu,
                       &uu[n], lduu, &ierr);
            }
        }

    } else {
        for (i32 i = 0; i < 2 * n; i++) {
            lwork[i] = true;
        }

        if (l_us && !l_uu) {
            mb03za("Update", "Update", "Update", "Init", which,
                   select, n, s, lds, t, ldt, g, ldg, u1, ldu1, u2, ldu2,
                   v1, ldv1, v2, ldv2, us, ldus, wr, wi, m,
                   dwork, ldwork, &ierr);
            if (ierr != 0)
                goto error_handling;

            mb01ux("Left", "Lower", "Transpose", n, n, ONE,
                   &us[n + n * ldus], ldus, g, ldg, dwork, ldwork, &ierr);
            i32 opt1 = (i32)dwork[0];
            wrkopt = (wrkopt > opt1) ? wrkopt : opt1;

            mb01ux("Right", "Lower", "No Transpose", n, n, ONE,
                   &us[n * ldus], ldus, g, ldg, dwork, ldwork, &ierr);

            for (i32 j = 0; j < n; j++) {
                i32 len = j + 1;
                SLC_DAXPY(&len, &ONE, &g[j], &ldg, &g[j * ldg], &int1);
            }
            i32 pdw = 2 * n * n;
            i32 ldwork_sub = ldwork - pdw;

            SLC_DLACPY("All", &n, &n, v1, &ldv1, dwork, &n);
            i32 n2 = 2 * n;
            SLC_DLACPY("All", &n, &n, v2, &ldv2, &dwork[n], &n2);

            mb01ux("Right", "Upper", "No Transpose", n2, n, NEG_ONE,
                   us, ldus, dwork, n2, &dwork[pdw], ldwork_sub, &ierr);
            i32 opt2 = (i32)dwork[pdw] + pdw;
            wrkopt = (wrkopt > opt2) ? wrkopt : opt2;

            SLC_DLACPY("All", &n, &n, u2, &ldu2, us, &ldus);
            mb01ux("Right", "Upper", "No Transpose", n, n, ONE,
                   &us[n], ldus, us, ldus, &dwork[pdw], ldwork_sub, &ierr);

            for (i32 j = 0; j < n; j++) {
                SLC_DAXPY(&n, &ONE, &us[j * ldus], &int1, &dwork[n + 2 * j * n], &int1);
            }

            SLC_DLACPY("All", &n, &n, u1, &ldu1, us, &ldus);
            mb01ux("Right", "Upper", "No Transpose", n, n, NEG_ONE,
                   &us[n], ldus, us, ldus, &dwork[pdw], ldwork_sub, &ierr);

            for (i32 j = 0; j < n; j++) {
                SLC_DAXPY(&n, &NEG_ONE, &dwork[2 * j * n], &int1, &us[j * ldus], &int1);
            }

            SLC_DLACPY("All", &n, &n, &dwork[n], &n2, &us[n], &ldus);

            mb01ux("Right", "Lower", "No Transpose", n, n, ONE,
                   &us[n * ldus], ldus, v1, ldv1, dwork, ldwork, &ierr);
            mb01ux("Right", "Lower", "No Transpose", n, n, ONE,
                   &us[n * ldus], ldus, v2, ldv2, dwork, ldwork, &ierr);
            mb01ux("Right", "Lower", "No Transpose", n, n, ONE,
                   &us[n + n * ldus], ldus, u1, ldu1, dwork, ldwork, &ierr);
            mb01ux("Right", "Lower", "No Transpose", n, n, ONE,
                   &us[n + n * ldus], ldus, u2, ldu2, dwork, ldwork, &ierr);

            SLC_DLACPY("All", &n, &n, v1, &ldv1, &us[n * ldus], &ldus);
            SLC_DLACPY("All", &n, &n, v2, &ldv2, &us[n + n * ldus], &ldus);

            for (i32 j = 0; j < n; j++) {
                SLC_DAXPY(&n, &NEG_ONE, &u1[j * ldu1], &int1, &us[(n + j) * ldus], &int1);
            }
            for (i32 j = 0; j < n; j++) {
                SLC_DAXPY(&n, &NEG_ONE, &u2[j * ldu2], &int1, &us[n + (n + j) * ldus], &int1);
            }

            mb03td("Hamiltonian", "Update", lwork, &lwork[n], n,
                   s, lds, g, ldg, &us[n * ldus], ldus, &us[n + n * ldus], ldus,
                   wr, wi, m, dwork, ldwork, &ierr);
            if (ierr != 0) {
                *info = 4;
                return;
            }

            SLC_DLASCL("General", &int0, &int0, &ONE, &NEG_ONE, &n, &n,
                       &us[n + n * ldus], &ldus, &ierr);

        } else if (!l_us && l_uu) {
            mb03za("Update", "Update", "Update", "Init", which,
                   select, n, s, lds, t, ldt, g, ldg, u1, ldu1, u2, ldu2,
                   v1, ldv1, v2, ldv2, uu, lduu, wr, wi, m,
                   dwork, ldwork, &ierr);
            if (ierr != 0)
                goto error_handling;

            mb01ux("Left", "Lower", "Transpose", n, n, ONE,
                   &uu[n + n * lduu], lduu, g, ldg, dwork, ldwork, &ierr);
            i32 opt1 = (i32)dwork[0];
            wrkopt = (wrkopt > opt1) ? wrkopt : opt1;

            mb01ux("Right", "Lower", "No Transpose", n, n, ONE,
                   &uu[n * lduu], lduu, g, ldg, dwork, ldwork, &ierr);

            for (i32 j = 0; j < n; j++) {
                i32 len = j + 1;
                SLC_DAXPY(&len, &ONE, &g[j], &ldg, &g[j * ldg], &int1);
            }
            i32 pdw = 2 * n * n;
            i32 ldwork_sub = ldwork - pdw;

            SLC_DLACPY("All", &n, &n, v1, &ldv1, dwork, &n);
            i32 n2 = 2 * n;
            SLC_DLACPY("All", &n, &n, v2, &ldv2, &dwork[n], &n2);

            mb01ux("Right", "Upper", "No Transpose", n2, n, NEG_ONE,
                   uu, lduu, dwork, n2, &dwork[pdw], ldwork_sub, &ierr);
            i32 opt2 = (i32)dwork[pdw] + pdw;
            wrkopt = (wrkopt > opt2) ? wrkopt : opt2;

            SLC_DLACPY("All", &n, &n, u2, &ldu2, uu, &lduu);
            mb01ux("Right", "Upper", "No Transpose", n, n, NEG_ONE,
                   &uu[n], lduu, uu, lduu, &dwork[pdw], ldwork_sub, &ierr);

            for (i32 j = 0; j < n; j++) {
                SLC_DAXPY(&n, &ONE, &uu[j * lduu], &int1, &dwork[n + 2 * j * n], &int1);
            }

            SLC_DLACPY("All", &n, &n, u1, &ldu1, uu, &lduu);
            mb01ux("Right", "Upper", "No Transpose", n, n, ONE,
                   &uu[n], lduu, uu, lduu, &dwork[pdw], ldwork_sub, &ierr);

            for (i32 j = 0; j < n; j++) {
                SLC_DAXPY(&n, &NEG_ONE, &dwork[2 * j * n], &int1, &uu[j * lduu], &int1);
            }

            SLC_DLACPY("All", &n, &n, &dwork[n], &n2, &uu[n], &lduu);

            mb01ux("Right", "Lower", "No Transpose", n, n, ONE,
                   &uu[n * lduu], lduu, v1, ldv1, dwork, ldwork, &ierr);
            mb01ux("Right", "Lower", "No Transpose", n, n, ONE,
                   &uu[n * lduu], lduu, v2, ldv2, dwork, ldwork, &ierr);
            mb01ux("Right", "Lower", "No Transpose", n, n, ONE,
                   &uu[n + n * lduu], lduu, u1, ldu1, dwork, ldwork, &ierr);
            mb01ux("Right", "Lower", "No Transpose", n, n, ONE,
                   &uu[n + n * lduu], lduu, u2, ldu2, dwork, ldwork, &ierr);

            SLC_DLACPY("All", &n, &n, v1, &ldv1, &uu[n * lduu], &lduu);
            SLC_DLACPY("All", &n, &n, v2, &ldv2, &uu[n + n * lduu], &lduu);

            for (i32 j = 0; j < n; j++) {
                SLC_DAXPY(&n, &ONE, &u1[j * ldu1], &int1, &uu[(n + j) * lduu], &int1);
            }
            for (i32 j = 0; j < n; j++) {
                SLC_DAXPY(&n, &ONE, &u2[j * ldu2], &int1, &uu[n + (n + j) * lduu], &int1);
            }

            mb03td("Hamiltonian", "Update", lwork, &lwork[n], n,
                   s, lds, g, ldg, &uu[n * lduu], lduu, &uu[n + n * lduu], lduu,
                   wr, wi, m, dwork, ldwork, &ierr);
            if (ierr != 0) {
                *info = 4;
                return;
            }

            SLC_DLASCL("General", &int0, &int0, &ONE, &NEG_ONE, &n, &n,
                       &uu[n + n * lduu], &lduu, &ierr);

        } else {
            mb03za("Update", "Update", "Update", "Init", which,
                   select, n, s, lds, t, ldt, g, ldg, u1, ldu1, u2, ldu2,
                   v1, ldv1, v2, ldv2, us, ldus, wr, wi, m,
                   dwork, ldwork, &ierr);
            if (ierr != 0)
                goto error_handling;

            mb01ux("Left", "Lower", "Transpose", n, n, ONE,
                   &us[n + n * ldus], ldus, g, ldg, dwork, ldwork, &ierr);
            i32 opt1 = (i32)dwork[0];
            wrkopt = (wrkopt > opt1) ? wrkopt : opt1;

            mb01ux("Right", "Lower", "No Transpose", n, n, ONE,
                   &us[n * ldus], ldus, g, ldg, dwork, ldwork, &ierr);

            for (i32 j = 0; j < n; j++) {
                i32 len = j + 1;
                SLC_DAXPY(&len, &ONE, &g[j], &ldg, &g[j * ldg], &int1);
            }

            SLC_DLACPY("All", &n, &n, v1, &ldv1, uu, &lduu);
            SLC_DLACPY("All", &n, &n, v2, &ldv2, &uu[n], &lduu);
            i32 n2 = 2 * n;
            mb01ux("Right", "Upper", "No Transpose", n2, n, ONE,
                   us, ldus, uu, lduu, dwork, ldwork, &ierr);
            i32 opt2 = (i32)dwork[0];
            wrkopt = (wrkopt > opt2) ? wrkopt : opt2;

            SLC_DLACPY("All", &n, &n, u1, &ldu1, &uu[n * lduu], &lduu);
            SLC_DLACPY("All", &n, &n, u2, &ldu2, &uu[n + n * lduu], &lduu);
            mb01ux("Right", "Upper", "No Transpose", n2, n, ONE,
                   &us[n], ldus, &uu[n * lduu], lduu, dwork, ldwork, &ierr);

            SLC_DLASCL("General", &int0, &int0, &ONE, &NEG_ONE, &n, &n2,
                       &uu[n], &lduu, &ierr);

            SLC_DLACPY("All", &n2, &n, uu, &lduu, us, &ldus);
            for (i32 j = 0; j < n; j++) {
                SLC_DAXPY(&n2, &NEG_ONE, &uu[(n + j) * lduu], &int1, &us[j * ldus], &int1);
            }
            for (i32 j = 0; j < n; j++) {
                SLC_DAXPY(&n2, &ONE, &uu[(n + j) * lduu], &int1, &uu[j * lduu], &int1);
            }

            mb01ux("Right", "Lower", "No Transpose", n, n, ONE,
                   &us[n * ldus], ldus, v1, ldv1, dwork, ldwork, &ierr);
            mb01ux("Right", "Lower", "No Transpose", n, n, ONE,
                   &us[n * ldus], ldus, v2, ldv2, dwork, ldwork, &ierr);
            mb01ux("Right", "Lower", "No Transpose", n, n, ONE,
                   &us[n + n * ldus], ldus, u1, ldu1, dwork, ldwork, &ierr);
            mb01ux("Right", "Lower", "No Transpose", n, n, ONE,
                   &us[n + n * ldus], ldus, u2, ldu2, dwork, ldwork, &ierr);

            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < n; i++) {
                    f64 temp = v1[i + j * ldv1];
                    v1[i + j * ldv1] = temp - u1[i + j * ldu1];
                    u1[i + j * ldu1] = temp + u1[i + j * ldu1];
                }
            }
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < n; i++) {
                    f64 temp = v2[i + j * ldv2];
                    v2[i + j * ldv2] = temp - u2[i + j * ldu2];
                    u2[i + j * ldu2] = temp + u2[i + j * ldu2];
                }
            }

            SLC_DLASET("All", &n2, &n, &ZERO, &ONE, &us[n * ldus], &ldus);

            mb03td("Hamiltonian", "Update", lwork, &lwork[n], n,
                   s, lds, g, ldg, &us[n * ldus], ldus, &us[n + n * ldus], ldus,
                   wr, wi, m, dwork, ldwork, &ierr);
            if (ierr != 0) {
                *info = 4;
                return;
            }

            SLC_DGEMM("No Transpose", "No Transpose", &n, &n, &n, &ONE,
                      u1, &ldu1, &us[n * ldus], &ldus, &ZERO, &uu[n * lduu], &lduu);
            SLC_DGEMM("No Transpose", "No Transpose", &n, &n, &n, &NEG_ONE,
                      u2, &ldu2, &us[n + n * ldus], &ldus, &ONE, &uu[n * lduu], &lduu);
            SLC_DGEMM("No Transpose", "No Transpose", &n, &n, &n, &NEG_ONE,
                      u1, &ldu1, &us[n + n * ldus], &ldus, &ZERO, &uu[n + n * lduu], &lduu);
            SLC_DGEMM("No Transpose", "No Transpose", &n, &n, &n, &NEG_ONE,
                      u2, &ldu2, &us[n * ldus], &ldus, &ONE, &uu[n + n * lduu], &lduu);

            SLC_DLACPY("All", &n, &n, &us[n * ldus], &ldus, u1, &ldu1);
            SLC_DLACPY("All", &n, &n, &us[n + n * ldus], &ldus, u2, &ldu2);

            SLC_DGEMM("No Transpose", "No Transpose", &n, &n, &n, &ONE,
                      v1, &ldv1, u1, &ldu1, &ZERO, &us[n * ldus], &ldus);
            SLC_DGEMM("No Transpose", "No Transpose", &n, &n, &n, &NEG_ONE,
                      v2, &ldv2, u2, &ldu2, &ONE, &us[n * ldus], &ldus);
            SLC_DGEMM("No Transpose", "No Transpose", &n, &n, &n, &NEG_ONE,
                      v1, &ldv1, u2, &ldu2, &ZERO, &us[n + n * ldus], &ldus);
            SLC_DGEMM("No Transpose", "No Transpose", &n, &n, &n, &NEG_ONE,
                      v2, &ldv2, u1, &ldu1, &ONE, &us[n + n * ldus], &ldus);
        }

        if (lric) {
            if (lbal) {
                if (l_us) {
                    mb04di(balanc, "Positive", n, ilo, scale, mm, us, ldus,
                           &us[n], ldus, &ierr);
                }
                if (l_uu) {
                    mb04di(balanc, "Positive", n, ilo, scale, mm, uu, lduu,
                           &uu[n], lduu, &ierr);
                }
            }
            return;
        }

        if (lbal && lbef) {
            if (l_us) {
                mb04di(balanc, "Positive", n, ilo, scale, mm, us, ldus,
                       &us[n], ldus, &ierr);
            }
            if (l_uu) {
                mb04di(balanc, "Positive", n, ilo, scale, mm, uu, lduu,
                       &uu[n], lduu, &ierr);
            }
        }

        for (i32 j = 0; j < 2 * n; j++) {
            iwork[j] = 0;
        }

        if (l_us) {
            i32 n2 = 2 * n;
            i32 ldwork_qr = ldwork - 2 * n;
            SLC_DGEQP3(&n2, &mm, us, &ldus, iwork, dwork, &dwork[2 * n], &ldwork_qr, &ierr);
            i32 opt2 = (i32)dwork[2 * n] + 2 * n;
            wrkopt = (wrkopt > opt2) ? wrkopt : opt2;

            SLC_DORGQR(&n2, &mm, &n, us, &ldus, dwork, &dwork[2 * n], &ldwork_qr, &ierr);
            i32 opt3 = (i32)dwork[2 * n] + 2 * n;
            wrkopt = (wrkopt > opt3) ? wrkopt : opt3;
        }

        if (l_uu) {
            i32 n2 = 2 * n;
            i32 ldwork_qr = ldwork - 2 * n;
            SLC_DGEQP3(&n2, &mm, uu, &lduu, iwork, dwork, &dwork[2 * n], &ldwork_qr, &ierr);
            i32 opt2 = (i32)dwork[2 * n] + 2 * n;
            wrkopt = (wrkopt > opt2) ? wrkopt : opt2;

            SLC_DORGQR(&n2, &mm, &n, uu, &lduu, dwork, &dwork[2 * n], &ldwork_qr, &ierr);
            i32 opt3 = (i32)dwork[2 * n] + 2 * n;
            wrkopt = (wrkopt > opt3) ? wrkopt : opt3;
        }

        if (lbal && !lbef) {
            if (l_us) {
                mb04di(balanc, "Positive", n, ilo, scale, n, us, ldus,
                       &us[n], ldus, &ierr);
            }
            if (l_uu) {
                mb04di(balanc, "Positive", n, ilo, scale, n, uu, lduu,
                       &uu[n], lduu, &ierr);
            }
        }
    }

    SLC_DSCAL(m, &NEG_ONE, wr, &int1);
    dwork[0] = (f64)wrkopt;
    return;

error_handling:
    if (ierr == 1) {
        *info = 2;
    } else if (ierr == 2 || ierr == 4) {
        *info = 1;
    } else if (ierr == 3) {
        *info = 3;
    }
}
