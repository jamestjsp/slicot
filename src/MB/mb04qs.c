// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>

void mb04qs(const char *tranc, const char *trand, const char *tranu,
            i32 m, i32 n, i32 ilo,
            const f64 *v, i32 ldv, const f64 *w, i32 ldw,
            f64 *c, i32 ldc, f64 *d, i32 ldd,
            const f64 *cs, const f64 *tau,
            f64 *dwork, i32 ldwork, i32 *info) {

    const f64 ONE = 1.0;

    *info = 0;

    bool ltrc = (toupper((unsigned char)tranc[0]) == 'T') ||
                (toupper((unsigned char)tranc[0]) == 'C');
    bool ltrd = (toupper((unsigned char)trand[0]) == 'T') ||
                (toupper((unsigned char)trand[0]) == 'C');
    bool ltru = (toupper((unsigned char)tranu[0]) == 'T');

    i32 mh = (m - ilo) > 0 ? (m - ilo) : 0;

    if (!(ltrc || toupper((unsigned char)tranc[0]) == 'N')) {
        *info = -1;
    } else if (!(ltrd || toupper((unsigned char)trand[0]) == 'N')) {
        *info = -2;
    } else if (!(ltru || toupper((unsigned char)tranu[0]) == 'N')) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (ilo < 1 || ilo > m + 1) {
        *info = -6;
    } else if (ldv < (m > 1 ? m : 1)) {
        *info = -8;
    } else if (ldw < (m > 1 ? m : 1)) {
        *info = -10;
    } else if ((ltrc && ldc < (n > 1 ? n : 1)) ||
               (!ltrc && ldc < (m > 1 ? m : 1))) {
        *info = -12;
    } else if ((ltrd && ldd < (n > 1 ? n : 1)) ||
               (!ltrd && ldd < (m > 1 ? m : 1))) {
        *info = -14;
    }

    if (*info != 0) {
        return;
    }

    bool lquery = (ldwork == -1);

    if (lquery) {
        if (m <= ilo || n == 0) {
            dwork[0] = ONE;
        } else {
            i32 ierr;
            mb04qb(tranc, trand, tranu, "C", "C", mh, n, mh,
                   v, ldv, w, ldw, c, ldc, d, ldd, cs, tau,
                   dwork, -1, &ierr);
            i32 minwrk = (n > 1) ? n : 1;
            f64 opt = dwork[0];
            dwork[0] = (opt > (f64)minwrk) ? opt : (f64)minwrk;
        }
        return;
    }

    if (m <= ilo || n == 0) {
        dwork[0] = ONE;
        return;
    }

    i32 minwrk = (n > 1) ? n : 1;
    if (ldwork < minwrk) {
        dwork[0] = (f64)minwrk;
        *info = -18;
        return;
    }

    i32 ic, jc, id, jd;
    if (ltrc) {
        ic = 0;
        jc = ilo;  // Fortran ILO+1 -> C index ILO (0-based)
    } else {
        ic = ilo;  // Fortran ILO+1 -> C index ILO (0-based)
        jc = 0;
    }
    if (ltrd) {
        id = 0;
        jd = ilo;
    } else {
        id = ilo;
        jd = 0;
    }

    i32 ierr;
    // Fortran: V(ILO+1,ILO), W(ILO+1,ILO) -> C: v[ilo + (ilo-1)*ldv], w[ilo + (ilo-1)*ldw]
    // CS(2*ILO-1) -> C: cs[2*(ilo-1)] = cs[2*ilo-2]
    // TAU(ILO) -> C: tau[ilo-1]
    mb04qb(tranc, trand, tranu, "Columnwise", "Columnwise", mh, n, mh,
           &v[ilo + (ilo - 1) * ldv], ldv,
           &w[ilo + (ilo - 1) * ldw], ldw,
           &c[ic + jc * ldc], ldc,
           &d[id + jd * ldd], ldd,
           &cs[2 * ilo - 2], &tau[ilo - 1],
           dwork, ldwork, &ierr);
}
