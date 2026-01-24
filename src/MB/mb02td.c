#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <string.h>

void mb02td(const char *norm, i32 n, f64 hnorm, const f64 *h, i32 ldh,
            const i32 *ipiv, f64 *rcond, i32 *iwork, f64 *dwork, i32 *info)
{
    const f64 one = 1.0;
    const f64 zero = 0.0;

    i32 onenrm;
    char normin;
    i32 ix, j, jp, kase, kase1;
    f64 hinvnm, scale, smlnum, t;
    i32 isave[3];
    i32 int1 = 1;

    *info = 0;

    onenrm = (norm[0] == '1' || norm[0] == 'O' || norm[0] == 'o');
    if (!onenrm && norm[0] != 'I' && norm[0] != 'i') {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (hnorm < zero) {
        *info = -3;
    } else if (ldh < (n > 1 ? n : 1)) {
        *info = -5;
    }

    if (*info != 0) {
        return;
    }

    *rcond = zero;
    if (n == 0) {
        *rcond = one;
        return;
    } else if (hnorm == zero) {
        return;
    }

    smlnum = SLC_DLAMCH("Safe minimum");

    hinvnm = zero;
    normin = 'N';
    if (onenrm) {
        kase1 = 1;
    } else {
        kase1 = 2;
    }
    kase = 0;

    while (1) {
        SLC_DLACN2(&n, &dwork[n], dwork, iwork, &hinvnm, &kase, isave);
        if (kase == 0) {
            break;
        }

        if (kase == kase1) {
            for (j = 0; j < n - 1; j++) {
                jp = ipiv[j] - 1;
                t = dwork[jp];
                if (jp != j) {
                    dwork[jp] = dwork[j];
                    dwork[j] = t;
                }
                dwork[j + 1] = dwork[j + 1] - t * h[(j + 1) + j * ldh];
            }

            SLC_DLATRS("Upper", "No transpose", "Non-unit", &normin, &n,
                       h, &ldh, dwork, &scale, &dwork[2 * n], info);
        } else {
            SLC_DLATRS("Upper", "Transpose", "Non-unit", &normin, &n,
                       h, &ldh, dwork, &scale, &dwork[2 * n], info);

            for (j = n - 2; j >= 0; j--) {
                dwork[j] = dwork[j] - h[(j + 1) + j * ldh] * dwork[j + 1];
                jp = ipiv[j] - 1;
                if (jp != j) {
                    t = dwork[jp];
                    dwork[jp] = dwork[j];
                    dwork[j] = t;
                }
            }
        }

        normin = 'Y';
        if (scale != one) {
            ix = SLC_IDAMAX(&n, dwork, &int1) - 1;
            if (scale < fabs(dwork[ix]) * smlnum || scale == zero) {
                return;
            }
            SLC_DRSCL(&n, &scale, dwork, &int1);
        }
    }

    if (hinvnm != zero) {
        *rcond = (one / hinvnm) / hnorm;
    }
}
