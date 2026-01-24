#include "slicot.h"

static const f64 Y[15] = {
    0.14, 0.18, 0.22, 0.25, 0.29,
    0.32, 0.35, 0.39, 0.37, 0.58,
    0.73, 0.96, 1.34, 2.10, 4.39
};

void md03bf(
    i32* iflag,
    i32 m,
    i32 n,
    i32* ipar,
    i32 lipar,
    const f64* dpar1,
    i32 ldpar1,
    const f64* dpar2,
    i32 ldpar2,
    const f64* x,
    i32* nfevl,
    f64* e,
    f64* j,
    i32* ldj,
    f64* dwork,
    i32 ldwork,
    i32* info
)
{
    (void)m; (void)n; (void)lipar; (void)dpar1; (void)ldpar1;
    (void)dpar2; (void)ldpar2; (void)dwork; (void)ldwork;

    *info = 0;

    if (*iflag == 1) {
        for (i32 i = 1; i <= 15; i++) {
            f64 tmp1 = (f64)i;
            f64 tmp2 = (f64)(16 - i);
            f64 tmp3 = (i > 8) ? tmp2 : tmp1;
            e[i-1] = Y[i-1] - (x[0] + tmp1 / (x[1]*tmp2 + x[2]*tmp3));
        }
    } else if (*iflag == 2) {
        for (i32 i = 1; i <= 15; i++) {
            f64 tmp1 = (f64)i;
            f64 tmp2 = (f64)(16 - i);
            f64 tmp3 = (i > 8) ? tmp2 : tmp1;
            f64 tmp4 = (x[1]*tmp2 + x[2]*tmp3) * (x[1]*tmp2 + x[2]*tmp3);
            j[(i-1) + 0*(*ldj)] = -1.0;
            j[(i-1) + 1*(*ldj)] = tmp1*tmp2/tmp4;
            j[(i-1) + 2*(*ldj)] = tmp1*tmp3/tmp4;
        }
        *nfevl = 0;
    } else if (*iflag == 3) {
        *ldj = 15;
        ipar[0] = 15 * 3;
        ipar[1] = 0;
        ipar[2] = 0;
        ipar[3] = 4*3 + 1;
        ipar[4] = 4*3;
    }
}
