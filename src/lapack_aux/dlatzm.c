#include "slicot/lapack_aux.h"
#include "slicot_blas.h"
#include <string.h>
#include <math.h>

void slicot_dlatzm(const char *side, i32 m, i32 n, const f64 *v, i32 incv,
                   f64 tau, f64 *c1, f64 *c2, i32 ldc, f64 *work) {
    const f64 one = 1.0;
    const f64 zero = 0.0;
    i32 i_one = 1;

    if (m < 1 || n < 1 || tau == zero) {
        return;
    }

    if (*side == 'L' || *side == 'l') {
        i32 mm1 = m - 1;
        f64 neg_tau = -tau;

        SLC_DCOPY(&n, c1, &ldc, work, &i_one);
        SLC_DGEMV("T", &mm1, &n, &one, c2, &ldc, v, &incv, &one, work, &i_one);

        SLC_DAXPY(&n, &neg_tau, work, &i_one, c1, &ldc);
        SLC_DGER(&mm1, &n, &neg_tau, v, &incv, work, &i_one, c2, &ldc);
    } else if (*side == 'R' || *side == 'r') {
        i32 nm1 = n - 1;
        f64 neg_tau = -tau;

        SLC_DCOPY(&m, c1, &i_one, work, &i_one);
        SLC_DGEMV("N", &m, &nm1, &one, c2, &ldc, v, &incv, &one, work, &i_one);

        SLC_DAXPY(&m, &neg_tau, work, &i_one, c1, &i_one);
        SLC_DGER(&m, &nm1, &neg_tau, work, &i_one, v, &incv, c2, &ldc);
    }
}
