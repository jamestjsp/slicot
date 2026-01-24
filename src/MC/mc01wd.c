/**
 * @file mc01wd.c
 * @brief Polynomial quotient and remainder for quadratic denominator.
 *
 * Computes quotient Q(x) and remainder R(x) such that:
 *   P(x) = B(x) * Q(x) + R(x)
 * where B(x) = u1 + u2*x + x^2 is a quadratic polynomial.
 */

#include "slicot.h"

void mc01wd(const i32 dp, const f64 *p, const f64 u1, const f64 u2,
            f64 *q, i32 *info)
{
    *info = 0;

    if (dp < 0) {
        *info = -1;
        return;
    }

    i32 n = dp + 1;

    q[n - 1] = p[n - 1];

    if (n > 1) {
        f64 b = q[n - 1];
        q[n - 2] = p[n - 2] - u2 * b;

        if (n > 2) {
            f64 a = q[n - 2];

            for (i32 i = n - 3; i >= 0; i--) {
                f64 c = p[i] - u2 * a - u1 * b;
                q[i] = c;
                b = a;
                a = c;
            }
        }
    }
}
