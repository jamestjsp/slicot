#include "slicot.h"
#include <math.h>
#include <ctype.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void dk01md(const char *type, i32 n, f64 *a, i32 *info)
{
    *info = 0;

    char t = (char)toupper((unsigned char)type[0]);
    bool mtype = (t == 'M');
    bool ntype = (t == 'N');
    bool qtype = (t == 'Q');

    if (!mtype && !ntype && !qtype) {
        *info = -1;
        return;
    }
    if (n <= 0) {
        *info = -2;
        return;
    }

    // N=1 special case: single sample unchanged (window coefficient = 1.0)
    if (n == 1) {
        return;
    }

    f64 fn = (f64)(n - 1);

    if (mtype) {
        f64 temp = M_PI / fn;
        for (i32 i = 0; i < n; i++) {
            a[i] = a[i] * (0.54 + 0.46 * cos(temp * (f64)i));
        }
    } else if (ntype) {
        f64 temp = M_PI / fn;
        for (i32 i = 0; i < n; i++) {
            a[i] = a[i] * 0.5 * (1.0 + cos(temp * (f64)i));
        }
    } else {
        i32 n1 = (n - 1) / 2 + 1;
        for (i32 i = 0; i < n; i++) {
            f64 buf = (f64)i / fn;
            f64 temp = buf * buf;
            if (i < n1) {
                a[i] = a[i] * (1.0 - 2.0 * temp) * (1.0 - buf);
            } else {
                a[i] = a[i] * 2.0 * (1.0 - buf * temp);
            }
        }
    }
}
