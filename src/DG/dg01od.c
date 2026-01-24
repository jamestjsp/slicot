/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include <math.h>
#include <ctype.h>

void dg01od(const char *scr, const char *wght, i32 n, f64 *a, f64 *w, i32 *info)
{
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 FOUR = 4.0;

    *info = 0;

    char scr_c = (char)toupper((unsigned char)scr[0]);
    char wght_c = (char)toupper((unsigned char)wght[0]);

    bool lfwd = (scr_c == 'N') || (scr_c == 'I');
    bool lscr = (scr_c == 'I') || (scr_c == 'O');
    bool lwght = (wght_c == 'A');

    if (!lfwd && !lscr) {
        *info = -1;
        return;
    }
    if (!lwght && wght_c != 'N') {
        *info = -2;
        return;
    }

    i32 m = 0;
    i32 j = 0;
    if (n >= 1) {
        j = n;
        while ((j % 2) == 0) {
            j = j / 2;
            m = m + 1;
        }
        if (j != 1) {
            *info = -3;
            return;
        }
    } else if (n < 0) {
        *info = -3;
        return;
    }

    if (n <= 1) {
        return;
    }

    i32 i, l, len, p1, p2, q1, q2, r1, r2, s1, s2, wpos;
    f64 cf, sf, t1, t2, th;

    if (!lwght) {
        r1 = 0;
        len = 1;
        th = FOUR * atan(ONE) / (f64)n;

        for (l = 0; l < m - 2; l++) {
            len = 2 * len;
            th = TWO * th;
            cf = cos(th);
            sf = sin(th);
            w[r1] = cf;
            w[r1 + 1] = sf;
            r1 = r1 + 2;

            for (i = 0; i < len - 2; i += 2) {
                w[r1] = cf * w[i] - sf * w[i + 1];
                w[r1 + 1] = sf * w[i] + cf * w[i + 1];
                r1 = r1 + 2;
            }
        }

        p1 = 2;
        q1 = r1 - 2;

        for (l = m - 3; l >= 0; l--) {
            for (i = p1; i <= q1; i += 4) {
                w[r1] = w[i];
                w[r1 + 1] = w[i + 1];
                r1 = r1 + 2;
            }
            p1 = q1 + 4;
            q1 = r1 - 2;
        }
    }

    if (lfwd && !lscr) {
        j = 0;
        for (i = 0; i < n; i++) {
            if (j > i) {
                t1 = a[i];
                a[i] = a[j];
                a[j] = t1;
            }
            l = n / 2;
            while (j >= l && l >= 1) {
                j = j - l;
                l = l / 2;
            }
            j = j + l;
        }
    }

    if (lfwd) {
        for (j = 1; j < n; j += 2) {
            t1 = a[j];
            a[j] = a[j - 1] - t1;
            a[j - 1] = a[j - 1] + t1;
        }

        len = 1;
        wpos = n - 2 * m;

        for (l = 0; l < m - 1; l++) {
            len = 2 * len;
            p2 = 0;
            q2 = len;
            r2 = len / 2;
            s2 = r2 + q2;

            for (i = 0; i < n / (2 * len); i++) {
                t1 = a[q2];
                a[q2] = a[p2] - t1;
                a[p2] = a[p2] + t1;
                t1 = a[s2];
                a[s2] = a[r2] - t1;
                a[r2] = a[r2] + t1;

                p1 = p2 + 1;
                q1 = p1 + len;
                r1 = q1 - 2;
                s1 = r1 + len;

                for (j = wpos; j < wpos + len - 2; j += 2) {
                    cf = w[j];
                    sf = w[j + 1];
                    t1 = cf * a[q1] + sf * a[s1];
                    t2 = -cf * a[s1] + sf * a[q1];
                    a[q1] = a[p1] - t1;
                    a[p1] = a[p1] + t1;
                    a[s1] = a[r1] - t2;
                    a[r1] = a[r1] + t2;
                    p1 = p1 + 1;
                    q1 = q1 + 1;
                    r1 = r1 - 1;
                    s1 = s1 - 1;
                }

                p2 = p2 + 2 * len;
                q2 = q2 + 2 * len;
                r2 = r2 + 2 * len;
                s2 = s2 + 2 * len;
            }

            wpos = wpos - 2 * len + 2;
        }
    } else {
        wpos = 0;
        len = n;

        for (l = m - 2; l >= 0; l--) {
            len = len / 2;
            p2 = 0;
            q2 = len;
            r2 = len / 2;
            s2 = r2 + q2;

            for (i = 0; i < n / (2 * len); i++) {
                t1 = a[q2];
                a[q2] = a[p2] - t1;
                a[p2] = a[p2] + t1;
                t1 = a[s2];
                a[s2] = a[r2] - t1;
                a[r2] = a[r2] + t1;

                p1 = p2 + 1;
                q1 = p1 + len;
                r1 = q1 - 2;
                s1 = r1 + len;

                for (j = wpos; j < wpos + len - 2; j += 2) {
                    cf = w[j];
                    sf = w[j + 1];
                    t1 = a[p1] - a[q1];
                    t2 = a[r1] - a[s1];
                    a[p1] = a[p1] + a[q1];
                    a[r1] = a[r1] + a[s1];
                    a[q1] = cf * t1 + sf * t2;
                    a[s1] = -cf * t2 + sf * t1;
                    p1 = p1 + 1;
                    q1 = q1 + 1;
                    r1 = r1 - 1;
                    s1 = s1 - 1;
                }

                p2 = p2 + 2 * len;
                q2 = q2 + 2 * len;
                r2 = r2 + 2 * len;
                s2 = s2 + 2 * len;
            }

            wpos = wpos + len - 2;
        }

        for (j = 1; j < n; j += 2) {
            t1 = a[j];
            a[j] = a[j - 1] - t1;
            a[j - 1] = a[j - 1] + t1;
        }
    }
}
