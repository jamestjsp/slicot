/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include <stdio.h>
#include <string.h>

i32 ud01mz(i32 m, i32 n, i32 l, const c128 *a, i32 lda,
           const char *text, char *output, i32 output_size)
{
    if (m < 1) {
        return -1;
    }
    if (n < 1) {
        return -2;
    }
    if (l < 1 || l > 3) {
        return -3;
    }
    if (lda < m) {
        return -5;
    }

    i32 pos = 0;
    i32 remaining = output_size;

    i32 text_len = (i32)strlen(text);
    i32 ltext = (text_len < 72) ? text_len : 72;
    while (ltext > 1 && text[ltext - 1] == ' ') {
        ltext--;
    }

    i32 written = snprintf(output + pos, remaining, " %.*s (%5dX%5d)\n\n",
                           ltext, text, m, n);
    if (written >= remaining) return -6;
    pos += written;
    remaining -= written;

    i32 n1 = (n - 1) / l;
    i32 j1 = 0;
    i32 j2 = l - 1;

    for (i32 blk = 0; blk < n1; blk++) {
        written = snprintf(output + pos, remaining, "       ");
        if (written >= remaining) return -6;
        pos += written;
        remaining -= written;

        for (i32 jj = j1; jj <= j2; jj++) {
            written = snprintf(output + pos, remaining, "             %5d              ", jj + 1);
            if (written >= remaining) return -6;
            pos += written;
            remaining -= written;
        }
        written = snprintf(output + pos, remaining, "\n");
        if (written >= remaining) return -6;
        pos += written;
        remaining -= written;

        for (i32 i = 0; i < m; i++) {
            written = snprintf(output + pos, remaining, " %5d  ", i + 1);
            if (written >= remaining) return -6;
            pos += written;
            remaining -= written;

            for (i32 jj = j1; jj <= j2; jj++) {
                f64 re = creal(a[i + jj * lda]);
                f64 im = cimag(a[i + jj * lda]);
                const char *sign = (im >= 0.0) ? "+" : "-";
                f64 abs_im = (im >= 0.0) ? im : -im;
                written = snprintf(output + pos, remaining, "%15.7E%s%15.7Ei ", re, sign, abs_im);
                if (written >= remaining) return -6;
                pos += written;
                remaining -= written;
            }
            written = snprintf(output + pos, remaining, "\n");
            if (written >= remaining) return -6;
            pos += written;
            remaining -= written;
        }

        written = snprintf(output + pos, remaining, " \n");
        if (written >= remaining) return -6;
        pos += written;
        remaining -= written;

        j1 += l;
        j2 += l;
    }

    written = snprintf(output + pos, remaining, "       ");
    if (written >= remaining) return -6;
    pos += written;
    remaining -= written;

    for (i32 j = j1; j < n; j++) {
        written = snprintf(output + pos, remaining, "             %5d              ", j + 1);
        if (written >= remaining) return -6;
        pos += written;
        remaining -= written;
    }
    written = snprintf(output + pos, remaining, "\n");
    if (written >= remaining) return -6;
    pos += written;
    remaining -= written;

    for (i32 i = 0; i < m; i++) {
        written = snprintf(output + pos, remaining, " %5d  ", i + 1);
        if (written >= remaining) return -6;
        pos += written;
        remaining -= written;

        for (i32 jj = j1; jj < n; jj++) {
            f64 re = creal(a[i + jj * lda]);
            f64 im = cimag(a[i + jj * lda]);
            const char *sign = (im >= 0.0) ? "+" : "-";
            f64 abs_im = (im >= 0.0) ? im : -im;
            written = snprintf(output + pos, remaining, "%15.7E%s%15.7Ei ", re, sign, abs_im);
            if (written >= remaining) return -6;
            pos += written;
            remaining -= written;
        }
        written = snprintf(output + pos, remaining, "\n");
        if (written >= remaining) return -6;
        pos += written;
        remaining -= written;
    }

    written = snprintf(output + pos, remaining, " \n");
    if (written >= remaining) return -6;
    pos += written;
    remaining -= written;

    return 0;
}
