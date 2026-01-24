/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include <stdio.h>
#include <string.h>

i32 ud01nd(i32 mp, i32 np, i32 dp, i32 l, const f64 *p, i32 ldp1, i32 ldp2,
           const char *text, char *output, i32 output_size)
{
    if (mp < 1) {
        return -1;
    }
    if (np < 1) {
        return -2;
    }
    if (dp < 0) {
        return -3;
    }
    if (l < 1 || l > 5) {
        return -4;
    }
    if (ldp1 < mp) {
        return -6;
    }
    if (ldp2 < np) {
        return -7;
    }

    i32 text_len = (i32)strlen(text);
    if (text_len > 72) {
        text_len = 72;
    }
    while (text_len > 0 && text[text_len - 1] == ' ') {
        text_len--;
    }

    char *ptr = output;
    char *end = output + output_size;
    i32 written;

    for (i32 k = 0; k <= dp; k++) {
        if (text_len == 0) {
            written = snprintf(ptr, (size_t)(end - ptr), " \n");
        } else {
            written = snprintf(ptr, (size_t)(end - ptr),
                               "\n %.*s(%2d) (%2dX%2d)\n",
                               text_len, text, k, mp, np);
        }
        if (written < 0 || ptr + written >= end) return -100;
        ptr += written;

        i32 n1 = (np - 1) / l;
        i32 j1 = 0;
        i32 j2 = l - 1;
        if (j2 >= np) j2 = np - 1;

        for (i32 jblock = 0; jblock < n1; jblock++) {
            written = snprintf(ptr, (size_t)(end - ptr), "     ");
            if (written < 0 || ptr + written >= end) return -100;
            ptr += written;

            for (i32 jj = j1; jj <= j2; jj++) {
                written = snprintf(ptr, (size_t)(end - ptr), "      %2d       ", jj + 1);
                if (written < 0 || ptr + written >= end) return -100;
                ptr += written;
            }
            written = snprintf(ptr, (size_t)(end - ptr), "\n");
            if (written < 0 || ptr + written >= end) return -100;
            ptr += written;

            for (i32 i = 0; i < mp; i++) {
                written = snprintf(ptr, (size_t)(end - ptr), " %2d  ", i + 1);
                if (written < 0 || ptr + written >= end) return -100;
                ptr += written;

                for (i32 jj = j1; jj <= j2; jj++) {
                    f64 val = p[i + jj * ldp1 + k * ldp1 * ldp2];
                    written = snprintf(ptr, (size_t)(end - ptr), "%15.7E", val);
                    if (written < 0 || ptr + written >= end) return -100;
                    ptr += written;
                }
                written = snprintf(ptr, (size_t)(end - ptr), "\n");
                if (written < 0 || ptr + written >= end) return -100;
                ptr += written;
            }

            j1 += l;
            j2 += l;
            if (j2 >= np) j2 = np - 1;
        }

        written = snprintf(ptr, (size_t)(end - ptr), "     ");
        if (written < 0 || ptr + written >= end) return -100;
        ptr += written;

        for (i32 j = j1; j < np; j++) {
            written = snprintf(ptr, (size_t)(end - ptr), "      %2d       ", j + 1);
            if (written < 0 || ptr + written >= end) return -100;
            ptr += written;
        }
        written = snprintf(ptr, (size_t)(end - ptr), "\n");
        if (written < 0 || ptr + written >= end) return -100;
        ptr += written;

        for (i32 i = 0; i < mp; i++) {
            written = snprintf(ptr, (size_t)(end - ptr), " %2d  ", i + 1);
            if (written < 0 || ptr + written >= end) return -100;
            ptr += written;

            for (i32 jj = j1; jj < np; jj++) {
                f64 val = p[i + jj * ldp1 + k * ldp1 * ldp2];
                written = snprintf(ptr, (size_t)(end - ptr), "%15.7E", val);
                if (written < 0 || ptr + written >= end) return -100;
                ptr += written;
            }
            written = snprintf(ptr, (size_t)(end - ptr), "\n");
            if (written < 0 || ptr + written >= end) return -100;
            ptr += written;
        }
    }

    written = snprintf(ptr, (size_t)(end - ptr), " \n");
    if (written < 0 || ptr + written >= end) return -100;

    return 0;
}
