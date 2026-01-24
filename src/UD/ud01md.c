/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include <stdio.h>
#include <string.h>

i32 ud01md(i32 m, i32 n, i32 l, const f64 *a, i32 lda,
           const char *text, char *output, i32 output_size)
{
    if (m < 1) {
        return -1;
    }
    if (n < 1) {
        return -2;
    }
    if (l < 1 || l > 5) {
        return -3;
    }
    if (lda < m) {
        return -6;
    }

    i32 text_len = (i32)strlen(text);
    if (text_len > 72) {
        text_len = 72;
    }
    while (text_len > 1 && text[text_len - 1] == ' ') {
        text_len--;
    }

    char *ptr = output;
    char *end = output + output_size;
    i32 written;

    written = snprintf(ptr, (size_t)(end - ptr), " %.*s (%5dX%5d)\n\n",
                       text_len, text, m, n);
    if (written < 0 || ptr + written >= end) return -100;
    ptr += written;

    i32 n1 = (n - 1) / l;
    i32 j1 = 0;
    i32 j2 = l - 1;

    for (i32 jblock = 0; jblock < n1; jblock++) {
        written = snprintf(ptr, (size_t)(end - ptr), "        ");
        if (written < 0 || ptr + written >= end) return -100;
        ptr += written;

        for (i32 jj = j1; jj <= j2; jj++) {
            written = snprintf(ptr, (size_t)(end - ptr), "     %5d     ", jj + 1);
            if (written < 0 || ptr + written >= end) return -100;
            ptr += written;
        }
        written = snprintf(ptr, (size_t)(end - ptr), "\n");
        if (written < 0 || ptr + written >= end) return -100;
        ptr += written;

        for (i32 i = 0; i < m; i++) {
            written = snprintf(ptr, (size_t)(end - ptr), " %5d  ", i + 1);
            if (written < 0 || ptr + written >= end) return -100;
            ptr += written;

            for (i32 jj = j1; jj <= j2; jj++) {
                written = snprintf(ptr, (size_t)(end - ptr), "%15.7E", a[i + jj * lda]);
                if (written < 0 || ptr + written >= end) return -100;
                ptr += written;
            }
            written = snprintf(ptr, (size_t)(end - ptr), "\n");
            if (written < 0 || ptr + written >= end) return -100;
            ptr += written;
        }

        written = snprintf(ptr, (size_t)(end - ptr), " \n");
        if (written < 0 || ptr + written >= end) return -100;
        ptr += written;

        j1 += l;
        j2 += l;
    }

    written = snprintf(ptr, (size_t)(end - ptr), "        ");
    if (written < 0 || ptr + written >= end) return -100;
    ptr += written;

    for (i32 j = j1; j < n; j++) {
        written = snprintf(ptr, (size_t)(end - ptr), "     %5d     ", j + 1);
        if (written < 0 || ptr + written >= end) return -100;
        ptr += written;
    }
    written = snprintf(ptr, (size_t)(end - ptr), "\n");
    if (written < 0 || ptr + written >= end) return -100;
    ptr += written;

    for (i32 i = 0; i < m; i++) {
        written = snprintf(ptr, (size_t)(end - ptr), " %5d  ", i + 1);
        if (written < 0 || ptr + written >= end) return -100;
        ptr += written;

        for (i32 jj = j1; jj < n; jj++) {
            written = snprintf(ptr, (size_t)(end - ptr), "%15.7E", a[i + jj * lda]);
            if (written < 0 || ptr + written >= end) return -100;
            ptr += written;
        }
        written = snprintf(ptr, (size_t)(end - ptr), "\n");
        if (written < 0 || ptr + written >= end) return -100;
        ptr += written;
    }

    written = snprintf(ptr, (size_t)(end - ptr), " \n");
    if (written < 0 || ptr + written >= end) return -100;

    return 0;
}
