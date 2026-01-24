/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <string.h>

i32 ue01md(i32 ispec, const char *name, const char *opts,
           i32 n1, i32 n2, i32 n3)
{
    char subnam[7];
    char c2[3], c3[2];
    i32 nb, nbmin, nx;
    i32 int_m1 = -1;

    if (ispec == 1 || ispec == 2 || ispec == 3) {
        i32 result = 1;

        size_t len = strlen(name);
        if (len > 6) len = 6;

        for (size_t i = 0; i < len; i++) {
            subnam[i] = (char)toupper((unsigned char)name[i]);
        }
        for (size_t i = len; i < 6; i++) {
            subnam[i] = ' ';
        }
        subnam[6] = '\0';

        c2[0] = subnam[3];
        c2[1] = subnam[4];
        c2[2] = '\0';

        c3[0] = subnam[5];
        c3[1] = '\0';

        if (ispec == 1) {
            nb = 1;

            if (strcmp(c2, "4S") == 0 || strcmp(c2, "4T") == 0) {
                if (c3[0] == 'B') {
                    nb = SLC_ILAENV(&ispec, "DGEQRF", " ", &n1, &n2, &int_m1, &int_m1) / 2;
                } else if (c3[0] == 'T') {
                    nb = SLC_ILAENV(&ispec, "DGEHRD", " ", &n1, &n2, &n1, &int_m1) / 4;
                }
            } else if (strcmp(c2, "4P") == 0) {
                if (c3[0] == 'B') {
                    nb = SLC_ILAENV(&ispec, "DGEHRD", " ", &n1, &n2, &n1, &int_m1) / 2;
                }
            } else if (strcmp(c2, "4W") == 0 || strcmp(c2, "4Q") == 0) {
                if (c3[0] == 'D') {
                    nb = SLC_ILAENV(&ispec, "DORGQR", " ", &n1, &n2, &n3, &int_m1) / 2;
                } else if (c3[0] == 'B') {
                    nb = SLC_ILAENV(&ispec, "DORMQR", " ", &n1, &n2, &n3, &int_m1) / 2;
                }
            } else if (strcmp(c2, "4R") == 0) {
                if (c3[0] == 'B') {
                    nb = SLC_ILAENV(&ispec, "DGEHRD", " ", &n1, &n2, &n1, &int_m1) / 2;
                }
            }
            result = nb;
        } else if (ispec == 2) {
            nbmin = 2;

            if (strcmp(c2, "4S") == 0 || strcmp(c2, "4T") == 0) {
                if (c3[0] == 'B') {
                    i32 tmp = SLC_ILAENV(&ispec, "DGEQRF", " ", &n1, &n2, &int_m1, &int_m1) / 2;
                    nbmin = tmp > 2 ? tmp : 2;
                } else if (c3[0] == 'T') {
                    i32 tmp = SLC_ILAENV(&ispec, "DGEHRD", " ", &n1, &n2, &n1, &int_m1) / 4;
                    nbmin = tmp > 2 ? tmp : 2;
                }
            } else if (strcmp(c2, "4P") == 0) {
                if (c3[0] == 'B') {
                    i32 tmp = SLC_ILAENV(&ispec, "DGEHRD", " ", &n1, &n2, &n1, &int_m1) / 4;
                    nbmin = tmp > 2 ? tmp : 2;
                }
            } else if (strcmp(c2, "4W") == 0 || strcmp(c2, "4Q") == 0) {
                if (c3[0] == 'D') {
                    i32 tmp = SLC_ILAENV(&ispec, "DORGQR", " ", &n1, &n2, &n3, &int_m1) / 2;
                    nbmin = tmp > 2 ? tmp : 2;
                } else if (c3[0] == 'B') {
                    i32 tmp = SLC_ILAENV(&ispec, "DORMQR", " ", &n1, &n2, &n3, &int_m1) / 2;
                    nbmin = tmp > 2 ? tmp : 2;
                }
            } else if (strcmp(c2, "4R") == 0) {
                if (c3[0] == 'B') {
                    i32 tmp = SLC_ILAENV(&ispec, "DGEHRD", " ", &n1, &n2, &n1, &int_m1) / 4;
                    nbmin = tmp > 2 ? tmp : 2;
                }
            }
            result = nbmin;
        } else if (ispec == 3) {
            nx = 0;

            if (strcmp(c2, "4S") == 0 || strcmp(c2, "4T") == 0) {
                if (c3[0] == 'B') {
                    nx = SLC_ILAENV(&ispec, "DGEQRF", " ", &n1, &n2, &int_m1, &int_m1);
                } else if (c3[0] == 'T') {
                    nx = SLC_ILAENV(&ispec, "DGEHRD", " ", &n1, &n2, &n1, &int_m1) / 2;
                }
            } else if (strcmp(c2, "4P") == 0) {
                if (c3[0] == 'B') {
                    nx = SLC_ILAENV(&ispec, "DGEHRD", " ", &n1, &n2, &n1, &int_m1) / 2;
                }
            } else if (strcmp(c2, "4W") == 0 || strcmp(c2, "4Q") == 0) {
                if (c3[0] == 'D') {
                    nx = SLC_ILAENV(&ispec, "DORGQR", " ", &n1, &n2, &n3, &int_m1);
                } else if (c3[0] == 'B') {
                    nx = SLC_ILAENV(&ispec, "DORGQR", " ", &n1, &n2, &n3, &int_m1);
                }
            } else if (strcmp(c2, "4R") == 0) {
                if (c3[0] == 'B') {
                    nx = SLC_ILAENV(&ispec, "DGEHRD", " ", &n1, &n2, &n1, &int_m1) / 2;
                }
            }
            result = nx;
        }
        return result;
    } else if (ispec == 4) {
        return SLC_ILAENV(&ispec, "DHSEQR", opts, &n1, &n2, &n3, &int_m1);
    } else if (ispec == 8) {
        return SLC_ILAENV(&ispec, "DHSEQR", opts, &n1, &n2, &n3, &int_m1);
    } else {
        return -1;
    }
}
