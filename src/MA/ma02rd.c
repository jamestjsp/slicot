/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MA02RD - Sort vector D and rearrange E with same permutation
 *
 * Sorts the elements of an n-vector D in increasing (ID='I') or decreasing
 * (ID='D') order, and rearranges the elements of an n-vector E using the
 * same permutations.
 *
 * Uses Quick Sort, but reverts to Insertion sort on arrays of length <= 20.
 * Stack dimension limits N to about 2^32.
 *
 * Based on LAPACK DLASRT, but applies to E the same interchanges used for D.
 */

#include "slicot.h"
#include <ctype.h>

#define SELECT_THRESHOLD 20
#define STACK_SIZE 32

i32 ma02rd(const char id, i32 n, f64 *d, f64 *e) {
    char id_upper = toupper((unsigned char)id);
    i32 dir = -1;
    i32 info = 0;

    if (id_upper == 'D') {
        dir = 0;
    } else if (id_upper == 'I') {
        dir = 1;
    }

    if (dir == -1) {
        info = -1;
        return info;
    }
    if (n < 0) {
        info = -2;
        return info;
    }

    if (n <= 1) {
        return 0;
    }

    i32 stack[2][STACK_SIZE];
    i32 stkpnt = 0;
    stack[0][0] = 0;
    stack[1][0] = n - 1;

    while (stkpnt >= 0) {
        i32 start = stack[0][stkpnt];
        i32 endd = stack[1][stkpnt];
        stkpnt--;

        i32 len = endd - start;

        if (len <= SELECT_THRESHOLD && len > 0) {
            if (dir == 0) {
                for (i32 i = start + 1; i <= endd; i++) {
                    for (i32 j = i; j > start; j--) {
                        if (d[j] > d[j - 1]) {
                            f64 tmp = d[j];
                            d[j] = d[j - 1];
                            d[j - 1] = tmp;
                            tmp = e[j];
                            e[j] = e[j - 1];
                            e[j - 1] = tmp;
                        } else {
                            break;
                        }
                    }
                }
            } else {
                for (i32 i = start + 1; i <= endd; i++) {
                    for (i32 j = i; j > start; j--) {
                        if (d[j] < d[j - 1]) {
                            f64 tmp = d[j];
                            d[j] = d[j - 1];
                            d[j - 1] = tmp;
                            tmp = e[j];
                            e[j] = e[j - 1];
                            e[j - 1] = tmp;
                        } else {
                            break;
                        }
                    }
                }
            }
        } else if (len > SELECT_THRESHOLD) {
            f64 d1 = d[start];
            f64 d2 = d[endd];
            i32 mid = (start + endd) / 2;
            f64 d3 = d[mid];

            f64 dmnmx;
            if (d1 < d2) {
                if (d3 < d1) {
                    dmnmx = d1;
                } else if (d3 < d2) {
                    dmnmx = d3;
                } else {
                    dmnmx = d2;
                }
            } else {
                if (d3 < d2) {
                    dmnmx = d2;
                } else if (d3 < d1) {
                    dmnmx = d3;
                } else {
                    dmnmx = d1;
                }
            }

            i32 i, j;
            if (dir == 0) {
                i = start - 1;
                j = endd + 1;
                while (1) {
                    do {
                        j--;
                    } while (d[j] < dmnmx);
                    do {
                        i++;
                    } while (d[i] > dmnmx);
                    if (i < j) {
                        f64 tmp = d[i];
                        d[i] = d[j];
                        d[j] = tmp;
                        tmp = e[i];
                        e[i] = e[j];
                        e[j] = tmp;
                    } else {
                        break;
                    }
                }
            } else {
                i = start - 1;
                j = endd + 1;
                while (1) {
                    do {
                        j--;
                    } while (d[j] > dmnmx);
                    do {
                        i++;
                    } while (d[i] < dmnmx);
                    if (i < j) {
                        f64 tmp = d[i];
                        d[i] = d[j];
                        d[j] = tmp;
                        tmp = e[i];
                        e[i] = e[j];
                        e[j] = tmp;
                    } else {
                        break;
                    }
                }
            }

            if (j - start > endd - j - 1) {
                stkpnt++;
                stack[0][stkpnt] = start;
                stack[1][stkpnt] = j;
                stkpnt++;
                stack[0][stkpnt] = j + 1;
                stack[1][stkpnt] = endd;
            } else {
                stkpnt++;
                stack[0][stkpnt] = j + 1;
                stack[1][stkpnt] = endd;
                stkpnt++;
                stack[0][stkpnt] = start;
                stack[1][stkpnt] = j;
            }
        }
    }

    return 0;
}
