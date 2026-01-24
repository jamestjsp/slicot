/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

void mb01zd(const char* side, const char* uplo, const char* transt,
            const char* diag, i32 m, i32 n, i32 l, f64 alpha,
            const f64* t, i32 ldt, f64* h, i32 ldh, i32* info)
{
    const f64 one = 1.0;
    const f64 zero = 0.0;

    bool lside = (side[0] == 'L' || side[0] == 'l');
    bool upper = (uplo[0] == 'U' || uplo[0] == 'u');
    bool trans = (transt[0] == 'T' || transt[0] == 't' ||
                  transt[0] == 'C' || transt[0] == 'c');
    bool nounit = (diag[0] == 'N' || diag[0] == 'n');

    i32 nrowt = lside ? m : n;
    i32 m2 = upper ? m : n;

    *info = 0;
    if (!(lside || side[0] == 'R' || side[0] == 'r')) {
        *info = -1;
    } else if (!(upper || uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -2;
    } else if (!(trans || transt[0] == 'N' || transt[0] == 'n')) {
        *info = -3;
    } else if (!(nounit || diag[0] == 'U' || diag[0] == 'u')) {
        *info = -4;
    } else if (m < 0) {
        *info = -5;
    } else if (n < 0) {
        *info = -6;
    } else if (l < 0 || l > (m2 > 1 ? m2 - 1 : 0)) {
        *info = -7;
    } else if (ldt < (nrowt > 1 ? nrowt : 1)) {
        *info = -10;
    } else if (ldh < (m > 1 ? m : 1)) {
        *info = -12;
    }

    if (*info != 0) {
        return;
    }

    if (m == 0 || n == 0) {
        return;
    }

    if (alpha == zero) {
        for (i32 j = 0; j < n; j++) {
            i32 i1, i2;
            if (upper) {
                i1 = 0;
                i2 = (j + l + 1 < m) ? (j + l + 1) : m;
            } else {
                i1 = (j - l > 0) ? (j - l) : 0;
                i2 = m;
            }
            for (i32 i = i1; i < i2; i++) {
                h[i + j * ldh] = zero;
            }
        }
        return;
    }

    if (lside) {
        if (!trans) {
            if (upper) {
                for (i32 j = 0; j < n; j++) {
                    i32 kmax = (j + l + 1 < m) ? (j + l + 1) : m;
                    for (i32 k = 0; k < kmax; k++) {
                        if (h[k + j * ldh] != zero) {
                            f64 temp = alpha * h[k + j * ldh];
                            i32 km1 = k;
                            SLC_DAXPY(&km1, &temp, &t[0 + k * ldt], &(i32){1},
                                      &h[0 + j * ldh], &(i32){1});
                            if (nounit) {
                                temp = temp * t[k + k * ldt];
                            }
                            h[k + j * ldh] = temp;
                        }
                    }
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    i32 kmin = (j - l > 0) ? (j - l) : 0;
                    for (i32 k = m - 1; k >= kmin; k--) {
                        if (h[k + j * ldh] != zero) {
                            f64 temp = alpha * h[k + j * ldh];
                            h[k + j * ldh] = temp;
                            if (nounit) {
                                h[k + j * ldh] = h[k + j * ldh] * t[k + k * ldt];
                            }
                            i32 nmk = m - k - 1;
                            if (nmk > 0) {
                                SLC_DAXPY(&nmk, &temp, &t[(k + 1) + k * ldt], &(i32){1},
                                          &h[(k + 1) + j * ldh], &(i32){1});
                            }
                        }
                    }
                }
            }
        } else {
            if (upper) {
                for (i32 j = 0; j < n; j++) {
                    i32 i1 = j + l;
                    for (i32 i = m - 1; i >= 0; i--) {
                        f64 temp;
                        if (i > i1) {
                            i32 len = i1 + 1;
                            temp = SLC_DDOT(&len, &t[0 + i * ldt], &(i32){1},
                                            &h[0 + j * ldh], &(i32){1});
                        } else {
                            temp = h[i + j * ldh];
                            if (nounit) {
                                temp = temp * t[i + i * ldt];
                            }
                            i32 im1 = i;
                            if (im1 > 0) {
                                temp = temp + SLC_DDOT(&im1, &t[0 + i * ldt], &(i32){1},
                                                       &h[0 + j * ldh], &(i32){1});
                            }
                        }
                        h[i + j * ldh] = alpha * temp;
                    }
                }
            } else {
                i32 jmax = (m + l < n) ? (m + l) : n;
                for (i32 j = 0; j < jmax; j++) {
                    i32 i1 = j - l;
                    for (i32 i = 0; i < m; i++) {
                        f64 temp;
                        if (i < i1) {
                            i32 len = m - i1;
                            temp = SLC_DDOT(&len, &t[i1 + i * ldt], &(i32){1},
                                            &h[i1 + j * ldh], &(i32){1});
                        } else {
                            temp = h[i + j * ldh];
                            if (nounit) {
                                temp = temp * t[i + i * ldt];
                            }
                            i32 nmi = m - i - 1;
                            if (nmi > 0) {
                                temp = temp + SLC_DDOT(&nmi, &t[(i + 1) + i * ldt], &(i32){1},
                                                       &h[(i + 1) + j * ldh], &(i32){1});
                            }
                        }
                        h[i + j * ldh] = alpha * temp;
                    }
                }
            }
        }
    } else {
        if (!trans) {
            if (upper) {
                for (i32 j = n - 1; j >= 0; j--) {
                    i32 i2 = (j + l + 1 < m) ? (j + l + 1) : m;
                    f64 temp = alpha;
                    if (nounit) {
                        temp = temp * t[j + j * ldt];
                    }
                    SLC_DSCAL(&i2, &temp, &h[0 + j * ldh], &(i32){1});
                    for (i32 k = 0; k < j; k++) {
                        f64 coef = alpha * t[k + j * ldt];
                        SLC_DAXPY(&i2, &coef, &h[0 + k * ldh], &(i32){1},
                                  &h[0 + j * ldh], &(i32){1});
                    }
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    i32 i1 = (j - l > 0) ? (j - l) : 0;
                    i32 len = m - i1;
                    f64 temp = alpha;
                    if (nounit) {
                        temp = temp * t[j + j * ldt];
                    }
                    SLC_DSCAL(&len, &temp, &h[i1 + j * ldh], &(i32){1});
                    for (i32 k = j + 1; k < n; k++) {
                        f64 coef = alpha * t[k + j * ldt];
                        SLC_DAXPY(&len, &coef, &h[i1 + k * ldh], &(i32){1},
                                  &h[i1 + j * ldh], &(i32){1});
                    }
                }
            }
        } else {
            if (upper) {
                i32 m2val = (n + l < m) ? (n + l) : m;
                for (i32 k = 0; k < n; k++) {
                    i32 i1 = (k + l + 1 < m) ? (k + l + 1) : m;
                    i32 i2 = (k + l + 1 < m2val) ? (k + l + 1) : m2val;
                    for (i32 j = 0; j < k; j++) {
                        if (t[j + k * ldt] != zero) {
                            f64 temp = alpha * t[j + k * ldt];
                            SLC_DAXPY(&i1, &temp, &h[0 + k * ldh], &(i32){1},
                                      &h[0 + j * ldh], &(i32){1});
                            for (i32 i = i1; i < i2; i++) {
                                h[i + j * ldh] = temp * h[i + k * ldh];
                            }
                        }
                    }
                    f64 temp = alpha;
                    if (nounit) {
                        temp = temp * t[k + k * ldt];
                    }
                    if (temp != one) {
                        SLC_DSCAL(&i2, &temp, &h[0 + k * ldh], &(i32){1});
                    }
                }
            } else {
                for (i32 k = n - 1; k >= 0; k--) {
                    i32 i1 = (k - l > 0) ? (k - l) : 0;
                    i32 i2 = (k - l + 1 > 0) ? (k - l + 1) : 0;
                    i32 m2val = (i2 - 1 > 0) ? (i2 - 1) : 0;
                    i32 len_main = m - i2;
                    for (i32 j = k + 1; j < n; j++) {
                        if (t[j + k * ldt] != zero) {
                            f64 temp = alpha * t[j + k * ldt];
                            if (len_main > 0) {
                                SLC_DAXPY(&len_main, &temp, &h[i2 + k * ldh], &(i32){1},
                                          &h[i2 + j * ldh], &(i32){1});
                            }
                            for (i32 i = i1; i < (i32)m2val; i++) {
                                h[i + j * ldh] = temp * h[i + k * ldh];
                            }
                        }
                    }
                    f64 temp = alpha;
                    if (nounit) {
                        temp = temp * t[k + k * ldt];
                    }
                    if (temp != one) {
                        i32 len = m - i1;
                        SLC_DSCAL(&len, &temp, &h[i1 + k * ldh], &(i32){1});
                    }
                }
            }
        }
    }
}
