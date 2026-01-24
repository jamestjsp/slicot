/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

static inline i32 max_i32(i32 a, i32 b) { return a > b ? a : b; }

void ag07bd(const char* jobe, i32 n, i32 m,
            const f64* a, i32 lda, const f64* e, i32 lde,
            const f64* b, i32 ldb, const f64* c, i32 ldc,
            const f64* d, i32 ldd,
            f64* ai, i32 ldai, f64* ei, i32 ldei,
            f64* bi, i32 ldbi, f64* ci, i32 ldci,
            f64* di, i32 lddi, i32* info) {

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    bool unite = (jobe[0] == 'I' || jobe[0] == 'i');
    i32 nm = n + m;

    *info = 0;

    if (!(jobe[0] == 'G' || jobe[0] == 'g') && !unite) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (lda < max_i32(1, n)) {
        *info = -5;
    } else if (lde < 1 || (!unite && lde < n)) {
        *info = -7;
    } else if (ldb < max_i32(1, n)) {
        *info = -9;
    } else if (ldc < max_i32(1, m)) {
        *info = -11;
    } else if (ldd < max_i32(1, m)) {
        *info = -13;
    } else if (ldai < max_i32(1, nm)) {
        *info = -15;
    } else if (ldei < max_i32(1, nm)) {
        *info = -17;
    } else if (ldbi < max_i32(1, nm)) {
        *info = -19;
    } else if (ldci < max_i32(1, m)) {
        *info = -21;
    } else if (lddi < max_i32(1, m)) {
        *info = -23;
    }

    if (*info != 0) {
        return;
    }

    if (m == 0) {
        return;
    }

    sl_int n_int = n;
    sl_int m_int = m;
    sl_int lda_int = lda;
    sl_int lde_int = lde;
    sl_int ldb_int = ldb;
    sl_int ldc_int = ldc;
    sl_int ldd_int = ldd;
    sl_int ldai_int = ldai;
    sl_int ldei_int = ldei;
    sl_int ldbi_int = ldbi;
    sl_int ldci_int = ldci;
    sl_int lddi_int = lddi;
    sl_int nm_int = nm;

    SLC_DLACPY("Full", &n_int, &n_int, a, &lda_int, ai, &ldai_int);

    SLC_DLACPY("Full", &m_int, &n_int, c, &ldc_int, &ai[n], &ldai_int);

    SLC_DLACPY("Full", &n_int, &m_int, b, &ldb_int, &ai[n * ldai], &ldai_int);

    SLC_DLACPY("Full", &m_int, &m_int, d, &ldd_int, &ai[n + n * ldai], &ldai_int);

    if (unite) {
        SLC_DLASET("Full", &nm_int, &n_int, &ZERO, &ONE, ei, &ldei_int);
    } else {
        SLC_DLACPY("Full", &n_int, &n_int, e, &lde_int, ei, &ldei_int);
        SLC_DLASET("Full", &m_int, &n_int, &ZERO, &ZERO, &ei[n], &ldei_int);
    }
    SLC_DLASET("Full", &nm_int, &m_int, &ZERO, &ZERO, &ei[n * ldei], &ldei_int);

    SLC_DLASET("Full", &n_int, &m_int, &ZERO, &ZERO, bi, &ldbi_int);
    f64 neg_one = -ONE;
    SLC_DLASET("Full", &m_int, &m_int, &ZERO, &neg_one, &bi[n], &ldbi_int);

    SLC_DLASET("Full", &m_int, &n_int, &ZERO, &ZERO, ci, &ldci_int);
    SLC_DLASET("Full", &m_int, &m_int, &ZERO, &ONE, &ci[n * ldci], &ldci_int);

    SLC_DLASET("Full", &m_int, &m_int, &ZERO, &ZERO, di, &lddi_int);
}
