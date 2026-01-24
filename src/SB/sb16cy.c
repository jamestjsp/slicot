/**
 * @file sb16cy.c
 * @brief Cholesky factors of Grammians for coprime factors of state-feedback controller.
 */

#include "slicot.h"
#include "slicot_blas.h"

void sb16cy(
    const char* dico,
    const char* jobcf,
    const i32 n,
    const i32 m,
    const i32 p,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* c,
    const i32 ldc,
    const f64* f,
    const i32 ldf,
    const f64* g,
    const i32 ldg,
    f64* scalec,
    f64* scaleo,
    f64* s,
    const i32 lds,
    f64* r,
    const i32 ldr,
    f64* dwork,
    const i32 ldwork,
    i32* info)
{
    const f64 one = 1.0;

    bool discr = (*dico == 'D' || *dico == 'd');
    bool leftw = (*jobcf == 'L' || *jobcf == 'l');

    *info = 0;

    i32 mp;
    if (leftw) {
        mp = m;
    } else {
        mp = p;
    }

    i32 max_n_mp = n > mp ? n : mp;
    i32 min_n_mp = n < mp ? n : mp;
    i32 lw = n * (n + max_n_mp + min_n_mp + 6);

    if (!(*dico == 'C' || *dico == 'c' || discr)) {
        *info = -1;
    } else if (!(leftw || *jobcf == 'R' || *jobcf == 'r')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (p < 0) {
        *info = -5;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -7;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -9;
    } else if (ldc < (1 > p ? 1 : p)) {
        *info = -11;
    } else if (ldf < (1 > m ? 1 : m)) {
        *info = -13;
    } else if (ldg < (1 > n ? 1 : n)) {
        *info = -15;
    } else if (lds < (1 > n ? 1 : n)) {
        *info = -19;
    } else if (ldr < (1 > n ? 1 : n)) {
        *info = -21;
    } else if (ldwork < (1 > lw ? 1 : lw)) {
        *info = -23;
    }

    if (*info != 0) {
        return;
    }

    i32 min_nmp = n < m ? n : m;
    min_nmp = min_nmp < p ? min_nmp : p;
    if (min_nmp == 0) {
        *scalec = one;
        *scaleo = one;
        if (dwork) {
            dwork[0] = one;
        }
        return;
    }

    i32 kaw = 0;
    i32 ku = kaw + n * n;
    i32 max_n_mp_ldu = n > mp ? n : mp;
    i32 kwr = ku + n * max_n_mp_ldu;
    i32 kwi = kwr + n;
    i32 kw = kwi + n;

    SLC_DLACPY("Full", &n, &n, a, &lda, &dwork[kaw], &n);
    SLC_DGEMM("N", "N", &n, &n, &p, &one, g, &ldg, c, &ldc, &one, &dwork[kaw], &n);

    i32 ldu, me;
    if (leftw) {
        ldu = n > m ? n : m;
        me = m;
        SLC_DLACPY("Full", &m, &n, f, &ldf, &dwork[ku], &ldu);
    } else {
        ldu = n > p ? n : p;
        me = p;
        SLC_DLACPY("Full", &p, &n, c, &ldc, &dwork[ku], &ldu);
    }

    i32 ierr;
    char jobfac_n = 'N';
    char trans_n = 'N';
    i32 ldwork_sb03od = ldwork - kw;
    sb03od(dico, &jobfac_n, &trans_n, n, me, &dwork[kaw], n,
           r, ldr, &dwork[ku], ldu, scaleo, &dwork[kwr],
           &dwork[kwi], &dwork[kw], ldwork_sb03od, &ierr);
    if (ierr != 0) {
        if (ierr == 2) {
            *info = 2;
        } else if (ierr == 1) {
            *info = 4;
        } else if (ierr == 6) {
            *info = 1;
        }
        return;
    }

    i32 wrkopt = (i32)dwork[kw] + kw;

    // Zero the lower triangle of R (which has Schur Q data), then copy upper from dwork
    for (i32 j = 0; j < n; j++) {
        for (i32 i = j + 1; i < n; i++) {
            r[i + j * ldr] = 0.0;
        }
    }
    SLC_DLACPY("Upper", &n, &n, &dwork[ku], &ldu, r, &ldr);

    SLC_DLACPY("Full", &n, &n, a, &lda, &dwork[kaw], &n);
    SLC_DGEMM("N", "N", &n, &n, &m, &one, b, &ldb, f, &ldf, &one, &dwork[kaw], &n);

    ldu = n;
    if (leftw) {
        me = m;
        SLC_DLACPY("Full", &n, &m, b, &ldb, &dwork[ku], &ldu);
    } else {
        me = p;
        SLC_DLACPY("Full", &n, &p, g, &ldg, &dwork[ku], &ldu);
    }

    char trans_t = 'T';
    ldwork_sb03od = ldwork - kw;
    sb03od(dico, &jobfac_n, &trans_t, n, me, &dwork[kaw], n,
           s, lds, &dwork[ku], ldu, scalec, &dwork[kwr],
           &dwork[kwi], &dwork[kw], ldwork_sb03od, &ierr);
    if (ierr != 0) {
        if (ierr == 2) {
            *info = 3;
        } else if (ierr == 1) {
            *info = 5;
        } else if (ierr == 6) {
            *info = 1;
        }
        return;
    }

    i32 wrkopt2 = (i32)dwork[kw] + kw;
    if (wrkopt2 > wrkopt) {
        wrkopt = wrkopt2;
    }

    // Zero the lower triangle of S (which has Schur Q data), then copy upper from dwork
    for (i32 j = 0; j < n; j++) {
        for (i32 i = j + 1; i < n; i++) {
            s[i + j * lds] = 0.0;
        }
    }
    SLC_DLACPY("Upper", &n, &n, &dwork[ku], &ldu, s, &lds);

    dwork[0] = (f64)wrkopt;
}
