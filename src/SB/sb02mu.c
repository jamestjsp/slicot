/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB02MU - Hamiltonian/Symplectic Matrix Construction
 *
 * Constructs the 2n-by-2n Hamiltonian or symplectic matrix S associated
 * to the linear-quadratic optimization problem.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <ctype.h>

void sb02mu(
    const char* dico_str,
    const char* hinv_str,
    const char* uplo_str,
    const i32 n,
    f64* a,
    const i32 lda,
    const f64* g,
    const i32 ldg,
    const f64* q,
    const i32 ldq,
    f64* s,
    const i32 lds,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const i32 int1 = 1;

    char dico = toupper((unsigned char)dico_str[0]);
    char hinv = toupper((unsigned char)hinv_str[0]);
    char uplo = toupper((unsigned char)uplo_str[0]);

    bool discr = (dico == 'D');
    bool luplo = (uplo == 'U');
    bool lhinv;

    if (discr) {
        lhinv = (hinv == 'D');
    } else {
        lhinv = false;
    }

    i32 n2 = n + n;

    *info = 0;

    if (dico != 'C' && dico != 'D') {
        *info = -1;
    } else if (discr && hinv != 'D' && hinv != 'I') {
        *info = -2;
    } else if (uplo != 'U' && uplo != 'L') {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < 1 || (n > 0 && lda < n)) {
        *info = -6;
    } else if (ldg < 1 || (n > 0 && ldg < n)) {
        *info = -8;
    } else if (ldq < 1 || (n > 0 && ldq < n)) {
        *info = -10;
    } else if (lds < 1 || (n > 0 && lds < n2)) {
        *info = -12;
    } else {
        i32 minwrk = discr ? (4 * n > 2 ? 4 * n : 2) : 1;
        bool lquery = (ldwork == -1);

        if ((ldwork < 1 || (discr && ldwork < minwrk)) && !lquery) {
            *info = -15;
        } else if (discr) {
            i32 maxwrk;
            i32 query_info = 0;
            i32 opt_query = -1;
            SLC_DGETRI(&n, a, &lda, iwork, dwork, &opt_query, &query_info);
            maxwrk = (i32)dwork[0];
            if (maxwrk < minwrk) maxwrk = minwrk;
            dwork[0] = (f64)maxwrk;
            if (lquery) return;
        } else if (lquery) {
            dwork[0] = one;
            return;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0) {
        dwork[0] = one;
        if (discr) dwork[1] = one;
        return;
    }

    i32 np1 = n + 1;

    if (!lhinv) {
        SLC_DLACPY("F", &n, &n, a, &lda, s, &lds);

        for (i32 j = 0; j < n; j++) {
            i32 nj = n + j;
            if (luplo) {
                for (i32 i = 0; i <= j; i++) {
                    s[(n + i) + j * lds] = -q[i + j * ldq];
                }
                for (i32 i = j + 1; i < n; i++) {
                    s[(n + i) + j * lds] = -q[j + i * ldq];
                }
                for (i32 i = 0; i <= j; i++) {
                    s[i + nj * lds] = -g[i + j * ldg];
                }
                for (i32 i = j + 1; i < n; i++) {
                    s[i + nj * lds] = -g[j + i * ldg];
                }
            } else {
                for (i32 i = 0; i < j; i++) {
                    s[(n + i) + j * lds] = -q[j + i * ldq];
                }
                for (i32 i = j; i < n; i++) {
                    s[(n + i) + j * lds] = -q[i + j * ldq];
                }
                for (i32 i = 0; i < j; i++) {
                    s[i + nj * lds] = -g[j + i * ldg];
                }
                for (i32 i = j; i < n; i++) {
                    s[i + nj * lds] = -g[i + j * ldg];
                }
            }
        }

        if (!discr) {
            for (i32 j = 0; j < n; j++) {
                i32 nj = n + j;
                for (i32 i = 0; i < n; i++) {
                    s[(n + i) + nj * lds] = -a[j + i * lda];
                }
            }
            dwork[0] = one;
        }
    }

    if (discr) {
        if (lhinv) {
            for (i32 i = 0; i < n; i++) {
                SLC_DCOPY(&n, &a[i], &lda, &s[np1 - 1 + (n + i) * lds], &int1);
            }
        }

        f64 anorm = SLC_DLANGE("1", &n, &n, a, &lda, dwork);

        SLC_DGETRF(&n, &n, a, &lda, iwork, info);

        if (*info > 0) {
            dwork[1] = zero;
            return;
        }

        f64 rcond;
        SLC_DGECON("1", &n, a, &lda, &anorm, &rcond, dwork, &iwork[n], info);

        f64 eps = SLC_DLAMCH("E");
        if (rcond < eps) {
            *info = n + 1;
            dwork[1] = rcond;
            return;
        }

        if (lhinv) {
            if (luplo) {
                for (i32 j = 0; j < n - 1; j++) {
                    SLC_DCOPY(&(i32){j + 1}, &q[j * ldq], &int1, &s[np1 - 1 + j * lds], &int1);
                    SLC_DCOPY(&(i32){n - j - 1}, &q[j + (j + 1) * ldq], &ldq, &s[(np1 - 1 + j + 1) + j * lds], &int1);
                }
                SLC_DCOPY(&n, &q[(n - 1) * ldq], &int1, &s[np1 - 1 + (n - 1) * lds], &int1);
            } else {
                SLC_DCOPY(&n, q, &int1, &s[np1 - 1], &int1);
                for (i32 j = 1; j < n; j++) {
                    SLC_DCOPY(&j, &q[j], &ldq, &s[np1 - 1 + j * lds], &int1);
                    SLC_DCOPY(&(i32){n - j}, &q[j + j * ldq], &int1, &s[(np1 - 1 + j) + j * lds], &int1);
                }
            }

            SLC_DGETRS("T", &n, &n, a, &lda, iwork, &s[np1 - 1], &lds, info);

            for (i32 j = 0; j < n - 1; j++) {
                SLC_DSWAP(&(i32){n - j - 1}, &s[(np1 + j) + j * lds], &int1, &s[(n + j) + (j + 1) * lds], &lds);
            }

            if (luplo) {
                for (i32 j = 0; j < n - 1; j++) {
                    SLC_DCOPY(&(i32){j + 1}, &g[j * ldg], &int1, &s[(n + j) * lds], &int1);
                    SLC_DCOPY(&(i32){n - j - 1}, &g[j + (j + 1) * ldg], &ldg, &s[(j + 1) + (n + j) * lds], &int1);
                }
                SLC_DCOPY(&n, &g[(n - 1) * ldg], &int1, &s[(n2 - 1) * lds], &int1);
            } else {
                SLC_DCOPY(&n, g, &int1, &s[np1 - 1 + (np1 - 1) * lds], &int1);
                for (i32 j = 1; j < n; j++) {
                    SLC_DCOPY(&j, &g[j], &ldg, &s[(n + j) * lds], &int1);
                    SLC_DCOPY(&(i32){n - j}, &g[j + j * ldg], &int1, &s[j + (n + j) * lds], &int1);
                }
            }

            SLC_DGEMM("N", "N", &n, &n, &n, &one, &s[np1 - 1], &lds, &s[(np1 - 1) * lds], &lds, &one, &s[(np1 - 1) + (np1 - 1) * lds], &lds);

            SLC_DGETRS("N", &n, &n, a, &lda, iwork, &s[(np1 - 1) * lds], &lds, info);

            SLC_DGETRI(&n, a, &lda, iwork, dwork, &ldwork, info);

            SLC_DLACPY("F", &n, &n, a, &lda, s, &lds);
        } else {
            SLC_DGETRS("N", &n, &n, a, &lda, iwork, &s[(np1 - 1) * lds], &lds, info);

            for (i32 j = 0; j < n - 1; j++) {
                SLC_DSWAP(&(i32){n - j - 1}, &s[(j + 1) + (n + j) * lds], &int1, &s[j + (np1 + j) * lds], &lds);
            }

            SLC_DGEMM("N", "N", &n, &n, &n, &one, &s[(np1 - 1) * lds], &lds, &s[np1 - 1], &lds, &one, s, &lds);

            SLC_DGETRS("T", &n, &n, a, &lda, iwork, &s[np1 - 1], &lds, info);

            SLC_DGETRI(&n, a, &lda, iwork, dwork, &ldwork, info);

            for (i32 j = 0; j < n; j++) {
                SLC_DCOPY(&n, &a[j], &lda, &s[(np1 - 1) + (n + j) * lds], &int1);
            }
        }

        i32 query_info = 0;
        i32 opt_query = -1;
        SLC_DGETRI(&n, a, &lda, iwork, dwork, &opt_query, &query_info);
        i32 maxwrk = (i32)dwork[0];
        i32 minwrk = 4 * n > 2 ? 4 * n : 2;
        if (maxwrk < minwrk) maxwrk = minwrk;
        dwork[0] = (f64)maxwrk;
        dwork[1] = rcond;
    }
}
