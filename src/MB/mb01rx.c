// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"
#include <stdbool.h>
#include <ctype.h>

i32 slicot_mb01rx(
    char side,
    char uplo,
    char trans,
    i32 m,
    i32 n,
    f64 alpha,
    f64 beta,
    f64 *r,
    i32 ldr,
    const f64 *a,
    i32 lda,
    const f64 *b,
    i32 ldb
) {
    const f64 zero = 0.0;
    const f64 one = 1.0;

    side = (char)toupper((unsigned char)side);
    uplo = (char)toupper((unsigned char)uplo);
    trans = (char)toupper((unsigned char)trans);

    bool lside = (side == 'L');
    bool luplo = (uplo == 'U');
    bool ltrans = (trans == 'T' || trans == 'C');

    // Validate parameters
    i32 info = 0;

    if (!lside && side != 'R') {
        info = -1;
    } else if (!luplo && uplo != 'L') {
        info = -2;
    } else if (!ltrans && trans != 'N') {
        info = -3;
    } else if (m < 0) {
        info = -4;
    } else if (n < 0) {
        info = -5;
    } else if (ldr < (m > 0 ? m : 1)) {
        info = -9;
    } else {
        // Validate LDA
        i32 lda_min;
        if ((lside && !ltrans) || (!lside && ltrans)) {
            lda_min = m;
        } else {
            lda_min = n;
        }
        if (lda < (lda_min > 0 ? lda_min : 1)) {
            info = -11;
        }
    }

    if (info == 0) {
        // Validate LDB
        i32 ldb_min = lside ? n : m;
        if (ldb < (ldb_min > 0 ? ldb_min : 1)) {
            info = -13;
        }
    }

    if (info != 0) {
        return info;
    }

    // Quick return
    if (m == 0) {
        return 0;
    }

    // Special cases: beta=0 or n=0
    if (beta == zero || n == 0) {
        if (alpha == zero) {
            // Set triangle to zero
            SLC_DLASET(&uplo, &m, &m, &zero, &zero, r, &ldr);
        } else if (alpha != one) {
            // Scale triangle by alpha
            SLC_DLASCL(&uplo, &(i32){0}, &(i32){0}, &one, &alpha, &m, &m, r, &ldr, &info);
        }
        return 0;
    }

    // General case: beta != 0, n > 0
    // Compute triangle of R = alpha*R + beta*op(A)*B or R = alpha*R + beta*B*op(A)

    if (lside) {
        // R = alpha*R + beta*op(A)*B
        if (luplo) {
            // Upper triangle
            if (ltrans) {
                // R = alpha*R + beta*A'*B
                // For each column j: R(1:j,j) = alpha*R(1:j,j) + beta*A'(1:j,:)*B(:,j)
                //                               = alpha*R(1:j,j) + beta*A(:,1:j)'*B(:,j)
                for (i32 j = 0; j < m; j++) {
                    i32 len = j + 1;
                    SLC_DGEMV("Transpose", &n, &len, &beta, a, &lda, &b[j * ldb], &(i32){1}, &alpha, &r[j * ldr], &(i32){1});
                }
            } else {
                // R = alpha*R + beta*A*B
                // For each column j: R(1:j,j) = alpha*R(1:j,j) + beta*A(1:j,:)*B(:,j)
                for (i32 j = 0; j < m; j++) {
                    i32 len = j + 1;
                    SLC_DGEMV("NoTranspose", &len, &n, &beta, a, &lda, &b[j * ldb], &(i32){1}, &alpha, &r[j * ldr], &(i32){1});
                }
            }
        } else {
            // Lower triangle
            if (ltrans) {
                // R = alpha*R + beta*A'*B
                // For each column j: R(j:m,j) = alpha*R(j:m,j) + beta*A'(j:m,:)*B(:,j)
                //                               = alpha*R(j:m,j) + beta*A(:,j:m)'*B(:,j)
                for (i32 j = 0; j < m; j++) {
                    i32 len = m - j;
                    SLC_DGEMV("Transpose", &n, &len, &beta, &a[j * lda], &lda, &b[j * ldb], &(i32){1}, &alpha, &r[j + j * ldr], &(i32){1});
                }
            } else {
                // R = alpha*R + beta*A*B
                // For each column j: R(j:m,j) = alpha*R(j:m,j) + beta*A(j:m,:)*B(:,j)
                for (i32 j = 0; j < m; j++) {
                    i32 len = m - j;
                    SLC_DGEMV("NoTranspose", &len, &n, &beta, &a[j], &lda, &b[j * ldb], &(i32){1}, &alpha, &r[j + j * ldr], &(i32){1});
                }
            }
        }
    } else {
        // R = alpha*R + beta*B*op(A)
        if (luplo) {
            // Upper triangle
            if (ltrans) {
                // R = alpha*R + beta*B*A'
                // For each column j: R(1:j,j) = alpha*R(1:j,j) + beta*B(1:j,:)*A'(:,j)
                //                               = alpha*R(1:j,j) + beta*B(1:j,:)*A(j,:)'
                // A(j,:) is row j of A: start at a[j], stride lda
                for (i32 j = 0; j < m; j++) {
                    i32 len = j + 1;
                    SLC_DGEMV("NoTranspose", &len, &n, &beta, b, &ldb, &a[j], &lda, &alpha, &r[j * ldr], &(i32){1});
                }
            } else {
                // R = alpha*R + beta*B*A
                // For each column j: R(1:j,j) = alpha*R(1:j,j) + beta*B(1:j,:)*A(:,j)
                // A(:,j) is column j of A: start at a[j*lda], stride 1
                for (i32 j = 0; j < m; j++) {
                    i32 len = j + 1;
                    SLC_DGEMV("NoTranspose", &len, &n, &beta, b, &ldb, &a[j * lda], &(i32){1}, &alpha, &r[j * ldr], &(i32){1});
                }
            }
        } else {
            // Lower triangle
            if (ltrans) {
                // R = alpha*R + beta*B*A'
                // For each column j: R(j:m,j) = alpha*R(j:m,j) + beta*B(j:m,:)*A'(:,j)
                //                               = alpha*R(j:m,j) + beta*B(j:m,:)*A(j,:)'
                // A(j,:) is row j of A: start at a[j], stride lda
                for (i32 j = 0; j < m; j++) {
                    i32 len = m - j;
                    SLC_DGEMV("NoTranspose", &len, &n, &beta, &b[j], &ldb, &a[j], &lda, &alpha, &r[j + j * ldr], &(i32){1});
                }
            } else {
                // R = alpha*R + beta*B*A
                // For each column j: R(j:m,j) = alpha*R(j:m,j) + beta*B(j:m,:)*A(:,j)
                // A(:,j) is column j of A: start at a[j*lda], stride 1
                for (i32 j = 0; j < m; j++) {
                    i32 len = m - j;
                    SLC_DGEMV("NoTranspose", &len, &n, &beta, &b[j], &ldb, &a[j * lda], &(i32){1}, &alpha, &r[j + j * ldr], &(i32){1});
                }
            }
        }
    }

    return 0;
}

void mb01rx(const char* side, const char* uplo, const char* trans, i32 m,
            i32 n, f64 alpha, f64 beta, f64* r, i32 ldr, const f64* a,
            i32 lda, const f64* b, i32 ldb, i32* info)
{
    *info = slicot_mb01rx(side[0], uplo[0], trans[0], m, n, alpha, beta,
                          r, ldr, a, lda, b, ldb);
}
