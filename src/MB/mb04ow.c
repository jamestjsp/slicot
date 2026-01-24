/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"

/**
 * @brief Performs a QR factorization update.
 *
 * MB04OW performs the QR factorization
 *
 *      ( U  ) = Q*( R ),  where  U = ( U1  U2 ),  R = ( R1  R2 ),
 *      ( x' )     ( 0 )              ( 0   T  )       ( 0   R3 )
 *
 * where U and R are (m+n)-by-(m+n) upper triangular matrices, x is
 * an m+n element vector, U1 is m-by-m, T is n-by-n, stored
 * separately, and Q is an (m+n+1)-by-(m+n+1) orthogonal matrix.
 *
 * The matrix ( U1 U2 ) must be supplied in the m-by-(m+n) upper
 * trapezoidal part of the array A and this is overwritten by the
 * corresponding part ( R1 R2 ) of R. The remaining upper triangular
 * part of R, R3, is overwritten on the array T.
 *
 * The transformations performed are also applied to the (m+n+1)-by-p
 * matrix ( B' C' d )' (' denotes transposition), where B, C, and d'
 * are m-by-p, n-by-p, and 1-by-p matrices, respectively.
 *
 * @param[in] m The number of rows of the matrix ( U1  U2 ). M >= 0.
 * @param[in] n The order of the matrix T. N >= 0.
 * @param[in] p The number of columns of the matrices B and C. P >= 0.
 * @param[in,out] a Array of dimension (LDA, N+M). On entry, the leading M-by-(M+N) 
 *                  upper trapezoidal part contains ( U1 U2 ). On exit, ( R1 R2 ).
 * @param[in] lda The leading dimension of the array A. LDA >= max(1,M).
 * @param[in,out] t Array of dimension (LDT, N). On entry, the leading N-by-N 
 *                  upper triangular part contains T. On exit, R3.
 * @param[in] ldt The leading dimension of the array T. LDT >= max(1,N).
 * @param[in,out] x Array of dimension (1+(M+N-1)*INCX). On entry, the vector x. 
 *                  On exit, the content is changed (destroyed).
 * @param[in] incx The increment for the elements of X. INCX > 0.
 * @param[in,out] b Array of dimension (LDB, P). On entry, B. On exit, transformed B.
 * @param[in] ldb The leading dimension of the array B. LDB >= max(1,M) if P > 0.
 * @param[in,out] c Array of dimension (LDC, P). On entry, C. On exit, transformed C.
 * @param[in] ldc The leading dimension of the array C. LDC >= max(1,N) if P > 0.
 * @param[in,out] d Array of dimension (1+(P-1)*INCD). On entry, the vector d. 
 *                  On exit, transformed d.
 * @param[in] incd The increment for the elements of D. INCD > 0.
 */
void mb04ow(i32 m, i32 n, i32 p, f64 *a, i32 lda, f64 *t, i32 ldt, 
            f64 *x, i32 incx, f64 *b, i32 ldb, f64 *c, i32 ldc, 
            f64 *d, i32 incd)
{
    /* Local variables */
    f64 ci, si, temp;
    i32 i, ix, mn;
    i32 inc_one = 1;

    /* Executable Statements */
    /* For efficiency reasons, the parameters are not checked. */

    mn = m + n;

    if (incx > 1) {
        /* Code for increment INCX > 1. */
        ix = 0; /* 0-based index for X */
        
        if (m > 0) {
            for (i = 0; i < m - 1; i++) {
                SLC_DLARTG(&a[i + i * lda], &x[ix], &ci, &si, &temp);
                a[i + i * lda] = temp;
                ix += incx;
                
                /* Apply rotation to row of A and vector X */
                /* A(I, I+1) is &a[i + (i+1)*lda] */
                /* Length is MN - (I+1) = mn - 1 - i */
                i32 len = mn - 1 - i;
                SLC_DROT(&len, &a[i + (i + 1) * lda], &lda, &x[ix], &incx, &ci, &si);
                
                if (p > 0) {
                    SLC_DROT(&p, &b[i], &ldb, d, &incd, &ci, &si);
                }
            }
            
            /* Last row of U1/A */
            SLC_DLARTG(&a[(m - 1) + (m - 1) * lda], &x[ix], &ci, &si, &temp);
            a[(m - 1) + (m - 1) * lda] = temp;
            ix += incx;
            
            if (n > 0) {
                SLC_DROT(&n, &a[(m - 1) + m * lda], &lda, &x[ix], &incx, &ci, &si);
            }
            if (p > 0) {
                SLC_DROT(&p, &b[m - 1], &ldb, d, &incd, &ci, &si);
            }
        }
        
        if (n > 0) {
            for (i = 0; i < n - 1; i++) {
                SLC_DLARTG(&t[i + i * ldt], &x[ix], &ci, &si, &temp);
                t[i + i * ldt] = temp;
                ix += incx;
                
                i32 len = n - 1 - i;
                SLC_DROT(&len, &t[i + (i + 1) * ldt], &ldt, &x[ix], &incx, &ci, &si);
                
                if (p > 0) {
                    SLC_DROT(&p, &c[i], &ldc, d, &incd, &ci, &si);
                }
            }
            
            SLC_DLARTG(&t[(n - 1) + (n - 1) * ldt], &x[ix], &ci, &si, &temp);
            t[(n - 1) + (n - 1) * ldt] = temp;
            
            if (p > 0) {
                SLC_DROT(&p, &c[n - 1], &ldc, d, &incd, &ci, &si);
            }
        }
        
    } else if (incx == 1) {
        /* Code for increment INCX = 1. */
        if (m > 0) {
            for (i = 0; i < m - 1; i++) {
                /* X[i] corresponds to X(I) */
                SLC_DLARTG(&a[i + i * lda], &x[i], &ci, &si, &temp);
                a[i + i * lda] = temp;
                
                i32 len = mn - 1 - i;
                /* X[i+1] is next element */
                SLC_DROT(&len, &a[i + (i + 1) * lda], &lda, &x[i + 1], &inc_one, &ci, &si);
                
                if (p > 0) {
                    SLC_DROT(&p, &b[i], &ldb, d, &incd, &ci, &si);
                }
            }
            
            /* Last row of U1 */
            SLC_DLARTG(&a[(m - 1) + (m - 1) * lda], &x[m - 1], &ci, &si, &temp);
            a[(m - 1) + (m - 1) * lda] = temp;
            
            if (n > 0) {
                SLC_DROT(&n, &a[(m - 1) + m * lda], &lda, &x[m], &inc_one, &ci, &si);
            }
            if (p > 0) {
                SLC_DROT(&p, &b[m - 1], &ldb, d, &incd, &ci, &si);
            }
        }
        
        if (n > 0) {
            /* IX starts at m */
            ix = m;
            
            for (i = 0; i < n - 1; i++) {
                SLC_DLARTG(&t[i + i * ldt], &x[ix], &ci, &si, &temp);
                t[i + i * ldt] = temp;
                ix++;
                
                i32 len = n - 1 - i;
                SLC_DROT(&len, &t[i + (i + 1) * ldt], &ldt, &x[ix], &inc_one, &ci, &si);
                
                if (p > 0) {
                    SLC_DROT(&p, &c[i], &ldc, d, &incd, &ci, &si);
                }
            }
            
            SLC_DLARTG(&t[(n - 1) + (n - 1) * ldt], &x[ix], &ci, &si, &temp);
            t[(n - 1) + (n - 1) * ldt] = temp;
            
            if (p > 0) {
                SLC_DROT(&p, &c[n - 1], &ldc, d, &incd, &ci, &si);
            }
        }
    }
}
