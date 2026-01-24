/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_UD_H
#define SLICOT_UD_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Read/copy coefficients of a matrix polynomial.
 *
 * Copies the coefficients of a matrix polynomial from input data to a 3D array:
 *     P(s) = P(0) + P(1)*s + ... + P(dp-1)*s^(dp-1) + P(dp)*s^dp
 *
 * The input data is organized as (DP+1) coefficient matrices, each MP-by-NP,
 * stored row by row (matching Fortran file READ order).
 *
 * @param[in] mp Number of rows of each coefficient matrix (mp >= 1)
 * @param[in] np Number of columns of each coefficient matrix (np >= 1)
 * @param[in] dp Degree of the matrix polynomial (dp >= 0)
 * @param[in] data Input data array, dimension (mp * np * (dp + 1))
 *                 Contains coefficient matrices P(0), P(1), ..., P(dp)
 *                 stored row by row
 * @param[out] p Output array, dimension (ldp1, ldp2, dp + 1)
 *               P(i,j,k) contains coefficient of s^(k-1) for element (i,j)
 *               where i = 1..mp, j = 1..np, k = 1..dp+1
 * @param[in] ldp1 Leading dimension of P (ldp1 >= mp)
 * @param[in] ldp2 Second dimension of P (ldp2 >= np)
 * @param[out] info Exit code:
 *                  0 = success
 *                  -1 = mp < 1
 *                  -2 = np < 1
 *                  -3 = dp < 0
 *                  -6 = ldp1 < mp
 *                  -7 = ldp2 < np
 */
void ud01bd(i32 mp, i32 np, i32 dp, const f64 *data, f64 *p,
            i32 ldp1, i32 ldp2, i32 *info);

/**
 * @brief Read/construct a sparse matrix from COO format data.
 *
 * Initializes an M-by-N matrix to zero, then assigns nonzero elements
 * from sparse COO (Coordinate) format input arrays.
 *
 * The original Fortran routine reads from a file; this C implementation
 * accepts the sparse data directly as arrays of row indices, column indices,
 * and values.
 *
 * @param[in] m Number of rows of the matrix A (m >= 0)
 * @param[in] n Number of columns of the matrix A (n >= 0)
 * @param[in] nnz Number of nonzero entries to assign
 * @param[in] rows Array of row indices (1-based, Fortran convention), dimension (nnz)
 * @param[in] cols Array of column indices (1-based, Fortran convention), dimension (nnz)
 * @param[in] vals Array of values, dimension (nnz)
 * @param[out] a Output matrix, dimension (lda, n)
 *               The leading M-by-N part contains the sparse matrix
 * @param[in] lda Leading dimension of A (lda >= max(1, m))
 * @param[out] info Exit code:
 *                  0 = success
 *                  -1 = m < 0
 *                  -2 = n < 0
 *                  -3 = nnz < 0
 *                  -8 = lda < max(1, m)
 *                  1 = warning: some indices were out of bounds (skipped)
 */
void ud01dd(i32 m, i32 n, i32 nnz, const i32 *rows, const i32 *cols,
            const f64 *vals, f64 *a, i32 lda, i32 *info);

/**
 * @brief Print an M-by-N real matrix row by row.
 *
 * Prints the elements of A to 7 significant figures, with column headers.
 * The matrix is printed in blocks of L columns.
 *
 * @param[in] m Number of rows of matrix A (m >= 1)
 * @param[in] n Number of columns of matrix A (n >= 1)
 * @param[in] l Number of elements per line (1 <= l <= 5)
 * @param[in] a Array of dimension (lda, n), the matrix to print
 * @param[in] lda Leading dimension of A (lda >= m)
 * @param[in] text Title caption (up to 72 characters)
 * @param[out] output Output buffer for formatted string
 * @param[in] output_size Size of output buffer
 * @return Exit code:
 *         0 = success
 *         -1 = m < 1
 *         -2 = n < 1
 *         -3 = l < 1 or l > 5
 *         -6 = lda < m
 *         -100 = output buffer too small
 */
i32 ud01md(i32 m, i32 n, i32 l, const f64 *a, i32 lda,
           const char *text, char *output, i32 output_size);

/**
 * @brief Read sparse matrix polynomial coefficients.
 *
 * Constructs a matrix polynomial from sparse element specifications:
 *     P(s) = P(0) + P(1)*s + ... + P(dp-1)*s^(dp-1) + P(dp)*s^dp
 *
 * Each nonzero polynomial element P_{i,j}(s) is specified by its row index,
 * column index, degree, and coefficients. All unspecified elements are zero.
 *
 * @param[in] mp Number of rows of each coefficient matrix (mp >= 1)
 * @param[in] np Number of columns of each coefficient matrix (np >= 1)
 * @param[in] dp Maximum degree of the matrix polynomial (dp >= 0)
 * @param[in] nelem Number of nonzero polynomial elements
 * @param[in] rows Array of row indices (1-based), dimension (nelem)
 * @param[in] cols Array of column indices (1-based), dimension (nelem)
 * @param[in] degrees Array of polynomial degrees for each element, dimension (nelem)
 * @param[in] coeffs Concatenated coefficients for all elements
 *                   Element e has (degrees[e]+1) coefficients: c_0, c_1, ..., c_d
 *                   representing P_{i,j}(s) = c_0 + c_1*s + ... + c_d*s^d
 * @param[out] p Output array, dimension (ldp1, ldp2, dp + 1)
 *               P(i,j,k) contains coefficient of s^(k-1) for element (i,j)
 * @param[in] ldp1 Leading dimension of P (ldp1 >= mp)
 * @param[in] ldp2 Second dimension of P (ldp2 >= np)
 * @param[out] info Exit code:
 *                  0 = success
 *                  -1 = mp < 1
 *                  -2 = np < 1
 *                  -3 = dp < 0
 *                  -10 = ldp1 < mp
 *                  -11 = ldp2 < np
 *                  1 = warning: some indices/degrees out of bounds (skipped)
 */
void ud01cd(i32 mp, i32 np, i32 dp, i32 nelem, const i32 *rows,
            const i32 *cols, const i32 *degrees, const f64 *coeffs,
            f64 *p, i32 ldp1, i32 ldp2, i32 *info);

/**
 * @brief Print an M-by-N complex matrix row by row.
 *
 * Prints the elements of A to 7 significant figures, with column headers.
 * The matrix is printed in blocks of L columns.
 *
 * @param[in] m Number of rows of matrix A (m >= 1)
 * @param[in] n Number of columns of matrix A (n >= 1)
 * @param[in] l Number of elements per line (1 <= l <= 3)
 * @param[in] a Array of dimension (lda, n), the complex matrix to print
 * @param[in] lda Leading dimension of A (lda >= m)
 * @param[in] text Title caption (up to 72 characters)
 * @param[out] output Output buffer for formatted string
 * @param[in] output_size Size of output buffer
 * @return Exit code:
 *         0 = success
 *         -1 = m < 1
 *         -2 = n < 1
 *         -3 = l < 1 or l > 3
 *         -5 = lda < m
 *         -6 = output buffer too small
 */
i32 ud01mz(i32 m, i32 n, i32 l, const c128 *a, i32 lda,
           const char *text, char *output, i32 output_size);

/**
 * @brief Print the coefficient matrices of a matrix polynomial.
 *
 * Prints the MP-by-NP coefficient matrices of a matrix polynomial:
 *     P(s) = P(0) + P(1)*s + ... + P(dp-1)*s^(dp-1) + P(dp)*s^dp
 *
 * The elements are output to 7 significant figures. Each coefficient
 * matrix is printed with a title showing the degree (TEXT(k)).
 *
 * @param[in] mp Number of rows of each coefficient matrix (mp >= 1)
 * @param[in] np Number of columns of each coefficient matrix (np >= 1)
 * @param[in] dp Degree of the matrix polynomial (dp >= 0)
 * @param[in] l Number of elements per line (1 <= l <= 5)
 * @param[in] p Array of dimension (ldp1, ldp2, dp + 1)
 *              P(i,j,k) contains coefficient of s^(k-1) for element (i,j)
 * @param[in] ldp1 Leading dimension of P (ldp1 >= mp)
 * @param[in] ldp2 Second dimension of P (ldp2 >= np)
 * @param[in] text Title caption (up to 72 characters)
 *                 If empty/blank, coefficient matrices separated by blank line
 * @param[out] output Output buffer for formatted string
 * @param[in] output_size Size of output buffer
 * @return Exit code:
 *         0 = success
 *         -1 = mp < 1
 *         -2 = np < 1
 *         -3 = dp < 0
 *         -4 = l < 1 or l > 5
 *         -6 = ldp1 < mp
 *         -7 = ldp2 < np
 *         -100 = output buffer too small
 */
i32 ud01nd(i32 mp, i32 np, i32 dp, i32 l, const f64 *p, i32 ldp1, i32 ldp2,
           const char *text, char *output, i32 output_size);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_UD_H */
