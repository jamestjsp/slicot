/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_AG_H
#define SLICOT_AG_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Descriptor inverse of a state-space or descriptor representation.
 *
 * Computes the inverse (Ai-lambda*Ei, Bi, Ci, Di) of a given
 * descriptor system (A-lambda*E, B, C, D) using the formulas:
 *
 *     Ei = ( E  0 )    Ai = ( A  B )    Bi = (  0 )
 *          ( 0  0 )         ( C  D )         ( -I )
 *
 *     Ci = ( 0  I )    Di = 0
 *
 * The inverse system has order N+M.
 *
 * @param[in] jobe 'G' for general E matrix, 'I' for identity E
 * @param[in] n Order of square matrices A and E (n >= 0)
 * @param[in] m Number of system inputs/outputs (m >= 0)
 * @param[in] a State matrix, dimension (lda, n)
 * @param[in] lda Leading dimension of a (lda >= max(1, n))
 * @param[in] e Descriptor matrix, dimension (lde, n). Not referenced if jobe='I'
 * @param[in] lde Leading dimension of e (lde >= max(1, n) if jobe='G', lde >= 1 if jobe='I')
 * @param[in] b Input matrix, dimension (ldb, m)
 * @param[in] ldb Leading dimension of b (ldb >= max(1, n))
 * @param[in] c Output matrix, dimension (ldc, n)
 * @param[in] ldc Leading dimension of c (ldc >= max(1, m))
 * @param[in] d Feedthrough matrix, dimension (ldd, m)
 * @param[in] ldd Leading dimension of d (ldd >= max(1, m))
 * @param[out] ai State matrix of inverse, dimension (ldai, n+m)
 * @param[in] ldai Leading dimension of ai (ldai >= max(1, n+m))
 * @param[out] ei Descriptor matrix of inverse, dimension (ldei, n+m)
 * @param[in] ldei Leading dimension of ei (ldei >= max(1, n+m))
 * @param[out] bi Input matrix of inverse, dimension (ldbi, m)
 * @param[in] ldbi Leading dimension of bi (ldbi >= max(1, n+m))
 * @param[out] ci Output matrix of inverse, dimension (ldci, n+m)
 * @param[in] ldci Leading dimension of ci (ldci >= max(1, m))
 * @param[out] di Feedthrough matrix of inverse, dimension (lddi, m)
 * @param[in] lddi Leading dimension of di (lddi >= max(1, m))
 * @param[out] info 0 = success, -i = parameter i invalid
 */
void ag07bd(const char* jobe, i32 n, i32 m,
            const f64* a, i32 lda, const f64* e, i32 lde,
            const f64* b, i32 ldb, const f64* c, i32 ldc,
            const f64* d, i32 ldd,
            f64* ai, i32 ldai, f64* ei, i32 ldei,
            f64* bi, i32 ldbi, f64* ci, i32 ldci,
            f64* di, i32 lddi, i32* info);

/**
 * @brief Zeros and Kronecker structure of a descriptor system pencil.
 *
 * Extracts from the system pencil
 *
 *     S(lambda) = ( A - lambda*E   B )
 *                 (      C         D )
 *
 * a regular pencil Af-lambda*Ef which has the finite Smith zeros of S(lambda)
 * as generalized eigenvalues. Also computes the orders of infinite Smith zeros
 * and determines the singular and infinite Kronecker structure.
 *
 * @param[in] equil 'S' to balance the system, 'N' for no balancing
 * @param[in] l Number of rows of A, B, E (l >= 0)
 * @param[in] n Number of columns of A, E, C (n >= 0)
 * @param[in] m Number of columns of B (m >= 0)
 * @param[in] p Number of rows of C (p >= 0)
 * @param[in,out] a State matrix, dimension (lda, n).
 *                  On exit: leading nfz-by-nfz contains Af
 * @param[in] lda Leading dimension of a (lda >= max(1, l))
 * @param[in,out] e Descriptor matrix, dimension (lde, n).
 *                  On exit: leading nfz-by-nfz contains Ef
 * @param[in] lde Leading dimension of e (lde >= max(1, l))
 * @param[in,out] b Input matrix, dimension (ldb, m). Destroyed on exit.
 * @param[in] ldb Leading dimension of b (ldb >= max(1,l) if m>0, else ldb >= 1)
 * @param[in,out] c Output matrix, dimension (ldc, n). Destroyed on exit.
 * @param[in] ldc Leading dimension of c (ldc >= max(1, p))
 * @param[in] d Direct transmission matrix, dimension (ldd, m)
 * @param[in] ldd Leading dimension of d (ldd >= max(1, p))
 * @param[out] nfz Number of finite zeros
 * @param[out] nrank Normal rank of system pencil
 * @param[out] niz Number of infinite zeros
 * @param[out] dinfz Maximum multiplicity of infinite Smith zeros
 * @param[out] nkror Number of right Kronecker indices
 * @param[out] ninfe Number of elementary infinite blocks
 * @param[out] nkrol Number of left Kronecker indices
 * @param[out] infz Integer array (n+1): infz[i] = # of infinite divisors of degree i+1
 * @param[out] kronr Integer array (n+m+1): right Kronecker indices
 * @param[out] infe Integer array (1+min(l+p,n+m)): multiplicities of infinite eigenvalues
 * @param[out] kronl Integer array (l+p+1): left Kronecker indices
 * @param[in] tol Tolerance for rank decisions (tol <= 0 uses default, tol < 1)
 * @param[out] iwork Integer workspace, dimension (n+max(1,m)).
 *                   On exit: iwork[0] = normal rank of transfer function
 * @param[out] dwork Workspace. On exit dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace size (-1 for query).
 *                   ldwork >= max(4*(l+n), LDW) if equil='S',
 *                   ldwork >= LDW if equil='N', where
 *                   LDW = max(l+p,m+n)*(m+n) + max(1,5*max(l+p,m+n))
 * @param[out] info 0 = success, -i = parameter i invalid
 */
void ag08bd(const char* equil, i32 l, i32 n, i32 m, i32 p,
            f64* a, i32 lda, f64* e, i32 lde,
            f64* b, i32 ldb, f64* c, i32 ldc, const f64* d, i32 ldd,
            i32* nfz, i32* nrank, i32* niz, i32* dinfz,
            i32* nkror, i32* ninfe, i32* nkrol,
            i32* infz, i32* kronr, i32* infe, i32* kronl,
            f64 tol, i32* iwork, f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Zeros and Kronecker structure of a complex descriptor system pencil.
 *
 * Extracts from the system pencil
 *
 *     S(lambda) = ( A - lambda*E   B )
 *                 (      C         D )
 *
 * a regular pencil Af-lambda*Ef which has the finite Smith zeros of S(lambda)
 * as generalized eigenvalues. Also computes the orders of infinite Smith zeros
 * and determines the singular and infinite Kronecker structure.
 *
 * This is the complex version of AG08BD.
 *
 * @param[in] equil 'S' to balance the system, 'N' for no balancing
 * @param[in] l Number of rows of A, B, E (l >= 0)
 * @param[in] n Number of columns of A, E, C (n >= 0)
 * @param[in] m Number of columns of B (m >= 0)
 * @param[in] p Number of rows of C (p >= 0)
 * @param[in,out] a Complex state matrix, dimension (lda, n).
 *                  On exit: leading nfz-by-nfz contains Af
 * @param[in] lda Leading dimension of a (lda >= max(1, l))
 * @param[in,out] e Complex descriptor matrix, dimension (lde, n).
 *                  On exit: leading nfz-by-nfz contains Ef
 * @param[in] lde Leading dimension of e (lde >= max(1, l))
 * @param[in,out] b Complex input matrix, dimension (ldb, m). Destroyed on exit.
 * @param[in] ldb Leading dimension of b (ldb >= max(1,l) if m>0, else ldb >= 1)
 * @param[in,out] c Complex output matrix, dimension (ldc, n). Destroyed on exit.
 * @param[in] ldc Leading dimension of c (ldc >= max(1, p))
 * @param[in] d Complex direct transmission matrix, dimension (ldd, m)
 * @param[in] ldd Leading dimension of d (ldd >= max(1, p))
 * @param[out] nfz Number of finite zeros
 * @param[out] nrank Normal rank of system pencil
 * @param[out] niz Number of infinite zeros
 * @param[out] dinfz Maximum multiplicity of infinite Smith zeros
 * @param[out] nkror Number of right Kronecker indices
 * @param[out] ninfe Number of elementary infinite blocks
 * @param[out] nkrol Number of left Kronecker indices
 * @param[out] infz Integer array (n+1): infz[i] = # of infinite divisors of degree i+1
 * @param[out] kronr Integer array (n+m+1): right Kronecker indices
 * @param[out] infe Integer array (1+min(l+p,n+m)): multiplicities of infinite eigenvalues
 * @param[out] kronl Integer array (l+p+1): left Kronecker indices
 * @param[in] tol Tolerance for rank decisions (tol <= 0 uses default, tol < 1)
 * @param[out] iwork Integer workspace, dimension (n+max(1,m)).
 *                   On exit: iwork[0] = normal rank of transfer function
 * @param[out] dwork Real workspace, dimension (LDWORK).
 *                   LDWORK >= max(4*(l+n), 2*max(l+p,m+n)) if equil='S',
 *                   LDWORK >= 2*max(l+p,m+n) if equil='N'.
 * @param[out] zwork Complex workspace. On exit zwork[0] = optimal lzwork.
 * @param[in] lzwork Complex workspace size (-1 for query).
 * @param[out] info 0 = success, -i = parameter i invalid
 */
void ag08bz(const char* equil, i32 l, i32 n, i32 m, i32 p,
            c128* a, i32 lda, c128* e, i32 lde,
            c128* b, i32 ldb, c128* c, i32 ldc, const c128* d, i32 ldd,
            i32* nfz, i32* nrank, i32* niz, i32* dinfz,
            i32* nkror, i32* ninfe, i32* nkrol,
            i32* infz, i32* kronr, i32* infe, i32* kronl,
            f64 tol, i32* iwork, f64* dwork, c128* zwork, i32 lzwork,
            i32* info);

/**
 * @brief Extract reduced descriptor system pencil preserving finite Smith zeros.
 *
 * Extracts from the (N+P)-by-(M+N) descriptor system pencil
 *
 *     S(lambda) = ( B   A - lambda*E  )
 *                 ( D        C        )
 *
 * with E nonsingular and upper triangular a (NR+PR)-by-(M+NR) "reduced"
 * descriptor system pencil
 *
 *     Sr(lambda) = ( Br  Ar-lambda*Er )
 *                  ( Dr     Cr        )
 *
 * having the same finite Smith zeros as S(lambda) but with Dr a PR-by-M
 * full row rank left upper trapezoidal matrix, and Er an NR-by-NR upper
 * triangular nonsingular matrix.
 *
 * @param[in] first True if called first time, false for already reduced system
 * @param[in] n Number of rows of B, columns of C, order of A and E (n >= 0)
 * @param[in] m Number of columns of B and D (m >= 0, m <= p if first=false)
 * @param[in] p Number of rows of C and D (p >= 0)
 * @param[in] svlmax Maximum singular value estimate for rank decisions (svlmax >= 0)
 * @param[in,out] abcd Compound matrix [B A; D C], dimension (ldabcd, m+n).
 *                     On exit: reduced [Br Ar; Dr Cr]
 * @param[in] ldabcd Leading dimension of abcd (ldabcd >= max(1, n+p))
 * @param[in,out] e Upper triangular matrix E, dimension (lde, n).
 *                  On exit: reduced Er
 * @param[in] lde Leading dimension of e (lde >= max(1, n))
 * @param[out] nr Order of reduced matrices Ar and Er
 * @param[out] pr Rank of resulting matrix Dr
 * @param[out] ninfz Number of infinite zeros (0 if first=false)
 * @param[out] dinfz Maximum multiplicity of infinite zeros (0 if first=false)
 * @param[out] nkronl Maximum dimension of left Kronecker blocks
 * @param[out] infz Integer array (n), infz[i] = number of infinite zeros of degree i+1
 * @param[out] kronl Integer array (n+1), left Kronecker block counts
 * @param[in] tol Tolerance for rank decisions (tol <= 0 uses default)
 * @param[out] iwork Integer workspace, dimension (m). Not used if first=false.
 * @param[out] dwork Workspace. On exit dwork[0] = optimal ldwork.
 * @param[in] ldwork Workspace size (-1 for query)
 * @param[out] info 0 = success, -i = parameter i invalid
 */
void ag08by(bool first, i32 n, i32 m, i32 p, f64 svlmax,
            f64* abcd, i32 ldabcd, f64* e, i32 lde,
            i32* nr, i32* pr, i32* ninfz, i32* dinfz, i32* nkronl,
            i32* infz, i32* kronl, f64 tol, i32* iwork,
            f64* dwork, i32 ldwork, i32* info);

/**
 * @brief Extract reduced descriptor system pencil preserving finite Smith zeros (complex).
 *
 * Extracts from the (N+P)-by-(M+N) complex descriptor system pencil
 *
 *     S(lambda) = ( B   A - lambda*E  )
 *                 ( D        C        )
 *
 * with E nonsingular and upper triangular a (NR+PR)-by-(M+NR) "reduced"
 * descriptor system pencil
 *
 *     Sr(lambda) = ( Br  Ar-lambda*Er )
 *                  ( Dr     Cr        )
 *
 * having the same finite Smith zeros as S(lambda) but with Dr a PR-by-M
 * full row rank left upper trapezoidal matrix, and Er an NR-by-NR upper
 * triangular nonsingular matrix.
 *
 * @param[in] first True if called first time, false for already reduced system
 * @param[in] n Number of rows of B, columns of C, order of A and E (n >= 0)
 * @param[in] m Number of columns of B and D (m >= 0, m <= p if first=false)
 * @param[in] p Number of rows of C and D (p >= 0)
 * @param[in] svlmax Maximum singular value estimate for rank decisions (svlmax >= 0)
 * @param[in,out] abcd Complex compound matrix [B A; D C], dimension (ldabcd, m+n).
 *                     On exit: reduced [Br Ar; Dr Cr]
 * @param[in] ldabcd Leading dimension of abcd (ldabcd >= max(1, n+p))
 * @param[in,out] e Complex upper triangular matrix E, dimension (lde, n).
 *                  On exit: reduced Er
 * @param[in] lde Leading dimension of e (lde >= max(1, n))
 * @param[out] nr Order of reduced matrices Ar and Er
 * @param[out] pr Rank of resulting matrix Dr
 * @param[out] ninfz Number of infinite zeros (0 if first=false)
 * @param[out] dinfz Maximum multiplicity of infinite zeros (0 if first=false)
 * @param[out] nkronl Maximum dimension of left Kronecker blocks
 * @param[out] infz Integer array (n), infz[i] = number of infinite zeros of degree i+1
 * @param[out] kronl Integer array (n+1), left Kronecker block counts
 * @param[in] tol Tolerance for rank decisions (tol <= 0 uses default)
 * @param[out] iwork Integer workspace, dimension (m). Not used if first=false.
 * @param[out] dwork Real workspace, dimension 2*max(m,p) if first, 2*p otherwise
 * @param[out] zwork Complex workspace. On exit zwork[0] = optimal lzwork.
 * @param[in] lzwork Complex workspace size (-1 for query)
 * @param[out] info 0 = success, -i = parameter i invalid
 */
void ag8byz(bool first, i32 n, i32 m, i32 p, f64 svlmax,
            c128* abcd, i32 ldabcd, c128* e, i32 lde,
            i32* nr, i32* pr, i32* ninfz, i32* dinfz, i32* nkronl,
            i32* infz, i32* kronl, f64 tol, i32* iwork,
            f64* dwork, c128* zwork, i32 lzwork, i32* info);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_AG_H */
