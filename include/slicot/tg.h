/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_TG_H
#define SLICOT_TG_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Balance the matrices of a descriptor system pencil.
 *
 * Balances the system pencil S = ((A B), (C 0)) - lambda*((E 0), (0 0))
 * corresponding to the descriptor triple (A-lambda*E, B, C) via diagonal
 * similarity transformations (Dl*A*Dr - lambda*Dl*E*Dr, Dl*B, C*Dr).
 *
 * @param[in] job Controls which matrices are balanced:
 *                'A' = All matrices (A, E, B, C)
 *                'B' = B, A, and E only (not C)
 *                'C' = C, A, and E only (not B)
 *                'N' = A and E only (not B or C)
 * @param[in] l Number of rows of A, B, E (L >= 0)
 * @param[in] n Number of columns of A, E, C (N >= 0)
 * @param[in] m Number of columns of B (M >= 0)
 * @param[in] p Number of rows of C (P >= 0)
 * @param[in] thresh Threshold for ignoring small elements (THRESH >= 0)
 * @param[in,out] a DOUBLE PRECISION array, dimension (lda,n)
 *                  In: L-by-N state dynamics matrix A
 *                  Out: Balanced matrix Dl*A*Dr
 * @param[in] lda Leading dimension of A (lda >= max(1,l))
 * @param[in,out] e DOUBLE PRECISION array, dimension (lde,n)
 *                  In: L-by-N descriptor matrix E
 *                  Out: Balanced matrix Dl*E*Dr
 * @param[in] lde Leading dimension of E (lde >= max(1,l))
 * @param[in,out] b DOUBLE PRECISION array, dimension (ldb,m)
 *                  In: L-by-M input/state matrix B
 *                  Out: Balanced matrix Dl*B (if job='A' or 'B')
 * @param[in] ldb Leading dimension of B (ldb >= max(1,l) if m>0, else >= 1)
 * @param[in,out] c DOUBLE PRECISION array, dimension (ldc,n)
 *                  In: P-by-N state/output matrix C
 *                  Out: Balanced matrix C*Dr (if job='A' or 'C')
 * @param[in] ldc Leading dimension of C (ldc >= max(1,p))
 * @param[out] lscale DOUBLE PRECISION array, dimension (l)
 *                    Left scaling factors Dl(i), i=1,...,l
 * @param[out] rscale DOUBLE PRECISION array, dimension (n)
 *                    Right scaling factors Dr(j), j=1,...,n
 * @param[out] dwork DOUBLE PRECISION array, dimension (3*(l+n))
 * @param[out] info Exit code (0 = success, <0 = invalid parameter -i)
 */
void tg01ad(
    const char* job, const i32 l, const i32 n, const i32 m, const i32 p,
    const f64 thresh,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* lscale, f64* rscale,
    f64* dwork,
    i32* info
);

/**
 * @brief Balance the matrices of a complex descriptor system pencil.
 *
 * Balances the system pencil S = ((A B), (C 0)) - lambda*((E 0), (0 0))
 * corresponding to the descriptor triple (A-lambda*E, B, C) via diagonal
 * similarity transformations (Dl*A*Dr - lambda*Dl*E*Dr, Dl*B, C*Dr).
 * This is the complex version of tg01ad.
 *
 * @param[in] job Controls which matrices are balanced:
 *                'A' = All matrices (A, E, B, C)
 *                'B' = B, A, and E only (not C)
 *                'C' = C, A, and E only (not B)
 *                'N' = A and E only (not B or C)
 * @param[in] l Number of rows of A, B, E (L >= 0)
 * @param[in] n Number of columns of A, E, C (N >= 0)
 * @param[in] m Number of columns of B (M >= 0)
 * @param[in] p Number of rows of C (P >= 0)
 * @param[in] thresh Threshold for ignoring small elements (THRESH >= 0).
 *                   Magnitude is computed as |real| + |imag|.
 * @param[in,out] a COMPLEX*16 array, dimension (lda,n)
 *                  In: L-by-N state dynamics matrix A
 *                  Out: Balanced matrix Dl*A*Dr
 * @param[in] lda Leading dimension of A (lda >= max(1,l))
 * @param[in,out] e COMPLEX*16 array, dimension (lde,n)
 *                  In: L-by-N descriptor matrix E
 *                  Out: Balanced matrix Dl*E*Dr
 * @param[in] lde Leading dimension of E (lde >= max(1,l))
 * @param[in,out] b COMPLEX*16 array, dimension (ldb,m)
 *                  In: L-by-M input/state matrix B
 *                  Out: Balanced matrix Dl*B (if job='A' or 'B')
 * @param[in] ldb Leading dimension of B (ldb >= max(1,l) if m>0, else >= 1)
 * @param[in,out] c COMPLEX*16 array, dimension (ldc,n)
 *                  In: P-by-N state/output matrix C
 *                  Out: Balanced matrix C*Dr (if job='A' or 'C')
 * @param[in] ldc Leading dimension of C (ldc >= max(1,p))
 * @param[out] lscale DOUBLE PRECISION array, dimension (l)
 *                    Left scaling factors Dl(i), i=1,...,l
 * @param[out] rscale DOUBLE PRECISION array, dimension (n)
 *                    Right scaling factors Dr(j), j=1,...,n
 * @param[out] dwork DOUBLE PRECISION array, dimension (3*(l+n))
 * @param[out] info Exit code (0 = success, <0 = invalid parameter -i)
 */
void tg01az(
    const char* job, const i32 l, const i32 n, const i32 m, const i32 p,
    const f64 thresh,
    c128* a, const i32 lda,
    c128* e, const i32 lde,
    c128* b, const i32 ldb,
    c128* c, const i32 ldc,
    f64* lscale, f64* rscale,
    f64* dwork,
    i32* info
);

/**
 * @brief Reduce descriptor system to generalized Hessenberg form.
 *
 * Reduces matrices A and E of the descriptor system pencil
 * S = (A, B; C, 0) - lambda*(E, 0; 0, 0) to generalized upper Hessenberg form
 * using orthogonal transformations: Q' * A * Z = H, Q' * E * Z = T.
 *
 * @param[in] jobe 'G' = E is general, 'U' = E is upper triangular
 * @param[in] compq 'N' = no Q, 'I' = init Q to I, 'V' = update Q
 * @param[in] compz 'N' = no Z, 'I' = init Z to I, 'V' = update Z
 * @param[in] n Order of A, E (N >= 0)
 * @param[in] m Number of columns of B (M >= 0)
 * @param[in] p Number of rows of C (P >= 0)
 * @param[in] ilo Lower active row/column index (1-based)
 * @param[in] ihi Upper active row/column index (1-based)
 * @param[in,out] a N-by-N matrix, returns H
 * @param[in] lda Leading dimension of A
 * @param[in,out] e N-by-N matrix, returns T
 * @param[in] lde Leading dimension of E
 * @param[in,out] b N-by-M matrix, returns Q'*B
 * @param[in] ldb Leading dimension of B
 * @param[in,out] c P-by-N matrix, returns C*Z
 * @param[in] ldc Leading dimension of C
 * @param[in,out] q N-by-N orthogonal matrix
 * @param[in] ldq Leading dimension of Q
 * @param[in,out] z N-by-N orthogonal matrix
 * @param[in] ldz Leading dimension of Z
 * @param[out] dwork Workspace
 * @param[in] ldwork Workspace size
 * @param[out] info Exit code (0 = success, <0 = invalid parameter -i)
 */
void tg01bd(const char *jobe, const char *compq, const char *compz,
            i32 n, i32 m, i32 p, i32 ilo, i32 ihi,
            f64 *a, i32 lda, f64 *e, i32 lde,
            f64 *b, i32 ldb, f64 *c, i32 ldc,
            f64 *q, i32 ldq, f64 *z, i32 ldz,
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Orthogonal reduction of descriptor system to SVD-like coordinate form.
 *
 * Computes orthogonal transformation matrices Q and Z such that the transformed
 * descriptor system (Q'*A*Z - lambda*Q'*E*Z, Q'*B, C*Z) is in SVD-like form:
 *
 *            ( A11  A12 )             ( Er  0 )
 *   Q'*A*Z = (          ) ,  Q'*E*Z = (       )
 *            ( A21  A22 )             (  0  0 )
 *
 * where Er is upper triangular and invertible. Optionally reduces A22 to:
 *
 *        ( Ar  X )                ( Ar  0 )
 *  A22 = (       )  (JOBA='T') or (       )  (JOBA='R')
 *        (  0  0 )                (  0  0 )
 *
 * with Ar upper triangular invertible, X full or zero.
 *
 * @param[in] compq Controls Q computation:
 *                  'N' = do not compute Q
 *                  'I' = initialize Q to identity and return Q
 *                  'U' = update existing Q (Q := Q1*Q)
 * @param[in] compz Controls Z computation:
 *                  'N' = do not compute Z
 *                  'I' = initialize Z to identity and return Z
 *                  'U' = update existing Z (Z := Z1*Z)
 * @param[in] joba Controls A22 reduction:
 *                 'N' = do not reduce A22
 *                 'R' = reduce A22 to SVD-like upper triangular form
 *                 'T' = reduce A22 to upper trapezoidal form
 * @param[in] l Number of rows of A, B, E (L >= 0)
 * @param[in] n Number of columns of A, E, C (N >= 0)
 * @param[in] m Number of columns of B (M >= 0)
 * @param[in] p Number of rows of C (P >= 0)
 * @param[in,out] a DOUBLE PRECISION array, dimension (lda,n)
 *                  In: L-by-N state dynamics matrix A
 *                  Out: Transformed matrix Q'*A*Z
 * @param[in] lda Leading dimension of A (lda >= max(1,l))
 * @param[in,out] e DOUBLE PRECISION array, dimension (lde,n)
 *                  In: L-by-N descriptor matrix E
 *                  Out: Transformed matrix Q'*E*Z in SVD-like form
 * @param[in] lde Leading dimension of E (lde >= max(1,l))
 * @param[in,out] b DOUBLE PRECISION array, dimension (ldb,m)
 *                  In: L-by-M input/state matrix B
 *                  Out: Transformed matrix Q'*B
 * @param[in] ldb Leading dimension of B (ldb >= max(1,l) if m>0, else >= 1)
 * @param[in,out] c DOUBLE PRECISION array, dimension (ldc,n)
 *                  In: P-by-N state/output matrix C
 *                  Out: Transformed matrix C*Z
 * @param[in] ldc Leading dimension of C (ldc >= max(1,p))
 * @param[in,out] q DOUBLE PRECISION array, dimension (ldq,l)
 *                  If compq='I': Out: orthogonal matrix Q
 *                  If compq='U': In: Q1, Out: Q1*Q
 *                  If compq='N': Not referenced
 * @param[in] ldq Leading dimension of Q (ldq >= max(1,l) if compq!='N', else >= 1)
 * @param[in,out] z DOUBLE PRECISION array, dimension (ldz,n)
 *                  If compz='I': Out: orthogonal matrix Z
 *                  If compz='U': In: Z1, Out: Z1*Z
 *                  If compz='N': Not referenced
 * @param[in] ldz Leading dimension of Z (ldz >= max(1,n) if compz!='N', else >= 1)
 * @param[out] ranke Rank of matrix E (order of Er)
 * @param[out] rnka22 Rank of A22 (order of Ar, if joba='R' or 'T')
 * @param[in] tol Tolerance for rank determination (0 < tol < 1)
 *                If tol <= 0, uses default: l*n*eps
 * @param[out] iwork INTEGER array, dimension (n)
 * @param[out] dwork DOUBLE PRECISION array, dimension (ldwork)
 *                   On exit, dwork[0] returns optimal ldwork
 * @param[in] ldwork Length of dwork (>= max(1, n+p, min(l,n)+max(3*n-1,m,l)))
 *                   If ldwork=-1, workspace query (returns optimal size in dwork[0])
 * @param[out] info Exit code (0 = success, <0 = invalid parameter -i)
 */
void tg01fd(
    const char* compq, const char* compz, const char* joba,
    const i32 l, const i32 n, const i32 m, const i32 p,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* ranke, i32* rnka22,
    const f64 tol,
    i32* iwork, f64* dwork, const i32 ldwork,
    i32* info
);

/**
 * @brief Unitary reduction of complex descriptor system to SVD-like coordinate form.
 *
 * Computes for the complex descriptor system (A-lambda*E,B,C) the unitary
 * transformation matrices Q and Z such that the transformed system
 * (Q'*A*Z-lambda*Q'*E*Z, Q'*B, C*Z) is in SVD-like coordinate form with:
 *
 *                  ( A11  A12 )             ( Er  0 )
 *         Q'*A*Z = (          ) ,  Q'*E*Z = (       ) ,
 *                  ( A21  A22 )             (  0  0 )
 *
 * where Er is an upper triangular invertible matrix, and ' denotes
 * the conjugate transpose.
 *
 * Optionally, the A22 matrix can be further reduced to the form
 *                  ( Ar  X )
 *            A22 = (       ) ,
 *                  (  0  0 )
 * with Ar an upper triangular invertible matrix, and X either full or zero.
 *
 * @param[in] compq Controls Q computation:
 *                  'N' = do not compute Q
 *                  'I' = initialize Q to identity and compute Q
 *                  'U' = update Q1 to Q1*Q (Q1 must be provided on entry)
 * @param[in] compz Controls Z computation:
 *                  'N' = do not compute Z
 *                  'I' = initialize Z to identity and compute Z
 *                  'U' = update Z1 to Z1*Z (Z1 must be provided on entry)
 * @param[in] joba Controls A22 reduction:
 *                 'N' = do not reduce A22
 *                 'R' = reduce A22 to SVD-like upper triangular form (X=0)
 *                 'T' = reduce A22 to upper trapezoidal form
 * @param[in] l Number of rows of A, B, E (L >= 0)
 * @param[in] n Number of columns of A, E, C (N >= 0)
 * @param[in] m Number of columns of B (M >= 0)
 * @param[in] p Number of rows of C (P >= 0)
 * @param[in,out] a L-by-N complex state matrix; on exit, transformed Q'*A*Z
 * @param[in] lda Leading dimension of A (>= max(1,L))
 * @param[in,out] e L-by-N complex descriptor matrix; on exit, transformed Q'*E*Z
 * @param[in] lde Leading dimension of E (>= max(1,L))
 * @param[in,out] b L-by-M complex input matrix; on exit, transformed Q'*B
 * @param[in] ldb Leading dimension of B (>= max(1,L) if M>0, >= 1 if M=0)
 * @param[in,out] c P-by-N complex output matrix; on exit, transformed C*Z
 * @param[in] ldc Leading dimension of C (>= max(1,P))
 * @param[in,out] q L-by-L complex unitary matrix Q
 * @param[in] ldq Leading dimension of Q (>= 1 if compq='N', >= max(1,L) otherwise)
 * @param[in,out] z N-by-N complex unitary matrix Z
 * @param[in] ldz Leading dimension of Z (>= 1 if compz='N', >= max(1,N) otherwise)
 * @param[out] ranke Estimated rank of E (order of Er)
 * @param[out] rnka22 Estimated rank of A22 (order of Ar) if joba='R' or 'T'
 * @param[in] tol Tolerance for rank determination (tol <= 0 uses default L*N*eps)
 * @param[out] iwork Integer workspace, dimension (N)
 * @param[out] dwork Double workspace, dimension (2*N)
 * @param[out] zwork Complex workspace, dimension (lzwork)
 * @param[in] lzwork Complex workspace size (>= max(1, N+P, min(L,N)+max(3*N-1,M,L)))
 *                   If lzwork=-1, workspace query mode
 * @param[out] info Exit code (0 = success, <0 = invalid parameter -i)
 */
void tg01fz(
    const char* compq, const char* compz, const char* joba,
    const i32 l, const i32 n, const i32 m, const i32 p,
    c128* a, const i32 lda,
    c128* e, const i32 lde,
    c128* b, const i32 ldb,
    c128* c, const i32 ldc,
    c128* q, const i32 ldq,
    c128* z, const i32 ldz,
    i32* ranke, i32* rnka22,
    const f64 tol,
    i32* iwork, f64* dwork, c128* zwork, const i32 lzwork,
    i32* info
);

/**
 * @brief Orthogonal reduction of descriptor system to controllability staircase form.
 *
 * Given the descriptor system (A-lambda*E,B,C) with matrices in partitioned form,
 * this routine reduces the pair (A1-lambda*E1,B1) to controllability staircase form,
 * separating controllable and uncontrollable finite eigenvalues.
 *
 * The reduction produces:
 *   Qc'*[B1 A1-lambda*E1]*diag(I,Zc) =
 *       ( Bc Ac-lambda*Ec      *         )
 *       ( 0     0         Anc-lambda*Enc )
 *
 * where (Bc, Ac-lambda*Ec) has full row rank NR for all finite lambda.
 *
 * @param[in] compq Controls Q computation:
 *                  'N' = do not compute Q
 *                  'I' = initialize Q to identity and return Q
 *                  'U' = update existing Q (Q := Q1*Q)
 * @param[in] compz Controls Z computation:
 *                  'N' = do not compute Z
 *                  'I' = initialize Z to identity and return Z
 *                  'U' = update existing Z (Z := Z1*Z)
 * @param[in] l Number of rows of A, B, E (L >= 0)
 * @param[in] n Number of columns of A, E, C (N >= 0)
 * @param[in] m Number of columns of B (M >= 0)
 * @param[in] p Number of rows of C (P >= 0)
 * @param[in] n1 Order of subsystem to be reduced (0 <= N1 <= min(L,N))
 * @param[in] lbe Number of nonzero sub-diagonals of E1 (0 <= LBE <= max(0,N1-1))
 * @param[in,out] a L-by-N state matrix A; on exit, transformed matrix
 * @param[in] lda Leading dimension of A (>= max(1,L))
 * @param[in,out] e L-by-N descriptor matrix E; on exit, transformed matrix
 * @param[in] lde Leading dimension of E (>= max(1,L))
 * @param[in,out] b L-by-M input matrix B; on exit, transformed matrix
 * @param[in] ldb Leading dimension of B (>= max(1,L))
 * @param[in,out] c P-by-N output matrix C; on exit, transformed matrix
 * @param[in] ldc Leading dimension of C (>= max(1,P))
 * @param[in,out] q L-by-L orthogonal transformation matrix Q
 * @param[in] ldq Leading dimension of Q (>= 1 if compq='N', >= L otherwise)
 * @param[in,out] z N-by-N orthogonal transformation matrix Z
 * @param[in] ldz Leading dimension of Z (>= 1 if compz='N', >= N otherwise)
 * @param[out] nr Order of controllable part (NR = sum of RTAU)
 * @param[out] nrblck Number of full row rank blocks in staircase form
 * @param[out] rtau Array of dimension (N1); RTAU(i) is row dimension of block i
 * @param[in] tol Tolerance for rank determination (tol <= 0 uses default L*N*eps)
 * @param[out] iwork Integer workspace, dimension (M)
 * @param[out] dwork Double workspace, dimension (max(N,L,2*M))
 * @param[out] info Exit code (0 = success, <0 = invalid parameter -i)
 */
/**
 * @brief Staircase controllability form for multi-input descriptor system.
 *
 * Given descriptor system (A-lambda*E,B,C) with A, E, B partitioned:
 *     A = (A1 X1; 0 X2), E = (E1 Y1; 0 Y2), B = (B1 B2; 0 0)
 * reduces pair (A1-lambda*E1,[B1 B2]) to staircase controllability form.
 *
 * @param[in] compq 'N': don't compute Q; 'I': init Q to I, return Qc;
 *                  'U': update existing Q with Q*Qc
 * @param[in] compz 'N': don't compute Z; 'I': init Z to I, return Zc;
 *                  'U': update existing Z with Z*Zc
 * @param[in] l Number of rows of A, E, B (L >= 0)
 * @param[in] n Number of columns of A, E, C (N >= 0)
 * @param[in] m1 Number of columns of B1 (M1 >= 0)
 * @param[in] m2 Number of columns of B2 (M2 >= 0)
 * @param[in] p Number of rows of C (P >= 0)
 * @param[in] n1 Order of subsystem to reduce (0 <= N1 <= min(L,N))
 * @param[in] lbe Number of nonzero sub-diagonals of E1 (0 <= LBE <= max(0,N1-1))
 * @param[in,out] a L-by-N state matrix; on exit transformed
 * @param[in] lda Leading dimension of A (>= max(1,L))
 * @param[in,out] e L-by-N descriptor matrix; on exit transformed with upper triangular Ec, Enc
 * @param[in] lde Leading dimension of E (>= max(1,L))
 * @param[in,out] b L-by-(M1+M2) input matrix; on exit transformed
 * @param[in] ldb Leading dimension of B (>= max(1,L))
 * @param[in,out] c P-by-N output matrix; on exit C*Zc
 * @param[in] ldc Leading dimension of C (>= max(1,P))
 * @param[in,out] q L-by-L transformation Q; if compq='U' must be orthogonal on entry
 * @param[in] ldq Leading dimension of Q (>= 1, >= L if compq='I'/'U')
 * @param[in,out] z N-by-N transformation Z; if compz='U' must be orthogonal on entry
 * @param[in] ldz Leading dimension of Z (>= 1, >= N if compz='I'/'U')
 * @param[out] nr Order of controllable part (0 <= NR <= N1)
 * @param[out] nrblck Number of staircase blocks (always even)
 * @param[out] rtau Array of block dimensions (size 2*N1)
 * @param[in] tol Tolerance for rank determination (<=0 uses default L*N*eps)
 * @param[out] iwork Integer workspace, size M1+M2
 * @param[out] dwork Double workspace; on exit dwork[0] = optimal ldwork
 * @param[in] ldwork Workspace size; -1 for query
 * @param[out] info Exit code (0=success, <0=invalid param -i)
 */
void tg01hu(
    const char* compq, const char* compz,
    const i32 l, const i32 n, const i32 m1, const i32 m2, const i32 p,
    const i32 n1, const i32 lbe,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* nr, i32* nrblck, i32* rtau,
    const f64 tol,
    i32* iwork, f64* dwork, const i32 ldwork,
    i32* info
);

void tg01hx(
    const char* compq, const char* compz,
    const i32 l, const i32 n, const i32 m, const i32 p,
    const i32 n1, const i32 lbe,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* nr, i32* nrblck, i32* rtau,
    const f64 tol,
    i32* iwork, f64* dwork,
    i32* info
);

/**
 * @brief Blocked version of controllability staircase reduction for descriptor systems.
 *
 * Reduces the descriptor pair (A1-lambda*E1, B1) to controllability staircase form.
 * Uses block algorithms for the QR reduction of E and panel-based row transformations.
 *
 * @param[in] compq Specifies Q computation:
 *                  'N' = do not compute Q
 *                  'I' = Q is initialized to identity, orthogonal Q returned
 *                  'U' = Q must contain orthogonal Q1 on entry, Q1*Q returned
 * @param[in] compz Specifies Z computation:
 *                  'N' = do not compute Z
 *                  'I' = Z is initialized to identity, orthogonal Z returned
 *                  'U' = Z must contain orthogonal Z1 on entry, Z1*Z returned
 * @param[in] l Number of rows of A, E, B (L >= 0)
 * @param[in] n Number of columns of A, E, C (N >= 0)
 * @param[in] m Number of columns of B (M >= 0)
 * @param[in] p Number of rows of C (P >= 0)
 * @param[in] n1 Order of subsystem to reduce (0 <= N1 <= min(L,N))
 * @param[in] lbe Number of nonzero subdiagonals of E1 (0 <= LBE <= max(0,N1-1))
 * @param[in,out] a L-by-N state matrix; on exit Qc'*A*diag(Zc,I)
 * @param[in] lda Leading dimension of A (>= max(1,L))
 * @param[in,out] e L-by-N descriptor matrix; on exit Qc'*E*diag(Zc,I)
 * @param[in] lde Leading dimension of E (>= max(1,L))
 * @param[in,out] b L-by-M input matrix; on exit Qc'*B
 * @param[in] ldb Leading dimension of B (>= max(1,L))
 * @param[in,out] c P-by-N output matrix; on exit C*Zc
 * @param[in] ldc Leading dimension of C (>= max(1,P))
 * @param[in,out] q L-by-L orthogonal transformation matrix Q
 * @param[in] ldq Leading dimension of Q (>= 1; >= L if COMPQ='I' or 'U')
 * @param[in,out] z N-by-N orthogonal transformation matrix Z
 * @param[in] ldz Leading dimension of Z (>= 1; >= N if COMPZ='I' or 'U')
 * @param[out] nr Order of controllable part
 * @param[out] nrblck Number of full row rank blocks in staircase form
 * @param[out] rtau Integer array, dimension (N1), staircase block row dimensions
 * @param[in] tol Tolerance for rank determination (<=0 uses default L*N*eps)
 * @param[out] iwork Integer workspace, size M
 * @param[out] dwork Double workspace; on exit dwork[0] = optimal ldwork
 * @param[in] ldwork Workspace size; -1 for query
 * @param[out] info Exit code (0=success, <0=invalid param -i)
 */
void tg01hy(
    const char* compq, const char* compz,
    const i32 l, const i32 n, const i32 m, const i32 p,
    const i32 n1, const i32 lbe,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* nr, i32* nrblck, i32* rtau,
    const f64 tol,
    i32* iwork, f64* dwork, const i32 ldwork,
    i32* info
);

/**
 * @brief Orthogonal reduction of descriptor system (C,A-lambda E) to RQ-coordinate form.
 *
 * Computes orthogonal transformation matrix Z such that the transformed
 * descriptor system pair (C*Z, A*Z - lambda*E*Z) has E*Z in upper trapezoidal form.
 *
 * @param[in] compz Controls Z computation:
 *                  'N' = do not compute Z
 *                  'I' = initialize Z to identity and return Z
 *                  'U' = update existing Z (Z := Z1*Z)
 * @param[in] l Number of rows of A and E (L >= 0)
 * @param[in] n Number of columns of A, E, and C (N >= 0)
 * @param[in] p Number of rows of C (P >= 0)
 * @param[in,out] a DOUBLE PRECISION array, dimension (lda,n)
 *                  In: L-by-N state dynamics matrix A
 *                  Out: Transformed matrix A*Z
 * @param[in] lda Leading dimension of A (lda >= max(1,l))
 * @param[in,out] e DOUBLE PRECISION array, dimension (lde,n)
 *                  In: L-by-N descriptor matrix E
 *                  Out: Transformed matrix E*Z in upper trapezoidal form
 * @param[in] lde Leading dimension of E (lde >= max(1,l))
 * @param[in,out] c DOUBLE PRECISION array, dimension (ldc,n)
 *                  In: P-by-N state/output matrix C
 *                  Out: Transformed matrix C*Z
 * @param[in] ldc Leading dimension of C (ldc >= max(1,p))
 * @param[in,out] z DOUBLE PRECISION array, dimension (ldz,n)
 *                  If compz='I': Out: orthogonal matrix Z
 *                  If compz='U': In: Z1, Out: Z1*Z
 *                  If compz='N': Not referenced
 * @param[in] ldz Leading dimension of Z (ldz >= 1 if compz='N', >= max(1,n) otherwise)
 * @param[out] dwork DOUBLE PRECISION array, dimension (ldwork)
 *                   On exit, dwork[0] returns optimal ldwork
 * @param[in] ldwork Length of dwork (>= max(1, min(l,n) + max(l,n,p)))
 * @param[out] info Exit code (0 = success, <0 = invalid parameter -i)
 */
void tg01dd(
    const char* compz,
    const i32 l, const i32 n, const i32 p,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* c, const i32 ldc,
    f64* z, const i32 ldz,
    f64* dwork, const i32 ldwork,
    i32* info
);

/**
 * @brief Orthogonal reduction of descriptor system to SVD coordinate form.
 *
 * Computes orthogonal matrices Q and Z such that Q'*A*Z and Q'*E*Z are
 * in SVD coordinate form, with E transformed to diagonal form containing
 * its singular values.
 *
 * @param[in] joba 'N' = no A22 reduction, 'R' = reduce A22 to SVD form
 * @param[in] l Number of rows of A, B, E (L >= 0)
 * @param[in] n Number of columns of A, E, C (N >= 0)
 * @param[in] m Number of columns of B (M >= 0)
 * @param[in] p Number of rows of C (P >= 0)
 * @param[in,out] a L-by-N state matrix; on exit Q'*A*Z
 * @param[in] lda Leading dimension of A (>= max(1,L))
 * @param[in,out] e L-by-N descriptor matrix; on exit Q'*E*Z in SVD form
 * @param[in] lde Leading dimension of E (>= max(1,L))
 * @param[in,out] b L-by-M input matrix; on exit Q'*B
 * @param[in] ldb Leading dimension of B (>= max(1,L) if M>0, else >= 1)
 * @param[in,out] c P-by-N output matrix; on exit C*Z
 * @param[in] ldc Leading dimension of C (>= max(1,P))
 * @param[out] q L-by-L left orthogonal transformation matrix
 * @param[in] ldq Leading dimension of Q (>= max(1,L))
 * @param[out] z N-by-N right orthogonal transformation matrix
 * @param[in] ldz Leading dimension of Z (>= max(1,N))
 * @param[out] ranke Effective rank of E
 * @param[out] rnka22 Effective rank of A22 (if JOBA='R')
 * @param[in] tol Tolerance for rank determination (TOL < 1.0; if <= 0, default used)
 * @param[out] dwork Workspace array
 * @param[in] ldwork Size of dwork (>= max(1, min(L,N) + max(3*min(L,N)+max(L,N), 5*min(L,N), M, P)))
 * @param[out] info 0=success, <0=invalid param -i, >0=SVD convergence failure
 */
void tg01ed(
    const char* joba,
    const i32 l, const i32 n, const i32 m, const i32 p,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* ranke, i32* rnka22,
    const f64 tol,
    f64* dwork, const i32 ldwork,
    i32* info
);

/**
 * @brief Reduce descriptor system to QR-coordinate form.
 *
 * Reduces the descriptor system pair (A-lambda E, B) to QR-coordinate form
 * by computing an orthogonal transformation matrix Q such that Q'*E is
 * upper trapezoidal.
 *
 * @param[in] compq Controls Q computation:
 *                  'N' = do not compute Q
 *                  'I' = initialize Q to identity and return Q
 *                  'U' = update existing Q (Q := Q1*Q)
 * @param[in] l Number of rows of A, B, E (L >= 0)
 * @param[in] n Number of columns of A, E (N >= 0)
 * @param[in] m Number of columns of B (M >= 0)
 * @param[in,out] a L-by-N state matrix; on exit Q'*A
 * @param[in] lda Leading dimension of A (>= max(1,L))
 * @param[in,out] e L-by-N descriptor matrix; on exit upper trapezoidal Q'*E
 * @param[in] lde Leading dimension of E (>= max(1,L))
 * @param[in,out] b L-by-M input matrix; on exit Q'*B
 * @param[in] ldb Leading dimension of B (>= max(1,L) if M>0, else >= 1)
 * @param[in,out] q L-by-L orthogonal matrix Q
 * @param[in] ldq Leading dimension of Q (>= max(1,L) if compq!='N', else >= 1)
 * @param[out] dwork Workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size (>= max(1, min(L,N) + max(L,N,M)))
 * @param[out] info Exit code (0 = success, <0 = invalid parameter -i)
 */
void tg01cd(const char *compq, i32 l, i32 n, i32 m,
            f64 *a, i32 lda, f64 *e, i32 lde,
            f64 *b, i32 ldb, f64 *q, i32 ldq,
            f64 *dwork, i32 ldwork, i32 *info);

/**
 * @brief Reduced descriptor representation without non-dynamic modes.
 *
 * Finds a reduced descriptor representation (Ar-lambda*Er,Br,Cr,Dr)
 * without non-dynamic modes for a descriptor representation
 * (A-lambda*E,B,C,D). Optionally, the reduced descriptor system can
 * be put into a standard form with the leading diagonal block
 * of Er identity.
 *
 * @param[in] jobs Controls whether to transform Er's leading diagonal block:
 *                 'S' = make Er with leading diagonal identity
 *                 'D' = keep Er unreduced or upper triangular
 * @param[in] l Number of rows of A, E, B (L >= 0)
 * @param[in] n Number of columns of A, E, C (N >= 0)
 * @param[in] m Number of columns of B (M >= 0)
 * @param[in] p Number of rows of C (P >= 0)
 * @param[in,out] a L-by-N state matrix; on exit, LR-by-NR reduced matrix Ar
 * @param[in] lda Leading dimension of A (>= max(1,L))
 * @param[in,out] e L-by-N descriptor matrix; on exit, LR-by-NR reduced matrix Er
 * @param[in] lde Leading dimension of E (>= max(1,L))
 * @param[in,out] b L-by-M input matrix; on exit, LR-by-M reduced matrix Br
 * @param[in] ldb Leading dimension of B (>= max(1,L))
 * @param[in,out] c P-by-N output matrix; on exit, P-by-NR reduced matrix Cr
 * @param[in] ldc Leading dimension of C (>= max(1,P))
 * @param[in,out] d P-by-M feedthrough matrix; on exit, modified Dr
 * @param[in] ldd Leading dimension of D (>= max(1,P))
 * @param[out] lr Number of reduced differential equations
 * @param[out] nr Dimension of reduced descriptor state vector
 * @param[out] ranke Estimated rank of matrix E
 * @param[out] infred Information on reduction:
 *                     >= 0: achieved order reduction (SVD-like form)
 *                     < 0: no reduction achieved, original restored
 * @param[in] tol Tolerance for rank determination (tol <= 0 uses default L*N*eps)
 * @param[out] iwork Integer workspace, dimension (N)
 * @param[out] dwork Double workspace, dimension (ldwork)
 * @param[in] ldwork Workspace size (>= max(1, N+P, min(L,N)+max(3*N-1,M,L)))
 *                   If ldwork=-1, workspace query mode
 * @param[out] info Exit code (0 = success, <0 = invalid parameter -i)
 */
void tg01gd(
    const char* jobs,
    const i32 l, const i32 n, const i32 m, const i32 p,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* d, const i32 ldd,
    i32* lr, i32* nr, i32* ranke, i32* infred,
    const f64 tol,
    i32* iwork, f64* dwork, const i32 ldwork,
    i32* info
);

/**
 * @brief Orthogonal reduction of a descriptor system to controllability staircase form.
 *
 * Computes orthogonal transformation matrices Q and Z which reduce the N-th order
 * descriptor system (A-lambda*E,B,C) to the form
 *
 *           ( Ac  *  )             ( Ec  *  )           ( Bc )
 *  Q'*A*Z = (        ) ,  Q'*E*Z = (        ) ,  Q'*B = (    ) ,
 *           ( 0  Anc )             ( 0  Enc )           ( 0  )
 *
 *     C*Z = ( Cc Cnc ) ,
 *
 * where the NCONT-th order descriptor system (Ac-lambda*Ec,Bc,Cc) is finite and/or
 * infinite controllable. The pencil Anc - lambda*Enc is regular of order N-NCONT
 * and contains the uncontrollable finite and/or infinite eigenvalues.
 *
 * @param[in] jobcon Controls which eigenvalues to separate:
 *                   'C' = separate both finite and infinite uncontrollable eigenvalues
 *                   'F' = separate only finite uncontrollable eigenvalues
 *                   'I' = separate only nonzero finite and infinite uncontrollable eigenvalues
 * @param[in] compq Specifies Q computation:
 *                  'N' = do not compute Q
 *                  'I' = Q is initialized to identity, orthogonal Q returned
 *                  'U' = Q must contain orthogonal Q1 on entry, Q1*Q returned
 * @param[in] compz Specifies Z computation:
 *                  'N' = do not compute Z
 *                  'I' = Z is initialized to identity, orthogonal Z returned
 *                  'U' = Z must contain orthogonal Z1 on entry, Z1*Z returned
 * @param[in] n Dimension of descriptor state vector (N >= 0)
 * @param[in] m Dimension of descriptor system input (M >= 0)
 * @param[in] p Dimension of descriptor system output (P >= 0)
 * @param[in,out] a N-by-N state matrix; on exit, transformed Q'*A*Z
 * @param[in] lda Leading dimension of A (>= max(1,N))
 * @param[in,out] e N-by-N descriptor matrix; on exit, transformed Q'*E*Z
 * @param[in] lde Leading dimension of E (>= max(1,N))
 * @param[in,out] b N-by-M input matrix; on exit, transformed Q'*B
 * @param[in] ldb Leading dimension of B (>= max(1,N))
 * @param[in,out] c P-by-N output matrix; on exit, transformed C*Z
 * @param[in] ldc Leading dimension of C (>= max(1,P))
 * @param[in,out] q N-by-N orthogonal transformation matrix Q
 * @param[in] ldq Leading dimension of Q (>= 1; >= N if COMPQ='I' or 'U')
 * @param[in,out] z N-by-N orthogonal transformation matrix Z
 * @param[in] ldz Leading dimension of Z (>= 1; >= N if COMPZ='I' or 'U')
 * @param[out] ncont Order of controllable part (Ac, Ec)
 * @param[out] niucon Number of uncontrollable infinite eigenvalues (for JOBCON='C')
 *                    Set to 0 for JOBCON='F' or 'I'
 * @param[out] nrblck Number of full row rank blocks in staircase form
 * @param[out] rtau Integer array, dimension (N), staircase block row dimensions
 * @param[in] tol Tolerance for rank determinations (TOL > 0 for user value, else default)
 *                Must be < 1
 * @param[out] iwork Integer workspace, dimension (M)
 * @param[out] dwork Double workspace, dimension (max(N, 2*M))
 * @param[out] info Exit code: 0 = success, <0 = invalid parameter -i
 */
void tg01hd(
    const char* jobcon, const char* compq, const char* compz,
    const i32 n, const i32 m, const i32 p,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* ncont, i32* niucon, i32* nrblck, i32* rtau,
    const f64 tol,
    i32* iwork, f64* dwork,
    i32* info
);

/**
 * @brief Orthogonal equivalence transformation of SISO descriptor system.
 *
 * Computes for a single-input single-output descriptor system (A, E, B, C)
 * with E upper triangular, a transformed system (Q'*A*Z, Q'*E*Z, Q'*B, C*Z)
 * via orthogonal equivalence transformation, so that Q'*B has only the first
 * element nonzero and Q'*E*Z remains upper triangular.
 *
 * @param[in] jobe Specifies E structure:
 *                 'U' = E is upper triangular
 *                 'I' = E is identity (not given)
 * @param[in] compc Transform C:
 *                  'C' = Transform output matrix C
 *                  'N' = Do not transform C
 * @param[in] compq Accumulate Q:
 *                  'N' = Do not form Q
 *                  'I' = Initialize Q to identity, return orthogonal Q
 *                  'U' = Update given Q1, return Q1*Q
 * @param[in] compz Accumulate Z:
 *                  'N' = Do not form Z
 *                  'I' = Initialize Z to identity, return orthogonal Z
 *                  'U' = Update given Z1, return Z1*Z
 * @param[in] n Dimension of state vector (N >= 0)
 * @param[in,out] a N-by-N state matrix; on exit Q'*A*Z
 * @param[in] lda Leading dimension of A (>= max(1,N))
 * @param[in,out] e N-by-N descriptor matrix (if JOBE='U'); on exit Q'*E*Z
 * @param[in] lde Leading dimension of E (>= max(1,N) if JOBE='U', >= 1 if 'I')
 * @param[in,out] b N-element input vector; on exit Q'*B with only b[0] nonzero
 * @param[in,out] c N-element output vector with stride INCC; on exit C*Z
 * @param[in] incc Stride of C (if COMPC='C', INCC > 0)
 * @param[in,out] q N-by-N orthogonal matrix Q
 * @param[in] ldq Leading dimension of Q (>= 1; >= N if COMPQ='I' or 'U')
 * @param[in,out] z N-by-N orthogonal matrix Z
 * @param[in] ldz Leading dimension of Z (>= 1; >= N if COMPZ='I' or 'U')
 * @param[out] info Exit code: 0 = success, <0 = invalid parameter -i
 */
void tg01kd(
    const char* jobe, const char* compc, const char* compq, const char* compz,
    const i32 n,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b,
    f64* c, const i32 incc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* info
);

/**
 * @brief Orthogonal reduction of descriptor system to observability staircase form.
 *
 * TG01ID computes orthogonal transformation matrices Q and Z which
 * reduce the N-th order descriptor system (A-lambda*E,B,C) to the form:
 *
 *   Q'*A*Z = ( Ano  * )    Q'*E*Z = ( Eno  * )    Q'*B = ( Bno )
 *            ( 0   Ao )             ( 0   Eo )           ( Bo  )
 *
 *   C*Z = ( 0   Co )
 *
 * where the NOBSV-th order descriptor system (Ao-lambda*Eo,Bo,Co)
 * is finite and/or infinite observable.
 *
 * @param[in] jobobs Specifies observability form:
 *                   'O': separate both finite and infinite unobservable eigenvalues
 *                   'F': separate only finite unobservable eigenvalues
 *                   'I': separate only nonzero finite and infinite unobservable eigenvalues
 * @param[in] compq  Specifies whether Q is computed:
 *                   'N': do not compute Q
 *                   'I': initialize Q to identity, return orthogonal Q
 *                   'U': on entry Q contains orthogonal Q1, return Q1*Q
 * @param[in] compz  Specifies whether Z is computed:
 *                   'N': do not compute Z
 *                   'I': initialize Z to identity, return orthogonal Z
 *                   'U': on entry Z contains orthogonal Z1, return Z1*Z
 * @param[in] n      Order of matrices A and E (N >= 0)
 * @param[in] m      Number of columns of B (M >= 0)
 * @param[in] p      Number of rows of C (P >= 0)
 * @param[in,out] a  N-by-N state matrix A (transformed to Q'*A*Z on exit)
 * @param[in] lda    Leading dimension of A (LDA >= max(1,N))
 * @param[in,out] e  N-by-N descriptor matrix E (transformed to Q'*E*Z on exit)
 * @param[in] lde    Leading dimension of E (LDE >= max(1,N))
 * @param[in,out] b  N-by-M input matrix B (transformed to Q'*B on exit)
 * @param[in] ldb    Leading dimension of B
 * @param[in,out] c  P-by-N output matrix C (transformed to C*Z on exit)
 * @param[in] ldc    Leading dimension of C (LDC >= max(1,M,P))
 * @param[in,out] q  N-by-N left transformation matrix
 * @param[in] ldq    Leading dimension of Q
 * @param[in,out] z  N-by-N right transformation matrix
 * @param[in] ldz    Leading dimension of Z
 * @param[out] nobsv Order of observable part
 * @param[out] niuobs Number of unobservable infinite eigenvalues (JOBOBS='O')
 * @param[out] nlblck Number of full column rank blocks in staircase form
 * @param[out] ctau  Column dimensions of full rank blocks (dimension N)
 * @param[in] tol    Tolerance for rank determination (TOL < 1)
 * @param[out] iwork Integer workspace (dimension P)
 * @param[out] dwork Double workspace (dimension max(N, 2*P))
 * @param[out] info  Error indicator (0 = success, -i = i-th argument invalid)
 */
void tg01id(
    const char* jobobs, const char* compq, const char* compz,
    const i32 n, const i32 m, const i32 p,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* nobsv, i32* niuobs, i32* nlblck, i32* ctau,
    const f64 tol,
    i32* iwork, f64* dwork,
    i32* info
);

/**
 * @brief Find irreducible descriptor representation.
 *
 * Finds a reduced (controllable, observable, or irreducible) descriptor
 * representation (Ar-lambda*Er,Br,Cr) for an original descriptor representation
 * (A-lambda*E,B,C). The pencil Ar-lambda*Er is in an upper block Hessenberg
 * form, with either Ar or Er upper triangular.
 *
 * @param[in] job Controls which parts to remove:
 *                'I' = irreducible (remove both uncontrollable and unobservable)
 *                'C' = controllable (remove uncontrollable only)
 *                'O' = observable (remove unobservable only)
 * @param[in] systyp Type of descriptor system:
 *                   'R' = rational transfer function
 *                   'S' = proper (standard) transfer function
 *                   'P' = polynomial transfer function
 * @param[in] equil Whether to perform preliminary scaling:
 *                  'S' = perform scaling
 *                  'N' = do not perform scaling
 * @param[in] n Dimension of descriptor state vector (N >= 0)
 * @param[in] m Dimension of input vector (M >= 0)
 * @param[in] p Dimension of output vector (P >= 0)
 * @param[in,out] a N-by-N state matrix; on exit, NR-by-NR reduced matrix Ar
 * @param[in] lda Leading dimension of A (>= max(1,N))
 * @param[in,out] e N-by-N descriptor matrix; on exit, NR-by-NR reduced matrix Er
 * @param[in] lde Leading dimension of E (>= max(1,N))
 * @param[in,out] b N-by-M input matrix (N-by-max(M,P) if JOB!='C');
 *                  on exit, NR-by-M reduced matrix Br
 * @param[in] ldb Leading dimension of B (>= max(1,N))
 * @param[in,out] c P-by-N output matrix (max(M,P)-by-N if JOB!='C');
 *                  on exit, P-by-NR reduced matrix Cr
 * @param[in] ldc Leading dimension of C (>= max(1,M,P) if N>0, >= 1 if N=0)
 * @param[out] nr Order of reduced descriptor representation
 * @param[out] infred Integer array, dimension 7:
 *                    [0-3]: Order reduction in phases 1-4 (-1 if not performed)
 *                    [4]: Number of nonzero sub-diagonals of Ar
 *                    [5]: Number of nonzero sub-diagonals of Er
 *                    [6]: Number of blocks in staircase form
 * @param[in] tol Tolerance for rank determination (default: N*N*eps if tol <= 0)
 * @param[out] iwork Integer workspace, size c*N+max(M,P) where c=2 if JOB='I'
 *                   or SYSTYP='R', c=1 otherwise
 * @param[out] dwork Double workspace
 * @param[in] ldwork Workspace size:
 *                   >= max(8*N,2*M,2*P) if EQUIL='S'
 *                   >= max(N,2*M,2*P) if EQUIL='N'
 *                   For better accuracy: >= 2*N*N+N*M+N*P+max(N,2*M,2*P)
 * @param[out] info Exit code (0 = success, <0 = invalid parameter -i)
 */
void tg01jd(
    const char* job, const char* systyp, const char* equil,
    const i32 n, const i32 m, const i32 p,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    i32* nr, i32* infred,
    const f64 tol,
    i32* iwork, f64* dwork, const i32 ldwork,
    i32* info
);

/**
 * @brief Reduced form for irreducible/controllable/observable descriptor system.
 *
 * Computes a reduced (controllable, observable, or irreducible) descriptor
 * representation (Ar-lambda*Er,Br,Cr) for a regular descriptor system
 * (A-lambda*E,B,C).
 *
 * @param[in] job 'I' irreducible, 'C' controllable, 'O' observable
 * @param[in] systyp 'R' remove finite+infinite, 'S' remove finite, 'P' remove infinite
 * @param[in] equil 'S' scale, 'N' no scaling
 * @param[in] cksing 'C' check singularity, 'N' no check
 * @param[in] restor 'R' restore original for max accuracy, 'N' no restore
 * @param[in] n System order (>= 0)
 * @param[in] m Number of inputs (>= 0)
 * @param[in] p Number of outputs (>= 0)
 * @param[in,out] a State matrix (n x n)
 * @param[in] lda Leading dimension of A (>= max(1,n))
 * @param[in,out] e Descriptor matrix (n x n)
 * @param[in] lde Leading dimension of E (>= max(1,n))
 * @param[in,out] b Input matrix (n x m)
 * @param[in] ldb Leading dimension of B (>= max(1,n))
 * @param[in,out] c Output matrix (p x n)
 * @param[in] ldc Leading dimension of C (>= max(1,p))
 * @param[out] nr Reduced system order
 * @param[out] infred Reduction info array (7 elements)
 * @param[in] tol Tolerance array (3 elements)
 * @param[out] iwork Integer workspace (2*n + max(m,p))
 * @param[out] dwork Real workspace (ldwork)
 * @param[in] ldwork Workspace size
 * @param[out] info Exit code: 0=success, <0=invalid param, >0=algorithm error
 */
void tg01jy(
    const char* job, const char* systyp, const char* equil,
    const char* cksing, const char* restor,
    const i32 n, const i32 m, const i32 p,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    i32* nr, i32* infred,
    const f64* tol,
    i32* iwork, f64* dwork, const i32 ldwork,
    i32* info
);

/**
 * @brief Finite-infinite decomposition of descriptor system.
 *
 * Computes orthogonal transformation matrices Q and Z which reduce
 * the regular pole pencil A-lambda*E of the descriptor system
 * (A-lambda*E,B,C) to the form (if JOB = 'F'):
 *
 *            ( Af  *  )             ( Ef  *  )
 *   Q'*A*Z = (        ) ,  Q'*E*Z = (        ) ,
 *            ( 0   Ai )             ( 0   Ei )
 *
 * or to the form (if JOB = 'I'):
 *
 *            ( Ai  *  )             ( Ei  *  )
 *   Q'*A*Z = (        ) ,  Q'*E*Z = (        ) ,
 *            ( 0   Af )             ( 0   Ef )
 *
 * where Af-lambda*Ef (Ef nonsingular, upper triangular) contains
 * finite eigenvalues, and Ai-lambda*Ei (Ai nonsingular, upper triangular)
 * contains infinite eigenvalues in staircase form.
 *
 * @param[in] job 'F' = finite-infinite, 'I' = infinite-finite separation
 * @param[in] joba 'H' = reduce Af to Hessenberg, 'N' = keep Af unreduced
 * @param[in] compq 'N' = no Q, 'I' = initialize Q to I, 'U' = update Q1*Q
 * @param[in] compz 'N' = no Z, 'I' = initialize Z to I, 'U' = update Z1*Z
 * @param[in] n Order of A and E (n >= 0)
 * @param[in] m Number of columns of B (m >= 0)
 * @param[in] p Number of rows of C (p >= 0)
 * @param[in,out] a N-by-N state matrix; returns Q'*A*Z
 * @param[in] lda Leading dimension of A (>= max(1,n))
 * @param[in,out] e N-by-N descriptor matrix; returns Q'*E*Z
 * @param[in] lde Leading dimension of E (>= max(1,n))
 * @param[in,out] b N-by-M input matrix; returns Q'*B
 * @param[in] ldb Leading dimension of B (>= max(1,n))
 * @param[in,out] c P-by-N output matrix; returns C*Z
 * @param[in] ldc Leading dimension of C (>= max(1,p) if JOB='F', >= max(1,m,p) if JOB='I')
 * @param[in,out] q N-by-N left transformation matrix
 * @param[in] ldq Leading dimension of Q (>= 1 if COMPQ='N', >= max(1,n) otherwise)
 * @param[in,out] z N-by-N right transformation matrix
 * @param[in] ldz Leading dimension of Z (>= 1 if COMPZ='N', >= max(1,n) otherwise)
 * @param[out] nf Order of Af and Ef (number of finite eigenvalues)
 * @param[out] nd Number of non-dynamic infinite eigenvalues
 * @param[out] niblck Number of infinite blocks minus one (0 if nd=0)
 * @param[out] iblck Array of size N; iblck[i] = dimension of i-th block
 * @param[in] tol Tolerance for rank decisions (use <= 0 for default)
 * @param[out] iwork Integer workspace (n)
 * @param[out] dwork Double workspace (ldwork)
 * @param[in] ldwork Workspace size (>= n + max(3*n, m, p) if n > 0, >= 1 otherwise)
 * @param[out] info 0=success, <0=invalid param, 1=pencil not regular
 */
void tg01ld(
    const char* job, const char* joba, const char* compq, const char* compz,
    const i32 n, const i32 m, const i32 p,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* nf, i32* nd, i32* niblck, i32* iblck,
    const f64 tol,
    i32* iwork, f64* dwork, const i32 ldwork,
    i32* info
);

/**
 * @brief Finite-infinite decomposition of structured descriptor system.
 *
 * Helper routine for TG01LD. Reduces a regular pole pencil A-lambda*E
 * of descriptor system (A-lambda*E,B,C) with A and E in structured form
 * (from TG01FD) to finite-infinite separated form.
 *
 * @param[in] compq true = compute/update Q, false = do not compute Q
 * @param[in] compz true = compute/update Z, false = do not compute Z
 * @param[in] n Order of A and E (n >= 0)
 * @param[in] m Number of columns of B (m >= 0)
 * @param[in] p Number of rows of C (p >= 0)
 * @param[in] ranke Rank of E matrix from TG01FD
 * @param[in] rnka22 Rank of A22 submatrix from TG01FD
 * @param[in,out] a N-by-N state matrix
 * @param[in] lda Leading dimension of A (>= max(1,n))
 * @param[in,out] e N-by-N descriptor matrix
 * @param[in] lde Leading dimension of E (>= max(1,n))
 * @param[in,out] b N-by-M input matrix
 * @param[in] ldb Leading dimension of B (>= max(1,n))
 * @param[in,out] c P-by-N output matrix
 * @param[in] ldc Leading dimension of C (>= max(1,p))
 * @param[in,out] q N-by-N left transformation matrix
 * @param[in] ldq Leading dimension of Q
 * @param[in,out] z N-by-N right transformation matrix
 * @param[in] ldz Leading dimension of Z
 * @param[out] nf Number of finite eigenvalues
 * @param[out] niblck Number of infinite blocks minus one
 * @param[out] iblck Dimension of infinite blocks
 * @param[in] tol Tolerance for rank decisions
 * @param[out] iwork Integer workspace (n)
 * @param[out] dwork Double workspace (ldwork)
 * @param[in] ldwork Workspace size
 * @param[out] info 0=success, 1=pencil not regular
 */
void tg01ly(
    const bool compq, const bool compz,
    const i32 n, const i32 m, const i32 p,
    const i32 ranke, const i32 rnka22,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* nf, i32* niblck, i32* iblck,
    const f64 tol,
    i32* iwork, f64* dwork, const i32 ldwork,
    i32* info
);

/**
 * @brief Complex SISO descriptor system equivalence transformation.
 *
 * Computes for a single-input single-output descriptor system (A, E, B, C)
 * with E upper triangular, a transformed system (Q'*A*Z, Q'*E*Z, Q'*B, C*Z)
 * via unitary equivalence transformation, so that Q'*B has only the first
 * element nonzero and Q'*E*Z remains upper triangular.
 *
 * Uses Givens rotations to annihilate the last N-1 elements of B in reverse
 * order while preserving the upper triangular form of E.
 *
 * @param[in] jobe Specifies E matrix type:
 *                 'U' = E is upper triangular
 *                 'I' = E is assumed identity and not referenced
 * @param[in] compc Transform C matrix:
 *                  'C' = Transform output matrix C
 *                  'N' = Do not transform C
 * @param[in] compq Accumulate Q transformations:
 *                  'N' = Do not form Q
 *                  'I' = Q initialized to identity, orthogonal Q returned
 *                  'U' = Given Q updated by orthogonal transformations
 * @param[in] compz Accumulate Z transformations:
 *                  'N' = Do not form Z
 *                  'I' = Z initialized to identity, orthogonal Z returned
 *                  'U' = Given Z updated by orthogonal transformations
 * @param[in] n Dimension of descriptor state vector (N >= 0)
 * @param[in,out] a COMPLEX*16 array, dimension (LDA,N)
 *                  In: N-by-N state matrix A
 *                  Out: Transformed matrix Q'*A*Z
 * @param[in] lda Leading dimension of A (>= max(1,N))
 * @param[in,out] e COMPLEX*16 array, dimension (LDE,*)
 *                  In: N-by-N upper triangular descriptor matrix E (if JOBE='U')
 *                  Out: Upper triangular part of Q'*E*Z (if JOBE='U')
 *                  Not referenced if JOBE='I'
 * @param[in] lde Leading dimension of E (>= max(1,N) if JOBE='U', >= 1 if JOBE='I')
 * @param[in,out] b COMPLEX*16 array, dimension (N)
 *                  In: Input vector B
 *                  Out: Transformed Q'*B with B[1:N-1] = 0
 * @param[in,out] c COMPLEX*16 array, dimension ((N-1)*INCC+1)
 *                  In: Output vector C at indices 0, INCC, ..., (N-1)*INCC (if COMPC='C')
 *                  Out: Transformed C*Z (if COMPC='C')
 *                  Not referenced if COMPC='N'
 * @param[in] incc Increment between successive C values (> 0 if COMPC='C')
 * @param[in,out] q COMPLEX*16 array, dimension (LDQ,*)
 *                  In: Given Q1 if COMPQ='U'
 *                  Out: Orthogonal transformation matrix (Q1*Q if COMPQ='U')
 *                  Not referenced if COMPQ='N'
 * @param[in] ldq Leading dimension of Q (>= max(1,N) if COMPQ != 'N', else >= 1)
 * @param[in,out] z COMPLEX*16 array, dimension (LDZ,*)
 *                  In: Given Z1 if COMPZ='U'
 *                  Out: Orthogonal transformation matrix (Z1*Z if COMPZ='U')
 *                  Not referenced if COMPZ='N'
 * @param[in] ldz Leading dimension of Z (>= max(1,N) if COMPZ != 'N', else >= 1)
 * @param[out] info Exit code: 0 = success, <0 = invalid parameter -i
 */
void tg01kz(
    const char* jobe, const char* compc, const char* compq, const char* compz,
    const i32 n,
    c128* a, const i32 lda,
    c128* e, const i32 lde,
    c128* b,
    c128* c, const i32 incc,
    c128* q, const i32 ldq,
    c128* z, const i32 ldz,
    i32* info
);

/**
 * @brief Block-diagonal decomposition of descriptor system in generalized Schur form.
 *
 * @param[in] jobt 'D' = compute Q, Z; 'I' = compute inv(Q), inv(Z)
 * @param[in] n Order of A, E (n >= 0)
 * @param[in] m Number of columns of B (m >= 0)
 * @param[in] p Number of rows of C (p >= 0)
 * @param[in] ndim Dimension of leading diagonal blocks (0 <= ndim <= n)
 * @param[in,out] a State matrix in Schur form (n x n)
 * @param[in] lda Leading dimension of A (>= max(1,n))
 * @param[in,out] e Descriptor matrix, upper triangular (n x n)
 * @param[in] lde Leading dimension of E (>= max(1,n))
 * @param[in,out] b Input matrix (n x m)
 * @param[in] ldb Leading dimension of B (>= max(1,n))
 * @param[in,out] c Output matrix (p x n)
 * @param[in] ldc Leading dimension of C (>= max(1,p))
 * @param[in,out] q Left transformation matrix (n x n)
 * @param[in] ldq Leading dimension of Q (>= max(1,n))
 * @param[in,out] z Right transformation matrix (n x n)
 * @param[in] ldz Leading dimension of Z (>= max(1,n))
 * @param[out] iwork Integer workspace (n+6)
 * @param[out] info Exit code: 0=success, <0=invalid param, 1=separation failed
 */
void tg01nx(
    const char* jobt,
    const i32 n, const i32 m, const i32 p, const i32 ndim,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* iwork,
    i32* info
);

/**
 * @brief Orthogonal equivalence transformation of SISO descriptor system.
 *
 * Computes for a single-input single-output descriptor system, given by
 * the system matrix:
 *
 *     [ D     C    ]
 *     [ B  A - s*E ]
 *
 * with E upper triangular, a transformed system (Q'*A*Z, Q'*E*Z, Q'*B, C*Z),
 * via orthogonal equivalence transformation, so that Q'*B has only the first
 * element nonzero and Q'*E*Z remains upper triangular.
 *
 * @param[in] jobe Specifies E structure:
 *                 'U' = E is upper triangular
 *                 'I' = E is identity (not given)
 * @param[in] n Dimension of state vector (N >= 0)
 * @param[in,out] dcba DOUBLE PRECISION array, dimension (LDDCBA, N+1)
 *                     In: Contains [D C; B A] where D is scalar, C is 1xN,
 *                         B is Nx1, A is NxN
 *                     Out: Contains [D C*Z; Q'*B Q'*A*Z]
 * @param[in] lddcba Leading dimension of DCBA (>= N+1)
 * @param[in,out] e DOUBLE PRECISION array, dimension (LDE, *)
 *                  In: N-by-N upper triangular matrix E (if JOBE='U')
 *                  Out: Upper triangular Q'*E*Z (if JOBE='U')
 *                  Not referenced if JOBE='I'
 * @param[in] lde Leading dimension of E (>= max(1,N) if JOBE='U', >= 1 if JOBE='I')
 * @param[out] info Exit code: 0 = success, <0 = invalid parameter -i
 */
void tg01oa(
    const char* jobe,
    const i32 n,
    f64* dcba, const i32 lddcba,
    f64* e, const i32 lde,
    i32* info
);

/**
 * @brief Unitary equivalence transformation of complex SISO descriptor system.
 *
 * Computes for a single-input single-output descriptor system with complex
 * matrices, given by the system matrix:
 *
 *     [ D     C    ]
 *     [ B  A - s*E ]
 *
 * with E upper triangular, a transformed system (Q'*A*Z, Q'*E*Z, Q'*B, C*Z),
 * via unitary equivalence transformation, so that Q'*B has only the first
 * element nonzero and Q'*E*Z remains upper triangular.
 *
 * @param[in] jobe Specifies E structure:
 *                 'U' = E is upper triangular
 *                 'I' = E is identity (not given)
 * @param[in] n Dimension of state vector (N >= 0)
 * @param[in,out] dcba COMPLEX*16 array, dimension (LDDCBA, N+1)
 *                     In: Contains [D C; B A] where D is scalar, C is 1xN,
 *                         B is Nx1, A is NxN
 *                     Out: Contains [D C*Z; Q'*B Q'*A*Z]
 * @param[in] lddcba Leading dimension of DCBA (>= N+1)
 * @param[in,out] e COMPLEX*16 array, dimension (LDE, *)
 *                  In: N-by-N upper triangular matrix E (if JOBE='U')
 *                  Out: Upper triangular Q'*E*Z (if JOBE='U')
 *                  Not referenced if JOBE='I'
 * @param[in] lde Leading dimension of E (>= max(1,N) if JOBE='U', >= 1 if JOBE='I')
 * @param[out] info Exit code: 0 = success, <0 = invalid parameter -i
 */
void tg01ob(
    const char* jobe,
    const i32 n,
    c128* dcba, const i32 lddcba,
    c128* e, const i32 lde,
    i32* info
);

/**
 * @brief Finite-infinite generalized real Schur form decomposition of descriptor system.
 *
 * Computes orthogonal transformation matrices Q and Z which reduce the
 * regular pole pencil A-lambda*E of the descriptor system (A-lambda*E,B,C)
 * to the form (if JOB = 'F'):
 *
 *            ( Af  *  )             ( Ef  *  )
 *   Q'*A*Z = (        ) ,  Q'*E*Z = (        )
 *            ( 0   Ai )             ( 0   Ei )
 *
 * or to the form (if JOB = 'I'):
 *
 *            ( Ai  *  )             ( Ei  *  )
 *   Q'*A*Z = (        ) ,  Q'*E*Z = (        )
 *            ( 0   Af )             ( 0   Ef )
 *
 * where (Af,Ef) is in generalized real Schur form with Ef nonsingular
 * and upper triangular and Af in real Schur form. The subpencil
 * Af-lambda*Ef contains the finite eigenvalues.
 *
 * @param[in] job 'F' = finite-infinite, 'I' = infinite-finite separation
 * @param[in] n Order of A and E (n >= 0)
 * @param[in] m Number of columns of B (m >= 0)
 * @param[in] p Number of rows of C (p >= 0)
 * @param[in,out] a N-by-N state matrix; returns Q'*A*Z
 * @param[in] lda Leading dimension of A (>= max(1,n))
 * @param[in,out] e N-by-N descriptor matrix; returns Q'*E*Z
 * @param[in] lde Leading dimension of E (>= max(1,n))
 * @param[in,out] b N-by-M input matrix; returns Q'*B
 * @param[in] ldb Leading dimension of B (>= max(1,n))
 * @param[in,out] c P-by-N output matrix; returns C*Z
 * @param[in] ldc Leading dimension of C (>= max(1,p))
 * @param[out] alphar Real parts of generalized eigenvalues
 * @param[out] alphai Imaginary parts of generalized eigenvalues
 * @param[out] beta Denominators of generalized eigenvalues
 * @param[out] q N-by-N left orthogonal transformation
 * @param[in] ldq Leading dimension of Q (>= max(1,n))
 * @param[out] z N-by-N right orthogonal transformation
 * @param[in] ldz Leading dimension of Z (>= max(1,n))
 * @param[out] nf Order of finite eigenvalue subpencil
 * @param[out] nd Number of non-dynamic infinite eigenvalues
 * @param[out] niblck Number of infinite blocks minus one (0 if nd=0)
 * @param[out] iblck Block dimensions array (size n)
 * @param[in] tol Tolerance for rank decisions (use <= 0 for default)
 * @param[out] iwork Integer workspace (n)
 * @param[out] dwork Double workspace (ldwork)
 * @param[in] ldwork Workspace size (>= 4*n if n>0, >= 1 otherwise)
 * @param[out] info 0=success, <0=invalid param, 1=pencil not regular, 2=QZ failed
 */
void tg01md(
    const char* job, const i32 n, const i32 m, const i32 p,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* alphar, f64* alphai, f64* beta,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* nf, i32* nd, i32* niblck, i32* iblck,
    const f64 tol,
    i32* iwork, f64* dwork, const i32 ldwork,
    i32* info
);

/**
 * @brief Reduce SISO descriptor system so feedthrough has large magnitude.
 *
 * Computes for a single-input single-output descriptor system, given by
 * the system matrix:
 *
 *     [ D     C    ]
 *     [ B  A - s*E ]
 *
 * with E nonsingular, a reduced system matrix:
 *
 *     [ d     c    ]
 *     [ b  a - s*e ]
 *
 * such that d has a "sufficiently" large magnitude.
 *
 * @param[in] jobe Specifies E matrix type:
 *                 'G' = E is a general nonsingular matrix
 *                 'I' = E is assumed identity (not given)
 * @param[in] n Dimension of descriptor state vector (N >= 0)
 * @param[in,out] dcba DOUBLE PRECISION array, dimension (LDDCBA, N+1)
 *                     In: Contains [D C; B A] where D is scalar, C is 1xN,
 *                         B is Nx1, A is NxN
 *                     Out: Leading (NZ+1)x(NZ+1) part contains reduced
 *                          [d c; b a]
 * @param[in] lddcba Leading dimension of DCBA (>= N+1)
 * @param[in,out] e DOUBLE PRECISION array, dimension (LDE, *)
 *                  In: N-by-N nonsingular descriptor matrix E (if JOBE='G')
 *                  Out: Leading NZ-by-NZ part contains reduced e (if JOBE='G')
 *                  Not referenced if JOBE='I'
 * @param[in] lde Leading dimension of E (>= max(1,N) if JOBE='G', >= 1 if JOBE='I')
 * @param[out] nz Order of the reduced system
 * @param[out] g Gain of the reduced system
 * @param[in] tol Tolerance for determining if d is large enough.
 *                If tol <= 0, default tol = EPS^(3/4) is used.
 * @param[out] dwork DOUBLE PRECISION workspace, dimension (LDWORK)
 *                   On exit, dwork[0] = optimal LDWORK
 * @param[in] ldwork Length of DWORK:
 *                   >= 2*N+1 if JOBE='G'
 *                   >= N+1 if JOBE='I'
 *                   If ldwork=-1, workspace query mode
 * @param[out] info Exit code: 0 = success, <0 = invalid parameter -i
 */
void tg01od(
    const char* jobe,
    const i32 n,
    f64* dcba, const i32 lddcba,
    f64* e, const i32 lde,
    i32* nz,
    f64* g,
    const f64 tol,
    f64* dwork, const i32 ldwork,
    i32* info
);

/**
 * @brief Reduce complex SISO descriptor system for sufficiently large feedthrough.
 *
 * Computes for a single-input single-output descriptor system with complex
 * elements, given by the system matrix:
 *
 *     [ D     C    ]
 *     [ B  A - s*E ]
 *
 * with E nonsingular, a reduced system matrix:
 *
 *     [ d     c    ]
 *     [ b  a - s*e ]
 *
 * such that d has a "sufficiently" large magnitude.
 *
 * Uses Householder transformations and Givens rotations to process the matrices.
 * If E is a general matrix, it is first triangularized using QR decomposition.
 *
 * @param[in] jobe Specifies E matrix type:
 *                 'G' = E is a general matrix
 *                 'I' = E is identity (not given)
 * @param[in] n Dimension of state vector (N >= 0)
 * @param[in,out] dcba COMPLEX*16 array, dimension (LDDCBA, N+1)
 *                     In: Contains [D C; B A] where D is scalar, C is 1xN,
 *                         B is Nx1, A is NxN
 *                     Out: Leading (NZ+1)x(NZ+1) contains reduced [d c; b a]
 * @param[in] lddcba Leading dimension of DCBA (>= N+1)
 * @param[in,out] e COMPLEX*16 array, dimension (LDE, *)
 *                  In: N-by-N nonsingular descriptor matrix E (if JOBE='G')
 *                  Out: NZ-by-NZ reduced descriptor matrix e (if JOBE='G')
 *                  Not referenced if JOBE='I'
 * @param[in] lde Leading dimension of E (>= max(1,N) if JOBE='G', >= 1 if JOBE='I')
 * @param[out] nz Order of the reduced system
 * @param[out] g Gain of the reduced system
 * @param[in] tol Tolerance for determining if d is sufficiently large
 *                If TOL <= 0, default TOLDEF = EPS^(3/4) is used
 * @param[out] zwork COMPLEX*16 workspace, dimension (LZWORK)
 *                   On exit, ZWORK[0] returns optimal LZWORK
 * @param[in] lzwork Workspace size
 *                   >= 2*N+1 if JOBE='G'
 *                   >= N+1   if JOBE='I'
 *                   If LZWORK=-1, workspace query is performed
 * @param[out] info Exit code: 0 = success, <0 = parameter -i is invalid
 */
void tg01oz(
    const char* jobe,
    const i32 n,
    c128* dcba, const i32 lddcba,
    c128* e, const i32 lde,
    i32* nz,
    c128* g,
    const f64 tol,
    c128* zwork, const i32 lzwork,
    i32* info
);

/**
 * @brief Bi-domain spectral splitting of a subpencil of a descriptor system.
 *
 * Computes orthogonal transformation matrices Q and Z which reduce the
 * regular pole pencil A-lambda*E of the descriptor system (A-lambda*E,B,C)
 * to generalized real Schur form with ordered generalized eigenvalues.
 * The pair (A,E) is reduced to:
 *
 *            ( *  *  *  * )             ( *  *  *  * )
 *   Q'*A*Z = ( 0  A1 *  * ) ,  Q'*E*Z = ( 0  E1 *  * )
 *            ( 0  0  A2 * )             ( 0  0  E2 * )
 *            ( 0  0  0  * )             ( 0  0  0  * )
 *
 * where subpencil A1-lambda*E1 contains eigenvalues in domain of interest
 * and A2-lambda*E2 contains eigenvalues outside the domain of interest.
 *
 * @param[in] dico Type of system: 'C' = continuous-time, 'D' = discrete-time
 * @param[in] stdom Domain type: 'S' = stability (left half plane/inside circle),
 *                  'U' = instability (right half plane/outside circle)
 * @param[in] jobae Shape of (A,E): 'S' = generalized Schur form, 'G' = general
 * @param[in] compq 'I' = initialize Q to identity, 'U' = update existing Q
 *                  (COMPQ='U' not allowed when JOBAE='G')
 * @param[in] compz 'I' = initialize Z to identity, 'U' = update existing Z
 *                  (COMPZ='U' not allowed when JOBAE='G')
 * @param[in] n Order of matrices A and E (N >= 0)
 * @param[in] m Number of columns of B (M >= 0)
 * @param[in] p Number of rows of C (P >= 0)
 * @param[in] nlow Lower boundary index for subpencil (0 <= NLOW <= NSUP <= N if JOBAE='S',
 *                 NLOW = min(1,N) if JOBAE='G')
 * @param[in] nsup Upper boundary index for subpencil (NSUP = N if JOBAE='G')
 * @param[in] alpha Domain boundary: real part (continuous) or modulus (discrete, >= 0)
 * @param[in,out] a N-by-N state dynamics matrix. On exit: Q'*A*Z in Schur form
 * @param[in] lda Leading dimension of A (>= max(1,N))
 * @param[in,out] e N-by-N descriptor matrix. On exit: Q'*E*Z upper triangular
 * @param[in] lde Leading dimension of E (>= max(1,N))
 * @param[in,out] b N-by-M input matrix. On exit: Q'*B
 * @param[in] ldb Leading dimension of B (>= max(1,N))
 * @param[in,out] c P-by-N output matrix. On exit: C*Z
 * @param[in] ldc Leading dimension of C (>= max(1,P))
 * @param[in,out] q N-by-N orthogonal transformation matrix Q
 * @param[in] ldq Leading dimension of Q (>= max(1,N))
 * @param[in,out] z N-by-N orthogonal transformation matrix Z
 * @param[in] ldz Leading dimension of Z (>= max(1,N))
 * @param[out] ndim Number of eigenvalues in domain of interest
 * @param[out] alphar Real parts of generalized eigenvalues (size N)
 * @param[out] alphai Imaginary parts of generalized eigenvalues (size N)
 * @param[out] beta Denominators of generalized eigenvalues (size N)
 * @param[out] dwork Workspace (size LDWORK)
 * @param[in] ldwork Workspace size (>= 8*N+16 if JOBAE='G', >= 4*N+16 if JOBAE='S',
 *                   or -1 for workspace query)
 * @param[out] info 0=success, <0=invalid param -i, 1=QZ failed, 2=ordering failed
 */
void tg01pd(
    const char* dico, const char* stdom, const char* jobae,
    const char* compq, const char* compz,
    const i32 n, const i32 m, const i32 p,
    const i32 nlow, const i32 nsup, const f64 alpha,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* ndim,
    f64* alphar, f64* alphai, f64* beta,
    f64* dwork, const i32 ldwork,
    i32* info
);

/**
 * @brief Three-domain spectral splitting of a descriptor system.
 *
 * Computes orthogonal transformation matrices Q and Z which reduce
 * the regular pole pencil A-lambda*E of the descriptor system
 * (A-lambda*E,B,C) to generalized real Schur form with ordered
 * generalized eigenvalues. The pair (A,E) is reduced to the form:
 *
 *            ( A1  *   *  )             ( E1  *   *  )
 *   Q'*A*Z = ( 0   A2  *  ) ,  Q'*E*Z = ( 0   E2  *  )
 *            ( 0   0   A3 )             ( 0   0   E3 )
 *
 * where the subpencils Ak-lambda*Ek, for k = 1, 2, 3, contain the
 * generalized eigenvalues which belong to certain domains of interest.
 *
 * @param[in] dico System type: 'C' = continuous-time, 'D' = discrete-time
 * @param[in] stdom Domain type: 'S' = stability (left half-plane/inside circle),
 *                  'U' = instability (right half-plane/outside circle),
 *                  'N' = whole domain except infinity
 * @param[in] jobfi Eigenvalue ordering: 'F' = finite first, 'I' = infinite first
 * @param[in] n Order of matrices A and E (N >= 0)
 * @param[in] m Number of columns of B (M >= 0)
 * @param[in] p Number of rows of C (P >= 0)
 * @param[in] alpha Domain boundary: real part (continuous) or modulus (discrete, >= 0)
 * @param[in,out] a N-by-N state dynamics matrix. On exit: Q'*A*Z in Schur form
 * @param[in] lda Leading dimension of A (>= max(1,N))
 * @param[in,out] e N-by-N descriptor matrix. On exit: Q'*E*Z upper triangular
 * @param[in] lde Leading dimension of E (>= max(1,N))
 * @param[in,out] b N-by-M input matrix. On exit: Q'*B
 * @param[in] ldb Leading dimension of B (>= max(1,N))
 * @param[in,out] c P-by-N output matrix. On exit: C*Z
 * @param[in] ldc Leading dimension of C (>= max(1,P))
 * @param[out] n1 Number of eigenvalues in first block (A1,E1)
 * @param[out] n2 Number of eigenvalues in second block (A2,E2)
 * @param[out] n3 Number of eigenvalues in third block (A3,E3)
 * @param[out] nd Number of non-dynamic infinite eigenvalues
 * @param[out] niblck Number of infinite blocks minus one (if ND > 0)
 * @param[out] iblck Block dimensions for infinite part (size N)
 * @param[out] q N-by-N left orthogonal transformation matrix
 * @param[in] ldq Leading dimension of Q (>= max(1,N))
 * @param[out] z N-by-N right orthogonal transformation matrix
 * @param[in] ldz Leading dimension of Z (>= max(1,N))
 * @param[out] alphar Real parts of generalized eigenvalues (size N)
 * @param[out] alphai Imaginary parts of generalized eigenvalues (size N)
 * @param[out] beta Denominators of generalized eigenvalues (size N)
 * @param[in] tol Tolerance for rank decisions (< 1, <= 0 for default)
 * @param[out] iwork Integer workspace (size N)
 * @param[out] dwork Workspace (size LDWORK)
 * @param[in] ldwork Workspace size (>= 4*N if STDOM='N', >= 4*N+16 otherwise,
 *                   or -1 for workspace query)
 * @param[out] info 0=success, <0=invalid param -i, 1=not regular,
 *                  2=QZ failed, 3=ordering failed
 */
void tg01qd(
    const char* dico, const char* stdom, const char* jobfi,
    const i32 n, const i32 m, const i32 p, const f64 alpha,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    i32* n1, i32* n2, i32* n3, i32* nd, i32* niblck, i32* iblck,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    f64* alphar, f64* alphai, f64* beta,
    const f64 tol,
    i32* iwork, f64* dwork, const i32 ldwork,
    i32* info
);

/**
 * @brief Finite-infinite block-diagonal decomposition of a descriptor system.
 *
 * Computes equivalence transformation matrices Q and Z which reduce
 * the regular pole pencil A-lambda*E of the descriptor system
 * (A-lambda*E,B,C) to block-diagonal form:
 *
 * JOB = 'F':
 *            ( Af  0  )             ( Ef  0  )
 *    Q*A*Z = (        ) ,   Q*E*Z = (        )
 *            ( 0   Ai )             ( 0   Ei )
 *
 * JOB = 'I':
 *            ( Ai  0  )             ( Ei  0  )
 *    Q*A*Z = (        ) ,   Q*E*Z = (        )
 *            ( 0   Af )             ( 0   Ef )
 *
 * where (Af,Ef) is in generalized real Schur form with Ef nonsingular
 * and upper triangular. The subpencil Af-lambda*Ef contains the finite
 * eigenvalues. The pair (Ai,Ei) is in generalized real Schur form with
 * Ai nonsingular and Ei nilpotent (upper triangular).
 *
 * @param[in] job Separation type: 'F' = finite-infinite, 'I' = infinite-finite
 * @param[in] jobt Transformation type: 'D' = direct Q, Z;
 *                 'I' = inverse inv(Q), inv(Z)
 * @param[in] n Order of matrices A and E (N >= 0)
 * @param[in] m Number of columns of B (M >= 0)
 * @param[in] p Number of rows of C (P >= 0)
 * @param[in,out] a N-by-N state dynamics matrix. On exit: transformed matrix
 * @param[in] lda Leading dimension of A (>= max(1,N))
 * @param[in,out] e N-by-N descriptor matrix. On exit: transformed matrix
 * @param[in] lde Leading dimension of E (>= max(1,N))
 * @param[in,out] b N-by-M input matrix. On exit: Q*B or inv(Q)*B
 * @param[in] ldb Leading dimension of B (>= max(1,N))
 * @param[in,out] c P-by-N output matrix. On exit: C*Z or C*inv(Z)
 * @param[in] ldc Leading dimension of C (>= max(1,P))
 * @param[out] alphar Real parts of finite eigenvalues (size N)
 * @param[out] alphai Imaginary parts of finite eigenvalues (size N)
 * @param[out] beta Denominators of finite eigenvalues (size N)
 * @param[out] q N-by-N left transformation matrix Q (or inv(Q))
 * @param[in] ldq Leading dimension of Q (>= max(1,N))
 * @param[out] z N-by-N right transformation matrix Z (or inv(Z))
 * @param[in] ldz Leading dimension of Z (>= max(1,N))
 * @param[out] nf Order of finite part (Af, Ef)
 * @param[out] nd Number of non-dynamic infinite eigenvalues
 * @param[out] niblck Number of infinite blocks minus one (or 0 if ND=0)
 * @param[out] iblck Block dimensions (size N)
 * @param[in] tol Tolerance for rank decisions (0 = default = N^2*EPS)
 * @param[out] iwork Integer workspace (size N+6)
 * @param[out] dwork Workspace (size LDWORK)
 * @param[in] ldwork Workspace size (>= 4*N if N>0, >= 1 if N=0, or -1 for query)
 * @param[out] info 0=success, <0=invalid param -i, 1=not regular,
 *                  2=QZ failed, 3=Sylvester eq. failed
 */
void tg01nd(
    const char* job, const char* jobt,
    const i32 n, const i32 m, const i32 p,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* alphar, f64* alphai, f64* beta,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    i32* nf, i32* nd, i32* niblck, i32* iblck,
    const f64 tol,
    i32* iwork, f64* dwork, const i32 ldwork,
    i32* info
);

/**
 * @brief Reduce descriptor system to generalized real Schur form
 *
 * Reduces the pair (A,E) to a real generalized Schur form using an orthogonal
 * equivalence transformation (A,E) <-- (Q'*A*Z, Q'*E*Z) and applies the
 * transformation to matrices B and C: B <-- Q'*B and C <-- C*Z.
 *
 * @param[in] n Order of matrices A and E (N >= 0)
 * @param[in] m Number of columns of B (M >= 0)
 * @param[in] p Number of rows of C (P >= 0)
 * @param[in,out] a N-by-N state dynamics matrix. On exit: Q'*A*Z in upper
 *                  quasi-triangular form (elements below first subdiagonal = 0)
 * @param[in] lda Leading dimension of A (>= max(1,N))
 * @param[in,out] e N-by-N descriptor matrix. On exit: Q'*E*Z upper triangular
 * @param[in] lde Leading dimension of E (>= max(1,N))
 * @param[in,out] b N-by-M input matrix. On exit: Q'*B
 * @param[in] ldb Leading dimension of B (>= max(1,N))
 * @param[in,out] c P-by-N output matrix. On exit: C*Z
 * @param[in] ldc Leading dimension of C (>= max(1,P))
 * @param[out] q N-by-N left orthogonal transformation matrix
 * @param[in] ldq Leading dimension of Q (>= max(1,N))
 * @param[out] z N-by-N right orthogonal transformation matrix
 * @param[in] ldz Leading dimension of Z (>= max(1,N))
 * @param[out] alphar Real parts of generalized eigenvalues (size N)
 * @param[out] alphai Imaginary parts of generalized eigenvalues (size N)
 * @param[out] beta Denominators of generalized eigenvalues (size N)
 * @param[out] dwork Workspace (size LDWORK). On exit, DWORK[0] = optimal LDWORK
 * @param[in] ldwork Workspace size (>= 8*N+16)
 * @param[out] info 0=success, <0=invalid param -i, >0=QZ algorithm failed
 */
void tg01wd(
    const i32 n, const i32 m, const i32 p,
    f64* a, const i32 lda,
    f64* e, const i32 lde,
    f64* b, const i32 ldb,
    f64* c, const i32 ldc,
    f64* q, const i32 ldq,
    f64* z, const i32 ldz,
    f64* alphar, f64* alphai, f64* beta,
    f64* dwork, const i32 ldwork,
    i32* info
);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_TG_H */
