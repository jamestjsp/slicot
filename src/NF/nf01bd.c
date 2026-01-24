/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>

/**
 * @brief Calculate the Jacobian of the Wiener system.
 *
 * Calculates the Jacobian dy/dX of the Wiener system.
 *
 * @param[in] cjte 'C' to compute J'*e, 'N' to skip.
 * @param[in] nsmp Number of training samples.
 * @param[in] m Length of each input sample.
 * @param[in] l Length of each output sample.
 * @param[in,out] ipar Integer parameters (n, nn).
 * @param[in] lipar Length of ipar.
 * @param[in,out] x Parameter vector.
 * @param[in] lx Length of x.
 * @param[in] u Input samples.
 * @param[in] ldu Leading dimension of u.
 * @param[in] e Error vector (if cjte='C').
 * @param[out] j Jacobian matrix.
 * @param[in,out] ldj Leading dimension of j.
 * @param[out] jte J'*e product (if cjte='C').
 * @param[out] dwork Workspace.
 * @param[in] ldwork Length of dwork.
 * @param[out] info Exit code.
 */
void nf01bd(const char *cjte, i32 nsmp, i32 m, i32 l, i32 *ipar, i32 lipar, 
            f64 *x, i32 lx, f64 *u, i32 ldu, f64 *e, f64 *j, i32 *ldj, 
            f64 *jte, f64 *dwork, i32 ldwork, i32 *info)
{
    /* Local variables */
    i32 ac, bd, bsn, i, ix, iy, jw, k, kcol, ldac, lpar, lths, n, nn, nsml, nths, z;
    f64 eps, h, parsav;
    bool wjte;
    f64 zero = 0.0, one = 1.0, epsfcn = 0.0;
    i32 inc_1 = 1, inc_0 = 0;

    wjte = (cjte[0] == 'C' || cjte[0] == 'c');
    n = ipar[0];
    nn = ipar[1];
    bsn = nn * (l + 2) + 1;
    nsml = nsmp * l;
    nths = bsn * l;
    lths = n * (m + l + 1) + l * m;
    lpar = nths + lths;

    *info = 0;
    if (!(wjte || cjte[0] == 'N' || cjte[0] == 'n')) *info = -1;
    else if (nsmp < 0) *info = -2;
    else if (m < 0) *info = -3;
    else if (l < 0) *info = -4;
    else if (nn < 0) *info = -5;
    else if (lipar < 2) *info = -6;
    else if (ipar[0] < 0) {
        /* Special case: return size */
        ipar[0] = nsml * ((n < 0 ? -n : n) * (m + l + 1) + l * m + bsn);
        *ldj = (nsml > 1) ? nsml : 1;
        return;
    }
    else if (lx < lpar) *info = -8;
    else if (ldu < (nsmp > 1 ? nsmp : 1)) *info = -10;
    else if (*ldj < (nsml > 1 ? nsml : 1)) *info = -13;
    else {
        ldac = n + l;
        i32 jw_size;
        if (m > 0) {
            jw_size = (n * ldac > n + m + l) ? n * ldac : n + m + l;
        } else {
            jw_size = (n * ldac > l) ? n * ldac : l;
        }
        
        i32 min_ldwork = 2 * nsml + ((2 * nn > ldac * (n + m) + 2 * n + jw_size) 
                                     ? 2 * nn 
                                     : ldac * (n + m) + 2 * n + jw_size);
                                     
        if (ldwork < min_ldwork) *info = -16;
    }

    if (*info != 0) {
        i32 err_code = -(*info);
        SLC_XERBLA("NF01BD", &err_code);
        return;
    }

    if (nsmp == 0 || l == 0) {
        if (wjte && lpar >= 1) {
            jte[0] = zero;
            SLC_DCOPY(&lpar, &jte[0], &inc_0, &jte[0], &inc_1);
        }
        return;
    }

    /* Compute output of linear part */
    iy = 0;
    z = iy + nsml;
    ac = z + nsml;
    bd = ac + ldac * n;
    ix = bd + ldac * m;
    jw = ix + n;

    i32 lwork_tb = ldwork - jw;
    tb01vy("Apply", n, m, l, &x[nths], lths, &dwork[ac], ldac, 
           &dwork[bd], ldac, &dwork[ac + n], ldac, &dwork[bd + n], ldac, 
           &dwork[ix], &dwork[jw], lwork_tb, info);

    i32 lwork_tf = ldwork - jw;
    tf01mx(n, m, l, nsmp, &dwork[ac], ldac, u, ldu, &dwork[ix], 
           &dwork[z], nsmp, &dwork[jw], lwork_tf, info);

    /* Analytical Jacobian for nonlinear part */
    jw = ac;
    i32 lwork_nf = ldwork - jw;
    
    if (wjte) {
        for (i = 0; i < l; i++) {
            /* J is NSML x NCOLJ.
               Compressed form: J(i*NSMP, 1) starts block i?
               Fortran: J(I*NSMP+1, 1).
               In C: &j[(i*nsmp) + 0*(*ldj)]. Stride ldj?
               No, J is LDJ x NCOLJ. Column major.
               Submatrix for output i corresponds to parameters wb(i).
               wb(i) has BSN params.
               J block: rows i*NSMP to (i+1)*NSMP-1.
               Cols 0 to BSN-1.
               So &j[(i*nsmp) + 0*(*ldj)].
               Wait, J structure in compressed form:
               [ J_wb_1  J_theta_1 ]
               [ ...     ...       ]
               
               J_wb_i is NSMP x BSN.
               J_theta_i is NSMP x LTHS.
               
               NF01BY fills J_wb_i?
               "J(I*NSMP+1,1)" -> start of block row i, col 0.
               And JTE part: "JTE(I*BSN+1)".
            */
            nf01by(cjte, nsmp, l, 1, &ipar[1], lipar - 1, &x[i * bsn], bsn, 
                   &dwork[z], nsmp, &e[i * nsmp], 
                   &j[i * nsmp], *ldj, &jte[i * bsn], &dwork[jw], lwork_nf, info);
        }
    } else {
        for (i = 0; i < l; i++) {
            nf01by(cjte, nsmp, l, 1, &ipar[1], lipar - 1, &x[i * bsn], bsn, 
                   &dwork[z], nsmp, dwork, /* e dummy */
                   &j[i * nsmp], *ldj, dwork, /* jte dummy */
                   &dwork[jw], lwork_nf, info);
        }
    }

    /* Output with unchanged parameters (for finite difference) */
    /* Result into DWORK(IY) */
    nf01ay(nsmp, l, l, &ipar[1], lipar - 1, x, nths, &dwork[z], nsmp, 
           &dwork[iy], nsmp, &dwork[jw], lwork_nf, info);

    /* Numerical Jacobian for linear part parameters (theta) */
    jw = z;
    i32 lwork_ad = ldwork - jw;
    f64 eps_mach = SLC_DLAMCH("Epsilon");
    if (epsfcn > eps_mach) eps_mach = epsfcn;
    eps = sqrt(eps_mach);

    /* Loop over theta parameters (indices nths to lpar-1) */
    for (k = nths; k < lpar; k++) {
        kcol = k - nths + bsn; /* Column index in J for theta */
        /* J col kcol. Pointer: &j[0 + kcol*(*ldj)]?
           Wait, compressed J has:
           Cols 0..BSN-1: wb parameters.
           Cols BSN..BSN+LTHS-1: theta parameters.
           
           But "J has the block form ... returned without zero blocks".
           "NCOLJ = NN*(L + 2) + 1 + N*(M + L + 1) + L*M."
           This matches BSN + LTHS.
           
           For row block i, we have J_wb_i (cols 0..BSN-1) and J_theta_i (cols BSN..end).
           Wait, J_wb_i is different for each i.
           Are they stacked?
           "J has the block form ... returned without zero blocks"
           
           The documentation says:
           "The leading NSMP*L-by-NCOLJ part ... contains the Jacobian ... in compressed form".
           NCOLJ = BSN + LTHS.
           
           So J is (NSMP*L) x (BSN + LTHS).
           
           NF01BY fills cols 0 to BSN-1.
           For row i (samples for output i), it fills J[row, 0..BSN-1].
           
           Now we fill cols BSN to end.
           kcol goes from BSN to BSN+LTHS-1.
           
           We perturb X[k].
           Call NF01AD to get perturbed output Y_new (all L outputs).
           J[row, kcol] = (Y_new[row] - Y_old[row]) / h.
           Row goes from 0 to NSML-1.
           Y_new is in DWORK(JW)? No, passed as argument 10 to NF01AD.
           Fortran passes `J(1, KCOL)` as Y argument to NF01AD.
           So NF01AD writes directly into the column of J.
           Then we update J in place: J = (J - Y_old)/h.
        */
        
        parsav = x[k];
        if (parsav == zero) h = eps;
        else h = eps * fabs(parsav);
        
        x[k] += h;
        
        /* J column pointer: &j[0 + kcol * (*ldj)] */
        nf01ad(nsmp, m, l, ipar, lipar, x, lpar, u, ldu, 
               &j[kcol * (*ldj)], nsmp, &dwork[jw], lwork_ad, info);
               
        x[k] = parsav;
        
        for (i = 0; i < nsml; i++) {
            /* j[i + kcol*ldj] contains new Y[i] */
            /* dwork[iy + i] contains old Y[i] */
            j[i + kcol * (*ldj)] = (j[i + kcol * (*ldj)] - dwork[iy + i]) / h;
        }
    }

    if (wjte) {
        /* Compute last part of J'e in JTE */
        /* JTE(NTHS+1) corresponds to theta part */
        /* J_theta is submatrix J[0:nsml, bsn:end] */
        /* GEMV J_theta^T * E */
        /* J_theta ptr: &j[bsn * (*ldj)] */
        SLC_DGEMV("Transpose", &nsml, &lths, &one, &j[bsn * (*ldj)], ldj, 
                  e, &inc_1, &zero, &jte[nths], &inc_1);
    }
}
