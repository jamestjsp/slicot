// SPDX-License-Identifier: BSD-3-Clause
// SG02CX - Line search parameter minimizing Riccati residual norm

#include "slicot.h"
#include "slicot_blas.h"
#include <ctype.h>
#include <stdbool.h>
#include <math.h>

void sg02cx(
    const char* jobe, const char* flag, const char* jobg, const char* uplo,
    const char* trans,
    const i32 n, const i32 m,
    const f64* e, const i32 lde,
    const f64* r, const i32 ldr,
    const f64* s, const i32 lds,
    f64* g, const i32 ldg,
    f64* alpha, f64* rnorm,
    f64* dwork, const i32 ldwork,
    i32* iwarn, i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 two = 2.0;
    const f64 three = 3.0;
    const f64 four = 4.0;
    const f64 six = 6.0;
    const f64 half = 0.5;
    i32 int1 = 1;

    char jobe_c = (char)toupper((unsigned char)jobe[0]);
    char flag_c = (char)toupper((unsigned char)flag[0]);
    char jobg_c = (char)toupper((unsigned char)jobg[0]);
    char uplo_c = (char)toupper((unsigned char)uplo[0]);
    char trans_c = (char)toupper((unsigned char)trans[0]);

    bool ljobe = (jobe_c == 'G');
    bool lflag = (flag_c == 'M');
    bool ljobg = (jobg_c == 'G');
    bool ljobf = (jobg_c == 'F');
    bool ljobh = (jobg_c == 'H');
    bool luplo = (uplo_c == 'U');
    bool ltrans = (trans_c == 'T' || trans_c == 'C');
    bool ljobl = ljobf || ljobh;

    *iwarn = 0;
    *info = 0;

    i32 nn = n * n;
    i32 nm = n * m;
    i32 nmin, nopt;
    bool lcnd = false;

    if (!ljobe && jobe_c != 'I') {
        *info = -1;
    } else if (!lflag && flag_c != 'P') {
        *info = -2;
    } else if (!ljobg && jobg_c != 'D' && !ljobl) {
        *info = -3;
    } else if (!luplo && uplo_c != 'L') {
        *info = -4;
    } else if (!ltrans && trans_c != 'N') {
        *info = -5;
    } else if (n < 0) {
        *info = -6;
    } else if (!ljobg && m < 0) {
        *info = -7;
    } else if (lde < 1 || (ljobe && !ljobl && lde < n)) {
        *info = -9;
    } else if (ldr < (n > 1 ? n : 1)) {
        *info = -11;
    } else if (lds < 1 || (ljobh && lds < m) || (!ljobl && lds < n)) {
        *info = -13;
    } else if (ldg < (n > 1 ? n : 1)) {
        *info = -15;
    } else {
        bool lquery = (ldwork == -1);

        if (ljobl) {
            nmin = nn + (nn > 51 ? nn : 51);
            if (lquery) nopt = nmin;
        } else if (ljobg) {
            if (ljobe) {
                nmin = nn + (2 * nn > 51 ? 2 * nn : 51);
            } else {
                nmin = nn + (nn > 51 ? nn : 51);
            }
            if (lquery) nopt = nmin;
        } else {
            nm = n * m;
            if (ljobe) {
                i32 max_nn_51 = nn > 51 ? nn : 51;
                i32 min_2nn_nm = 2 * nn < nm ? 2 * nn : nm;
                nmin = nn + (max_nn_51 > min_2nn_nm ? max_nn_51 : min_2nn_nm);
                lcnd = (2 * m > 3 * n);
                if (lquery) {
                    nopt = lcnd ? 3 * nn : nn + nm;
                }
            } else {
                i32 max_nn_nm = nn > nm ? nn : nm;
                nmin = nn + (max_nn_nm > 51 ? max_nn_nm : 51);
                lcnd = (m > 3 * n);
                if (lquery) {
                    nopt = lcnd ? 2 * nn : nn + nm;
                }
            }
            if (lquery && nopt < nmin) nopt = nmin;
        }

        if (lquery) {
            f64 tmp_alpha = 0.0, tmp_beta = 0.0, tmp_gamma = 0.0, tmp_delta = 0.0;
            f64 mc_work[3];
            i32 mc_ldwork = -1;
            i32 mc_info;
            mc01xd(tmp_alpha, tmp_beta, tmp_gamma, tmp_delta,
                   mc_work, mc_work, mc_work, mc_work, mc_ldwork, &mc_info);
            i32 mc_opt = nn + 9 + (i32)mc_work[0];
            if (mc_opt > nopt) nopt = mc_opt;
            dwork[0] = (f64)nopt;
            return;
        } else if (ldwork == -2) {
            dwork[0] = (f64)nmin;
            return;
        } else if (ldwork < nmin) {
            *info = -19;
            dwork[0] = (f64)nmin;
        }
    }

    if (*info != 0) {
        return;
    }

    if (n == 0 || (!ljobg && m == 0)) {
        *alpha = one;
        *rnorm = zero;
        return;
    }

    const char* nt = "N";
    const char* tr = "T";
    const char* side;
    const char* ntrans;

    if (ljobe) {
        if (ltrans) {
            side = "R";
            ntrans = nt;
        } else {
            side = "L";
            ntrans = tr;
        }
    }

    i32 sp = nn;

    if (ljobl) {
        if (ljobf) {
            SLC_DSYRK(uplo, nt, &n, &m, &one, g, &ldg, &zero, dwork, &n);
        } else {
            i32 mb_info;
            mb01rb("L", uplo, nt, n, m, zero, one, dwork, n, g, ldg, s, lds, &mb_info);
        }
    } else if (ljobg) {
        if (ljobe) {
            SLC_DSYMM(side, uplo, &n, &n, &one, s, &lds, e, &lde, &zero, &dwork[sp], &n);
            i32 mb_info;
            mb01ru(uplo, ntrans, n, n, zero, one, dwork, n, &dwork[sp], n, g, ldg,
                   &dwork[sp + nn], nn, &mb_info);
        } else {
            i32 mb_info;
            mb01ru(uplo, nt, n, n, zero, one, dwork, n, s, lds, g, ldg,
                   &dwork[sp], nn, &mb_info);
        }
    } else {
        bool wwt = (n >= m);
        bool use1;

        if (ljobe) {
            use1 = lcnd && ldwork >= 3 * nn;

            if (use1) {
                SLC_DSYMM(side, uplo, &n, &n, &one, s, &lds, e, &lde, &zero, &dwork[sp], &n);
                SLC_DSYRK(uplo, nt, &n, &m, &one, g, &ldg, &zero, dwork, &n);
                i32 mb_info;
                mb01ru(uplo, ntrans, n, n, zero, one, dwork, n, &dwork[sp], n, dwork, n,
                       &dwork[sp + nn], nn, &mb_info);
                SLC_DSCAL(&n, &half, dwork, &(i32){n + 1});
            } else if (wwt) {
                SLC_DSYMM("L", uplo, &n, &m, &one, s, &lds, g, &ldg, &zero, dwork, &n);
                SLC_DGEMM(ntrans, nt, &n, &m, &n, &one, e, &lde, dwork, &n, &zero, &dwork[sp], &n);
            } else {
                SLC_DSYMM(side, uplo, &n, &n, &one, s, &lds, e, &lde, &zero, dwork, &n);
                if (ltrans) {
                    SLC_DGEMM(nt, nt, &n, &m, &n, &one, dwork, &n, g, &ldg, &zero, &dwork[sp], &n);
                } else {
                    SLC_DGEMM(tr, nt, &m, &n, &n, &one, g, &ldg, dwork, &n, &zero, &dwork[sp], &m);
                }
            }
        } else {
            use1 = lcnd || ldwork < nn + nm;

            if (use1) {
                SLC_DSYRK(uplo, nt, &n, &m, &one, g, &ldg, &zero, dwork, &n);
                i32 mb_info;
                mb01ru(uplo, nt, n, n, zero, one, dwork, n, s, lds, dwork, n,
                       &dwork[sp], nn, &mb_info);
                SLC_DSCAL(&n, &half, dwork, &(i32){n + 1});
            } else {
                SLC_DSYMM("L", uplo, &n, &m, &one, s, &lds, g, &ldg, &zero, &dwork[sp], &n);
            }
        }

        if (!use1) {
            if (wwt || !ljobe || ltrans) {
                SLC_DSYRK(uplo, nt, &n, &m, &one, &dwork[sp], &n, &zero, dwork, &n);
            } else {
                SLC_DSYRK(uplo, ntrans, &n, &m, &one, &dwork[sp], &m, &zero, dwork, &n);
            }
        }
    }

    f64 pb = zero;
    i32 critnr = 0;

    f64 vnorm = SLC_DLANSY("F", uplo, &n, dwork, &n, dwork);
    f64 rnorm_init = SLC_DLANSY("F", uplo, &n, r, &ldr, dwork);

    if (rnorm_init == zero) {
        *alpha = zero;
        *rnorm = zero;
        return;
    }

    f64 mx = one;
    if (rnorm_init > mx) mx = rnorm_init;
    if (vnorm > mx) mx = vnorm;

    f64 pa = (rnorm_init / mx) * (rnorm_init / mx);

    i32 idx = 0;
    if (luplo) {
        for (i32 j = 0; j < n; j++) {
            f64 dot = SLC_DDOT(&j, &r[j * ldr], &int1, &dwork[idx], &int1);
            pb += two * dot / mx / mx + (r[j + j * ldr] / mx) * (dwork[idx + j] / mx);
            idx += n;
        }
    } else {
        for (i32 j = 0; j < n - 1; j++) {
            i32 len = n - j - 1;
            f64 dot = SLC_DDOT(&len, &r[(j + 1) + j * ldr], &int1, &dwork[idx + 1], &int1);
            pb += (r[j + j * ldr] / mx) * (dwork[idx] / mx) + two * dot / mx / mx;
            idx += n + 1;
        }
        pb += (r[(n - 1) + (n - 1) * ldr] / mx) * (dwork[idx] / mx);
    }

    f64 poly_alpha = -two * pa;
    f64 beta, gamma;
    if (lflag) {
        beta = two * (pa - two * pb);
        gamma = six * pb;
    } else {
        beta = two * (pa + two * pb);
        gamma = -six * pb;
    }
    f64 delta = four * (vnorm / mx) * (vnorm / mx);

    i32 evrpos = sp;
    i32 evipos = evrpos + 3;
    i32 evqpos = evipos + 3;
    i32 rpos = evqpos + 3;

    i32 mc_info;
    mc01xd(poly_alpha, beta, gamma, delta, &dwork[evrpos], &dwork[evipos],
           &dwork[evqpos], &dwork[rpos], ldwork - rpos, &mc_info);

    if (mc_info != 0) {
        *info = 1;
        *alpha = one;
        *rnorm = vnorm;
        return;
    }

    f64 crn[2], crd[2];

    for (i32 j = 0; j < 3; j++) {
        if (dwork[evipos + j] == zero) {
            pa = dwork[evrpos + j];
            pb = dwork[evqpos + j];
            if (pa >= zero && pb > zero && pa <= two * pb) {
                mx = fabs(beta);
                if (fabs(gamma) > mx) mx = fabs(gamma);
                if (delta > mx) mx = delta;
                if (fabs(pa) > mx) mx = fabs(pa);
                if (fabs(pb) > mx) mx = fabs(pb);

                if (mx > zero) {
                    f64 pa_scaled = pa / mx;
                    f64 pb_scaled = pb / mx;
                    f64 pc = (beta / mx) * pb_scaled * pb_scaled +
                             two * (gamma / mx) * pa_scaled * pb_scaled +
                             three * (delta / mx) * pa_scaled * pa_scaled;
                    if (pc > zero) {
                        crn[critnr] = dwork[evrpos + j];
                        crd[critnr] = dwork[evqpos + j];
                        critnr++;
                    }
                }
            }
        }
    }

    if (critnr == 0) {
        *iwarn = 2;
        *alpha = one;
        *rnorm = vnorm;
        return;
    }

    *alpha = crn[0] / crd[0];
    pa = one - *alpha;

    SLC_DLACPY(uplo, &n, &n, r, &ldr, &dwork[sp], &n);
    SLC_DLASCL(uplo, &int1, &int1, &one, &pa, &n, &n, &dwork[sp], &n, &mc_info);

    if (lflag) {
        pa = -(*alpha) * (*alpha);
    } else {
        pa = (*alpha) * (*alpha);
    }

    idx = 0;
    if (luplo) {
        for (i32 j = 0; j < n; j++) {
            i32 len = j + 1;
            SLC_DAXPY(&len, &pa, &dwork[idx], &int1, &dwork[sp + idx], &int1);
            idx += n;
        }
    } else {
        for (i32 j = 0; j < n; j++) {
            i32 len = n - j;
            SLC_DAXPY(&len, &pa, &dwork[idx], &int1, &dwork[sp + idx], &int1);
            idx += n + 1;
        }
    }

    *rnorm = SLC_DLANSY("F", uplo, &n, &dwork[sp], &n, dwork);

    if (critnr == 2) {
        f64 beta_val = crn[1] / crd[1];
        pb = one - beta_val;

        SLC_DLACPY(uplo, &n, &n, r, &ldr, &dwork[sp], &n);
        SLC_DLASCL(uplo, &int1, &int1, &one, &pb, &n, &n, &dwork[sp], &n, &mc_info);

        if (lflag) {
            pb = -beta_val * beta_val;
        } else {
            pb = beta_val * beta_val;
        }

        idx = 0;
        if (luplo) {
            for (i32 j = 0; j < n; j++) {
                i32 len = j + 1;
                SLC_DAXPY(&len, &pb, &dwork[idx], &int1, &dwork[sp + idx], &int1);
                idx += n;
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                i32 len = n - j;
                SLC_DAXPY(&len, &pb, &dwork[idx], &int1, &dwork[sp + idx], &int1);
                idx += n + 1;
            }
        }

        vnorm = SLC_DLANSY("F", uplo, &n, &dwork[sp], &n, dwork);

        if (vnorm < *rnorm) {
            *alpha = beta_val;
            *rnorm = vnorm;
        }
    }
}
