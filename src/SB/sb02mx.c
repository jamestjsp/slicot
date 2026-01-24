/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB02MX - Extended Riccati preprocessing routine
 *
 * Computes:
 *   G = B*R^(-1)*B'
 *   A_bar = A +/- op(B*R^(-1)*L')
 *   Q_bar = Q +/- L*R^(-1)*L'
 *
 * Extended version of SB02MT with TRANS, FLAG, and DEF parameters.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <ctype.h>

void sb02mx(
    const char* jobg_str,
    const char* jobl_str,
    const char* fact_str,
    const char* uplo_str,
    const char* trans_str,
    const char* flag_str,
    const char* def_str,
    const i32 n,
    const i32 m,
    f64* a,
    const i32 lda,
    f64* b,
    const i32 ldb,
    f64* q,
    const i32 ldq,
    f64* r,
    const i32 ldr,
    f64* l,
    const i32 ldl,
    i32* ipiv,
    i32* oufact,
    f64* g,
    const i32 ldg,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const i32 int1 = 1;

    char jobg = toupper((unsigned char)jobg_str[0]);
    char jobl = toupper((unsigned char)jobl_str[0]);
    char fact = toupper((unsigned char)fact_str[0]);
    char uplo = toupper((unsigned char)uplo_str[0]);
    char trans = toupper((unsigned char)trans_str[0]);
    char flag_ch = toupper((unsigned char)flag_str[0]);
    char def = toupper((unsigned char)def_str[0]);

    bool ljobg = (jobg == 'G');
    bool ljobl = (jobl == 'N');
    bool lfactc = (fact == 'C');
    bool lfactu = (fact == 'U');
    bool luplou = (uplo == 'U');
    bool ltrans = (trans == 'N');
    bool lflag = (flag_ch == 'M');
    bool ldef = (def == 'D');
    bool lnfact = (!lfactc && !lfactu);

    *info = 0;

    if (!ljobg && jobg != 'N') {
        *info = -1;
    } else if (!ljobl && jobl != 'Z') {
        *info = -2;
    } else if (lnfact && fact != 'N') {
        *info = -3;
    } else if (!luplou && uplo != 'L') {
        *info = -4;
    } else if (!ltrans && trans != 'T' && trans != 'C') {
        *info = -5;
    } else if (!lflag && flag_ch != 'P') {
        *info = -6;
    } else if (!ldef && def != 'I' && lnfact) {
        *info = -7;
    } else if (n < 0) {
        *info = -8;
    } else if (m < 0) {
        *info = -9;
    } else if (lda < 1 || (ljobl && lda < n)) {
        *info = -11;
    } else if (ldb < 1 || (n > 0 && ldb < n)) {
        *info = -13;
    } else if (ldq < 1 || (ljobl && ldq < n)) {
        *info = -15;
    } else if (ldr < 1 || (m > 0 && ldr < m)) {
        *info = -17;
    } else if (ldl < 1 || (ljobl && ldl < n)) {
        *info = -19;
    } else if (ldg < 1 || (ljobg && ldg < n)) {
        *info = -23;
    } else {
        i32 wrkmin;
        if (lfactc) {
            wrkmin = 1;
        } else if (lfactu) {
            wrkmin = (ljobg || ljobl) ? ((n * m > 1) ? n * m : 1) : 1;
        } else {
            if (ljobg || ljobl) {
                i32 nm = n * m;
                i32 tmp = 3 * m > nm ? 3 * m : nm;
                wrkmin = tmp > 2 ? tmp : 2;
            } else {
                wrkmin = 3 * m > 2 ? 3 * m : 2;
            }
        }
        if (ldwork == -1) {
            if (lnfact) {
                SLC_DSYTRF(&uplo, &m, r, &ldr, ipiv, dwork, &(i32){-1}, info);
                i32 wrkopt = (i32)dwork[0];
                dwork[0] = (f64)(wrkopt > wrkmin ? wrkopt : wrkmin);
            } else {
                dwork[0] = (f64)wrkmin;
            }
            return;
        } else if (ldwork == -2) {
            dwork[0] = (f64)wrkmin;
            return;
        } else if (ldwork < wrkmin) {
            *info = -26;
            dwork[0] = (f64)wrkmin;
        }
    }

    if (*info != 0) {
        return;
    }

    if (m == 0) {
        if (ljobg) {
            SLC_DLASET(&uplo, &n, &n, &zero, &zero, g, &ldg);
        }
        *oufact = 0;
        dwork[0] = (f64)(lnfact ? 2 : 1);
        if (lnfact) {
            dwork[1] = zero;
        }
        return;
    }

    i32 wrkopt = 1;
    f64 eps, rcond, rnorm;

    if (lnfact) {
        eps = SLC_DLAMCH("P");

        rnorm = SLC_DLANSY("1", &uplo, &m, r, &ldr, dwork);

        if (ldef) {
            SLC_DCOPY(&m, r, &(i32){ldr + 1}, dwork, &int1);
            ma02ed(uplo, m, r, ldr);

            SLC_DPOTRF(&uplo, &m, r, &ldr, info);

            if (*info == 0) {
                SLC_DPOCON(&uplo, &m, r, &ldr, &rnorm, &rcond, dwork, iwork, info);

                *oufact = 1;
                if (rcond < eps) {
                    *info = m + 1;
                    dwork[1] = rcond;
                    return;
                }
                wrkopt = 3 * m > wrkopt ? 3 * m : wrkopt;
            } else {
                SLC_DCOPY(&m, dwork, &int1, r, &(i32){ldr + 1});

                if (luplou) {
                    ma02ed('L', m, r, ldr);
                } else {
                    ma02ed('U', m, r, ldr);
                }
            }
        }

        if (!ldef || *info > 0) {
            SLC_DSYTRF(&uplo, &m, r, &ldr, ipiv, dwork, &ldwork, info);
            *oufact = 2;
            if (*info > 0) {
                dwork[1] = zero;
                return;
            }
            i32 opt = (i32)dwork[0];
            wrkopt = opt > wrkopt ? opt : wrkopt;

            SLC_DSYCON(&uplo, &m, r, &ldr, ipiv, &rnorm, &rcond, dwork, iwork, info);

            if (rcond < eps) {
                *info = m + 1;
                dwork[1] = rcond;
                return;
            }
        }
    } else if (lfactc) {
        *oufact = 1;
    } else {
        *oufact = 2;
    }

    if (n > 0) {
        const char* nt = "N";
        const char* tr = "T";
        f64 temp = zero;

        if (ljobl) {
            temp = lflag ? -one : one;
        }

        f64 bnorm = SLC_DLANGE("1", &n, &m, b, &ldb, dwork);
        bool bnzer = (bnorm > zero);

        if (*oufact == 1) {
            const char* transu = luplou ? nt : tr;

            if (bnzer) {
                SLC_DTRSM("R", &uplo, transu, "N", &n, &m, &one, r, &ldr, b, &ldb);

                if (ljobg) {
                    SLC_DSYRK(&uplo, nt, &n, &m, &one, b, &ldb, &zero, g, &ldg);
                }
            }

            if (ljobl) {
                SLC_DTRSM("R", &uplo, transu, "N", &n, &m, &one, r, &ldr, l, &ldl);

                if (bnzer) {
                    if (ltrans) {
                        SLC_DGEMM(nt, tr, &n, &n, &m, &temp, b, &ldb, l, &ldl, &one, a, &lda);
                    } else {
                        SLC_DGEMM(nt, tr, &n, &n, &m, &temp, l, &ldl, b, &ldb, &one, a, &lda);
                    }
                }

                SLC_DSYRK(&uplo, nt, &n, &m, &temp, l, &ldl, &one, q, &ldq);
            }
        } else {
            if (bnzer) {
                if (ljobg || !ltrans) {
                    for (i32 j = 0; j < m; j++) {
                        SLC_DCOPY(&n, &b[j * ldb], &int1, &dwork[j], &m);
                    }

                    SLC_DSYTRS(&uplo, &m, &n, r, &ldr, ipiv, dwork, &m, info);
                }

                if (ljobg) {
                    mb01rb("L", &uplo, nt, n, m, zero, one, g, ldg, b, ldb, dwork, m, info);
                }
            }

            if (ljobl) {
                if (!ltrans && bnzer) {
                    SLC_DGEMM(nt, nt, &n, &n, &m, &temp, l, &ldl, dwork, &m, &one, a, &lda);
                }

                for (i32 j = 0; j < m; j++) {
                    SLC_DCOPY(&n, &l[j * ldl], &int1, &dwork[j], &m);
                }

                SLC_DSYTRS(&uplo, &m, &n, r, &ldr, ipiv, dwork, &m, info);

                if (ltrans && bnzer) {
                    SLC_DGEMM(nt, nt, &n, &n, &m, &temp, b, &ldb, dwork, &m, &one, a, &lda);
                }

                mb01rb("L", &uplo, nt, n, m, one, temp, q, ldq, l, ldl, dwork, m, info);
            }
        }
    }

    dwork[0] = (f64)wrkopt;
    if (lnfact) {
        dwork[1] = rcond;
    }
}
