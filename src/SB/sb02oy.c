/**
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SB02OY - Extended Hamiltonian/symplectic matrix pair construction
 *
 * Constructs extended matrix pairs for algebraic Riccati equations
 * and compresses them to 2N-by-2N using QL factorization.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <stdlib.h>
#include <ctype.h>

void sb02oy(
    const char* type_str,
    const char* dico_str,
    const char* jobb_str,
    const char* fact_str,
    const char* uplo_str,
    const char* jobl_str,
    const char* jobe_str,
    const i32 n,
    const i32 m,
    const i32 p,
    const f64* a,
    const i32 lda,
    const f64* b,
    const i32 ldb,
    const f64* q,
    const i32 ldq,
    const f64* r,
    const i32 ldr,
    const f64* l,
    const i32 ldl,
    const f64* e,
    const i32 lde,
    f64* af,
    const i32 ldaf,
    f64* bf,
    const i32 ldbf,
    const f64 tol,
    i32* iwork,
    f64* dwork,
    const i32 ldwork,
    i32* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 mone = -1.0;
    const i32 int1 = 1;

    char type_c = toupper((unsigned char)type_str[0]);
    char dico = toupper((unsigned char)dico_str[0]);
    char jobb = toupper((unsigned char)jobb_str[0]);
    char fact = toupper((unsigned char)fact_str[0]);
    char uplo = toupper((unsigned char)uplo_str[0]);
    char jobl = toupper((unsigned char)jobl_str[0]);
    char jobe = toupper((unsigned char)jobe_str[0]);

    bool optc = (type_c == 'O');
    bool discr = (dico == 'D');
    bool ljobb = (jobb == 'B');
    bool lfacn = (fact == 'N');
    bool lfacq = (fact == 'C');
    bool lfacr = (fact == 'D');
    bool lfacb = (fact == 'B');
    bool luplo = (uplo == 'U');
    bool ljobe = (jobe == 'I');
    bool ljobl = false;

    i32 n2 = n + n;
    i32 nm, nnm;
    if (ljobb) {
        ljobl = (jobl == 'Z');
        nm = n + m;
        nnm = n2 + m;
    } else {
        nm = n;
        nnm = n2;
    }
    i32 np1 = n;
    i32 n2p1 = n2;

    *info = 0;

    if (!optc && type_c != 'S') {
        *info = -1;
    } else if (!discr && dico != 'C') {
        *info = -2;
    } else if (!ljobb && jobb != 'G') {
        *info = -3;
    } else if (!lfacq && !lfacr && !lfacb && !lfacn) {
        *info = -4;
    } else if (!ljobb || lfacn) {
        if (!luplo && uplo != 'L') {
            *info = -5;
        }
    }
    if (*info == 0 && ljobb) {
        if (!ljobl && jobl != 'N') {
            *info = -6;
        }
    }
    if (*info == 0) {
        if (!ljobe && jobe != 'N') {
            *info = -7;
        } else if (n < 0) {
            *info = -8;
        }
    }
    if (*info == 0 && ljobb) {
        if (m < 0) {
            *info = -9;
        }
    }
    if (*info == 0 && (!lfacn || !optc)) {
        if (p < 0) {
            *info = -10;
        } else if (ljobb && !optc && p != m) {
            *info = -10;
        }
    }
    if (*info == 0) {
        i32 min_lda = n > 1 ? n : 1;
        if (lda < min_lda) {
            *info = -12;
        }
    }
    if (*info == 0) {
        i32 min_ldb = n > 1 ? n : 1;
        if (ldb < min_ldb) {
            *info = -14;
        }
    }
    if (*info == 0) {
        if ((lfacn || lfacr) && ldq < (n > 1 ? n : 1)) {
            *info = -16;
        } else if ((lfacq || lfacb) && ldq < (p > 1 ? p : 1)) {
            *info = -16;
        }
    }
    if (*info == 0) {
        if (ldr < 1) {
            *info = -18;
        }
    }
    if (*info == 0 && ljobb) {
        if ((lfacn || lfacq) && ldr < m) {
            *info = -18;
        } else if ((lfacr || lfacb) && ldr < p) {
            *info = -18;
        }
        if (*info == 0) {
            if (!ljobl && ldl < (n > 1 ? n : 1)) {
                *info = -20;
            } else if (ljobl && ldl < 1) {
                *info = -20;
            }
        }
    }
    if (*info == 0) {
        if (!ljobe && lde < (n > 1 ? n : 1)) {
            *info = -22;
        } else if (ljobe && lde < 1) {
            *info = -22;
        }
    }
    if (*info == 0) {
        i32 min_ldaf = nnm > 1 ? nnm : 1;
        if (ldaf < min_ldaf) {
            *info = -24;
        }
    }
    if (*info == 0) {
        if ((ljobb || discr || !ljobe) && ldbf < nnm) {
            *info = -26;
        } else if (ldbf < 1) {
            *info = -26;
        }
    }
    if (*info == 0) {
        if (ljobb) {
            i32 req1 = nnm > 3*m ? nnm : 3*m;
            if (ldwork < req1) {
                *info = -30;
            }
        } else if (ldwork < 1) {
            *info = -30;
        }
    }

    if (*info != 0) {
        return;
    }

    dwork[0] = one;
    if (n == 0) {
        return;
    }

    SLC_DLACPY("F", &n, &n, a, &lda, af, &ldaf);

    if (!lfacq && !lfacb) {
        SLC_DLACPY(&uplo, &n, &n, q, &ldq, &af[np1], &ldaf);
        if (luplo) {
            for (i32 j = 0; j < n - 1; j++) {
                i32 count = n - j - 1;
                SLC_DCOPY(&count, &q[j + (j+1)*ldq], &ldq, &af[np1 + j + 1 + j*ldaf], &int1);
            }
        } else {
            for (i32 j = 1; j < n; j++) {
                SLC_DCOPY(&j, &q[j], &ldq, &af[np1 + j*ldaf], &int1);
            }
        }
    } else {
        SLC_DSYRK("U", "T", &n, &p, &one, q, &ldq, &zero, &af[np1], &ldaf);
        for (i32 j = 1; j < n; j++) {
            SLC_DCOPY(&j, &af[np1 + j*ldaf], &int1, &af[np1 + j + 0*ldaf], &ldaf);
        }
    }

    if (ljobb) {
        if (ljobl) {
            SLC_DLASET("F", &m, &n, &zero, &zero, &af[n2p1], &ldaf);
        } else {
            for (i32 i = 0; i < n; i++) {
                SLC_DCOPY(&m, &l[i], &ldl, &af[n2p1 + i*ldaf], &int1);
            }
        }
    }

    if (discr || ljobb) {
        SLC_DLASET("F", &n, &n, &zero, &zero, &af[np1*ldaf], &ldaf);
    } else {
        if (luplo) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i <= j; i++) {
                    af[i + (np1+j)*ldaf] = -b[i + j*ldb];
                }
                for (i32 i = j + 1; i < n; i++) {
                    af[i + (np1+j)*ldaf] = -b[j + i*ldb];
                }
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < j; i++) {
                    af[i + (np1+j)*ldaf] = -b[j + i*ldb];
                }
                for (i32 i = j; i < n; i++) {
                    af[i + (np1+j)*ldaf] = -b[i + j*ldb];
                }
            }
        }
    }

    if (discr) {
        if (ljobe) {
            SLC_DLASET("F", &nm, &n, &zero, &mone, &af[np1 + np1*ldaf], &ldaf);
        } else {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < n; i++) {
                    af[(np1+i) + (np1+j)*ldaf] = -e[j + i*lde];
                }
            }
            if (ljobb) {
                SLC_DLASET("F", &m, &n, &zero, &zero, &af[n2p1 + np1*ldaf], &ldaf);
            }
        }
    } else {
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < n; i++) {
                af[(np1+i) + (np1+j)*ldaf] = a[j + i*lda];
            }
        }
        if (ljobb) {
            if (optc) {
                for (i32 j = 0; j < n; j++) {
                    SLC_DCOPY(&m, &b[j], &ldb, &af[n2p1 + (np1+j)*ldaf], &int1);
                }
            } else {
                SLC_DLACPY("F", &p, &n, q, &ldq, &af[n2p1 + np1*ldaf], &ldaf);
            }
        }
    }

    if (ljobb) {
        if (optc) {
            SLC_DLACPY("F", &n, &m, b, &ldb, &af[n2p1*ldaf], &ldaf);
        } else {
            for (i32 i = 0; i < p; i++) {
                SLC_DCOPY(&n, &q[i], &ldq, &af[(n2+i)*ldaf], &int1);
            }
        }

        if (ljobl) {
            SLC_DLASET("F", &n, &m, &zero, &zero, &af[np1 + n2p1*ldaf], &ldaf);
        } else {
            SLC_DLACPY("F", &n, &m, l, &ldl, &af[np1 + n2p1*ldaf], &ldaf);
        }

        if (!lfacr && !lfacb) {
            SLC_DLACPY(&uplo, &m, &m, r, &ldr, &af[n2p1 + n2p1*ldaf], &ldaf);
            if (luplo) {
                for (i32 j = 0; j < m - 1; j++) {
                    i32 count = m - j - 1;
                    SLC_DCOPY(&count, &r[j + (j+1)*ldr], &ldr, &af[n2p1 + j + 1 + (n2+j)*ldaf], &int1);
                }
            } else {
                for (i32 j = 1; j < m; j++) {
                    SLC_DCOPY(&j, &r[j], &ldr, &af[n2p1 + (n2+j)*ldaf], &int1);
                }
            }
        } else if (optc) {
            SLC_DSYRK("U", "T", &m, &p, &one, r, &ldr, &zero, &af[n2p1 + n2p1*ldaf], &ldaf);
            for (i32 j = 1; j < m; j++) {
                SLC_DCOPY(&j, &af[n2p1 + (n2+j)*ldaf], &int1, &af[n2p1 + j + n2p1*ldaf], &ldaf);
            }
        } else {
            for (i32 j = 0; j < m; j++) {
                for (i32 i = 0; i < p; i++) {
                    af[(n2+i) + (n2+j)*ldaf] = r[i + j*ldr] + r[j + i*ldr];
                }
            }
        }
    }

    if (!ljobb && !discr && ljobe) {
        return;
    }

    if (ljobe) {
        i32 npm = n + nm;
        SLC_DLASET("F", &npm, &n, &zero, &one, bf, &ldbf);
    } else {
        SLC_DLACPY("F", &n, &n, e, &lde, bf, &ldbf);
        SLC_DLASET("F", &nm, &n, &zero, &zero, &bf[np1], &ldbf);
    }

    if (!discr || ljobb) {
        SLC_DLASET("F", &n, &n, &zero, &zero, &bf[np1*ldbf], &ldbf);
    } else {
        if (luplo) {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i <= j; i++) {
                    bf[i + (np1+j)*ldbf] = b[i + j*ldb];
                }
                for (i32 i = j + 1; i < n; i++) {
                    bf[i + (np1+j)*ldbf] = b[j + i*ldb];
                }
            }
        } else {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < j; i++) {
                    bf[i + (np1+j)*ldbf] = b[j + i*ldb];
                }
                for (i32 i = j; i < n; i++) {
                    bf[i + (np1+j)*ldbf] = b[i + j*ldb];
                }
            }
        }
    }

    if (discr) {
        for (i32 j = 0; j < n; j++) {
            for (i32 i = 0; i < n; i++) {
                bf[(np1+i) + (np1+j)*ldbf] = -a[j + i*lda];
            }
        }
        if (ljobb) {
            if (optc) {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = 0; i < m; i++) {
                        bf[(n2+i) + (np1+j)*ldbf] = -b[j + i*ldb];
                    }
                }
            } else {
                for (i32 j = 0; j < n; j++) {
                    for (i32 i = 0; i < p; i++) {
                        bf[(n2+i) + (np1+j)*ldbf] = -q[i + j*ldq];
                    }
                }
            }
        }
    } else {
        if (ljobe) {
            SLC_DLASET("F", &nm, &n, &zero, &mone, &bf[np1 + np1*ldbf], &ldbf);
        } else {
            for (i32 j = 0; j < n; j++) {
                for (i32 i = 0; i < n; i++) {
                    bf[(np1+i) + (np1+j)*ldbf] = -e[j + i*lde];
                }
            }
            if (ljobb) {
                SLC_DLASET("F", &m, &n, &zero, &zero, &bf[n2p1 + np1*ldbf], &ldbf);
            }
        }
    }

    if (!ljobb) {
        return;
    }

    i32 itau = 0;
    i32 jwork = itau + m;
    i32 lwork_rem = ldwork - jwork;
    SLC_DGEQLF(&nnm, &m, &af[n2p1*ldaf], &ldaf, &dwork[itau], &dwork[jwork], &lwork_rem, info);
    i32 wrkopt = (i32)dwork[jwork];

    lwork_rem = ldwork - jwork;
    SLC_DORMQL("L", "T", &nnm, &n2, &m, &af[n2p1*ldaf], &ldaf, &dwork[itau], af, &ldaf, &dwork[jwork], &lwork_rem, info);
    i32 opt_tmp = (i32)dwork[jwork] + jwork;
    wrkopt = wrkopt > opt_tmp ? wrkopt : opt_tmp;

    lwork_rem = ldwork - jwork;
    SLC_DORMQL("L", "T", &nnm, &n2, &m, &af[n2p1*ldaf], &ldaf, &dwork[itau], bf, &ldbf, &dwork[jwork], &lwork_rem, info);

    f64 toldef = tol;
    if (toldef <= zero) {
        toldef = SLC_DLAMCH("E");
    }

    f64 rcond;
    SLC_DTRCON("1", "L", "N", &m, &af[n2p1 + n2p1*ldaf], &ldaf, &rcond, dwork, iwork, info);
    wrkopt = wrkopt > 3*m ? wrkopt : 3*m;

    if (rcond <= toldef) {
        *info = 1;
    }

    dwork[0] = (f64)wrkopt;
    dwork[1] = rcond;
}
