/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * MB03KA - Moving diagonal blocks in generalized periodic Schur form
 *
 * Reorders the diagonal blocks of a formal matrix product
 * T22_K^S(K) * T22_K-1^S(K-1) * ... * T22_1^S(1) of length K
 * in generalized periodic Schur form such that the block with
 * starting row index IFST is moved to row index ILST.
 */

#include "slicot.h"
#include "slicot_blas.h"
#include <math.h>
#include <stdlib.h>

void mb03ka(const char* compq, const i32* whichq, const bool ws,
            const i32 k, const i32 nc, const i32 kschur,
            i32* ifst, i32* ilst,
            const i32* n, const i32* ni, const i32* s,
            f64* t, const i32* ldt, const i32* ixt,
            f64* q, const i32* ldq, const i32* ixq,
            const f64* tol, i32* iwork, f64* dwork,
            const i32 ldwork, i32* info)
{
    const f64 ZERO = 0.0;

    i32 here, i, ip1, it, minwrk, nbf, nbl, nbnext;

    *info = 0;

    if (nc == 2) {
        nbf = 1;
        nbl = 1;
    } else if (nc == 3) {
        nbf = 1;
        nbl = 2;
    } else {
        nbf = 2;
        nbl = 2;
    }

    mb03kb(compq, whichq, ws, k, nc, kschur, 1, nbf, nbl, n, ni, s,
           t, ldt, ixt, q, ldq, ixq, tol, iwork, dwork, -1, info);

    minwrk = (i32)dwork[0];
    if (minwrk < 1) minwrk = 1;

    if (ldwork != -1 && ldwork < minwrk) {
        *info = -21;
    }

    if (ldwork == -1) {
        dwork[0] = (f64)minwrk;
        return;
    } else if (*info < 0) {
        i32 neg_info = 21;
        SLC_XERBLA("MB03KA", &neg_info);
        return;
    }

    i = kschur - 1;
    ip1 = i % k;
    if (ip1 < 0) ip1 += k;
    ip1 = (i + 1) % k;

    if (*ifst > 1) {
        if (s[i] == 1) {
            it = ixt[i] - 1 + (ni[i] + *ifst - 2) * ldt[i] + ni[ip1] + *ifst - 1;
        } else {
            it = ixt[i] - 1 + (ni[ip1] + *ifst - 2) * ldt[i] + ni[i] + *ifst - 1;
        }
        if (t[it] != ZERO) {
            (*ifst)--;
        }
    }

    nbf = 1;
    if (*ifst < nc) {
        if (s[i] == 1) {
            it = ixt[i] - 1 + (ni[i] + *ifst - 1) * ldt[i] + ni[ip1] + *ifst;
        } else {
            it = ixt[i] - 1 + (ni[ip1] + *ifst - 1) * ldt[i] + ni[i] + *ifst;
        }
        if (t[it] != ZERO) {
            nbf = 2;
        }
    }

    if (*ilst > 1) {
        if (s[i] == 1) {
            it = ixt[i] - 1 + (ni[i] + *ilst - 2) * ldt[i] + ni[ip1] + *ilst - 1;
        } else {
            it = ixt[i] - 1 + (ni[ip1] + *ilst - 2) * ldt[i] + ni[i] + *ilst - 1;
        }
        if (t[it] != ZERO) {
            (*ilst)--;
        }
    }

    nbl = 1;
    if (*ilst < nc) {
        if (s[i] == 1) {
            it = ixt[i] - 1 + (ni[i] + *ilst - 1) * ldt[i] + ni[ip1] + *ilst;
        } else {
            it = ixt[i] - 1 + (ni[ip1] + *ilst - 1) * ldt[i] + ni[i] + *ilst;
        }
        if (t[it] != ZERO) {
            nbl = 2;
        }
    }

    if (*ifst == *ilst) {
        return;
    }

    if (*ifst < *ilst) {
        if (nbf == 2 && nbl == 1) {
            (*ilst)--;
        }
        if (nbf == 1 && nbl == 2) {
            (*ilst)++;
        }

        here = *ifst;

        while (here < *ilst) {
            if (nbf == 1 || nbf == 2) {
                nbnext = 1;
                if (here + nbf + 1 <= nc) {
                    if (s[i] == 1) {
                        it = ixt[i] - 1 + (ni[i] + here + nbf - 1) * ldt[i] +
                             ni[ip1] + here + nbf;
                    } else {
                        it = ixt[i] - 1 + (ni[ip1] + here + nbf - 1) * ldt[i] +
                             ni[i] + here + nbf;
                    }
                    if (t[it] != ZERO) {
                        nbnext = 2;
                    }
                }

                mb03kb(compq, whichq, ws, k, nc, kschur, here, nbf, nbnext,
                       n, ni, s, t, ldt, ixt, q, ldq, ixq, tol, iwork,
                       dwork, ldwork, info);
                if (*info != 0) {
                    *ilst = here;
                    return;
                }
                here += nbnext;

                if (nbf == 2) {
                    if (s[i] == 1) {
                        it = ixt[i] - 1 + (ni[i] + here - 1) * ldt[i] + ni[ip1] + here;
                    } else {
                        it = ixt[i] - 1 + (ni[ip1] + here - 1) * ldt[i] + ni[i] + here;
                    }
                    if (t[it] == ZERO) {
                        nbf = 3;
                    }
                }
            } else {
                nbnext = 1;
                if (here + 3 <= nc) {
                    if (s[i] == 1) {
                        it = ixt[i] - 1 + (ni[i] + here + 1) * ldt[i] + ni[ip1] + here + 2;
                    } else {
                        it = ixt[i] - 1 + (ni[ip1] + here + 1) * ldt[i] + ni[i] + here + 2;
                    }
                    if (t[it] != ZERO) {
                        nbnext = 2;
                    }
                }

                mb03kb(compq, whichq, ws, k, nc, kschur, here + 1, 1, nbnext,
                       n, ni, s, t, ldt, ixt, q, ldq, ixq, tol, iwork,
                       dwork, ldwork, info);
                if (*info != 0) {
                    *ilst = here;
                    return;
                }

                if (nbnext == 1) {
                    mb03kb(compq, whichq, ws, k, nc, kschur, here, 1, nbnext,
                           n, ni, s, t, ldt, ixt, q, ldq, ixq, tol, iwork,
                           dwork, ldwork, info);
                    if (*info != 0) {
                        *ilst = here;
                        return;
                    }
                    here++;
                } else {
                    if (s[i] == 1) {
                        it = ixt[i] - 1 + (ni[i] + here) * ldt[i] + ni[ip1] + here + 1;
                    } else {
                        it = ixt[i] - 1 + (ni[ip1] + here) * ldt[i] + ni[i] + here + 1;
                    }
                    if (t[it] == ZERO) {
                        nbnext = 1;
                    }

                    if (nbnext == 2) {
                        mb03kb(compq, whichq, ws, k, nc, kschur, here, 1, nbnext,
                               n, ni, s, t, ldt, ixt, q, ldq, ixq, tol, iwork,
                               dwork, ldwork, info);
                        if (*info != 0) {
                            *ilst = here;
                            return;
                        }
                        here += 2;
                    } else {
                        mb03kb(compq, whichq, ws, k, nc, kschur, here, 1, 1,
                               n, ni, s, t, ldt, ixt, q, ldq, ixq, tol, iwork,
                               dwork, ldwork, info);
                        if (*info != 0) {
                            *ilst = here;
                            return;
                        }
                        mb03kb(compq, whichq, ws, k, nc, kschur, here + 1, 1, 1,
                               n, ni, s, t, ldt, ixt, q, ldq, ixq, tol, iwork,
                               dwork, ldwork, info);
                        if (*info != 0) {
                            *ilst = here + 1;
                            return;
                        }
                        here += 2;
                    }
                }
            }
        }
    } else {
        here = *ifst;

        while (here > *ilst) {
            if (nbf == 1 || nbf == 2) {
                nbnext = 1;
                if (here >= 3) {
                    if (s[i] == 1) {
                        it = ixt[i] - 1 + (ni[i] + here - 3) * ldt[i] + ni[ip1] + here - 2;
                    } else {
                        it = ixt[i] - 1 + (ni[ip1] + here - 3) * ldt[i] + ni[i] + here - 2;
                    }
                    if (t[it] != ZERO) {
                        nbnext = 2;
                    }
                }

                mb03kb(compq, whichq, ws, k, nc, kschur, here - nbnext, nbnext, nbf,
                       n, ni, s, t, ldt, ixt, q, ldq, ixq, tol, iwork,
                       dwork, ldwork, info);
                if (*info != 0) {
                    *ilst = here;
                    return;
                }
                here -= nbnext;

                if (nbf == 2) {
                    if (s[i] == 1) {
                        it = ixt[i] - 1 + (ni[i] + here - 1) * ldt[i] + ni[ip1] + here;
                    } else {
                        it = ixt[i] - 1 + (ni[ip1] + here - 1) * ldt[i] + ni[i] + here;
                    }
                    if (t[it] == ZERO) {
                        nbf = 3;
                    }
                }
            } else {
                nbnext = 1;
                if (here >= 3) {
                    if (s[i] == 1) {
                        it = ixt[i] - 1 + (ni[i] + here - 3) * ldt[i] + ni[ip1] + here - 2;
                    } else {
                        it = ixt[i] - 1 + (ni[ip1] + here - 3) * ldt[i] + ni[i] + here - 2;
                    }
                    if (t[it] != ZERO) {
                        nbnext = 2;
                    }
                }

                mb03kb(compq, whichq, ws, k, nc, kschur, here - nbnext, nbnext, 1,
                       n, ni, s, t, ldt, ixt, q, ldq, ixq, tol, iwork,
                       dwork, ldwork, info);
                if (*info != 0) {
                    *ilst = here;
                    return;
                }

                if (nbnext == 1) {
                    mb03kb(compq, whichq, ws, k, nc, kschur, here, nbnext, 1,
                           n, ni, s, t, ldt, ixt, q, ldq, ixq, tol, iwork,
                           dwork, ldwork, info);
                    if (*info != 0) {
                        *ilst = here;
                        return;
                    }
                    here--;
                } else {
                    if (s[i] == 1) {
                        it = ixt[i] - 1 + (ni[i] + here - 2) * ldt[i] + ni[ip1] + here - 1;
                    } else {
                        it = ixt[i] - 1 + (ni[ip1] + here - 2) * ldt[i] + ni[i] + here - 1;
                    }
                    if (t[it] == ZERO) {
                        nbnext = 1;
                    }

                    if (nbnext == 2) {
                        mb03kb(compq, whichq, ws, k, nc, kschur, here - 1, 2, 1,
                               n, ni, s, t, ldt, ixt, q, ldq, ixq, tol, iwork,
                               dwork, ldwork, info);
                        if (*info != 0) {
                            *ilst = here;
                            return;
                        }
                        here -= 2;
                    } else {
                        mb03kb(compq, whichq, ws, k, nc, kschur, here, 1, 1,
                               n, ni, s, t, ldt, ixt, q, ldq, ixq, tol, iwork,
                               dwork, ldwork, info);
                        if (*info != 0) {
                            *ilst = here;
                            return;
                        }
                        mb03kb(compq, whichq, ws, k, nc, kschur, here - 1, 1, 1,
                               n, ni, s, t, ldt, ixt, q, ldq, ixq, tol, iwork,
                               dwork, ldwork, info);
                        if (*info != 0) {
                            *ilst = here - 1;
                            return;
                        }
                        here -= 2;
                    }
                }
            }
        }
    }

    *ilst = here;
    dwork[0] = (f64)minwrk;
}
