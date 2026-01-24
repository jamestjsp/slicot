// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"

void mb04ty(bool updatq, bool updatz, i32 m, i32 n, i32 nblcks,
            const i32 *inuk, const i32 *imuk, f64 *a, i32 lda,
            f64 *e, i32 lde, f64 *q, i32 ldq, f64 *z, i32 ldz, i32 *info) {
    *info = 0;

    if (m <= 0 || n <= 0) {
        return;
    }

    // ISMUK = sum(i=1,...,k) MU(i)
    // ISNUK1 = sum(i=1,...,k-1) NU(i)
    i32 ismuk = 0;
    i32 isnuk1 = 0;

    for (i32 k = 0; k < nblcks; k++) {
        ismuk += imuk[k];
        isnuk1 += inuk[k];
    }

    // Note: ISNUK1 has not yet correct value - it's sum of all NU, not sum(k-1)

    i32 mukp1 = 0;  // MU(k+1) from previous iteration

    // Process blocks in reverse order: k = NBLCKS, ..., 1
    for (i32 k = nblcks - 1; k >= 0; k--) {
        i32 muk = imuk[k];
        i32 nuk = inuk[k];
        isnuk1 -= nuk;

        // Determine coordinates (1-based as used by MB04TW/MB04TV)
        i32 ifire = 1 + isnuk1;
        i32 ifice = 1 + ismuk;
        i32 ifica = ifice - muk;

        // Check: mu(k+1) > nu(k) is error
        if (mukp1 > nuk) {
            *info = 1;
            return;
        }

        // Reduce E(k) to upper triangular via row Givens (MB04TW)
        i32 tw_info = 0;
        mb04tw(updatq, m, n, nuk, mukp1, ifire, ifice, ifica,
               a, lda, e, lde, q, ldq, &tw_info);

        // Check: nu(k) > mu(k) is error
        if (nuk > muk) {
            *info = 2;
            return;
        }

        // Reduce A(k) to upper triangular via column Givens (MB04TV)
        i32 tv_info = 0;
        mb04tv(updatz, n, nuk, muk, ifire, ifica,
               a, lda, e, lde, z, ldz, &tv_info);

        ismuk -= muk;
        mukp1 = muk;
    }
}
