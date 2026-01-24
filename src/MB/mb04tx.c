// SPDX-License-Identifier: BSD-3-Clause

#include "slicot.h"
#include "slicot_blas.h"

void mb04tx(bool updatq, bool updatz, i32 m, i32 n, i32 *nblcks,
            i32 *inuk, i32 *imuk, f64 *a, i32 lda, f64 *e, i32 lde,
            f64 *q, i32 ldq, f64 *z, i32 ldz, i32 *mnei) {
    const f64 zero = 0.0;

    mnei[0] = 0;
    mnei[1] = 0;
    mnei[2] = 0;
    mnei[3] = 0;

    if (m <= 0 || n <= 0) {
        return;
    }

    // Initialization
    i32 ismuk = 0;
    i32 isnuk = 0;

    i32 nb = *nblcks;
    for (i32 k = 0; k < nb; k++) {
        ismuk += imuk[k];
        isnuk += inuk[k];
    }

    // MEPS, NEPS are dimensions of pencil s*E(eps)-A(eps)
    i32 meps = isnuk;
    i32 neps = ismuk;
    i32 minf = 0;
    i32 ninf = 0;

    // MUKP1 = mu(k+1). It is assumed that mu(NBLCKS+1) = 0
    i32 mukp1 = 0;

    // Process blocks from last to first (k = NBLCKS, ..., 1 in Fortran)
    for (i32 k = nb - 1; k >= 0; k--) {
        i32 nuk = inuk[k];
        i32 muk = imuk[k];

        // Reduce submatrix E(k,k+1) to square matrix
        // WHILE (NU(k) > MU(k+1))
        while (nuk > mukp1) {
            // sk1p1 = sum(i=k+1,...,p-1) NU(i)
            // tk1p1 = sum(i=k+1,...,p-1) MU(i)
            i32 sk1p1 = 0;
            i32 tk1p1 = 0;

            // Process blocks from k+1 to NBLCKS (ip = k+1, ..., nb-1 in 0-based)
            for (i32 ip = k + 1; ip < nb; ip++) {
                // tp1 = sum(i=1,...,p-1) MU(i) = ismuk + tk1p1
                // In Fortran: TP1 = ISMUK + TK1P1 (1-based column)
                i32 tp1 = ismuk + tk1p1;  // 1-based
                i32 ra = isnuk + sk1p1;   // 1-based row
                i32 ca = tp1;             // 1-based column

                i32 mup = imuk[ip];
                i32 nup = inuk[ip];
                i32 mup1 = nup;

                // Annihilate elements using column Givens rotations
                // Fortran: DO CJA = CA, CA + MUP - NUP - 1
                for (i32 cja = ca; cja <= ca + mup - nup - 1; cja++) {
                    f64 sc, ss;
                    // 0-based indices
                    i32 ra_idx = ra - 1;
                    i32 cja_idx = cja - 1;

                    // DROTG on A(RA, CJA) and A(RA, CJA+1)
                    SLC_DROTG(&a[ra_idx + cja_idx * lda],
                              &a[ra_idx + (cja_idx + 1) * lda], &sc, &ss);

                    // Apply transformations to A and E
                    // MB04TU operates on rows 1:RA-1 of columns CJA and CJA+1
                    mb04tu(ra - 1, &a[cja_idx * lda], 1,
                           &a[(cja_idx + 1) * lda], 1, sc, ss);

                    // Interchange: A(RA, CJA+1) = A(RA, CJA), A(RA, CJA) = 0
                    a[ra_idx + (cja_idx + 1) * lda] = a[ra_idx + cja_idx * lda];
                    a[ra_idx + cja_idx * lda] = zero;

                    // Apply to E (rows 1:RA)
                    mb04tu(ra, &e[cja_idx * lde], 1,
                           &e[(cja_idx + 1) * lde], 1, sc, ss);

                    // Update Z if needed
                    if (updatz) {
                        mb04tu(n, &z[cja_idx * ldz], 1,
                               &z[(cja_idx + 1) * ldz], 1, sc, ss);
                    }
                }

                // Annihilate remaining elements with alternating row/column rotations
                // CE = TP1 + MUP (1-based)
                // CA = CE - MUP1 - 1 (1-based)
                i32 ce = tp1 + mup;
                ca = ce - mup1 - 1;

                // Fortran: DO RJE = RA + 1, RA + MUP1
                for (i32 rje = ra + 1; rje <= ra + mup1; rje++) {
                    f64 sc, ss;
                    i32 cje = ce + 1;  // 1-based
                    i32 cja = ca + 1;  // 1-based

                    // 0-based indices
                    i32 rje_idx = rje - 1;
                    i32 cje_idx = cje - 1;
                    i32 cja_idx = cja - 1;

                    // Row transformation
                    // DROTG on E(RJE, CJE) and E(RJE-1, CJE)
                    SLC_DROTG(&e[rje_idx + cje_idx * lde],
                              &e[(rje_idx - 1) + cje_idx * lde], &sc, &ss);

                    // Apply to E (columns CJE+1:N)
                    i32 ncols = n - cje;
                    if (ncols > 0) {
                        mb04tu(ncols, &e[rje_idx + cje * lde], lde,
                               &e[(rje_idx - 1) + cje * lde], lde, sc, ss);
                    }

                    // Interchange E rows
                    e[(rje_idx - 1) + cje_idx * lde] = e[rje_idx + cje_idx * lde];
                    e[rje_idx + cje_idx * lde] = zero;

                    // Apply to A (columns CJA:N)
                    i32 ncols_a = n - cja + 1;
                    mb04tu(ncols_a, &a[rje_idx + cja_idx * lda], lda,
                           &a[(rje_idx - 1) + cja_idx * lda], lda, sc, ss);

                    // Update Q if needed
                    if (updatq) {
                        mb04tu(m, &q[rje_idx * ldq], 1,
                               &q[(rje_idx - 1) * ldq], 1, sc, ss);
                    }

                    // Column transformation
                    // DROTG on A(RJE, CJA) and A(RJE, CJA+1)
                    SLC_DROTG(&a[rje_idx + cja_idx * lda],
                              &a[rje_idx + (cja_idx + 1) * lda], &sc, &ss);

                    // Apply to A (rows 1:RJE-1)
                    mb04tu(rje - 1, &a[cja_idx * lda], 1,
                           &a[(cja_idx + 1) * lda], 1, sc, ss);

                    // Interchange A columns
                    a[rje_idx + (cja_idx + 1) * lda] = a[rje_idx + cja_idx * lda];
                    a[rje_idx + cja_idx * lda] = zero;

                    // Apply to E (rows 1:RJE)
                    mb04tu(rje, &e[cja_idx * lde], 1,
                           &e[(cja_idx + 1) * lde], 1, sc, ss);

                    // Update Z if needed
                    if (updatz) {
                        mb04tu(n, &z[cja_idx * ldz], 1,
                               &z[(cja_idx + 1) * ldz], 1, sc, ss);
                    }
                }

                sk1p1 += nup;
                tk1p1 += mup;
            }

            // Reduce dimensions
            muk--;
            nuk--;
            ismuk--;
            isnuk--;
            meps--;
            neps--;
            minf++;
            ninf++;
        }

        imuk[k] = muk;
        inuk[k] = nuk;

        // Prepare for next block
        isnuk -= nuk;
        ismuk -= muk;
        mukp1 = muk;
    }

    // If mu(NBLCKS) = 0, the last submatrix is empty
    if (nb > 0 && imuk[nb - 1] == 0) {
        (*nblcks)--;
    }

    // Store dimensions
    mnei[0] = meps;
    mnei[1] = neps;
    mnei[2] = minf;
    mnei[3] = ninf;
}
