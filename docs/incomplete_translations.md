# Incomplete SLICOT C Translations

Systematic audit of ~620 C source files against Fortran77 originals.
Generated 2025-02-07.

## Summary

| Category | Count | Action |
|----------|-------|--------|
| INCOMPLETE | 8 | Must fix — wrong/degraded results |
| STUB | 2 | Must fix — mostly unimplemented |
| DEGRADED | 5 | Should fix — correct but missing optimizations |
| FALSE_POSITIVE | ~30 | No action needed |

---

## INCOMPLETE — Major algorithmic sections missing

### ab13hd.c (4% ratio, 141/3439 lines)
**Missing:** 33 subroutine calls. Entire JOBE='G'/'C' (general/companion) code paths.
Only handles JOBE='I' (identity). Missing: DGEBAL, DGEHRD, DHSEQR, DGESVD, DGETRF,
DGETRS, DGGBAK, DGGBAL, DHGEQZ, DLASRT, DORMHR, DORMQR, DSYRK, DTGSEN, DTRCON,
DTRSM, MA02AD, MB01SD, MB02RD, MB02SD, MB02TD, MB03XD, MB04BP, SB04OD, TB01ID,
TG01AD, TG01BD, ZGESVD. The file explicitly notes "simplified implementation".
**Impact:** Returns wrong results for non-identity E matrices.

### ab09iy.c (34% ratio, 290/844 lines)
**Missing:** MB01WD, DSYEV (spectral factorization path for frequency-weighted
Gramians when alpha < 1). The basic weighted/unweighted Lyapunov paths are present
but the spectral reweighting for stability-enforcing alpha parameters is absent.
**Impact:** Wrong Gramians when ALPHAC < 1 or ALPHAO < 1 with non-trivial weights.

### ab13dd.c (66% ratio, 1241/1853 lines)
**Missing:** DLASRT, MB01SD. Missing eigenvalue sorting and symmetric matrix update
in the continuous-time structured singular value computation.
**Impact:** May produce incorrect mu upper bounds for continuous-time systems.

### mb01vd.c (7% ratio, 133/1678 lines)
**Missing:** 28-branch sparse optimization using DCOPY-based operations for all
UPLO/TRANS/TYPE combinations. Only basic dense path implemented.
**Impact:** Correct for dense matrices but may produce wrong results for sparse
structured operations where the Fortran has specialized handling.

### mb02pd.c (22% ratio, 124/540 lines)
**Missing:** DGEEQU (equilibration), DGERFS (iterative refinement), DLAQGE
(row/column scaling). Error bounds (ferr/berr) are zeroed instead of computed.
Equilibration (FACT='E') is silently ignored.
**Impact:** Reduced numerical accuracy. No iterative refinement means larger
backward errors for ill-conditioned systems.

### mb03bd.c (68% ratio, 1335/1954 lines)
**Missing:** Zero-chasing algorithm for periodic Schur decomposition (TODO marker at
line 1246). The main eigenvalue reordering logic exists but the ILO/IHI zero-chasing
sweep path is unimplemented.
**Impact:** May fail or produce wrong results for certain eigenvalue reordering cases.

### mb04rb.c (23% ratio, 80/347 lines)
**Missing:** MB01KD, MB04PA, DGEMM. Only handles basic case, missing the blocked
symplectic URV decomposition update algorithm.
**Impact:** Wrong results for non-trivial block operations.

### mb01uy.c (34% ratio, 158/457 lines)
**Missing:** DGEMM, DGEQRF. The Fortran has optimized paths using DGEMM for large
matrices and DGEQRF for QR-based symmetric rank-k update. C version only has
element-wise loops.
**Impact:** Correct for small matrices but missing the DGEMM/DGEQRF optimized path
that handles the actual computation differently for larger cases. Needs verification
that element-wise path covers all cases.

---

## STUB — Only parameter validation, most logic missing

### mb04pb.c (7% ratio, 27/341 lines)
**Missing:** DGEHRD, DGEMM, DSYR2K, MB04PA. Only has parameter validation.
Requires UE01MD (not yet translated) for blocked symplectic factorization.
**Impact:** Function does nothing. Blocked on UE01MD dependency.

### mb04tb.c (12% ratio, 89/687 lines)
**Missing:** DGEMM, DGEQRF, MB03XU. Delegates to mb04ts for unblocked case only.
The entire blocked algorithm (the primary purpose of the routine) is missing.
**Impact:** Only works for small matrices where unblocked path suffices.

---

## DEGRADED — Correct but missing optimizations

### mb03xp.c (16% ratio, 105/633 lines)
**Missing:** MB03YA, DLARNV, DLARFG, DLARFX, DGEMV, DTRMV (multishift QZ bulge
chasing). Delegates entirely to mb03yd (single-shift QZ).
**Impact:** Correct results but O(n) slower for large matrices. The multishift
optimization is ~300 lines of Fortran.

### mb02cu.c (71% ratio, 713/1001 lines)
**Missing:** DGELQ2, DGEQR2 (unblocked QR/LQ). Uses DGEQRF/DGELQF (blocked
versions) instead.
**Impact:** Correct. Blocked versions are actually preferred for large matrices.
Minor: unblocked versions may be slightly faster for very small matrices.

### mb03bb.c (120% ratio, 495/411 lines)
**Missing:** DLADIV, ZLARTG, ZROT. These are inlined as direct computations.
C version is actually longer than Fortran (complex arithmetic expanded).
**Impact:** Correct. The complex operations are correctly implemented inline.

### mc01td.c (46% ratio, 136/290 lines)
**Missing:** DRSCL (reciprocal scaling). Uses inline `1.0/scale` multiplication
instead.
**Impact:** Correct. DRSCL is a trivial BLAS-like operation.

### mb03bf.c (64% ratio, 84/130 lines)
**Missing:** DROT (Givens rotation application). Uses inline rotation code.
**Impact:** Correct. DROT is trivially inlined.

---

## FALSE_POSITIVE — No issues found

| File | Reason |
|------|--------|
| ab05rd.c | Full translation, low ratio due to verbose Fortran comments |
| ab09ed.c | Full translation, calls all required subroutines |
| ab08nz.c | Full translation with all complex variants |
| ab08mz.c | Full translation with all complex variants |
| dg01nd.c | Legitimate wrapper delegating to dg01md + dg01ny |
| fd01ad.c | Full translation with DLARTG inlined via SLC_DLARTG |
| ib01ad.c | Dispatcher calling ib01md/ib01od, all paths present |
| ib01px.c | Full translation, low ratio due to Fortran comment blocks |
| ib01py.c | Full translation, low ratio due to Fortran comment blocks |
| ib03ad.c | Full translation, "simplified" comment refers to workspace calc only |
| mb01ru.c | Full translation with DGEMM/DSYR2K inlined |
| mb03md.c | Full translation (bisection algorithm fully implemented) |
| mb03nd.c | Full translation (small routine, QR iteration) |
| mb03pd.c | Full translation (product eigenvalue computation) |
| mb03vd.c | Full translation (sorting eigenvalues of product) |
| mb04nd.c | Full translation (RQ/QR factorization update) |
| mb04od.c | Full translation (RQ factorization update) |
| mb04ty.c | Full translation (symplectic butterfly) |
| mc03nd.c | Full translation (polynomial GCD computation) |
| nf01bq.c | Full translation, parameter validation is complete |
| sb02ru.c | Calls mb02pd (which itself is incomplete, but sb02ru's translation is faithful) |
| sb03ou.c | Full translation (Cholesky factor of Lyapunov solution) |
| sb16cd.c | Full translation (coprime factorization controller) |
| sg03bd.c | Uses DGGES (modern replacement for deprecated DGEGS) |
| tb01ux.c | Full translation delegating to tb01ud + tb01xd |
| tb04ad.c | Full translation (transfer matrix computation) |
| tf01qd.c | Full translation (output response, simple loop) |
| tf01rd.c | Full translation (output response, simple loop) |
| tg01hd.c | Full translation (descriptor system reduction) |
| tg01jy.c | Uses tg01hx instead of tg01hy (equivalent replacement) |
| tg01nd.c | Full translation (descriptor system reduction) |

---

## Notes

- **Method:** Automated triage via subroutine call comparison + line-count ratio,
  followed by manual verification of each candidate (reading both C and Fortran sources).
- **Filters applied:** Inlined BLAS (DCOPY, DSCAL, DSWAP, DAXPY, DLASET, DLACPY,
  DLASCL, DROT, DLADIV, DRSCL), legitimate delegation patterns, modern LAPACK
  replacements (DGEGS→DGGES).
- **mb01vd** sparse optimization is the largest single gap (~1300 lines of Fortran).
- **ab13hd** is the most severe — 96% of Fortran logic missing.
