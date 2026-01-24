/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_UE_H
#define SLICOT_UE_H

#include "../slicot_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Get machine-specific parameters for SLICOT routines.
 *
 * Provides an extension of the LAPACK routine ILAENV to return
 * machine-specific parameters for SLICOT routines. The default values
 * aim to give good performance on a wide range of computers.
 *
 * @param[in] ispec Specifies the parameter to be returned:
 *                  = 1: optimal blocksize
 *                  = 2: minimum block size for block routine
 *                  = 3: crossover point (use unblocked for N < this)
 *                  = 4: number of shifts (product eigenvalue routine)
 *                  = 8: crossover point for multishift QR
 * @param[in] name Name of the calling subroutine (upper or lower case)
 * @param[in] opts Character options to the subroutine, concatenated
 * @param[in] n1 First problem dimension
 * @param[in] n2 Second problem dimension
 * @param[in] n3 Third problem dimension
 *
 * @return The function value according to ISPEC, or -1 for invalid ISPEC
 */
i32 ue01md(i32 ispec, const char *name, const char *opts,
           i32 n1, i32 n2, i32 n3);

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_UE_H */
