/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 1996-2025, The SLICOT Team (original Fortran77 code)
 * Copyright (c) 2025, slicot.c contributors (C11 translation)
 */

#ifndef SLICOT_TYPES_H
#define SLICOT_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <complex.h>

/**
 * @file slicot_types.h
 * @brief Type aliases for SLICOT C library
 *
 * Provides convenient type aliases following SLICUTLET conventions:
 * - i32: 32-bit signed integer (Fortran INTEGER)
 * - i64: 64-bit signed integer (for ILP64 BLAS/LAPACK)
 * - f64: 64-bit floating point (Fortran DOUBLE PRECISION)
 * - c128: 128-bit complex (Fortran COMPLEX*16)
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Integer types */
typedef int32_t i32;
typedef int64_t i64;

/* Floating point types */
typedef double f64;

/* Complex types */
typedef double complex c128;

/* Static assertions for ABI compatibility */
_Static_assert(sizeof(i32) == 4, "i32 must be 4 bytes");
_Static_assert(sizeof(i64) == 8, "i64 must be 8 bytes");
_Static_assert(sizeof(f64) == 8, "f64 must be 8 bytes");
_Static_assert(sizeof(c128) == 2 * sizeof(f64), "c128 must be two f64s");

#ifdef __cplusplus
}
#endif

#endif /* SLICOT_TYPES_H */
