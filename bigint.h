/*
 * ═══════════════════════════════════════════════════════════════════════════════
 * BIGINT LIBRARY — 4096-bit Arbitrary Precision Integer Arithmetic
 * ═══════════════════════════════════════════════════════════════════════════════
 * C port of bigint.asm for the HexState 6-State Quantum Processor Engine.
 * 64 limbs × 64 bits = 4096 bits.
 */

#ifndef BIGINT_H
#define BIGINT_H

#include <stdint.h>
#include <stddef.h>

#define BIGINT_LIMBS  64
#define BIGINT_BYTES  (BIGINT_LIMBS * 8)   /* 512 */
#define BIGINT_BITS   (BIGINT_LIMBS * 64)  /* 4096 */

typedef struct {
    uint64_t limbs[BIGINT_LIMBS];  /* Little-endian: limbs[0] is LSW */
} BigInt;

/* ─── Core Operations ─── */
void     bigint_clear(BigInt *a);
void     bigint_copy(BigInt *dst, const BigInt *src);
int      bigint_cmp(const BigInt *a, const BigInt *b);       /* 1, 0, -1 */
int      bigint_is_zero(const BigInt *a);                    /* 1 if zero */

/* ─── Arithmetic ─── */
void     bigint_add(BigInt *result, const BigInt *a, const BigInt *b);
void     bigint_sub(BigInt *result, const BigInt *a, const BigInt *b);
void     bigint_mul(BigInt *result, const BigInt *a, const BigInt *b);
void     bigint_div_mod(const BigInt *dividend, const BigInt *divisor,
                        BigInt *quotient, BigInt *remainder);

/* ─── Bit Operations ─── */
void     bigint_shl1(BigInt *a);
void     bigint_shr1(BigInt *a);
int      bigint_get_bit(const BigInt *a, uint32_t bit_index);
void     bigint_set_bit(BigInt *a, uint32_t bit_index);
void     bigint_clr_bit(BigInt *a, uint32_t bit_index);
uint32_t bigint_bitlen(const BigInt *a);

/* ─── Conversion ─── */
void     bigint_set_u64(BigInt *a, uint64_t val);
uint64_t bigint_to_u64(const BigInt *a);

/* ─── Higher-Level ─── */
void     bigint_gcd(BigInt *result, const BigInt *a, const BigInt *b);
void     bigint_pow_mod(BigInt *result, const BigInt *base,
                        const BigInt *exp, const BigInt *mod);

/* ─── String Conversion ─── */
/* Parse a decimal string into a BigInt. Returns 0 on success, -1 on error. */
int      bigint_from_decimal(BigInt *a, const char *str);
/* Write BigInt as decimal string into buf (must hold at least 1240 chars). */
void     bigint_to_decimal(char *buf, size_t bufsize, const BigInt *a);

#endif /* BIGINT_H */
