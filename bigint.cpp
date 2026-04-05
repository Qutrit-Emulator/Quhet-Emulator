#include "bigint.h"
#include <string.h>

extern "C" {

void bigint_clear(BigInt *a) {
    mpz_set_ui(a->z, 0);
}

void bigint_copy(BigInt *dst, const BigInt *src) {
    mpz_set(dst->z, src->z);
}

int bigint_is_zero(const BigInt *a) {
    return mpz_cmp_ui(a->z, 0) == 0;
}

int bigint_cmp(const BigInt *a, const BigInt *b) {
    return mpz_cmp(a->z, b->z);
}

void bigint_add(BigInt *result, const BigInt *a, const BigInt *b) {
    mpz_add(result->z, a->z, b->z);
}

void bigint_sub(BigInt *result, const BigInt *a, const BigInt *b) {
    mpz_sub(result->z, a->z, b->z);
}

void bigint_mul(BigInt *result, const BigInt *a, const BigInt *b) {
    mpz_mul(result->z, a->z, b->z);
}

void bigint_shl1(BigInt *a) {
    mpz_mul_2exp(a->z, a->z, 1);
}

void bigint_shr1(BigInt *a) {
    mpz_fdiv_q_2exp(a->z, a->z, 1);
}

int bigint_get_bit(const BigInt *a, uint32_t bit_index) {
    return mpz_tstbit(a->z, bit_index);
}

void bigint_set_bit(BigInt *a, uint32_t bit_index) {
    mpz_setbit(a->z, bit_index);
}

void bigint_clr_bit(BigInt *a, uint32_t bit_index) {
    mpz_clrbit(a->z, bit_index);
}

uint32_t bigint_bitlen(const BigInt *a) {
    return mpz_sizeinbase(a->z, 2);
}

void bigint_set_u64(BigInt *a, uint64_t val) {
    mpz_set_ui(a->z, val);
}

uint64_t bigint_to_u64(const BigInt *a) {
    if (mpz_fits_ulong_p(a->z)) {
        return mpz_get_ui(a->z);
    }
    return 0;
}

void bigint_div_mod(const BigInt *dividend, const BigInt *divisor,
                    BigInt *quotient, BigInt *remainder) {
    if (quotient && remainder) {
        mpz_tdiv_qr(quotient->z, remainder->z, dividend->z, divisor->z);
    } else if (quotient) {
        mpz_tdiv_q(quotient->z, dividend->z, divisor->z);
    } else if (remainder) {
        mpz_tdiv_r(remainder->z, dividend->z, divisor->z);
    }
}

void bigint_gcd(BigInt *result, const BigInt *a, const BigInt *b) {
    mpz_gcd(result->z, a->z, b->z);
}

void bigint_pow_mod(BigInt *result, const BigInt *base,
                    const BigInt *exp, const BigInt *mod) {
    mpz_powm(result->z, base->z, exp->z, mod->z);
}

int bigint_from_decimal(BigInt *a, const char *str) {
    return mpz_set_str(a->z, str, 10);
}

void bigint_to_decimal(char *buf, size_t bufsize, const BigInt *a) {
    mpz_get_str(buf, 10, a->z);
}

} // extern "C"
