# Kyber Reference Bridge - Algebraic Attack Mode
import sys
import hashlib
import binascii

class Kyber512:
    n = 256
    k = 2
    q = 3329

    @staticmethod
    def prf(seed, b_len, nonce):
        shake = hashlib.shake_256()
        shake.update(seed + bytes([nonce]))
        return shake.digest(b_len)
        
    @staticmethod
    def xof(seed, i, j, b_len):
        shake = hashlib.shake_128()
        shake.update(seed + bytes([i, j]))
        return shake.digest(b_len)

    @staticmethod
    def parse_t(t_bytes):
        res = []
        for i in range(0, len(t_bytes), 3):
            b0, b1, b2 = t_bytes[i], t_bytes[i+1], t_bytes[i+2]
            res.append(b0 | ((b1 & 0x0f) << 8))
            res.append(((b1 >> 4) | (b2 << 4)))
        return res

    @staticmethod
    def parse_a(rho):
        A = [[[] for _ in range(Kyber512.k)] for _ in range(Kyber512.k)]
        for i in range(Kyber512.k):
            for j in range(Kyber512.k):
                buf = Kyber512.xof(rho, j, i, 512*3)
                idx = 0
                coeffs = []
                while len(coeffs) < Kyber512.n:
                    d1 = buf[idx] | ((buf[idx+1] & 0x0f) << 8)
                    d2 = (buf[idx+1] >> 4) | (buf[idx+2] << 4)
                    idx += 3
                    if d1 < Kyber512.q: coeffs.append(d1)
                    if len(coeffs) < Kyber512.n and d2 < Kyber512.q: coeffs.append(d2)
                A[i][j] = coeffs
        return A
        
    zetas = [
        2285, 2586, 2560, 2221, 3285, 1642, 3162, 2731, 1224, 2529, 2374, 2031, 1424, 2737, 2895,  277,
        2935, 2398,  191, 1238, 3125,  714,  643, 2307, 3273, 2125, 1138, 2404, 3208, 1151, 1474,  946,
        3182, 1634, 1504, 2574, 1534,  298, 2011, 2387, 3034, 1452, 2187, 3128, 2808,  543, 1756, 1123,
        2441, 1335, 1629,  192, 1714, 1530, 2013,  138, 1845, 1968, 1644, 1066,  331, 2865, 2901,  188,
        1441, 3154, 2062, 1894,  287,  221, 2989, 3217, 3008, 2775, 1195,  710, 2289, 1157, 1813, 2977,
        2280, 2707,  259, 3011, 2816,   24, 1278, 1782,  965,  445, 1492,  673, 2017, 1848, 1521, 2505,
        2109,  340, 3110,   50, 1500,  802, 2728,  345, 2803,  253,  581, 1084,  104, 2005,  103,  538,
        2339, 2964,  554, 2697, 2656,  318,  982,  368, 2921, 2228,  196, 3192,  152, 1616, 1243, 1133,
    ]

    @staticmethod
    def basemul(a0, a1, b0, b1, zeta):
        q = Kyber512.q
        r0 = (a0 * b0 + a1 * b1 * zeta) % q
        r1 = (a0 * b1 + a1 * b0) % q
        return r0, r1

    @staticmethod
    def inv_ntt(a):
        q = Kyber512.q
        k = 127
        res = list(a)
        step = 2
        while step <= 128:
            half = step // 2
            for start in range(0, 256, step):
                zeta = Kyber512.zetas[k]
                k -= 1
                for j in range(start, start + half):
                    t = res[j]
                    res[j] = (t + res[j+half]) % q
                    res[j+half] = (zeta * (res[j+half] - t)) % q
            step *= 2
        f = 3303  # 128^{-1} mod 3329
        for i in range(256):
            res[i] = (res[i] * f) % q
        return res

def mod_inv(a, q=3329):
    return pow(a, q-2, q)

def solve_4x4(M, b, q=3329):
    """Solve 4x4 linear system Mx = b mod q via Gaussian elimination."""
    n = 4
    aug = [list(M[i]) + [b[i]] for i in range(n)]
    
    for col in range(n):
        # Find pivot
        pivot = -1
        for row in range(col, n):
            if aug[row][col] % q != 0:
                pivot = row
                break
        if pivot == -1:
            return None  # Singular
        aug[col], aug[pivot] = aug[pivot], aug[col]
        
        inv = mod_inv(aug[col][col], q)
        for j in range(n+1):
            aug[col][j] = (aug[col][j] * inv) % q
        
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(n+1):
                aug[row][j] = (aug[row][j] - factor * aug[col][j]) % q
    
    return [aug[i][n] % q for i in range(n)]

def centered(x, q=3329):
    """Map from [0, q) to centered representation [-q/2, q/2)."""
    if x > q // 2:
        return x - q
    return x

def parse_pk_and_attack(hex_str):
    hex_len = len(hex_str)
    if hex_len > 1600:
        hex_str = hex_str[-1600:]
        
    pk = binascii.unhexlify(hex_str)
    t_enc = pk[:768]
    rho = pk[768:]
    
    t_ntt = Kyber512.parse_t(t_enc)
    A_ntt = Kyber512.parse_a(rho)
    
    b_ntt = [t_ntt[i*256:(i+1)*256] for i in range(2)]
    
    # Generate full zeta table (256 entries) matching C init_zetas
    q = 3329
    zeta_prim = 17
    zetas_full = [0]*256
    for ii in range(128):
        br = 0
        for bit in range(7):
            if (ii >> bit) & 1:
                br |= (1 << (6 - bit))
        zetas_full[ii] = pow(zeta_prim, br, q)
        zetas_full[128 + ii] = zetas_full[ii]
    
    # ALGEBRAIC ATTACK: Solve 4x4 system at each of 128 frequency pairs
    s0_hat = [0]*256
    s1_hat = [0]*256
    
    for i in range(128):
        v0, v1 = 2*i, 2*i+1
        Z = zetas_full[64 + i]
        
        a00_0, a00_1 = A_ntt[0][0][v0], A_ntt[0][0][v1]
        a01_0, a01_1 = A_ntt[0][1][v0], A_ntt[0][1][v1]
        a10_0, a10_1 = A_ntt[1][0][v0], A_ntt[1][0][v1]
        a11_0, a11_1 = A_ntt[1][1][v0], A_ntt[1][1][v1]
        
        # 4x4 system: M * [s0_v0, s0_v1, s1_v0, s1_v1] = [b0_v0, b0_v1, b1_v0, b1_v1]
        M = [
            [a00_0, (Z*a00_1)%q, a01_0, (Z*a01_1)%q],
            [a00_1, a00_0,       a01_1, a01_0      ],
            [a10_0, (Z*a10_1)%q, a11_0, (Z*a11_1)%q],
            [a10_1, a10_0,       a11_1, a11_0      ],
        ]
        b_vec = [b_ntt[0][v0], b_ntt[0][v1], b_ntt[1][v0], b_ntt[1][v1]]
        
        sol = solve_4x4(M, b_vec, q)
        if sol is None:
            print(f"  [!] Singular system at pair {i}", file=sys.stderr)
            continue
        
        s0_hat[v0], s0_hat[v1] = sol[0], sol[1]
        s1_hat[v0], s1_hat[v1] = sol[2], sol[3]
    
    # Inverse NTT to get spatial domain candidates
    s0_spatial = Kyber512.inv_ntt(s0_hat)
    s1_spatial = Kyber512.inv_ntt(s1_hat)
    
    # Convert to centered representation and round to {-eta..eta}
    eta = 3
    s0_rounded = []
    s1_rounded = []
    for i in range(256):
        v0 = centered(s0_spatial[i])
        v1 = centered(s1_spatial[i])
        s0_rounded.append(max(-eta, min(eta, v0)))
        s1_rounded.append(max(-eta, min(eta, v1)))
    
    # Also compute b_spatial and A_spatial for verification
    b_spatial = [Kyber512.inv_ntt(b_ntt[i]) for i in range(2)]
    A_spatial = [[Kyber512.inv_ntt(A_ntt[i][j]) for j in range(2)] for i in range(2)]
    
    with open("kyber_ntt_dump.txt", "w") as f:
        f.write("2 256 3329\n")
        
        # Line 2: b_spatial
        for i in range(2):
            for e in b_spatial[i]:
                f.write(f"{e} ")
        f.write("\n")
        
        # Line 3: A_spatial (2x2 x 256)
        for i in range(2):
            for j in range(2):
                for e in A_spatial[i][j]:
                    f.write(f"{e} ")
        f.write("\n")
        
        # Line 4: b_ntt (2 x 256)
        for i in range(2):
            for e in b_ntt[i]:
                f.write(f"{e} ")
        f.write("\n")
        
        # Line 5: A_ntt (2x2 x 256)
        for i in range(2):
            for j in range(2):
                for e in A_ntt[i][j]:
                    f.write(f"{e} ")
        f.write("\n")
        
        # Line 6: basemul zetas (128 values: zetas_full[64..191])
        for i in range(128):
            f.write(f"{zetas_full[64+i]} ")
        f.write("\n")
    
    print(f"  [SPECTRAL] Dumped NTT-domain data for 128-slot spectral attack")

if __name__ == "__main__":
    hex_key = sys.argv[1]
    parse_pk_and_attack(hex_key)
