import numpy as np

# ------------------------
# Field parameters (KoalaBear)
# ------------------------
modulus = 2**31 - 2**24 + 1          # p = 2,130,706,433
_NR = np.uint64(3)                               # u^2 = 3   (in F_p)
_ALPHA0 = np.uint64(1)                           # v^2 = ALPHA0 + ALPHA1 * u
_ALPHA1 = np.uint64(1)
modulus32 = np.uint32(2**31 - 2**24 + 1)          # p = 2,130,706,433
modulus64 = np.uint64(2**31 - 2**24 + 1)          # p = 2,130,706,433

# ------------------------
# Small helpers
# ------------------------
def _u64(x): return x.astype(np.uint64, copy=False)
def _modp(x): return (x % modulus).astype(np.uint32, copy=False)
def _neg(x):  return _modp(modulus - _u64(x))

# ------------------------
# Base-field ops (F_p)
# ------------------------
def add(x, y):
    o = (x + y)
    return o - (o >= modulus32) * modulus32

def sub(x, y):
    o = (x + modulus32 - y)
    return o - (o >= modulus32) * modulus32

def mul(x, y):
    return _modp(_u64(x) * _u64(y))

def pow3(x):
    x = _u64(x)
    xx = (x * x) % modulus64
    return ((xx * x) % modulus64).astype(np.uint32, copy=False)

def sum(x, axis=0):
    return np.uint32(np.sum(_u64(x), axis=axis) % modulus)

def modinv(x):
    xf = _modp(_u64(x))  # ensure < p
    out = np.empty_like(xf, dtype=np.uint32)
    flat_x = xf.ravel()
    flat_o = out.ravel()
    for i in range(flat_x.size):
        xi = int(flat_x[i])
        flat_o[i] = 0 if xi == 0 else pow(xi, int(modulus) - 2, int(modulus))
    return out.reshape(xf.shape)

# ------------------------
# Quadratic extension ops (F_p[u] with u^2 = _NR)
# Represent a = a0 + a1*u as (..., 2) array [a0, a1]
# ------------------------
def _cplx_add(A, B):
    return np.stack((_modp(_u64(A[...,0]) + _u64(B[...,0])),
                     _modp(_u64(A[...,1]) + _u64(B[...,1]))), axis=-1)

def _cplx_sub(A, B):
    return np.stack((_modp(_u64(A[...,0]) + modulus - _u64(B[...,0])),
                     _modp(_u64(A[...,1]) + modulus - _u64(B[...,1]))), axis=-1)

def _cplx_mul(A, B):
    a0, a1 = _u64(A[...,0]), _u64(A[...,1])
    b0, b1 = _u64(B[...,0]), _u64(B[...,1])
    r0 = _modp(a0 * b0 + _NR * a1 * b1)          # (a0*b0 + NR*a1*b1) mod p
    r1 = _modp(a0 * b1 + a1 * b0)                # (a0*b1 + a1*b0)     mod p
    return np.stack((r0, r1), axis=-1)

def _cplx_neg(A):
    return np.stack((_neg(A[...,0]), _neg(A[...,1])), axis=-1)

def _cplx_square(A):
    # (a0 + a1 u)^2 = (a0^2 + NR*a1^2) + (2*a0*a1) u
    a0, a1 = _u64(A[...,0]), _u64(A[...,1])
    r0 = _modp(a0*a0 + _NR*a1*a1)
    r1 = _modp((a0*a1) << 1)
    return np.stack((r0, r1), axis=-1)

def _cplx_inv(A):
    # (a0 + a1 u)^(-1) = (a0 - a1 u) / (a0^2 - NR*a1^2)
    a0, a1 = _u64(A[...,0]), _u64(A[...,1])
    denom = _modp(a0*a0 + modulus - (_NR*a1*a1) % modulus)            # in F_p
    inv_d = modinv(denom)
    r0 = _modp(a0 * _u64(inv_d))
    r1 = _modp((modulus - a1) * _u64(inv_d))
    return np.stack((r0, r1), axis=-1)

# constant alpha = _ALPHA0 + _ALPHA1*u as a 2-vector
_ALPHA = np.array([np.uint32(_ALPHA0), np.uint32(_ALPHA1)], dtype=np.uint32)

t31m1 = np.uint64(2**31-1)
overflow = np.uint64(2**24 - 1)
p = 2**31 - 2**24 + 1

def weakmod(x):
    return ((x & t31m1) + overflow * (x >> 31))

def mul_ext(x, y):
    """
    Multiply in F_p[X]/(X^4 - 3).
    Pack as (...,4): a0 + a1*X + a2*X^2 + a3*X^3.
    """
    x = np.asarray(x, dtype=np.uint32)
    y = np.asarray(y, dtype=np.uint32)

    p  = np.uint64(2**31 - 2**24 + 1)
    NR = np.uint64(3)  # X^4 = 3

    X = x.astype(np.uint64, copy=False)
    Y = y.astype(np.uint64, copy=False)

    a0, a1, a2, a3 = X[...,0], X[...,1], X[...,2], X[...,3]
    b0, b1, b2, b3 = Y[...,0], Y[...,1], Y[...,2], Y[...,3]

    # 16 base-field muls
    ab00 = a0*b0; ab01 = a0*b1; ab02 = a0*b2; ab03 = a0*b3
    ab10 = a1*b0; ab11 = a1*b1; ab12 = a1*b2; ab13 = a1*b3
    ab20 = a2*b0; ab21 = a2*b1; ab22 = a2*b2; ab23 = a2*b3
    ab30 = a3*b0; ab31 = a3*b1; ab32 = a3*b2; ab33 = a3*b3

    z0 = ab00 + NR*weakmod(ab13 + ab22 + ab31)
    z1 = ab01 + ab10 + NR*weakmod(ab23 + ab32)
    z2 = ab02 + ab11 + ab20 + NR*weakmod(ab33)
    z3 = ab03 + ab12 + ab21 + ab30

    Z = np.empty(np.broadcast_shapes(x.shape, y.shape), dtype=np.uint32)
    Z[...,0] = (z0 % p)
    Z[...,1] = (z1 % p)
    Z[...,2] = (z2 % p)
    Z[...,3] = (z3 % p)
    return Z

def modinv_ext(x):
    """
    Inverse in F_p[X]/(X^4 - 3), packed as (...,4).
    Returns zeros where input is zero (no division-by-zero check).
    """
    x = np.asarray(x, dtype=np.uint32)
    assert x.dtype == np.uint32 and x.size % 4 == 0

    p  = np.uint64(2**31 - 2**24 + 1)
    NR = np.uint64(3)  # Y^2 = 3 where Y = X^2

    X = x.reshape(-1, 4).astype(np.uint64, copy=False)
    a0, a1, a2, a3 = X[:,0], X[:,1], X[:,2], X[:,3]

    # K helpers: K = F_p[Y]/(Y^2=3)
    def k_add(r1, i1, r2, i2):
        srr = r1 + r2
        sii = i1 + i2
        srr -= p * (srr >= p)
        sii -= p * (sii >= p)
        return srr, sii

    def k_sub(r1, i1, r2, i2):
        drr = r1 + p - r2
        dii = i1 + p - i2
        drr -= p * (drr >= p)
        dii -= p * (dii >= p)
        return drr, dii

    def k_mul(r1, i1, r2, i2):
        rr = (r1*r2 + NR*i1*i2) % p
        ii = (r1*i2 + i1*r2) % p
        return rr, ii

    def k_sqr(r, i):
        # (r + iY)^2 = (r^2 + 3 i^2) + (2 r i) Y
        rr = (r*r + NR*i*i) % p
        ii = ((r << 1) * i) % p
        return rr, ii

    def k_mulY(rr, ii):
        return (NR*ii) % p, rr

    def inv_fp(z):
        return modinv(z)

    def k_inv(r, i):
        # (r + iY)^{-1} = (r - iY) / (r^2 - 3 i^2)
        denom = weakmod(r*r + p - (NR * (i*i) % p))
        denom_inv = inv_fp(denom)
        cr = (r * denom_inv) % p
        ci = ((p - i) * denom_inv) % p
        return cr.astype(np.uint64), ci.astype(np.uint64)

    # Build A = a0 + a2 Y ; B = a1 + a3 Y  in K
    Ar, Ai = a0, a2
    Br, Bi = a1, a3

    # denom = A^2 - Y*B^2  in K
    A2_r, A2_i = k_sqr(Ar, Ai)
    B2_r, B2_i = k_sqr(Br, Bi)
    YB2_r, YB2_i = k_mulY(B2_r, B2_i)
    denom_r, denom_i = k_sub(A2_r, A2_i, YB2_r, YB2_i)

    invd_r, invd_i = k_inv(denom_r, denom_i)

    # (A - X B) * inv_d  in the outer quadratic
    # real = A * inv_d
    real_r, real_i = k_mul(Ar, Ai, invd_r, invd_i)
    # imag = (-B) * inv_d
    nBr = (p - Br) % p
    nBi = (p - Bi) % p
    imag_r, imag_i = k_mul(nBr, nBi, invd_r, invd_i)

    Z = np.empty_like(X, dtype=np.uint32)
    Z[:,0] = (real_r % p).astype(np.uint32)
    Z[:,2] = (real_i % p).astype(np.uint32)
    Z[:,1] = (imag_r % p).astype(np.uint32)
    Z[:,3] = (imag_i % p).astype(np.uint32)
    return Z.reshape(x.shape)

# ------------------------
# Convenience wrappers (match the original API surface)
# ------------------------
def zeros(shape):
    return np.zeros(shape, dtype=np.uint32)

def array(x):
    return np.array(x, dtype=np.uint32)

def arange(*args):
    return np.arange(*args, dtype=np.uint32)

def append(*args, axis=0):
    return np.concatenate(tuple(args), axis=axis).astype(np.uint32, copy=False)

def tobytes(x):
    return np.asarray(x, dtype=np.uint32).tobytes()

def eq(x, y):
    return np.array_equal(_modp(_u64(x)), _modp(_u64(y)))

def iszero(x):
    return not np.any(_modp(_u64(x)))

def zeros_like(obj):
    if isinstance(obj, np.ndarray):
        return np.zeros_like(obj, dtype=np.uint32)
    else:
        return obj.__class__.zeros(obj.shape)
