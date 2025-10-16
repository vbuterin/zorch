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

# ------------------------
# Degree-4 tower ops (F_p[u][v] / (u^2=_NR, v^2=_ALPHA0 + _ALPHA1*u))
# Pack x as (..., 4): x0 + x1*u  +  v*(x2 + x3*u)
# ------------------------
def mul_ext(x, y):
    """
    Multiply in F_p[u][v] with u^2=3 and v^2=1+u.
    Inputs x,y are uint32 arrays with length % 4 == 0, laid out as:
      x = [a0, a1, b0, b1,  a0, a1, b0, b1, ...] meaning (A + v*B)
      where A = a0 + a1*u, B = b0 + b1*u.
    Returns same-shape uint32 array.
    """
    x = np.asarray(x, dtype=np.uint32)
    y = np.asarray(y, dtype=np.uint32)

    p  = np.uint64(2**31 - 2**24 + 1)  # KoalaBear modulus
    NR = np.uint64(3)                  # u^2 = 3

    X = x.astype(np.uint64, copy=False)
    Y = y.astype(np.uint64, copy=False)

    a0, a1, b0, b1 = X[...,0], X[...,1], X[...,2], X[...,3]
    c0, c1, d0, d1 = Y[...,0], Y[...,1], Y[...,2], Y[...,3]

    # LL = A*C in F_p[u]
    ll_r = weakmod(a0*c0 + NR*a1*c1)
    ll_i = weakmod(a0*c1 + a1*c0)

    # RR = B*D in F_p[u]
    rr_r = weakmod(b0*d0 + NR*b1*d1)
    rr_i = weakmod(b0*d1 + b1*d0)

    # real = LL + alpha*RR, with alpha = 1 + u
    # alpha*RR = (rr_r + NR*rr_i) + (rr_i + rr_r)*u
    real_r = (ll_r + rr_r + NR*rr_i) % p
    real_i = (ll_i + rr_i + rr_r) % p

    # comb = (A+B)*(C+D) in F_p[u]
    e0 = (a0 + b0)
    e0 -= p * (e0 >= p)
    e1 = (a1 + b1)
    e1 -= p * (e1 >= p)
    f0 = (c0 + d0)
    f0 -= p * (f0 >= p)
    f1 = (c1 + d1)
    f1 -= p * (f1 >= p)
    comb_r = (e0*f0 + NR*e1*f1)
    comb_i = (e0*f1 + e1*f0)

    # imag = comb - LL - RR
    imag_r = (comb_r + (p << 26) - (ll_r + rr_r)) % p
    imag_i = (comb_i + (p << 26) - (ll_i + rr_i)) % p

    Z = np.empty(np.broadcast_shapes(x.shape, y.shape), dtype=np.uint32)
    Z[...,0] = real_r
    Z[...,1] = real_i
    Z[...,2] = imag_r
    Z[...,3] = imag_i
    return Z

def modinv_ext(x):
    x = np.asarray(x, dtype=np.uint32)
    assert x.dtype == np.uint32 and x.size % 4 == 0

    shp = x.shape
    X = x.reshape(-1, 4)

    A = np.stack((X[:,0], X[:,1]), axis=1)  # a0 + a1 u
    B = np.stack((X[:,2], X[:,3]), axis=1)  # b0 + b1 u

    A2   = _cplx_square(A)
    B2   = _cplx_square(B)
    aB2  = _cplx_mul(B2, _ALPHA)           # alpha * B^2
    denom = _cplx_sub(A2, aB2)             # A^2 - alpha*B^2   (in F_p[u])
    inv_d = _cplx_inv(denom)               # (in F_p[u])

    real = _cplx_mul(A, inv_d)             # (A) * inv_d
    imag = _cplx_mul(_cplx_neg(B), inv_d)  # (-B) * inv_d

    Z = np.empty_like(X)
    Z[:,0], Z[:,1] = real[:,0], real[:,1]
    Z[:,2], Z[:,3] = imag[:,0], imag[:,1]
    return Z.reshape(shp)

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
