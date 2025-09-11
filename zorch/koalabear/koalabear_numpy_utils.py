import numpy as np

# ------------------------
# Field parameters (KoalaBear)
# ------------------------
modulus = 2**31 - 2**24 + 1          # p = 2,130,706,433
_NR = np.uint64(3)                               # u^2 = 3   (in F_p)
_ALPHA0 = np.uint64(1)                           # v^2 = ALPHA0 + ALPHA1 * u
_ALPHA1 = np.uint64(1)

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
    return _modp(_u64(x) + _u64(y))

def sub(x, y):
    return _modp(_u64(x) + modulus - _u64(y))

def mul(x, y):
    return _modp(_u64(x) * _u64(y))

def pow3(x):
    xx = mul(x, x)
    return mul(xx, x)

def sum(x, axis=0):
    return np.uint32(np.sum(_u64(x) % modulus, axis=axis) % modulus)

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

# ------------------------
# Degree-4 tower ops (F_p[u][v] / (u^2=_NR, v^2=_ALPHA0 + _ALPHA1*u))
# Pack x as (..., 4): x0 + x1*u  +  v*(x2 + x3*u)
# ------------------------
def mul_ext(x, y):
    x = np.asarray(x, dtype=np.uint32)
    y = np.asarray(y, dtype=np.uint32)
    assert x.dtype == np.uint32 and y.dtype == np.uint32
    assert x.size % 4 == 0

    xy = (x.reshape(-1, 4), y.reshape(-1, 4))
    X, Y = (xy[0], xy[1])

    A = np.stack((X[:,0], X[:,1]), axis=1)  # a0 + a1 u
    B = np.stack((X[:,2], X[:,3]), axis=1)  # b0 + b1 u
    C = np.stack((Y[:,0], Y[:,1]), axis=1)
    D = np.stack((Y[:,2], Y[:,3]), axis=1)

    LL   = _cplx_mul(A, C)                         # A*C
    RR   = _cplx_mul(B, D)                         # B*D
    RR_a = _cplx_mul(RR, _ALPHA)                  # RR * alpha
    real = _cplx_add(LL, RR_a)                     # real  = LL + RR*alpha
    comb = _cplx_mul(_cplx_add(A, B), _cplx_add(C, D))
    imag = _cplx_sub(comb, _cplx_add(LL, RR))      # imag  = (A+B)(C+D) - LL - RR

    new_shape = np.broadcast_shapes(x.shape, y.shape)
    Z = np.zeros(new_shape, dtype=np.uint32)
    Z[...,0], Z[...,1] = real[...,0], real[...,1]
    Z[...,2], Z[...,3] = imag[...,0], imag[...,1]
    return Z.reshape(new_shape)

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
