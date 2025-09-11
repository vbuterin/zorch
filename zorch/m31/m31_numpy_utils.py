import numpy as np

modulus = 2**31 - 1
_M31_u32 = np.uint32(modulus)
_M31_u64 = np.uint64(modulus)


# ---------- low-level helpers (vectorized) ----------

def _as_u32(x):
    return np.asarray(x, dtype=np.uint32)

def _mod31(x):
    """
    Fast Mersenne reduction for arrays/ints using uint64 intermediates,
    then fold twice and final % to be safe.
    """
    x64 = x.astype(np.uint64, copy=False)
    t = (x64 & _M31_u64) + (x64 >> 31)
    t = (t & _M31_u64) + (t >> 31)
    t = t % _M31_u64
    return t.astype(np.uint32)

def _addmod(a, b):
    return _mod31(a.astype(np.uint64) + b.astype(np.uint64))

def _submod(a, b):
    # (a - b) mod p  ==  a + p - b  (all uint64 to avoid wrap)
    return _mod31(a.astype(np.uint64) + _M31_u64 - b.astype(np.uint64))

def _mulmod(a, b):
    # exact via 64-bit product, then % p
    prod = a.astype(np.uint64) * b.astype(np.uint64)
    return (prod % _M31_u64).astype(np.uint32)

def _shl1_mod(a):
    # (a << 1) mod p
    return ((a.astype(np.uint64) << 1) % _M31_u64).astype(np.uint32)


# ---------- public elementwise ops ----------

def add(x, y):
    """
    Elementwise (x + y) mod (2^31-1)
    """
    x = _as_u32(x); y = _as_u32(y)
    return _addmod(x, y)

def sub(x, y):
    """
    Elementwise (x - y) mod (2^31-1)
    """
    x = _as_u32(x); y = _as_u32(y)
    return _submod(x, y)

def pow5(x):
    """
    Elementwise x^5 mod (2^31-1)
    """
    x = _as_u32(x)
    x2 = _mulmod(x, x)        # x^2
    x4 = _mulmod(x2, x2)      # x^4
    return _mulmod(x4, x)     # x^5

def sum(x, axis=0):
    """
    Reduction sum(x) mod (2^31-1) -> uint32 scalar
    """
    x = _as_u32(x)
    s = np.sum(x.astype(np.uint64) % _M31_u64, dtype=np.uint64, axis=axis) % _M31_u64
    return np.uint32(s)

def mul(x, y):
    """
    Elementwise (x * y) mod (2^31-1)
    """
    x = _as_u32(x); y = _as_u32(y)
    return _mulmod(x, y)


# ---------- extension-field style ops ----------

def _multiply_complex(A0, A1, B0, B1):
    """
    Multiply (A0 + i*A1) * (B0 + i*B1) in F_p[i]/(i^2 = -1),
    all lanes are uint32 arrays under mod p.
    Returns (real, imag).
    """
    low  = _mulmod(A0, B0)
    high = _mulmod(A1, B1)
    med  = _mulmod(_addmod(A0, A1), _addmod(B0, B1))
    real = _submod(low, high)
    imag = _submod(med, _addmod(low, high))
    return real, imag

def mul_ext(x, y):
    """
    Vectorized 'extension' multiply, operating on groups of 4 uint32s.
    The layout matches the original cupy kernel:
      x = [x0, x1, x2, x3,  x0, x1, x2, x3, ...]
      y = [y0, y1, y2, y3,  y0, y1, y2, y3, ...]
    Returns same-shape uint32 array.
    """
    x = _as_u32(x); y = _as_u32(y)
    assert x.size % 4 == 0 and y.size % 4 == 0

    X = x.reshape(-1, 4)
    Y = y.reshape(-1, 4)

    x0, x1, x2, x3 = (X[:, 0], X[:, 1], X[:, 2], X[:, 3])
    y0, y1, y2, y3 = (Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3])

    # LL part
    LL_r, LL_i = _multiply_complex(x0, x1, y0, y1)

    # combined (Karatsuba middle)
    cA0 = _addmod(x0, x2)
    cA1 = _addmod(x1, x3)
    cB0 = _addmod(y0, y2)
    cB1 = _addmod(y1, y3)
    C_r, C_i = _multiply_complex(cA0, cA1, cB0, cB1)

    # RR part
    RR_r, RR_i = _multiply_complex(x2, x3, y2, y3)

    z0 = _addmod(_submod(LL_r, RR_r), _mod31(RR_i.astype(np.uint64) * 2))
    z1 = _submod(_submod(LL_i, RR_i), _mod31(RR_r.astype(np.uint64) * 2))
    z2 = _submod(C_r, _addmod(LL_r, RR_r))
    z3 = _submod(C_i, _addmod(LL_i, RR_i))

    new_shape = np.broadcast_shapes(x.shape, y.shape)
    Z = np.stack([z0, z1, z2, z3], axis=1).reshape(new_shape)
    return Z.astype(np.uint32)


# ---------- modular inverse ----------

def _modinv_vectorized_base(x):
    """
    Implements the same exponentiation schedule as the CUDA kernel:

    o = x
    pow = x^2
    for i in range(29):
        pow = pow^2
        o *= pow

    That evaluates x^(2^31 - 3) = x^(p-2) for p = 2^31 - 1.
    Returns uint32 array (0 maps to 0).
    """
    x = _as_u32(x)
    o = x.copy()
    pow_x = _mulmod(x, x)  # x^2
    for _ in range(29):
        pow_x = _mulmod(pow_x, pow_x)   # square
        o = _mulmod(o, pow_x)           # multiply
    return o

def modinv(x):
    """
    Elementwise modular inverse in F_p (p = 2^31-1).
    Matches the CUDA kernel behavior (0 -> 0).
    """
    x = _as_u32(x)
    return _modinv_vectorized_base(x)


# ---------- vectorized "extension" inverse on groups of 4 ----------

def modinv_ext(x):
    """
    Vectorized inverse on 4-limb groups, matching the CUDA kernel layout.
    x is uint32 array with size % 4 == 0.
    """
    x = _as_u32(x)
    assert x.size % 4 == 0, "Input length must be a multiple of 4"
    X = x.reshape(-1, 4)

    x0 = X[:, 0]; x1 = X[:, 1]; x2 = X[:, 2]; x3 = X[:, 3]

    x0_sq = _mulmod(x0, x0)
    x1_sq = _mulmod(x1, x1)
    x0x1  = _mulmod(x0, x1)

    x2_sq = _mulmod(x2, x2)
    x3_sq = _mulmod(x3, x3)
    x2x3  = _mulmod(x2, x3)

    r20 = _submod(x2_sq, x3_sq)
    r21 = _mod31((x2x3.astype(np.uint64) << 1))  # 2*x2x3 mod p

    # denom0 = (x0^2 - x1^2) + (r20 - 2*r21)  (all mod p)
    t1 = _submod(x0_sq, x1_sq)
    t2 = _submod(r20, _shl1_mod(r21))  # r20 - 2*r21
    denom0 = _addmod(t1, t2)

    # denom1 = 2*x0x1 + r21 + 2*r20  (all mod p)
    denom1 = _addmod(_addmod(_shl1_mod(x0x1), r21), _shl1_mod(r20))

    inv_denom_norm = modinv(_addmod(_mulmod(denom0, denom0),
                                    _mulmod(denom1, denom1)))
    inv_denom0 = _mulmod(denom0, inv_denom_norm)
    inv_denom1 = _mulmod(_submod(np.uint32(0), denom1), inv_denom_norm)  # -denom1 * inv_norm

    z0 = _submod(_mulmod(x0, inv_denom0), _mulmod(x1, inv_denom1))
    z1 = _addmod(_mulmod(x0, inv_denom1), _mulmod(x1, inv_denom0))
    z2 = _submod(_mulmod(x3, inv_denom1), _mulmod(x2, inv_denom0))
    z3 = _submod(np.uint32(0), _addmod(_mulmod(x2, inv_denom1), _mulmod(x3, inv_denom0)))

    Z = np.stack([z0, z1, z2, z3], axis=1).reshape(x.shape)
    return Z.astype(np.uint32)


# ---------- small utility wrappers (NumPy versions) ----------

def zeros(shape):
    return np.zeros(shape, dtype=np.uint32)

def array(x):
    return np.array(x, dtype=np.uint32)

def arange(*args):
    return np.arange(*args, dtype=np.uint32)

def append(*args, axis=0):
    return np.concatenate(args, axis=axis)

def tobytes(x):
    return np.asarray(x).tobytes()

def eq(x, y):
    x = _as_u32(x); y = _as_u32(y)
    return np.array_equal(x % _M31_u32, y % _M31_u32)

def iszero(x):
    x = _as_u32(x)
    return not np.any(x % _M31_u32)

def zeros_like(obj):
    if obj.__class__ == np.ndarray:
        return np.zeros_like(obj)
    else:
        return obj.__class__.zeros(obj.shape)
