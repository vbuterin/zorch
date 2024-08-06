import cupy as cp

MAX_SIZE = 32768

# Multiply v1 * v2 in the binary tower field
# See https://blog.lambdaclass.com/snarks-on-binary-fields-binius/
# for introduction to how binary tower fields work
#
# The general rule is that if i = b0b1...bk in binary, then the
# index-i bit is the product of all x_i where b_i=1, eg. the
# index-5 bit (32) is x_2 * x_0
#
# Multiplication involves multiplying these multivariate polynomials
# as usual, but with the reduction rule that:
# (x_0)^2 = x_0 + 1
# (x_{i+1})^2 = x_{i+1} * x_i + 1

def binmul(v1, v2, length=None):
    if v1 < 256 and v2 < 256 and rawmulcache[v1][v2] is not None:
        return rawmulcache[v1][v2]
    if v1 < 2 or v2 < 2:
        return v1 * v2
    if length is None:
        length = 1 << (max(v1, v2).bit_length() - 1).bit_length()
    halflen = length//2
    quarterlen = length//4
    halfmask = (1 << halflen)-1

    L1, R1 = v1 & halfmask, v1 >> halflen
    L2, R2 = v2 & halfmask, v2 >> halflen

    # Optimized special case (used to compute R1R2_high), sec III of
    # https://ieeexplore.ieee.org/document/612935
    if (L1, R1) == (0, 1):
        outR = binmul(1 << quarterlen, R2, halflen) ^ L2
        return R2 ^ (outR << halflen)

    # x_{i+1}^2 reduces to 1 + x_{i+1} * x_i
    # Uses Karatsuba to only require three sub-multiplications for each input
    # halving (R1R2_high doesn't count because of the above optimization)
    L1L2 = binmul(L1, L2, halflen)
    R1R2 = binmul(R1, R2, halflen)
    R1R2_high = binmul(1 << quarterlen, R1R2, halflen)
    Z3 = binmul(L1 ^ R1, L2 ^ R2, halflen)
    return (
        L1L2 ^
        R1R2 ^
        ((Z3 ^ L1L2 ^ R1R2 ^ R1R2_high) << halflen)
    )

rawmulcache = [[None for _ in range(256)] for _ in range(256)]

for i in range(256):
    for j in range(256):
        rawmulcache[i][j] = binmul(i, j)

def build_mul_table_small():
    table_low = cp.zeros((65536, 256), dtype=cp.uint16)
    table_high = cp.zeros((65536, 256), dtype=cp.uint16)
    
    for i in [2**x for x in range(16)]:
        top_p_of_2 = 0
        for j in range(1, 256):
            if (j & (j-1)) == 0:
                table_low[i, j] = binmul(i, j)
                table_high[i, j] = binmul(i, j << 8)
                top_p_of_2 = j
            else:
                for table in (table_low, table_high):
                    table[i][j] = table[i][top_p_of_2] ^ table[i][j-top_p_of_2]
    
    for i in [2**x for x in range(1, 16)]:
       for table in (table_low, table_high):
           table[i:2*i] = table[i] ^ table[:i]
    
    return table_low, table_high

def multiply_small(x, y):
    return mul_table[0][x, y & 255] ^ mul_table[1][x, y >> 8]

mul_table = build_mul_table_small()
mul = multiply_small
print("Built multiplication table (low memory option)")

assert mul(12345, 23456) == 65306

# Build a table mapping x -> 1/x
def build_inv_table():
    output = cp.ones(65536, dtype=cp.uint16)
    exponents = cp.arange(0, 65536, 1, dtype=cp.uint16)
    for i in range(15):
        exponents = mul(exponents, exponents)
        output = mul(exponents, output)
    return output

inv = build_inv_table()
print("Built inversion table")

assert mul(7890, inv[7890]) == 1

# Convert a 128-bit integer into the field element representation we
# use here, which is a length-8 vector of uint16's
def int_to_bigbin(value):
    return cp.array(
        [(value >> (k*16)) & 65535 for k in range(8)],
        dtype=cp.uint16
    )

# Convert a uint16-representation big field element into an int
def bigbin_to_int(value):
    return sum(int(x) << (16*i) for i,x in enumerate(value))

# Multiplying an element in the i'th level subfield by X_i can be done in
# an optimized way. See sec III of https://ieeexplore.ieee.org/document/612935
def mul_by_Xi(x, N):
    assert x.shape[-1] == N
    if N == 1:
        return mul(x, 256)
    L, R = x[..., :N//2], x[..., N//2:]
    outR = mul_by_Xi(R, N//2) ^ L
    return cp.concatenate((R, outR), axis=-1)

# Multiplies together two field elements, using the Karatsuba algorithm
def big_mul(x1, x2):
    N = x1.shape[-1]
    if N == 1:
        return mul(x1, x2)
    L1, L2 = x1[..., :N//2], x2[..., :N//2]
    R1, R2 = x1[..., N//2:], x2[..., N//2:]
    L1L2 = big_mul(L1, L2)
    R1R2 = big_mul(R1, R2)
    R1R2_high = mul_by_Xi(R1R2, N//2)
    Z3 = big_mul(L1 ^ R1, L2 ^ R2)
    o = cp.concatenate((
        L1L2 ^ R1R2,
        (Z3 ^ L1L2 ^ R1R2 ^ R1R2_high)
    ), axis=-1)
    return o

def zeros(shape):
    return cp.zeros(shape, dtype=cp.uint16)

def array(x):
    return cp.array(x, dtype=cp.uint16)

def arange(*args):
    return cp.arange(*args, dtype=cp.uint16)

def append(*args, axis=0):
    return cp.concatenate((*args,), axis=axis)

def tobytes(x):
    return x.tobytes()

xor = cp.ReductionKernel(
    'uint16 x',  # input params
    'uint16 y',  # output params
    'x',  # map
    '(a ^ b)',  # reduce
    'y = a',  # post-reduction map
    '0',  # identity value
    'xor'  # kernel name
)

def zeros_like(obj):
    if obj.__class__ == cp.ndarray:
        return cp.zeros_like(obj)
    else:
        return obj.__class__.zeros(obj.shape)
