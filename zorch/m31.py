import cupy as cp

M31 = modulus = 2**31-1

add = cp.ElementwiseKernel(
   'uint32 x, uint32 y',        # input argument list
   'uint32 z',                 # output argument list
   'z = (x + y); z = (z & 2147483647) + (z >> 31)',   # loop body code
   'add')            # kernel name

sub = cp.ElementwiseKernel(
   'uint32 x, uint32 y',        # input argument list
   'uint32 z',                 # output argument list
   'const unsigned int M31 = 2147483647; z = (x + M31 - y); z = (z & M31) + (z >> 31)',   # loop body code
   'sub')            # kernel name

pow5 = cp.ElementwiseKernel(
   'uint32 x',        # input argument list
   'uint32 o',                 # output argument list
   '''
    const unsigned int M31 = 2147483647;

    unsigned int z1 = (x * x);
    unsigned int z2 = __umulhi(x, x);
    unsigned int z = (z1 & M31) + (z1 >> 31) + z2 * 2;
    unsigned int x2 = (z & M31) + (z >> 31);

    z1 = (x * x2);
    z2 = __umulhi(x, x2);
    z = (z1 & M31) + (z1 >> 31) + z2 * 2;
    unsigned int x3 = (z & M31) + (z >> 31);

    z1 = (x2 * x3);
    z2 = __umulhi(x2, x3);
    z = (z1 & M31) + (z1 >> 31) + z2 * 2;
    o = (z & M31) + (z >> 31);
   ''',
   'pow5')            # kernel name

sum = cp.ReductionKernel(
    'uint32 x',  # input params
    'uint32 y',  # output params
    'x',  # map
    '(a + b) % 2147483647',  # reduce
    'y = a',  # post-reduction map
    '0',  # identity value
    'sum'  # kernel name
)

mul = cp.ElementwiseKernel(
   'uint32 x, uint32 y',        # input argument list
   'uint32 z',                 # output argument list
   '''
   unsigned int z1 = (x * y);
   unsigned int z2 = __umulhi(x, y);
   z = (z1 & 2147483647) + (z1 >> 31) + z2 * 2;
   z = (z & 2147483647) + (z >> 31)''',   # loop body code
   'mul')            # kernel name

kernel_code = r'''
const unsigned int M31 = 2147483647;

__device__ unsigned int submod(unsigned int x, unsigned int y) {
    unsigned int z = (x + M31 - y);
    return (z & M31) + (z >> 31);
}

__device__ unsigned int mulmod(unsigned int x, unsigned int y) {
    unsigned int z1 = (x * y);
    unsigned int z2 = __umulhi(x, y);
    unsigned int z = (z1 & M31) + (z1 >> 31) + z2 * 2;
    return (z & M31) + (z >> 31);
}

__device__ unsigned int mod31(unsigned int x) {
    return (x & M31) + (x >> 31);
}

__device__ void multiply_complex(unsigned int* o_r,
                                 unsigned int* o_i,
                                 unsigned int A0,
                                 unsigned int A1,
                                 unsigned int B0,
                                 unsigned int B1) {
    unsigned int low = mulmod(A0, B0);
    unsigned int high = mulmod(A1, B1);
    unsigned int med = mulmod(mod31(A0 + A1), mod31(B0 + B1));
    *o_r = submod(low, high);
    *o_i = submod(med, mod31(low + high));
}

extern "C" __global__
void vectorized_mulmod(const unsigned int* x,
                       const unsigned int* y,
                       unsigned int* z,
                       int num_blocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_blocks) {
        int base_idx = idx * 4;

        unsigned int o_LL_r, o_LL_i;
        multiply_complex(
            &o_LL_r, &o_LL_i,
            x[base_idx], x[base_idx + 1],
            y[base_idx], y[base_idx + 1]
        );

        unsigned int A_fold_0 = mod31(x[base_idx] + x[base_idx + 2]);
        unsigned int A_fold_1 = mod31(x[base_idx + 1] + x[base_idx + 3]);
        unsigned int B_fold_0 = mod31(y[base_idx] + y[base_idx + 2]);
        unsigned int B_fold_1 = mod31(y[base_idx + 1] + y[base_idx + 3]);
        unsigned int o_comb_r, o_comb_i;
        multiply_complex(
            &o_comb_r, &o_comb_i,
            A_fold_0, A_fold_1,
            B_fold_0, B_fold_1
        );

        unsigned int o_RR_r, o_RR_i;
        multiply_complex(
            &o_RR_r, &o_RR_i,
            x[base_idx + 2], x[base_idx + 3],
            y[base_idx + 2], y[base_idx + 3]
        );

        z[base_idx] = mod31(submod(o_LL_r, o_RR_r) + mod31(o_RR_i * 2));
        z[base_idx + 1] = submod(submod(o_LL_i, o_RR_i), mod31(o_RR_r * 2));
        z[base_idx + 2] = submod(o_comb_r, mod31(o_LL_r + o_RR_r));
        z[base_idx + 3] = submod(o_comb_i, mod31(o_LL_i + o_RR_i));
    }
}
'''

# Load the kernel
mul_ext_kernel = cp.RawKernel(kernel_code, 'vectorized_mulmod')

# Wrapper function
def mul_ext(x, y):
    x, y = cp.broadcast_arrays(x, y)
    assert x.shape == y.shape
    assert x.dtype == y.dtype == cp.uint32
    
    x_flat = x.ravel()
    y_flat = y.ravel()
    z = cp.zeros_like(x_flat)
    
    num_blocks = x_flat.size // 4
    threads_per_block = 256
    blocks_per_grid = (num_blocks + threads_per_block - 1) // threads_per_block
    
    mul_ext_kernel((blocks_per_grid,), (threads_per_block,), 
                  (x_flat, y_flat, z, num_blocks))
    
    return z.reshape(x.shape)

kernel_code = r'''
const unsigned int M31 = 2147483647;

__device__ unsigned int submod(unsigned int x, unsigned int y) {
    unsigned int z = (x + M31 - y);
    return (z & M31) + (z >> 31);
}

__device__ unsigned int mulmod(unsigned int x, unsigned int y) {
    unsigned int z1 = (x * y);
    unsigned int z2 = __umulhi(x, y);
    unsigned int z = (z1 & M31) + (z1 >> 31) + z2 * 2;
    return (z & M31) + (z >> 31);
}

__device__ unsigned int mod31(unsigned int x) {
    return (x & M31) + (x >> 31);
}

__device__ unsigned int modinv(unsigned int x) {
    unsigned int o = x;
    unsigned int pow_of_x = mulmod(x, x);
    for (int i = 0; i < 29; i++) {
        pow_of_x = mulmod(pow_of_x, pow_of_x);
        o = mulmod(o, pow_of_x);
    }
    return o;
};

extern "C" __global__
void vectorized_modinv(const unsigned int* x,
                       unsigned int* z,
                       int num_blocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_blocks) {
        int base_idx = idx * 4;

        unsigned int x0_sq = mulmod(x[base_idx], x[base_idx]);
        unsigned int x1_sq = mulmod(x[base_idx + 1], x[base_idx + 1]);
        unsigned int x0x1 = mulmod(x[base_idx], x[base_idx + 1]);
        unsigned int x2_sq = mulmod(x[base_idx + 2], x[base_idx + 2]);
        unsigned int x3_sq = mulmod(x[base_idx + 3], x[base_idx + 3]);
        unsigned int x2x3 = mulmod(x[base_idx + 2], x[base_idx + 3]);
        unsigned int r20 = submod(x2_sq, x3_sq);
        unsigned int r21 = mod31(x2x3 << 1);
        unsigned int denom0 = mod31(
            submod(x0_sq, x1_sq)
            + submod(r20, mod31(r21 << 1))
        );
        unsigned int denom1 = mod31(
            mod31(mod31(x0x1 << 1) + r21)
            + mod31(r20 << 1)
        );
        unsigned int inv_denom_norm = modinv(mod31(
            mulmod(denom0, denom0) + mulmod(denom1, denom1)
        ));
        unsigned int inv_denom0 = mulmod(denom0, inv_denom_norm);
        unsigned int inv_denom1 = mulmod(M31 - denom1, inv_denom_norm);

        z[base_idx] = submod(
            mulmod(x[base_idx], inv_denom0),
            mulmod(x[base_idx + 1], inv_denom1)
        );
        z[base_idx + 1] = mod31(
            mulmod(x[base_idx], inv_denom1)
            + mulmod(x[base_idx + 1], inv_denom0)
        );
        z[base_idx + 2] = submod(
            mulmod(x[base_idx + 3], inv_denom1),
            mulmod(x[base_idx + 2], inv_denom0)
        );
        z[base_idx + 3] = M31 - mod31(
            mulmod(x[base_idx + 2], inv_denom1)
            + mulmod(x[base_idx + 3], inv_denom0)
        );
    }
}

extern "C" __global__
void vectorized_basic_modinv(const unsigned int* x,
                       unsigned int* z,
                       int num_blocks) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_blocks) {
        z[idx] = modinv(x[idx]);
    }
}
'''

# Load the kernel
modinv_kernel = cp.RawKernel(kernel_code, 'vectorized_basic_modinv')
modinv_ext_kernel = cp.RawKernel(kernel_code, 'vectorized_modinv')

# Wrapper function
def modinv(x):
    assert x.dtype == cp.uint32
    
    x_flat = x.ravel()
    z = cp.zeros_like(x_flat)
    
    num_blocks = x_flat.size
    threads_per_block = 256
    blocks_per_grid = (num_blocks + threads_per_block - 1) // threads_per_block
    
    modinv_kernel((blocks_per_grid,), (threads_per_block,), 
                  (x_flat, z, num_blocks))
    
    return z.reshape(x.shape)

# Wrapper function
def modinv_ext(x):
    assert x.dtype == cp.uint32
    
    x_flat = x.ravel()
    z = cp.zeros_like(x_flat)
    
    num_blocks = x_flat.size // 4
    threads_per_block = 256
    blocks_per_grid = (num_blocks + threads_per_block - 1) // threads_per_block
    
    modinv_ext_kernel((blocks_per_grid,), (threads_per_block,), 
                      (x_flat, z, num_blocks))
    
    return z.reshape(x.shape)

def zeros(shape):
    return cp.zeros(shape, dtype=cp.uint32)

def array(x):
    return cp.array(x, dtype=cp.uint32)

def arange(*args):
    return cp.arange(*args, dtype=cp.uint32)

def append(*args):
    return cp.concatenate((*args,))

def tobytes(x):
    return x.tobytes()

def eq(x, y):
    return cp.array_equal(x % M31, y % M31)

def iszero(x):
    return not cp.any(x % M31)

def test():
    M31 = 2**31-1
    x_orig = 3 ** cp.arange(10**7, dtype=cp.uint32) % M31
    x = cp.copy(x_orig)
    for i in range(31):
        x = mul(x, x)
    assert cp.array_equal(x, mul(x_orig, x_orig))
    print("Multiplication tests passed")
    x0 = mul(sub(x[:10], x[10:20]), x[20:30])
    x1 = sub(mul(x[:10], x[20:30]), mul(x[10:20], x[20:30]))
    x2 = mul(add(x[:10], x[30:40]), x[20:30])
    x3 = add(mul(x[:10], x[20:30]), mul(x[30:40], x[20:30]))
    assert cp.array_equal(x0, x1)
    assert cp.array_equal(x2, x3)
    print("Add, mul and sub tests passed")
    x4_orig = 3 ** cp.arange(4 * 10**7, dtype=cp.uint32).reshape((10**7, 4)) % M31
    x4 = cp.copy(x4_orig)
    x5 = cp.copy(x4_orig)
    for i in range(4):
        x4 = mul_ext(x4, x4)
    for i in range(15):
        x5 = mul_ext(x5, x4_orig)
    assert cp.array_equal(x4, x5)
    print("Mul_ext tests passed")
    x6 = 3 ** cp.arange(10**6, dtype=cp.uint32) % M31
    x7 = modinv(x6)
    x8 = mul(x6, x7) % M31
    assert cp.array_equal(x8 - 1, cp.zeros_like(x8))
    print("Basic modinv tests passed")
    x9 = modinv_ext(x6.reshape((250000, 4)))
    x10 = mul_ext(x6.reshape((250000, 4)), x9) % M31
    assert cp.array_equal(x10[...,0] - 1, cp.zeros_like(x10[...,0]))
    assert cp.array_equal(x10[...,1:], cp.zeros_like(x10[...,1:]))
    print("Extended modinv tests passed")
