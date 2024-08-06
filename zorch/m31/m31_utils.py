import cupy as cp

modulus = 2**31-1

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
    'uint32 o',        # output argument list
    preamble='''
    const unsigned int M31 = 2147483647;

    unsigned int mulmod(unsigned int a, unsigned int b) {
        unsigned int z = (a * b);
        z = (z & M31) + (z >> 31) + __umulhi(a, b) * 2;
        return (z & M31) + (z >> 31);
    };
    ''',
    operation='''

    unsigned int xpow = mulmod(x, x);
    xpow = mulmod(xpow, xpow);
    o = mulmod(xpow, x);
    ''',
    name='pow5',            # kernel name
)

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
   z = (z1 & 2147483647) + (z1 >> 31) + __umulhi(x, y) * 2;
   z = (z & 2147483647) + (z >> 31)''',   # loop body code
   'mul')            # kernel name

_mul_ext = cp.ElementwiseKernel(
    'complex128 x, complex128 y',
    'complex128 z',
    preamble='''

const unsigned int M31 = 2147483647;

__device__ unsigned int submod(unsigned int x, unsigned int y) {
    unsigned int z = (x + M31 - y);
    return (z & M31) + (z >> 31);
}

__device__ unsigned int mulmod(unsigned int x, unsigned int y) {
    unsigned int z1 = (x * y);
    unsigned int z = (z1 & M31) + (z1 >> 31) + __umulhi(x, y) * 2;
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

    ''',
    operation='''

    thrust::complex<double> _x = x;
    thrust::complex<double> _y = y;

    unsigned int x0 = reinterpret_cast<unsigned int*>(&_x)[0];
    unsigned int x1 = reinterpret_cast<unsigned int*>(&_x)[1];
    unsigned int x2 = reinterpret_cast<unsigned int*>(&_x)[2];
    unsigned int x3 = reinterpret_cast<unsigned int*>(&_x)[3];

    unsigned int y0 = reinterpret_cast<unsigned int*>(&_y)[0];
    unsigned int y1 = reinterpret_cast<unsigned int*>(&_y)[1];
    unsigned int y2 = reinterpret_cast<unsigned int*>(&_y)[2];
    unsigned int y3 = reinterpret_cast<unsigned int*>(&_y)[3];

    unsigned int o_LL_r, o_LL_i;
    multiply_complex(
            &o_LL_r, &o_LL_i,
            x0, x1,
            y0, y1
    );

    unsigned int o_comb_r, o_comb_i;
    multiply_complex(
        &o_comb_r, &o_comb_i,
        mod31(x0 + x2), mod31(x1 + x3),
        mod31(y0 + y2), mod31(y1 + y3)
    );

    unsigned int o_RR_r, o_RR_i;
    multiply_complex(
        &o_RR_r, &o_RR_i,
        x2, x3,
        y2, y3
    );

    reinterpret_cast<unsigned int*>(&z)[0] = mod31(submod(o_LL_r, o_RR_r) + mod31(o_RR_i * 2));
    reinterpret_cast<unsigned int*>(&z)[1] = submod(submod(o_LL_i, o_RR_i), mod31(o_RR_r * 2));
    reinterpret_cast<unsigned int*>(&z)[2] = submod(o_comb_r, mod31(o_LL_r + o_RR_r));
    reinterpret_cast<unsigned int*>(&z)[3] = submod(o_comb_i, mod31(o_LL_i + o_RR_i));

    '''
)

def mul_ext(x, y):
    xc128 = x.view(cp.complex128)
    yc128 = y.view(cp.complex128)
    zc128 = _mul_ext(xc128, yc128)
    return zc128.view(cp.uint32)

kernel_code = r'''
const unsigned int M31 = 2147483647;

__device__ unsigned int mod31(unsigned int x) {
    return (x & M31) + (x >> 31);
}

__device__ unsigned int submod(unsigned int x, unsigned int y) {
    return mod31(x + M31 - y);
}

__device__ unsigned int mulmod(unsigned int x, unsigned int y) {
    unsigned int z1 = (x * y);
    unsigned int z2 = __umulhi(x, y);
    return mod31((z1 & M31) + (z1 >> 31) + z2 * 2);
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

def append(*args, axis=0):
    return cp.concatenate((*args,), axis=axis)

def tobytes(x):
    return x.tobytes()

def eq(x, y):
    return cp.array_equal(x % modulus, y % modulus)

def iszero(x):
    return not cp.any(x % modulus)

def zeros_like(obj):
    if obj.__class__ == cp.ndarray:
        return cp.zeros_like(obj)
    else:
        return obj.__class__.zeros(obj.shape)
