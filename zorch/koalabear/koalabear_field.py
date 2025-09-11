from .koalabear_numpy_utils import (
    zeros, arange, array, append, add, sub, mul, np, pow3, modinv, mul_ext,
    modinv_ext, modulus, sum as m31_sum
)

def mod31_py_obj(inp):
    if isinstance(inp, int):
        return inp % modulus
    else:
        return [mod31_py_obj(x) for x in inp]

class KoalaBear():
    def __init__(self, x):
        if isinstance(x, (int, list)):
            x = array(mod31_py_obj(x))
        elif isinstance(x, KoalaBear):
            x = x.value
        self.value = x
        assert self.value.dtype == np.uint32

    @classmethod
    def zeros(cls, shape):
        return cls(zeros(shape))

    @classmethod
    def arange(cls, *args):
        return cls(arange(*args))

    @classmethod
    def append(cls, *args, axis=0):
        return cls(append(*(x.value for x in args), axis=axis))

    @classmethod
    def sum(cls, arg, axis):
        return cls(m31_sum(arg.value, axis=axis))

    @property
    def shape(self):
        return self.value.shape

    def reshape(self, shape):
        return KoalaBear(self.value.reshape(shape))

    def swapaxes(self, ax1, ax2):
        return KoalaBear(self.value.swapaxes(ax1, ax2))

    def copy(self):
        return KoalaBear(np.copy(self.value))

    @property
    def ndim(self):
        return self.value.ndim

    def to_extended(self):
        o = zeros(self.value.shape + (4,))
        o[...,0] = self.value
        return ExtendedKoalaBear(o)

    def __getitem__(self, index):
        return KoalaBear(self.value[index])

    def __setitem__(self, index, value):
        if isinstance(value, int):
            self.value[index] = value
        elif isinstance(value, KoalaBear):
            self.value[index] = value.value
        else:
            raise Exception(f"Bad input for setitem: {value}")

    def __add__(self, other):
        if isinstance(other, ExtendedKoalaBear):
            return self.to_extended() + other
        elif isinstance(other, int):
            other = KoalaBear(other)
        return KoalaBear(add(self.value, other.value))

    def __neg__(self):
        return KoalaBear(modulus - self.value)

    def __sub__(self, other):
        if isinstance(other, ExtendedKoalaBear):
            return self.to_extended() - other
        elif isinstance(other, int):
            other = KoalaBear(other)
        return KoalaBear(sub(self.value, other.value))

    def __mul__(self, other):
        if isinstance(other, ExtendedKoalaBear):
            return ExtendedKoalaBear(mul(
                self.value.reshape(self.value.shape + (1,)),
                other.value
            ))
        elif isinstance(other, int):
            other = KoalaBear(other)
        return KoalaBear(mul(self.value, other.value))

    def __pow__(self, other):
        assert isinstance(other, int)
        if other == 3:
            # Optimize common special case
            return KoalaBear(pow3(self.value))
        elif other == 0:
            return KoalaBear(np.ones(self.value.shape))
        elif other == 1:
            return self
        elif other % 2 == 1:
            sub = self ** (other // 2)
            return sub * sub * self
        else:
            sub = self ** (other // 2)
            return sub * sub

    def inv(self):
        return KoalaBear(modinv(self.value))

    def __truediv__(self, other):
        if isinstance(other, int):
            other = KoalaBear(other)
        return self * other.inv()

    def __rtruediv__(self, other):
        return self.inv() * other

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = lambda self, other: -(self - other)

    def __repr__(self):
        return f'{self.value}'

    def __int__(self):
        return int(self.value)

    def __len__(self):
        return len(self.value)

    def tobytes(self):
        return (self.value % modulus).tobytes()

    def __eq__(self, other):
        if isinstance(other, int):
            other = KoalaBear(other)
        elif isinstance(other, ExtendedKoalaBear):
            return self.to_extended() == other
        shape = np.broadcast_shapes(self.value.shape, other.value.shape)
        return np.array_equal(
            np.broadcast_to(self.value, shape) % modulus,
            np.broadcast_to(other.value, shape) % modulus
        )

class ExtendedKoalaBear():
    def __init__(self, x):
        if isinstance(x, int):
            x = array([x % modulus, 0, 0, 0])
        elif isinstance(x, list):
            x = array(mod31_py_obj(x))
        elif isinstance(x, KoalaBear):
            x = x.to_extended().value
        elif isinstance(x, ExtendedKoalaBear):
            x = x.value
        assert x.shape[-1] == 4
        self.value = x
        assert self.value.dtype == np.uint32

    @classmethod
    def zeros(cls, shape):
        return cls(zeros(shape + (4,)))

    @classmethod
    def append(cls, *args, axis=0):
        return cls(append(*(x.value for x in args), axis=axis))

    @classmethod
    def sum(cls, arg, axis):
        adjusted_axis = axis if axis >= 0 else axis-1
        return cls(m31_sum(arg.value, axis=adjusted_axis))

    @property
    def shape(self):
        return self.value.shape[:-1]

    def reshape(self, shape):
        return ExtendedKoalaBear(self.value.reshape(shape + (4,)))

    def swapaxes(self, ax1, ax2):
        adjusted_ax1 = ax1 if ax1 >= 0 else ax1-1
        adjusted_ax2 = ax2 if ax2 >= 0 else ax2-1
        return ExtendedKoalaBear(self.value.swapaxes(adjusted_ax1, adjusted_ax2))

    def copy(self):
        return ExtendedKoalaBear(np.copy(self.value))

    @property
    def ndim(self):
        return self.value.ndim - 1

    def to_extended(self):
        return self

    def __getitem__(self, index):
        return ExtendedKoalaBear(self.value[index])

    def __setitem__(self, index, value):
        if isinstance(value, int):
            self.value[index] = value
        elif isinstance(value, KoalaBear):
            self.value[index] = value.to_extended().value
        elif isinstance(value, ExtendedKoalaBear):
            self.value[index] = value.value
        else:
            raise Exception(f"Bad input for setitem: {value}")

    def __add__(self, other):
        if isinstance(other, KoalaBear):
            other = other.to_extended()
        elif isinstance(other, int):
            other = ExtendedKoalaBear(other)
        return ExtendedKoalaBear(add(self.value, other.value))

    def __neg__(self):
        return ExtendedKoalaBear(modulus - self.value)

    def __sub__(self, other):
        if isinstance(other, KoalaBear):
            other = other.to_extended()
        elif isinstance(other, int):
            other = ExtendedKoalaBear(other)
        return ExtendedKoalaBear(sub(self.value, other.value))

    def __mul__(self, other):
        if isinstance(other, int):
            other = KoalaBear(other)
        if isinstance(other, KoalaBear):
            return ExtendedKoalaBear(mul(
                self.value,
                other.value.reshape(other.value.shape + (1,))
            ))
        return ExtendedKoalaBear(mul_ext(self.value, other.value))

    def __pow__(self, other):
        assert isinstance(other, int)
        if other == 0:
            return KoalaBear(np.ones(self.shape)).to_extended()
        elif other == 1:
            return self
        elif other % 2 == 1:
            sub = self ** (other // 2)
            return sub * sub * self
        else:
            sub = self ** (other // 2)
            return sub * sub

    def inv(self):
        return ExtendedKoalaBear(modinv_ext(self.value))

    def __truediv__(self, other):
        if isinstance(other, int):
            other = KoalaBear(other)
        return self * other.inv()

    def __rtruediv__(self, other):
        return self.inv() * other

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = lambda self, other: -(self - other)

    def __repr__(self):
        return f'{self.value}'

    def tobytes(self):
        return (self.value % modulus).tobytes()

    def __len__(self):
        return len(self.value)

    def __eq__(self, other):
        if isinstance(other, int):
            other = KoalaBear(other)
        if isinstance(other, KoalaBear):
            other = other.to_extended()
        shape = np.broadcast_shapes(self.value.shape, other.value.shape)
        return np.array_equal(
            np.broadcast_to(self.value, shape) % modulus,
            np.broadcast_to(other.value, shape) % modulus
        )

def matmul(a, b, assume_second_input_small=False):
    if not isinstance(a, (KoalaBear, ExtendedKoalaBear)):
        raise Exception("First input must be KoalaBear or extended KoalaBear")
    if not isinstance(b, (KoalaBear, ExtendedKoalaBear)):
        raise Exception("Second input must be KoalaBear or extended KoalaBear")
    if isinstance(a, ExtendedKoalaBear) and isinstance(b, ExtendedKoalaBear):
        raise Exception("inputs cannot both be extended")
    a_value = a.value if isinstance(a, KoalaBear) else a.value.swapaxes(-2, -1)
    if assume_second_input_small:
        data1 = np.matmul(a_value & 65535, b.value)
        data2 = np.matmul(a_value >> 16, b.value)
        o = add(data1, mul(data2, array(65536)))
    else:
        data1 = a_value.astype(np.uint64)
        data2 = b.value.astype(np.uint64)
        o1 = np.matmul(data1 & 65535, data2)
        o2 = np.matmul(data1 >> 16, data2)
        o = ((o1 + ((o2 % modulus) << 16)) % modulus).astype(np.uint32)
    if isinstance(a, KoalaBear) and isinstance(b, KoalaBear):
        return KoalaBear(o)
    elif isinstance(a, ExtendedKoalaBear):
        return ExtendedKoalaBear(o.swapaxes(-2, -1))
    elif isinstance(b, ExtendedKoalaBear):
        return ExtendedKoalaBear(o)
    else:
        raise Exception("wat")
