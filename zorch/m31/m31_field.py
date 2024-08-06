from .m31_utils import (
    zeros, arange, array, append, add, sub, mul, cp, pow5, modinv, mul_ext,
    modinv_ext, modulus, sum as m31_sum
)

def mod31_py_obj(inp):
    if isinstance(inp, int):
        return inp % modulus
    else:
        return [mod31_py_obj(x) for x in inp]

class M31():
    def __init__(self, x):
        if isinstance(x, (int, list)):
            x = array(mod31_py_obj(x))
        elif isinstance(x, M31):
            x = x.value
        self.value = x
        assert self.value.dtype == cp.uint32

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
        return M31(self.value.reshape(shape))

    def swapaxes(self, ax1, ax2):
        return M31(self.value.swapaxes(ax1, ax2))

    def copy(self):
        return M31(cp.copy(self.value))

    @property
    def ndim(self):
        return self.value.ndim

    def to_extended(self):
        o = zeros(self.value.shape + (4,))
        o[...,0] = self.value
        return ExtendedM31(o)

    def __getitem__(self, index):
        return M31(self.value[index])

    def __setitem__(self, index, value):
        if isinstance(value, int):
            self.value[index] = value
        elif isinstance(value, M31):
            self.value[index] = value.value
        else:
            raise Exception(f"Bad input for setitem: {value}")

    def __add__(self, other):
        if isinstance(other, ExtendedM31):
            return self.to_extended() + other
        elif isinstance(other, int):
            other = M31(other)
        return M31(add(self.value, other.value))

    def __neg__(self):
        return M31(modulus - self.value)

    def __sub__(self, other):
        if isinstance(other, ExtendedM31):
            return self.to_extended() - other
        elif isinstance(other, int):
            other = M31(other)
        return M31(sub(self.value, other.value))

    def __mul__(self, other):
        if isinstance(other, ExtendedM31):
            return ExtendedM31(mul(
                self.value.reshape(self.value.shape + (1,)),
                other.value
            ))
        elif isinstance(other, int):
            other = M31(other)
        return M31(mul(self.value, other.value))

    def __pow__(self, other):
        assert isinstance(other, int)
        if other == 5:
            # Optimize common special case
            return M31(pow5(self.value))
        elif other == 0:
            return M31(cp.ones(self.value.shape))
        elif other == 1:
            return self
        elif other % 2 == 1:
            sub = self ** (other // 2)
            return sub * sub * self
        else:
            sub = self ** (other // 2)
            return sub * sub

    def inv(self):
        return M31(modinv(self.value))

    def __truediv__(self, other):
        if isinstance(other, int):
            other = M31(other)
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
            other = M31(other)
        elif isinstance(other, ExtendedM31):
            return self.to_extended() == other
        shape = cp.broadcast_shapes(self.value.shape, other.value.shape)
        return cp.array_equal(
            cp.broadcast_to(self.value, shape) % modulus,
            cp.broadcast_to(other.value, shape) % modulus
        )

class ExtendedM31():
    def __init__(self, x):
        if isinstance(x, int):
            x = array([x % modulus, 0, 0, 0])
        elif isinstance(x, list):
            x = array(mod31_py_obj(x))
        elif isinstance(x, M31):
            x = x.value.to_extended()
        elif isinstance(x, ExtendedM31):
            x = x.value
        assert x.shape[-1] == 4
        self.value = x
        assert self.value.dtype == cp.uint32

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
        return ExtendedM31(self.value.reshape(shape + (4,)))

    def swapaxes(self, ax1, ax2):
        adjusted_ax1 = ax1 if ax1 >= 0 else ax1-1
        adjusted_ax2 = ax2 if ax2 >= 0 else ax2-1
        return ExtendedM31(self.value.swapaxes(adjusted_ax1, adjusted_ax2))

    def copy(self):
        return ExtendedM31(cp.copy(self.value))

    @property
    def ndim(self):
        return self.value.ndim - 1

    def to_extended(self):
        return self

    def __getitem__(self, index):
        return ExtendedM31(self.value[index])

    def __setitem__(self, index, value):
        if isinstance(value, int):
            self.value[index] = value
        elif isinstance(value, M31):
            self.value[index] = value.to_extended().value
        elif isinstance(value, ExtendedM31):
            self.value[index] = value.value
        else:
            raise Exception(f"Bad input for setitem: {value}")

    def __add__(self, other):
        if isinstance(other, M31):
            other = other.to_extended()
        elif isinstance(other, int):
            other = ExtendedM31(other)
        return ExtendedM31(add(self.value, other.value))

    def __neg__(self):
        return ExtendedM31(modulus - self.value)

    def __sub__(self, other):
        if isinstance(other, M31):
            other = other.to_extended()
        elif isinstance(other, int):
            other = ExtendedM31(other)
        return ExtendedM31(sub(self.value, other.value))

    def __mul__(self, other):
        if isinstance(other, int):
            other = M31(other)
        if isinstance(other, M31):
            return ExtendedM31(mul(
                self.value,
                other.value.reshape(other.value.shape + (1,))
            ))
        return ExtendedM31(mul_ext(self.value, other.value))

    def __pow__(self, other):
        assert isinstance(other, int)
        if other == 0:
            return M31(cp.ones(self.shape)).to_extended()
        elif other == 1:
            return self
        elif other % 2 == 1:
            sub = self ** (other // 2)
            return sub * sub * self
        else:
            sub = self ** (other // 2)
            return sub * sub

    def inv(self):
        return ExtendedM31(modinv_ext(self.value))

    def __truediv__(self, other):
        if isinstance(other, int):
            other = M31(other)
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
            other = M31(other)
        if isinstance(other, M31):
            other = other.to_extended()
        shape = cp.broadcast_shapes(self.value.shape, other.value.shape)
        return cp.array_equal(
            cp.broadcast_to(self.value, shape) % modulus,
            cp.broadcast_to(other.value, shape) % modulus
        )

def matmul(a, b, assume_second_input_small=False):
    if not isinstance(a, (M31, ExtendedM31)):
        raise Exception("First input must be M31 or extended M31")
    elif not isinstance(b, M31):
        raise Exception("Second input must be M31")
    a_value = a.value if isinstance(a, M31) else a.value.swapaxes(-2, -1)
    if assume_second_input_small:
        data1 = cp.matmul(a_value & 65535, b.value)
        data2 = cp.matmul(a_value >> 16, b.value)
        o = add(data1, mul(data2, array(65536)))
    else:
        data1 = a_value.astype(cp.uint64)
        data2 = b.value.astype(cp.uint64)
        o1 = cp.matmul(data1 & 65535, data2)
        o2 = cp.matmul(data1 >> 16, data2)
        o = ((o1 + ((o2 % modulus) << 16)) % modulus).astype(cp.uint32)
    if isinstance(a, M31):
        return M31(o)
    else:
        return ExtendedM31(o.swapaxes(-2, -1))
