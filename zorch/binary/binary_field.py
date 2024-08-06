from .utils import (
    mul, zeros, arange, array, xor, inv, cp, big_mul
)

class Binary():
    def __init__(self, x):
        if isinstance(x, (int, list)):
            x = array(x)
        elif isinstance(x, Binary):
            x = x.value
        self.value = x
        assert self.value.dtype == cp.uint16

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
        return cls(xor(arg.value, axis=axis))

    @property
    def shape(self):
        return self.value.shape

    def reshape(self, shape):
        return Binary(self.value.reshape(shape))

    def swapaxes(self, ax1, ax2):
        return Binary(self.value.swapaxes(ax1, ax2))

    def copy(self):
        return Binary(cp.copy(self.value))

    @property
    def ndim(self):
        return self.value.ndim

    def __getitem__(self, index):
        return Binary(self.value[index])

    def __setitem__(self, index, value):
        if isinstance(value, int):
            self.value[index] = value
        elif isinstance(value, Binary):
            self.value[index] = value.value
        else:
            raise Exception(f"Bad input for setitem: {value}")

    def to_extended(self, limbs):
        o = zeros(self.value.shape + (limbs,))
        o[...,0] = self.value
        return ExtendedBinary(o)

    def __add__(self, other):
        raise Exception("Use xor with binary fields")

    def __neg__(self):
        raise Exception("x = -x in binary fields")

    def __sub__(self, other):
        raise Exception("Use xor with binary fields")

    def __xor__(self, other):
        if isinstance(other, ExtendedBinary):
            return self.to_extended(other.limbs) ^ other
        elif isinstance(other, int):
            other = Binary(other)
        return Binary(self.value ^ other.value)

    def __mul__(self, other):
        if isinstance(other, ExtendedBinary):
            return ExtendedBinary(mul(
                self.value.reshape(self.value.shape + (1,)),
                other.value
            ))
        elif isinstance(other, int):
            other = Binary(other)
        return Binary(mul(self.value, other.value))

    def __pow__(self, other):
        assert isinstance(other, int)
        if other == 0:
            return Binary(cp.ones(self.value.shape))
        elif other == 1:
            return self
        elif other % 2 == 1:
            sub = self ** (other // 2)
            return sub * sub * self
        else:
            sub = self ** (other // 2)
            return sub * sub

    def inv(self):
        return Binary(inv[self.value])

    def __truediv__(self, other):
        if isinstance(other, int):
            other = Binary(other)
        return self * other.inv()

    def __rtruediv__(self, other):
        return self.inv() * other

    __rxor__ = __xor__
    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__

    def __repr__(self):
        return f'{self.value}'

    def __int__(self):
        return int(self.value)

    def __len__(self):
        return len(self.value)

    def tobytes(self):
        return self.value.tobytes()

    def __eq__(self, other):
        if isinstance(other, int):
            other = Binary(other)
        elif isinstance(other, ExtendedBinary):
            return self.to_extended(other.limbs) == other
        shape = cp.broadcast_shapes(self.value.shape, other.value.shape)
        return cp.array_equal(
            cp.broadcast_to(self.value, shape),
            cp.broadcast_to(other.value, shape)
        )

def match_limbs(a, b):
    if a.shape[-1] == b.shape[-1]:
        return a, b
    elif a.shape[-1] < b.shape[-1]:
        padding = b.shape[-1] - a.shape[-1]
        pad_params = [(0,0)] * (a.ndim - 1) + [(0, padding)]
        return cp.pad(a, pad_params, mode='constant', constant_values=0), b
    else:
        padding = a.shape[-1] - b.shape[-1]
        pad_params = [(0,0)] * (b.ndim - 1) + [(0, padding)]
        return a, cp.pad(b, pad_params, mode='constant', constant_values=0)

class ExtendedBinary():
    def __init__(self, x):
        if isinstance(x, int):
            raise Exception("Initializing directly from int not supported")
        elif isinstance(x, list):
            x = array(x)
        elif isinstance(x, Binary):
            raise Exception("Initializing directly from Binary not supported")
        elif isinstance(x, ExtendedBinary):
            x = x.value
        assert x.shape[-1] & (x.shape[-1] - 1) == 0
        self.value = x
        assert self.value.dtype == cp.uint16

    @classmethod
    def zeros(cls, shape):
        return cls(zeros(shape + (4,)))

    @classmethod
    def append(cls, *args, axis=0):
        return cls(append(*(x.value for x in args), axis=axis))

    @classmethod
    def sum(cls, arg, axis):
        adjusted_axis = axis if axis >= 0 else axis-1
        return cls(xor(arg.value, axis=adjusted_axis))

    @property
    def shape(self):
        return self.value.shape[:-1]

    @property
    def limbs(self):
        return self.value.shape[-1]

    def reshape(self, shape):
        return ExtendedBinary(self.value.reshape(shape + self.shape[-1:]))

    def swapaxes(self, ax1, ax2):
        adjusted_ax1 = ax1 if ax1 >= 0 else ax1-1
        adjusted_ax2 = ax2 if ax2 >= 0 else ax2-1
        return ExtendedBinary(self.value.swapaxes(adjusted_ax1, adjusted_ax2))

    def copy(self):
        return ExtendedBinary(cp.copy(self.value))

    @property
    def ndim(self):
        return self.value.ndim - 1

    def to_extended(self):
        return self

    def __getitem__(self, index):
        return ExtendedBinary(self.value[index])

    def __setitem__(self, index, value):
        if isinstance(value, int):
            self.value[index] = value
        elif isinstance(value, Binary):
            self.value[index] = value.to_extended(self.limbs).value
        elif isinstance(value, ExtendedBinary):
            self.value[index] = value.value
        else:
            raise Exception(f"Bad input for setitem: {value}")

    def __add__(self, other):
        raise Exception("Use xor with binary fields")

    def __neg__(self):
        raise Exception("x = -x in binary fields")

    def __sub__(self, other):
        raise Exception("Use xor with binary fields")

    def __xor__(self, other):
        if isinstance(other, Binary):
            other = other.to_extended(self.limbs)
        elif isinstance(other, int):
            other = Binary(other).to_extended(self.limbs)
        a, b = match_limbs(self.value, other.value)
        return ExtendedBinary(a ^ b)

    def __mul__(self, other):
        if isinstance(other, Binary):
            return ExtendedBinary(mul(
                self.value,
                other.value.reshape(other.value.shape + (1,))
            ))
        elif isinstance(other, int):
            return ExtendedBinary(mul(
                self.value,
                cp.array(other, dtype=cp.uint16)
            ))
        a, b = match_limbs(self.value, other.value)
        return ExtendedBinary(big_mul(a, b))

    def __pow__(self, other):
        assert isinstance(other, int)
        if other == 0:
            return ExtendedBinary(cp.ones(self.shape))
        elif other == 1:
            return self
        elif other % 2 == 1:
            sub = self ** (other // 2)
            return sub * sub * self
        else:
            sub = self ** (other // 2)
            return sub * sub

    def inv(self):
        # Waaaaay under-optimized, todo fix
        power = 2**(16 * self.limbs) - 2
        return self ** power

    def __truediv__(self, other):
        if isinstance(other, int):
            other = Binary(other)
        return self * other.inv()

    def __rtruediv__(self, other):
        return self.inv() * other

    __rxor__ = __xor__
    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__

    def __repr__(self):
        return f'{self.value}'

    def tobytes(self):
        return self.value.tobytes()

    def __len__(self):
        return len(self.value)

    def __eq__(self, other):
        if isinstance(other, int):
            other = Binary(other)
        if isinstance(other, Binary):
            other = other.to_extended(self.limbs)
        a, b = match_limbs(self.value, other.value)
        shape = cp.broadcast_shapes(a.shape, b.shape)
        return cp.array_equal(
            cp.broadcast_to(a, shape), cp.broadcast_to(b, shape)
        )
