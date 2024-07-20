import cupy as cp
from . import m31
M31 = modulus = 2**31-1

class Point():
    def __init__(self, x, y):
        assert x.shape == y.shape
        self.x = x
        self.y = y

    @classmethod
    def zeros(cls, shape):
        return cls(m31.zeros(shape) + 1, m31.zeros(shape))

    @property
    def shape(self):
        return self.x.shape

    def reshape(self, shape):
        return Point(
            self.x.reshape(shape),
            self.y.reshape(shape)
        )

    @property
    def ndim(self):
        return self.x.ndim

    def to_extended(self):
        ox = cp.zeros(self.x.shape + (4,), dtype=cp.uint32)
        ox[...,0] = self.x
        oy = cp.zeros(self.y.shape + (4,), dtype=cp.uint32)
        oy[...,0] = self.y
        return ExtendedPoint(ox, oy)

    def __getitem__(self, index):
        return Point(self.x[index], self.y[index])

    def __setitem__(self, index, value):
        assert self.__class__ == value.__class__
        self.x[index] = value.x
        self.y[index] = value.y

    def __add__(self, other):
        assert self.__class__ == other.__class__
        return Point(
            m31.sub(m31.mul(self.x, other.x), m31.mul(self.y, other.y)),
            m31.add(m31.mul(self.x, other.y), m31.mul(self.y, other.x))
        )

    def double(self):
        return Point(
            ((m31.mul(self.x, self.x) * 2) % M31 + M31 - 1) % M31,
            (2 * m31.mul(self.x, self.y)) % M31
        )

    def __repr__(self):
        return f'(x={self.x}, y={self.y})'

class ExtendedPoint():
    def __init__(self, x, y):
        assert x.shape == y.shape
        assert x.shape[-1] == 4
        self.x = x
        self.y = y

    @classmethod
    def zeros(cls, shape):
        x = m31.zeros(shape + (4,))
        x[...,0] = 1
        y = m31.zeros(shape + (4,))
        return cls(x, y)

    @property
    def shape(self):
        return self.x.shape[:-1]

    def reshape(self, shape):
        return ExtendedPoint(
            self.x.reshape(shape + (4,)),
            self.y.reshape(shape + (4,))
        )

    @property
    def ndim(self):
        return self.x.ndim - 1

    def to_extended(self):
        return self

    def __getitem__(self, index):
        return ExtendedPoint(self.x[index], self.y[index])

    def __setitem__(self, index, value):
        self.x[index] = value.x
        self.y[index] = value.y

    def __add__(self, other):
        assert self.__class__ == other.__class__
        return ExtendedPoint(
            m31.sub(
                m31.mul_ext(self.x, other.x),
                m31.mul_ext(self.y, other.y)
            ),
            m31.add(
                m31.mul_ext(self.x, other.y),
                m31.mul_ext(self.y, other.x)
            )
        )

    def double(self):
        return ExtendedPoint(
            ((m31.mul_ext(self.x, self.x) * 2) % M31 + M31 - 1) % M31,
            (2 * m31.mul_ext(self.x, self.y)) % M31
        )

    def __repr__(self):
        return f'(x={self.x}, y={self.y})'

Z = Point(
    cp.array(1, dtype=cp.uint32),
    cp.array(0, dtype=cp.uint32)
)

G = Point(
    cp.array(1268011823, dtype=cp.uint32),
    cp.array(2, dtype=cp.uint32)
)
