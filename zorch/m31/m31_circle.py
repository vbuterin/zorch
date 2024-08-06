from .m31_utils import modulus
from .m31_field import M31

class Point():
    def __init__(self, x, y):
        assert x.shape == y.shape
        self.x = x
        self.y = y

    @classmethod
    def zeros(cls, shape):
        return cls(M31.zeros(shape) + 1, M31.zeros(shape))

    @classmethod
    def append(cls, *args, axis=0):
        return cls(
            args[0].x.__class__.append(*(v.x for v in args), axis=axis),
            args[0].x.__class__.append(*(v.y for v in args), axis=axis)
        )

    @property
    def shape(self):
        return self.x.shape

    def reshape(self, shape):
        return Point(
            self.x.reshape(shape),
            self.y.reshape(shape)
        )

    def swapaxes(self, ax1, ax2):
        return Point(self.x.swapaxes(ax1, ax2), self.y.swapaxes(ax1, ax2))

    def copy(self):
        return Point(cp.copy(self.x), cp.copy(self.y))

    @property
    def ndim(self):
        return self.x.ndim

    def to_extended(self):
        return Point(self.x.to_extended(), self.y.to_extended())

    def __getitem__(self, index):
        return Point(self.x[index], self.y[index])

    def __setitem__(self, index, value):
        assert self.__class__ == value.__class__
        self.x[index] = value.x
        self.y[index] = value.y

    def __add__(self, other):
        assert self.__class__ == other.__class__
        return Point(
            self.x * other.x - self.y * other.y,
            self.x * other.y + self.y * other.x
        )

    def double(self):
        return Point(
            self.x * self.x * 2 - 1,
            self.x * self.y * 2
        )

    def __repr__(self):
        return f'(x={self.x}, y={self.y})'

    def __len__(self):
        return len(self.value)

    def tobytes(self):
        if isinstance(self.x, ExtendedM31) or isinstance(self.y, ExtendedM31):
            self.x = self.x.to_extended()
            self.y = self.y.to_extended()
        return self.x.tobytes() + self.y.tobytes()

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

Z = Point(M31(1), M31(0))

G = Point(M31(1268011823), M31(2))
