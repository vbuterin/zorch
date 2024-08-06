from .utils import cp, arange
from .binary_field import Binary, ExtendedBinary

def test():
    x_orig = Binary(3 ** arange(10**4))
    x = x_orig.copy()
    for i in range(16):
        x = x * x
    assert x_orig == x
    assert (
        x[:10] * (x[10:20] ^ x[20:30]) ==
        (x[:10] * x[10:20]) ^ (x[:10] * x[20:30])
    )
    assert (x ^ x[::-1]) ^ x[::-1] == x
    print("Basic arithmetic tests passed")
    x4_orig = ExtendedBinary(
        3 ** arange(8 * 10**5).reshape((10**5, 8))
    )
    x4 = ExtendedBinary(cp.copy(x4_orig.value))
    x5 = ExtendedBinary(cp.copy(x4_orig.value))
    for i in range(4):
        x4 = x4 * x4
    for i in range(15):
        x5 *= x4_orig
    assert x4 == x5
    assert (
        x4[:10] * (x4[10:20] ^ x4[20:30]) ==
        (x4[:10] * x4[10:20]) ^ (x4[:10] * x4[20:30])
    )
    assert (
        x4[:10] * (x[10:20] ^ x4[20:30]) ==
        (x4[:10] * x[10:20]) ^ (x4[:10] * x4[20:30])
    )
    print("Extended arithmetic tests passed")
    x6 = Binary(3 ** arange(10**4))
    x7 = x6.inv()
    x8 = x6 * x7
    assert x8 == 1
    assert x8 ^ 1 == 0
    print("Basic modinv tests passed")
    assert (x4.inv() * x4_orig).inv() * x4_orig == x4
    assert (x4[:10000].inv() * x).inv() * x == x4[:10000]
    x9 = ExtendedBinary(x6.value.reshape((2500, 4)))
    x10 = x9.inv()
    assert x9 * x10 == 1
    assert x9 * x10 ^ 1 == 0
    print("Extended modinv tests passed")
    x11 = ExtendedBinary([1,2,3,4,0,0,0,0])
    x11_short = ExtendedBinary([1,2,3,4])
    example = ExtendedBinary([5,6,7,8,9,10,11,12])
    assert x11 * example == x11_short * example
    assert x11 ^ example == x11_short ^ example
    print("Different-length tests passed")
