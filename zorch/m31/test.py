from .m31_field import M31, ExtendedM31, matmul
from .m31_utils import modulus, cp, arange
from .m31_circle import Point, G, Z

def test():
    x_orig = M31(3 ** arange(10**7) % modulus)
    x = M31(x_orig.value.copy())
    for i in range(31):
        x = x * x
    assert x == x_orig * x_orig
    assert (
        x[:10] * (x[10:20] + x[20:30]) ==
        (x[:10] * x[10:20]) + (x[:10] * x[20:30])
    )
    assert x ** 5 == x * x * x * x * x
    assert (x.inv() * x) == 1
    assert (x + x_orig) - x_orig == x
    print("Basic arithmetic tests passed")
    x4_orig = ExtendedM31(
        3 ** arange(4 * 10**7, dtype=cp.uint16).reshape((10**7, 4)) % modulus
    )
    x4 = ExtendedM31(cp.copy(x4_orig.value))
    x5 = ExtendedM31(cp.copy(x4_orig.value))
    for i in range(4):
        x4 = x4 * x4
    for i in range(15):
        x5 *= x4_orig
    assert x4 == x5
    assert (
        x4[:10] * (x4[10:20] + x4[20:30]) ==
        (x4[:10] * x4[10:20]) + (x4[:10] * x4[20:30])
    )
    print("Extended arithmetic tests passed")
    x6 = M31(3 ** arange(10**6) % modulus)
    x7 = x6.inv()
    x8 = x6 * x7
    assert x8 == 1
    assert x8 - 1 == 0
    print("Basic modinv tests passed")
    assert (x4.inv() * x4_orig).inv() * x4_orig == x4
    assert (x4.inv() * x).inv() * x == x4
    x9 = ExtendedM31(x6.value.reshape((250000, 4)))
    x10 = x9.inv()
    assert x9 * x10 == 1
    assert x9 * x10 - 1 == 0
    print("Extended modinv tests passed")
    x = G
    for i in range(31):
        x = x.double()
    assert x == Z
    x = Point.zeros(1)
    coeff = G
    for i in range(4):
        double_x = x.double()
        x = Point.zeros(x.shape[0] * 2)
        x[::2] = double_x
        x[1::2] = double_x + G
    for i in range(15):
        assert x[i+1] == x[i] + G
    ext_point = Point(
        ExtendedM31([968417241, 1522700037, 1711331479, 520782658]),
        ExtendedM31([950082908, 1835034903, 1779185035, 1647796460])
    )
    x = Point.zeros(1).to_extended()
    coeff = ext_point
    for i in range(4):
        double_x = x.double()
        x = Point.zeros(x.shape[0] * 2).to_extended()
        x[::2] = double_x
        x[1::2] = double_x + ext_point
    for i in range(15):
        assert x[i+1] == x[i] + ext_point
        assert x[i+1] + G == (x[i] + G) + ext_point
    print("Point arithmetic tests passed")
    a = M31([123, 456000])
    m1 = M31([[3, 4], [2, 3]])
    m2 = M31([[3, -4], [-2, 3]])
    med = matmul(a, m1) * 10
    o = matmul(med, m2) * 10
    assert o == M31([12300, 45600000])
    a2 = ExtendedM31([[1,2,3,4],[5,6,7,800000]])
    med2 = matmul(a2, m1, assume_second_input_small=True) * 10
    o2 = matmul(med2, m2) * 10
    assert o2 == ExtendedM31([[100,200,300,400],[500,600,700,80000000]])
    print("Matrix multiplication tests passed")

if __name__ == '__main__':
    test()
