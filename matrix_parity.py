from numpy import bitwise_xor
from numpy.random import randint
from time import time

bit_length = 32
size = 2 ** 14

dtype_from_bit = {
    8: 'uint8',
    16: 'uint16',
    32: 'uint32',
    64: 'uint64',
}

matrix = randint(
    0, 1 << bit_length, (size,) * 2, dtype_from_bit[bit_length])

begin = time()
parity0 = bitwise_xor.reduce(matrix)
parity1 = bitwise_xor.reduce(matrix, 1)
end = time()

print(end - begin)
print(matrix.__sizeof__())
