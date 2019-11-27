from time import time
import numpy

bit_length = 32
size = 2 ** 14

print('bit', bit_length)
print('size', size)

dtype_from_bit = {
    8: 'uint8',
    16: 'uint16',
    32: 'uint32',
    64: 'uint64',
}

begin_gen = time()
matrix = numpy.random.randint(
    0, 1 << bit_length, (size, size), dtype_from_bit[bit_length]
)
end_gen = time()
print('mem', matrix.__sizeof__())
print('gen', end_gen - begin_gen)

begin_cpu = time()
parity0 = numpy.bitwise_xor.reduce(matrix)
parity1 = numpy.bitwise_xor.reduce(matrix, 1)
end_cpu = time()
print('cpu', end_cpu - begin_cpu)
