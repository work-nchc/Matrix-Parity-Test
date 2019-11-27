from time import time, sleep
import numpy
import cupy

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

parity_kernel = cupy.ReductionKernel(
    'T x',
    'T y',
    'x',
    'a ^ b',
    'y = a',
    '0',
    'parity'
)

def check(n):
    count = 0
    while n:
        count += 1
        n &= n - 1
    return count

begin_gen = time()
matrix = numpy.random.randint(
    0, 1 << bit_length, (size, size), dtype_from_bit[bit_length]
)
end_gen = time()
print('mem', matrix.__sizeof__())
print('gen', end_gen - begin_gen)

begin_mv = time()
matrix_gpu = cupy.asarray(matrix)
end_mv = time()
print('move', end_mv - begin_mv)

sleep(2)

begin_gpu = time()
parity0_in_gpu = parity_kernel(matrix_gpu, axis=0)
parity1_in_gpu = parity_kernel(matrix_gpu, axis=1)
parity0_from_gpu = parity0_in_gpu.get()
parity1_from_gpu = parity1_in_gpu.get()
end_gpu = time()
print('gpu', end_gpu - begin_gpu)

begin_cpu = time()
parity0 = numpy.bitwise_xor.reduce(matrix)
parity1 = numpy.bitwise_xor.reduce(matrix, 1)
end_cpu = time()
print('cpu', end_cpu - begin_cpu)

print(
    'check',
    sum(map(check, parity0_from_gpu ^ parity0))
    + sum(map(check, parity1_from_gpu ^ parity1)),
)
