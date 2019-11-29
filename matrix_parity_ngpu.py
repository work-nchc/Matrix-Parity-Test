from time import time
from os import environ
environ['PATH'] = environ['PATH'] + 'N:\\nchc\\cuda101\\bin;'
import numpy
import cupy

bit_length = 32
size = 2 ** 16
n_gpu = 8

print('bit', bit_length)
print('size', size)
print('n_gpu', n_gpu)

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
    0, 1 << bit_length, (n_gpu, size, size), dtype_from_bit[bit_length]
)
end_gen = time()
print('mem', matrix.__sizeof__())
print('gen', end_gen - begin_gen)

matrix_gpu = [None] * n_gpu
parity0 = [None] * n_gpu
parity1 = [None] * n_gpu
parity0_in_gpu = [None] * n_gpu
parity1_in_gpu = [None] * n_gpu
parity0_from_gpu = [None] * n_gpu
parity1_from_gpu = [None] * n_gpu

begin_mv = time()
for i in range(n_gpu):
    with cupy.cuda.Device(i):
        matrix_gpu[i] = cupy.asarray(matrix[i])
cupy.cuda.Stream.null.synchronize()
end_mv = time()
print('move', end_mv - begin_mv)

begin_gpu = time()
for i in range(n_gpu):
    with cupy.cuda.Device(i):
        parity0_in_gpu[i] = parity_kernel(matrix_gpu[i], axis=0)
        parity1_in_gpu[i] = parity_kernel(matrix_gpu[i], axis=1)
for i in range(n_gpu):
    with cupy.cuda.Device(i):
        parity0_from_gpu[i] = parity0_in_gpu[i].get()
        parity1_from_gpu[i] = parity1_in_gpu[i].get()
end_gpu = time()
print('gpu', end_gpu - begin_gpu)

begin_cpu = time()
for i in range(n_gpu):
    parity0[i] = numpy.bitwise_xor.reduce(matrix[i])
    parity1[i] = numpy.bitwise_xor.reduce(matrix[i], 1)
end_cpu = time()
print('cpu', end_cpu - begin_cpu)

for i in range(n_gpu):
    print(
        'check',
        sum(map(check, parity0_from_gpu[i] ^ parity0[i]))
        + sum(map(check, parity1_from_gpu[i] ^ parity1[i])),
    )
