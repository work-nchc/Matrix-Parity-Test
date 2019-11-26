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

matrix = numpy.random.randint(
    0, 1 << bit_length, (size,) * 2, dtype_from_bit[bit_length])
print('mem', matrix.__sizeof__())

begin_mv = time()
matrix_gpu = cupy.asarray(matrix)
end_mv = time()
print('move', end_mv - begin_mv)

sleep(1)

begin_gpu = time()
parity0_gpu = parity_kernel(matrix_gpu, axis=0)
parity1_gpu = parity_kernel(matrix_gpu, axis=1)
end_gpu = time()
print('gpu', end_gpu - begin_gpu)

begin = time()
parity0 = numpy.bitwise_xor.reduce(matrix)
parity1 = numpy.bitwise_xor.reduce(matrix, 1)
end = time()
print('cpu', end - begin)

print(
    'check',
    (parity0_gpu.get() ^ parity0).sum() + (parity1_gpu.get() ^ parity1).sum())
