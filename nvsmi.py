from subprocess import run, PIPE
from os import chdir
chdir('N:/nchc/nv')
print(run(('nvidia-smi.exe',), stdout=PIPE).stdout.decode())
print(run((
    'nvidia-smi.exe', 'nvlink',
#    '-s',
#    '-c',
    '-p',
), stdout=PIPE).stdout.decode())
