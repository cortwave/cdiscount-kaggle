import subprocess
import json
from time import time

with open('/home/arseny/gpu.log', 'a') as out:
    while True:
        s = subprocess.run(['nvidia-smi', '--format=csv', '--query-gpu=power.draw,utilization.gpu,temperature.gpu'],
                           stdout=subprocess.PIPE)
        head, gpu0, gpu1, _ = s.stdout.decode().split('\n')

        for i, gpu in enumerate((gpu0, gpu1)):
            power, util, temp = gpu.split(', ')
            out.write(json.dumps({'power': float(power.split(' ')[0]),
                                  'util': float(util.split(' ')[0]),
                                  'temp': float(temp.split(' ')[0]),
                                  'time': str(time()),
                                  'gpu': i,
                                  }) + '\n')