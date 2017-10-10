import subprocess
import json
from time import time, sleep
from fire import Fire

FILE_NAME = './gpu.log'


def write_log(file_name=FILE_NAME, interval=0):
    with open(file_name, 'a+') as out:
        while True:
            s = subprocess.run(['nvidia-smi', '--format=csv', '--query-gpu=power.draw,utilization.gpu,temperature.gpu'],
                               stdout=subprocess.PIPE)
            gpus = s.stdout.decode().split('\n')[1:-1]
            for i, gpu in enumerate(gpus):
                power, util, temp = gpu.split(', ')
                out.write(json.dumps({'power': float(power.split(' ')[0]),
                                      'util': float(util.split(' ')[0]),
                                      'temp': float(temp.split(' ')[0]),
                                      'time': str(time()),
                                      'gpu': i,
                                      }) + '\n')
                out.flush()
            if interval != 0:
                sleep(interval)


if __name__ == '__main__':
    Fire(write_log)
