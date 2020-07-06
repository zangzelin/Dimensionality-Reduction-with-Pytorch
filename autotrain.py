import os
from multiprocessing import Process, Manager
import numpy as np
import signal
import time
from itertools import product
import subprocess
# parameter analysis for SAGloss


lr_list = [1e-2, ]
perplexity_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
# MAER_list = [0.0, 0.01, 0.1, 0.3, 0.5, 1, 2, 4, 8]


losp_list = list(product(lr_list, perplexity_list))

pool_number = 16

c_pn = pool_number-1
gpunum = 0
for lr, perplexity in losp_list:
    txt = "CUDA_VISIBLE_DEVICES={} ".format(gpunum) +\
        "python -u ./main.py --lr {} --numberClass {}".format(
            lr, perplexity
    )
    print(txt)
    # os.system(txt)
    time.sleep(2)
    if gpunum < 7:
        gpunum += 1
    else:
        gpunum = 0

    if c_pn == 0:
        print('222', c_pn)
        c_pn = pool_number-1
        child = subprocess.Popen(txt, shell=True)
        child.wait()
        # subprocess.Popen()
    else:
        print('111', c_pn)
        child = subprocess.Popen(txt, shell=True)
        # child.wait(2)
        c_pn -= 1

    # cmd.append()
