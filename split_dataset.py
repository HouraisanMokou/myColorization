import argparse
import os.path
import shutil

####
# split test and val
####
import random
import sys

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./dataset/ilsvr2012')
    parser.add_argument('--random', type=int, default=2022)
    args, unknown = parser.parse_known_args()
    origin = os.path.join(args.path, 'test_val')

    test = os.path.join(args.path, 'test')
    val = os.path.join(args.path, 'val')
    for f in [test, val]:
        os.makedirs(f)

    # split train to
    files = os.listdir(origin)
    length = len(files)
    print(length)
    length2=int(length//5)
    test_idx=np.random.permutation(length)[:length2]
    for idx,file in enumerate(files):
        if idx in test_idx:
            shutil.move(
                os.path.join(origin,file),
                os.path.join(test,file)
            )
        else:
            shutil.move(
                os.path.join(origin,file),
                os.path.join(val,file)
            )
        if idx%10==0:
            print('.', end='')
            sys.stdout.flush()



