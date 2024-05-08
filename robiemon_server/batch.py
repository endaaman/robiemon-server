import os
import sys
from glob import glob

import numpy as np
from PIL import Image
from matplotlib import cm as colormap

def gen_mask():
    ff = glob('data/results/bt/*/cam.png')
    for f in ff:
        d = os.path.dirname(f)
        image = Image.open(f)
        cam_mask = np.array(image)/255
        Image.fromarray(np.uint8(colormap.jet(cam_mask)*255)).save(os.path.join(d,'cam_jet.png'))
        Image.fromarray(np.uint8(colormap.inferno(cam_mask)*255)).save(os.path.join(d,'cam_inferno.png'))

def bar():
    print('Executing bar')

def batch():
    try:
        command = sys.argv[1]
        if command in globals() and callable(globals()[command]):
            globals()[command]()
        else:
            print(f'No such command: {command}')
    except IndexError:
        print('No command provided')

if __name__ == '__main__':
    batch()
