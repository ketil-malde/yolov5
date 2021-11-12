#!/usr/bin/python3

# Script to wrap yolo v5
# images must be placed in ./images/foobar.jpg (or png, etc)
# labels must be placed in ./labels/foobar.txt, format <classno> <x1> <y1> <x2> <y2> - in fraction of image

import os
import pwd
import cv2

IMAGENAME='yolov5'

USERID=os.getuid()
GROUPID=os.getgid()
CWD=os.getcwd()
USERNAME=pwd.getpwuid(USERID).pw_name
RUNTIME='--gpus device=0'

def docker_run(args=''):
    os.system(f'docker run {RUNTIME} --ipc=host --rm --user {USERID}:{GROUPID} -v {CWD}:/project -it {USERNAME}-{IMAGENAME} {args}')

def docker_build(args=''):
    os.system(f'docker build --build-arg user={USERNAME} --build-arg uid={USERID} --build-arg gid={GROUPID} -t {USERNAME}-{IMAGENAME} {args}')

from ast import literal_eval

def prep_data():
    '''Extract info from annotations.csv and write text files in ./labels'''
    # class, center x, center y, width, heights - as image fractions
    os.makedirs('labels', exist_ok=True)
    curim = None
    of = None
    dx = None
    dy = None
    tf = open('train.txt', 'w')
    with open('annotations.csv', 'r') as f:
        for l in f.readlines():
            im, cls, bbox, mask = l.split('\t')
            if im != curim:
                if of is not None: of.close()
                curim = im
                tmpimg = cv2.imread(f'images/{im}')
                tf.write(f'./images/{im}\n')
                dy,dx,C = tmpimg.shape
                of = open('labels/'+im[:-4]+'.txt', 'w')
            x1, y1, x2, y2 = literal_eval(bbox)
            # print(im, cls, x1, y1, x2, y2, mask)
            cx = (float(x1)+float(x2))/2
            cy = (float(y1)+float(y2))/2
            w  = float(x2)-float(x1)
            h  = float(y2)-float(y1)

            of.write(f'0 {cx/dx} {cy/dy} {w/dx} {h/dy}\n')
    of.close()
    tf.close()

class Model:
    def __init__(self, conf, mypath):
        self.myconf = conf
        self.mypath = mypath

    def build(self):
        '''Build the docker instance'''
        prep_data()
        print('Building docker...')
        docker_build(self.mypath)
        
    def train(self):
        '''Train the network'''
        docker_run(f"python3 /usr/src/app/train.py --hyp={self.myconf['hyper']} --weights={self.myconf['weights']} --data={self.myconf['dataset']}")

    def check(self):
        '''Verify that data is in place and that the output doesn't exist'''
        pass

    def predict(self, wgths, target, output):
        '''Run a trained network on the data in target'''
        docker_run(f"python3 /usr/src/app/detect.py --weights={wgths} --source={target} --name={output}")

    def test(self):
        '''Run tests'''
        docker_run('python3 /src/test.py')

    def status(self):
        '''Print the current training status'''
        # check if docker exists
        # check if network is trained (and validation accuracy?)
        # check if data is present for training
        # check if test data is present
        # check if test output is present (and report)
        pass

if __name__ == '__main__':
    import argparse
    import sys
    
    if sys.argv[1] == 'build':
        docker_build()
    else:
        error('Usage: {sys.argv[0]} [check,status,train,predict] ...')

