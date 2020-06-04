#!/usr/bin/env python
import os
import glob
import shutil

script_root = os.path.dirname(os.path.realpath(__file__))

def delete_files(pattern):
    fileList = glob.glob(os.path.join(script_root, "data", pattern))
    for file in fileList:
        try:
            if os.path.isdir(file):
                shutil.rmtree(file)
            else:
                os.remove(file)
        except:
            print("Error while deleting file : ", file)

delete_files("model*")
delete_files("cifar_trace*")
delete_files("events*")
delete_files("eval*")
delete_files("checkpoint*")