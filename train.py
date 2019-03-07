import matplotlib.pyplot as plt
import numpy as np


from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/ksyolo.cfg", "load": "bin/tiny-yolo-voc.weights", "batch":64, "epoch": 20, "train": True, "annotation": "C:\\Users\\ldsbd\\Documents\\College\\0Directed Study\\darkflow\\train\\anno\\", "dataset": "C:\\Users\\ldsbd\\Documents\\College\\0Directed Study\\darkflow\\train\\images\\"}
TFNet(options).train()

options = {"model": "cfg/ksyolo.cfg", "load": -1}
tfnet2 = TFNet(options)

tfnet2.load_from_ckpt()

