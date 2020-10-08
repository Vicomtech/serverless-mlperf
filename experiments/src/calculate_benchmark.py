import argparse
import csv
# import numpy as np
# import matplotlib.pyplot as plt
import os.path
from os import path
from utils import benchmark_utils as bu
def parse_args():
    parser = argparse.ArgumentParser(description='Run this script to get a text graph of '
                                                 'SSD model from TensorFlow Object Detection API. '
                                                 'Then pass it with .pb file to cv::dnn::readNetFromTensorflow function.')
    parser.add_argument('--framework', required=True, help='framework.')
    parser.add_argument('--benchtype', required=True, help='mobilenet or ssd')
    parser.add_argument('--memory', required=True, help='[768,1536,2240,3008]')
    parser.add_argument('--path', required=True, help='mobilenet or ssd')
    args = parser.parse_args()
    return args
args = parse_args()
data_handler = bu.BUtils()
benchark_results = data_handler.get_benchmark_from_path(args.path,
                                args.framework,
                                args.benchtype,
                                args.memory, dump_to_file=True)
data_handler.printBenchmark()
