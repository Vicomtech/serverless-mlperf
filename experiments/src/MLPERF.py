import json
import argparse
import pathlib
from os import listdir
from os.path import isfile, join
import os, sys
import shutil
import boto3
from utils import benchmark_utils as bu


def create_directory(current_path):
    if os.path.isdir(current_path):
        shutil.rmtree(current_path)
    pathlib.Path(current_path).mkdir(parents=True, exist_ok=True)
    # os.mkdir(current_path, mode=0o777)parents=True, exist_ok=True
#
# --engine_and_format ie-ir --cv_task classification
# --memory_type 0
# --bucket_name your_bucket_name --profile=your_profile
def parse_args():
    parser = argparse.ArgumentParser(description='Run this script to get a text graph of '
                                                 'SSD model from TensorFlow Object Detection API. '
                                                 'Then pass it with .pb file to cv::dnn::readNetFromTensorflow function.')
    parser.add_argument('--engine_and_format', required=True, help='["ie-ir", "ocv-cf", "ocv-tf"].')
    parser.add_argument('--cv_task', required=True, help='[classification, object detection]')
    parser.add_argument('--memory_type', required=True, help='[0,1,2,3]')
    parser.add_argument('--profile', required=True, help='--profile=default')
    parser.add_argument('--bucket_name', required=True, help='your s3 bucketname ')
    args = parser.parse_args()
    return args


# detele the S3 bucket files from completed/output/ and input
def delete_files(profile, bucket_name):
    aws_base = "aws s3 rm "
    aws_folder1 = "s3://" + bucket_name + "/completed/"
    aws_folder2 = "s3://" + bucket_name + "/output/"
    aws_folder3 = "s3://" + bucket_name + "/input/"
    aws_extra = " --recursive --exclude '*' --include '*.json' " + profile
    aws_command1 = aws_base + aws_folder1 + aws_extra
    print(aws_command1)
    aws_command2 = aws_base + aws_folder2 + aws_extra
    print(aws_command2)
    aws_command3 = aws_base + aws_folder3 + aws_extra
    print(aws_command3)
    os.system(aws_command1)
    os.system(aws_command2)
    os.system(aws_command3)


# uploads .json data S3 bucket files trigger the functions.
def warm_up_upload_files(engine_and_format,
                         cv_task,
                         memory,
                         profile,
                         bucket_name,
                         input_folder):
    print("***************system warm up ****************")
    dict_to_folder = {"ie-ir": "IR", "ocv-cf": "CAFFE", "ocv-tf": "TF"}
    target_folder = cv_task + "_" + dict_to_folder[engine_and_format]
    warm_up_iterations = 2
    local_path = os.path.join(os.getcwd(), "experiments",
                              "exper_input",
                              "memory_evolution",
                              target_folder,
                              memory)
    aws_base = "aws s3 cp "
    onlyfiles = [f for f in listdir(local_path) if isfile(join(local_path, f))]
    warm_up_counter = 0
    for file in onlyfiles:
        if warm_up_counter < warm_up_iterations:
            file_to_upload = local_path + "/" + file
            aws_folder = " s3://" + bucket_name + "/" + input_folder
            aws_extra = " --include '*.json'" + profile
            aws_command = aws_base + file_to_upload + " " + aws_folder + aws_extra
            print(aws_command)
            os.system(aws_command)
            warm_up_counter = warm_up_counter + 1


def upload_files(current_engine_and_format,
                 cv_task,
                 memory,
                 profile,
                 bucket_name,
                 input_folder):
    target_folder = generate_target_folder(current_engine_and_format, cv_task)
    local_path = os.path.join(os.getcwd(), "experiments",
                              "exper_input",
                              "memory_evolution",
                              target_folder,
                              memory)
    if local_path == None:
        raise Exception("please provide the local path to your .json files to s3")
    aws_base = "aws s3 cp "
    local_folder = local_path
    aws_folder = " s3://" + bucket_name + "/" + input_folder
    aws_extra = " --recursive --include '*.json' " + profile
    aws_command = aws_base + local_folder + aws_folder + aws_extra
    print(aws_command)
    os.system(aws_command)


def generate_target_folder(current_engine_and_format, cv_task):
    dict_to_folder = {"ie-ir": "IR", "ocv-cf": "CAFFE", "ocv-tf": "TF"}
    target_folder = cv_task + "_" + dict_to_folder[current_engine_and_format]
    return target_folder


def remove_files_and_folders(current_engine_and_format, cv_task, memory):
    target_folder = generate_target_folder(current_engine_and_format, cv_task)
    local_target_path = os.path.join(os.getcwd(), "experiments", "exper_results", "memory_evolution", target_folder,
                                     memory)
    create_directory(local_target_path)


def download_files(current_engine_and_format, cv_task, memory, profile):
    aws_base = "aws s3 cp "
    target_folder = generate_target_folder(current_engine_and_format, cv_task)
    local_target_path = os.path.join(os.getcwd(), "experiments", "exper_results", "memory_evolution", target_folder,
                                     memory)
    if not os.path.isdir(local_target_path):
        raise Exception("bad output path")
    aws_extra = " --recursive --include '*.json' " + profile
    aws_folder = " s3://"+ bucket_name + "/output"
    aws_command = aws_base + aws_folder + " " + local_target_path + aws_extra
    print(aws_command)
    os.system(aws_command)
    return local_target_path

def warmup(current_engine_and_format,
           current_bench_type,
           profile,
           bucket_name,
           current_memory,
           S3_trigger_data_input_folder):
    print("--------MLPERF WARMUP-------------")
    delete_files(profile, bucket_name)
    warm_up_upload_files(current_engine_and_format, current_bench_type,
                         current_memory,
                         profile,
                         bucket_name,
                         S3_trigger_data_input_folder)
    duration = 15
    remove_files_and_folders(current_engine_and_format, current_bench_type, current_memory)
    os.system("sleep " + str(duration))
    print ("warm-up finished")

def execute_benchmark(current_engine_and_format,
                      current_bench_type,
                      profile,
                      bucket_name,
                      current_memory,
                      S3_trigger_data_input_folder):
    delete_files(profile, bucket_name)
    upload_files(current_engine_and_format, current_bench_type,
                 current_memory,
                 profile,
                 bucket_name,
                 S3_trigger_data_input_folder)
    duration = 150
    remove_files_and_folders(current_engine_and_format, current_bench_type, current_memory)
    os.system("sleep " + str(duration))
    return download_files(current_engine_and_format, current_bench_type, current_memory, profile)

args = parse_args()
memory = ["768", "1536", "2240", "3008"]


if int(args.memory_type) >= len(memory) and int(args.memory_type):
    raise Exception("bad argument entered to memory please provide [0,1,2,3]")
current_memory = memory[int(args.memory_type)]
if args.engine_and_format == "ie-ir" or args.engine_and_format == "ocv-cf" or args.engine_and_format == "ocv-tf":
    current_engine_and_format = args.engine_and_format
else:
    raise Exception("bad argument entered to engine_and_format: current_value {} options [ov, tf,caffe]")
if args.cv_task == "classification" or args.cv_task == "object_detection":
    current_cv_task = args.cv_task
else:
    raise Exception("bad argument entered to engine_and_format: current_value {} options [ov, tf,caffe]")
bucket_name = args.bucket_name
profile = "--profile=" + args.profile
S3_trigger_data_input_folder = "input/"

warmup(current_engine_and_format,
       current_cv_task,
       profile,
       bucket_name,
       current_memory,
       S3_trigger_data_input_folder)
result_data_folder = execute_benchmark(current_engine_and_format,
                                       current_cv_task,
                                       profile,
                                       bucket_name,
                                       current_memory,
                                       S3_trigger_data_input_folder)
data_handler = bu.BUtils()
benchark_results = data_handler.get_benchmark_from_path(result_data_folder,
                                                        current_engine_and_format,
                                                        current_cv_task,
                                                        memory[int(args.memory_type)],
                                                        dump_to_file=True)
data_handler.printBenchmark()
