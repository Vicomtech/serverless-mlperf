import json
import argparse
from os import listdir
from os.path import isfile, join
import os
import shutil
import pathlib


def parse_args():
    parser = argparse.ArgumentParser(description='Run this script to generate the LOADGEN experiment input ')
    parser.add_argument('--num', required=True, help='the quantity of the input images.')
    parser.add_argument('--input_dataset', required=True, help='where the dataset images are located')
    parser.add_argument('--dataset_name', required=True, help='which is the database name coco or imagenet')
    args = parser.parse_args()
    return args


def generate_folders(dataset_name):
    output_paths = []
    path = os.path.join(os.getcwd(), "experiments", "exper_input")
    memory = ["768", "1536", "2240", "3008"]
    model = ["IR", "TF", "CAFFE"]
    type = {"imagenet":"classification", "coco":"object_detection"}

    folder = "memory_evolution"
    total_path = os.path.join(path, folder)
    if os.path.isdir(total_path):
        print("directory exists")
    else:
        create_directory(total_path)

    print("creating {} folder in {}".format(folder, path))
    # for t in type[dataset_name]:
    for fr in model:
        bench_folder_name = type[dataset_name] + "_" + fr
        create_directory(os.path.join(total_path, bench_folder_name))
        for mem in memory:
            current_path = os.path.join(total_path, bench_folder_name, mem)
            create_directory(current_path)
            output_paths.append(current_path)
    return output_paths


def create_directory(current_path):
    if os.path.isdir(current_path):
        shutil.rmtree(current_path)
    pathlib.Path(current_path).mkdir(parents=True, exist_ok=True)


def generate_json_files(output_path_list,
                        num,
                        db_images_path,
                        db_name):
    db_files_path = db_images_path
    counter = 0
    if not os.path.exists(db_images_path):
        raise Exception("The image path for dataset images does not exist.")
    onlyfiles = [f for f in listdir(db_files_path) if isfile(join(db_files_path, f))]
    for file in onlyfiles:
        if counter <= num:
            for out_path in output_path_list:
                decomposed_path = out_path.split("/")
                current_value = decomposed_path[len(decomposed_path) - 2].split("_")
                completed_path, input_path, output_path = create_json_parameters(db_name,
                                                                                 current_value[0],
                                                                                 current_value[1])
                if db_name == "coco" and current_value[0] == "object":
                    generate_and_write_json_file(completed_path, file, input_path, out_path, output_path)
                if db_name == "imagenet" and current_value[0] == "classification":
                    generate_and_write_json_file(completed_path, file, input_path, out_path, output_path)
        counter = counter + 1


def generate_and_write_json_file(completed_path, file, input_path, out_path, output_path):
    json_results = dict()
    json_results["OutputPath"] = output_path
    json_results["CompletedPath"] = completed_path
    json_results["ImageFilenames"] = []
    json_results["ImageFilenames"].append(input_path + "/" + file)
    json_filename = os.path.join(out_path, file.split(".")[0] + ".json")
    with open(json_filename, 'w') as fp:
        json.dump(json_results, fp)


def create_json_parameters(db_name,
                           cv_task,
                           engine_and_format):
    input_path = db_name
    output_path = "output/"
    completed_path = "completed/"
    # output_path = "output/" + cv_task + "_" + engine_and_format
    # completed_path = "completed/" + cv_task + "_" + engine_and_format
    return completed_path, input_path, output_path


args = parse_args()
if (args.dataset_name != "coco") and (args.dataset_name != "imagenet"):
    raise Exception("unknown database name {} [use coco or imagenet values]".format(args.dataset_name))
memory = ["768", "1536", "2240", "3008"]
model = ["IR", "TF", "CAFFE"]
output_path_list = generate_folders(args.dataset_name )
generate_json_files(output_path_list,
                    int(args.num),
                    args.input_dataset,
                    args.dataset_name)
