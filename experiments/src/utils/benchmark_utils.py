import os
from os import listdir
from os.path import isfile, join
import json
import numpy as np
import argparse
import csv
import numpy as np
import time
# import matplotlib.pyplot as plt
# # import matplotlib.axes.Axes.boxplot
# # import matplotlib.pyplot.boxplot
from datetime import datetime


def prepare_data_for_latencies_barchart(path_to_file, prefix="latency"):
    return_data = dict()
    mobilenet_openvino = []
    mobilenet_tf = []
    mobilenet_caffe = []
    ssd_openvino = []
    ssd_tf = []
    ssd_caffe = []
    if os.path.isfile(path_to_file):
        with open(path_to_file, mode='r') as csv_file:
            reader = csv.DictReader(csv_file)
            line_count = 0
            for row in reader:
                if row["model_name"].find("mobilenet_openvino") >= 0:
                    mobilenet_openvino.append(round(float(row[prefix]),1))
                if row["model_name"].find("mobilenet_tf") >= 0:
                    mobilenet_tf.append(round(float(row[prefix]),1))
                if row["model_name"].find("mobilenet_caffe") >= 0:
                    mobilenet_caffe.append(round(float(row[prefix]),1))
                if row["model_name"].find("ssd_openvino") >= 0:
                    ssd_openvino.append(round(float(row[prefix]),1))
                if row["model_name"].find("ssd_tf") >= 0:
                    ssd_tf.append(round(float(row[prefix]),1))
                if row["model_name"].find("ssd_caffe") >= 0:
                    ssd_caffe.append(round(float(row[prefix]),1))
                line_count = line_count + 1
            print("lines {}".format(line_count))
            return {"mobilenet": {"openvino": mobilenet_openvino,
                                  "tf": mobilenet_tf,
                                  "caffe": mobilenet_caffe},
                    "ssd": {
                        "openvino": ssd_openvino,
                        "tf": ssd_tf,
                        "caffe": ssd_caffe,
                    }
                    }
    else:
        raise Exception("wrong csv file please check if the path is correct {}".format(path_to_file))


class BUtils():
    def calculate_benchmark(self, framework,
                            model,
                            memory,
                            number_of_files,
                            latencies,
                            start_times,
                            end_times):

        return_result = dict()
        total_queries = number_of_files
        if len(latencies) > 0:
            latency_90 = np.percentile(latencies, 90)
        else:
            latency_90 = 0
            print("WARNING. latency array empty")

        if len(start_times) > 0 and len(end_times) > 0:
            minimum_start = np.min(start_times)
            finish_end = np.max(end_times)
            latency_variance = np.std(latencies)
            latency_50 = np.percentile(latencies, 50)
            time_difference = float((finish_end - minimum_start))

            qps = (float)(total_queries / time_difference)
        else:
            print("WARNING. time array empty")
            minimum_start = 0
            finish_end = 1
            latency_50 = 0
            latency_variance = 0
            qps = 0

        return_result["framework"] = framework
        return_result["model"] = model
        return_result["memory"] = memory
        return_result["latencies"] = {"latency_50": latency_50,
                                      "latency_90": latency_90,
                                      "variance": latency_variance,
                                      "latency_list": latencies}
        return_result["throughput"] = {"QPS": qps,
                                       "min_start": minimum_start,
                                       "max_end": finish_end,
                                       "start_time_list": start_times,
                                       "finish_time_list": end_times}
        return return_result

    def get_benchmark_from_path(self, path,
                                framework,
                                model,
                                memory,
                                dump_to_file=False):
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        self.number_of_files = len(onlyfiles)
        latencies = []
        start_times = []
        end_times = []
        file_counter = 0
        for file in onlyfiles:
            if ".json" in file:
                with open(path + "/" + file, 'r') as json_file:
                    data = json.load(json_file)
                    if data is None:
                        raise Exception("data is none in {}".format(file))
                    # print("reading {} numfile {}".format(file, file_counter))
                    latencies.append(data["inf_perf"][0]["forward"])
                    start_times.append(data["start_time"])
                    end_times.append(data["finish_time"])
                file_counter = file_counter + 1
        if len(onlyfiles) > 0:
            self.return_result = self.calculate_benchmark(
                framework,
                model,
                memory,
                self.number_of_files,
                latencies,
                start_times,
                end_times)
        else:
            self.return_result = self.get_dummy_benchmark_model(framework, model, memory)
        if dump_to_file:
            self.dumpBenchmarkToFile(self.return_result, model, framework, memory, )

        return self.return_result

    def dumpBenchmarkToFile(self, results, model, fr, memory):
        csv_file = model + "_" + fr + "_" + memory + ".csv"
        csv_columns = ["latency", "start_time", "end_time", "formated_start", "formated_end"]
        latencies_list = results["latencies"]["latency_list"]
        start_time_list = results["throughput"]["start_time_list"]
        end_time_list = results["throughput"]["finish_time_list"]
        with open(csv_file, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(csv_columns)
            for i in range(0, len(latencies_list)):
                writer.writerow([latencies_list[i],
                                 start_time_list[i],
                                 end_time_list[i],
                                 time.ctime(start_time_list[i]),
                                 time.ctime(end_time_list[i]),
                                 ])

    def get_dummy_benchmark_model(self, framework, model, memory):
        return_result = dict()
        return_result["framework"] = framework
        return_result["model"] = model
        return_result["memory"] = memory
        return_result["latencies"] = {"mean": 0,
                                      "latency_90": 0,
                                      "variance": [],
                                      "latency_list": 0}
        return_result["throughput"] = {"QPS": 0,
                                       "min_start": 0,
                                       "max_end": 0,
                                       "start_time_list": [],
                                       "finish_time_list": []}

    def printBenchmark(self):
        print("**********benchmarked unit***********")
        print("benchmark {}_{}".format(self.return_result["model"], self.return_result["framework"]))
        print("memory {}".format(self.return_result["memory"]))
        print("latencyes 50% -> {} 90% -> {}".format(self.return_result["latencies"]["latency_50"],
                                                     self.return_result["latencies"]["latency_90"]))
        print("QPS: {}".format(self.return_result["throughput"]["QPS"]))
        print("number of files: {}".format(self.number_of_files))
        print("Time difference {} s".format(
            self.return_result["throughput"]["max_end"] - self.return_result["throughput"]["min_start"]))
