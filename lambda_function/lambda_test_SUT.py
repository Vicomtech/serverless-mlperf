import json
import time
import cv2
from dldt_tools.processing_layer import HandlerApp
def choose_engine_and_format(engine_and_format, cv_task):
    if cv_task == "classification":
        models_path = "mobilenetv1/FP32/"
        if engine_and_format == "ie-ir":
            backend = "mobilenet-ov-runtime"
        elif engine_and_format == "ocv-tf":
            backend = "mobilenet-tf-runtime"
        elif engine_and_format == "ocv-cf":
            backend = "mobilenet-caffe-runtime"
    if cv_task == "object_detection":
        models_path = "ssd-mobilenetv1/FP32/"
        if engine_and_format == "ie-ir":
            backend = "ssd-mobilenet-ov-runtime"
        elif engine_and_format == "ocv-tf":
            backend = "ssd-mobilenet-tf-runtime"
        elif engine_and_format == "ocv-cf":
            backend = "ssd-mobilenet-caffe-runtime"
    backend_handler = HandlerApp(engine_and_format)
    backend_handler.init(backend, models_path, make_profiling = True)
    return backend_handler
t0 = time.time()
engine_and_format = "ie-ir" # ie-ir, ocv-cf, ocv-tf
cv_task = "classification" # classification, object_detection
backend = choose_engine_and_format(engine_and_format, cv_task)

def lambda_handler(event, context):
    init_time = time.time()
    print(event)
    h_var = backend.init_handler_variables(event)
    backend.make_aws_inference(h_var)
    finish_time = time.time()
    h_var["OutputData"]["start_time"] = init_time
    h_var["OutputData"]["finish_time"] = finish_time
    backend.deliver_output_data(h_var)
    return {
        'statusCode': 200,
        'body': json.dumps("hello world")
    }
