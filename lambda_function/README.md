# lambda_test_SUT.py
-------------------------------------------------------------------------------
This lambda function depends on two AWS layers **cv2** and **handler_utils** processing_layer. A lambda function has two scopes, global and function scope, for example:
```python
  import xxxx
  GLOBAL_SCOPE
  def lambda_handler(event, context):
    FUNCTION SCOPE.
```
The interesting part of GLOBAL_SCOPE is that all instances created in this scope, are shared between function instances (FUNCTION_SCOPE).

In the lambda function there is a function called chooseFramework() which defines the type of vision task along with the framework, and then Loads DNN models, initializes datasets (Imagenet, COCO) and loads the selected DNN framework into the GLOBAL SCOPE.  
```python
def chooseFramework(framework, cv_type):
    if cv_type == "classification":
        models_path = "mobilenetv1/FP32/"
        if framework == "openvino":
            backend = "mobilenet-ov-runtime"
        elif framework == "tensorflow":
            backend = "mobilenet-tf-runtime"
        elif framework == "caffe":
            backend = "mobilenet-caffe-runtime"
    if cv_type == "object_detection":
        models_path = "ssd-mobilenetv1/FP32/"
        if framework == "openvino":
            backend = "ssd-mobilenet-ov-runtime"
        elif framework == "tensorflow":
            backend = "ssd-mobilenet-tf-runtime"
        elif framework == "caffe":
            backend = "ssd-mobilenet-caffe-runtime"
    backend_handler = HandlerApp(framework)
    backend_handler.init(backend, models_path, make_profiling = True)
    return backend_handler
```
