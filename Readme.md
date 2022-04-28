# Yolov5-Executor

This is a [SegServe](https://github.com/hip-satomi/SegServe) executor for the [Yolov5](https://github.com/ultralytics/yolov5.git) object detection method.

## Local testing

Make sure you have [anaconda](https://www.anaconda.com/products/distribution) installed and an active environment with [`mlflow`](https://pypi.org/project/mlflow/). Then execute
```bash
pip install mlflow
mlflow run ./ -e main -P input_images=<path/to/your/image> -P model=<yoloc5 model name, e.g. yolov5n>
```
The resulting object detection should be written to `output.json` and logged as an artifact in the mlflow run.

## Intended Usage

The wrapper is used to deploy the Yolov5 methods in the [SegServe](https://github.com/hip-satomi/SegServe) runtime environment. SegServe can be used to host 3rd party segmentation/detection algorithms and execute them on a central computer while providing a REST interface for clients. Therefore, end-users do not need any powerful hardware/GPU.
