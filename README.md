<!-- markdownlint-disable MD033 -->

# Using the IBM Z Accelerated for NVIDIA Triton™ Inference Server Container Image

# Table of contents

- [Overview](#overview)
- [Downloading the IBM Z Accelerated for NVIDIA Triton™ Inference server Container Image](#container)
- [Container Image Contents](#contents)
- [IBM Z Accelerated for NVIDIA Triton™ Inference Server container image usage](#launch-container)
- [Security and Deployment Guidelines](#security-and-deployment-guidelines)
- [IBM Z Accelerated for NVIDIA Triton™ Inference Server Backends](#triton-server-backends)
- [Triton Model Analyzer](#triton-model-analyzer)
- [REST APIs](#triton-server-restapi)
- [MLflow Integration](#mlflow-integration)
- [Model Validation](#model-validation)
- [Using the Code Samples](#code-samples)
- [Additional Topics](#additional-topics)
- [Limitations and Known Issues](#limitations-known-issues)
- [Versions and Release Cadence](#versioning)
- [Frequently Asked Questions](#faq)
- [Technical Support](#contact)
- [Licenses](#licenses)

# Overview <a id="overview"></a>

[Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2410/user-guide/docs/user_guide/architecture.html) is
an open source fast, scalable, and open-source AI inference server, by
standardizing model deployment and execution streamlined and optimized for high
performance. The Triton Inference Server can deploy AI models such as deep
learning (DL) and machine learning (ML).

Client program initiates the inference request to Triton Inference Server.
Inference requests arrive at the server via either HTTP/REST or GRPC or by the C
API and are then routed to the appropriate per-model scheduler. Triton
implements multiple scheduling and batching algorithms that can be configured
per model-by-model basis. Each model's scheduler optionally performs batching of
inference requests and then passes the requests to the corresponding backend
module based on the model type. The backend performs inferencing using the
inputs (duly pre-processed) to produce the requested outputs. The outputs are
then returned to the client program.

The Triton Inference Server provides a backend API that exposes a C API along with backend libraries and frameworks in an optimized way, enabling support for ONNX-MLIR on IBM Z.

Triton Inference Server python backend enables pre-processing, post-processing
and server models written in python programming language. This enables IBM Snap
ML support on IBM Z.


The models being served by Triton Inference Server can be queried and controlled
by a dedicated model management API that is available by HTTP/REST or GRPC
protocol, or by the C API.

The model repository is a file-system based repository of the models that Triton Inference Server will make available for a
deployment.


Please visit the section
[Downloading the IBM Z Accelerated for NVIDIA Triton™ Inference Server container image](#container)<!--markdownlint-disable-line MD013 -->
to get started.

# Downloading the IBM Z Accelerated for NVIDIA Triton™ Inference Server container image <a id="container"></a> <!--markdownlint-disable-line MD013 -->

Downloading the IBM Z Accelerated for NVIDIA Triton™ Inference Server container
image requires credentials for the IBM Z and LinuxONE Container Registry,
[icr.io](https://icr.io).

Documentation on obtaining credentials to `icr.io` is located
[here](https://ibm.github.io/ibm-z-oss-hub/main/main.html).

---

Once credentials to `icr.io` are obtained and have been used to login to the
registry, you may pull (download) the IBM Z Accelerated for NVIDIA Triton™
Inference Server container image with the following code block:

```bash
docker pull icr.io/ibmz/ibmz-accelerated-for-nvidia-triton-inference-server:X.Y.Z
```

In the `docker pull` command illustrated above, the version specified above is
`X.Y.Z`. This is based on the version available in the
[IBM Z and LinuxONE Container Registry](https://ibm.github.io/ibm-z-oss-hub/containers/ibmz-accelerated-for-nvidia-triton-inference-server.html).

---

To remove the IBM Z Accelerated for NVIDIA Triton™ Inference Server container
image, please follow the commands in the code block:

```bash
# Find the Image ID from the image listing
docker images

# Remove the image
docker rmi <IMAGE ID>
```

---

\*_Note. This documentation will refer to image/containerization commands in
terms of Docker. If you are utilizing Podman, please replace `docker` with
`podman` when using our example code snippets._

# Container Image Contents <a id="contents"></a>

To view a brief overview of the operating system version, software versions and
content installed in the container, as well as any release notes for each
released container image version, please visit the `releases` section of this
GitHub Repository, or you can click
[here](https://github.com/IBM/ibmz-accelerated-for-nvidia-triton-inference-server/releases/).

# IBM Z Accelerated for NVIDIA Triton™ Inference Server container image usage <a id="launch-container"></a> <!--markdownlint-disable-line MD013 -->

For documentation on how models can be served with Triton Inference Server, please visit the official 

[Open Source Triton Inference Server documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2410/user-guide/docs/index.html).

For brief examples on deploying models with Triton Inference Server, please
visit our [samples section](#code-samples)

## Launch IBM Z Accelerated for NVIDIA Triton™ Inference Server container

Launching and maintaining IBM Z Accelerated Triton™ Inference Server revolves
around official

[quick start](https://github.com/triton-inference-server/server/blob/r24.07/docs/getting_started/quickstart.md)
tutorial.

This documentation will cover :

- [Creating a Model Repository](#create-model-repository)
- [Launching IBM Z Accelerated for NVIDIA Triton™ Inference Server](#start-triton-server)
- [Deploying the model](#security-and-deployment-guidelines)

## Creating a Model Repository <a id="create-model-repository"></a>

User can follow the steps describe at
[model repository](https://github.com/triton-inference-server/server/blob/r24.07/docs/user_guide/model_repository.md)
to create a model repository. The following steps would launch the Triton
Inference Server.

## Launching IBM Z Accelerated for NVIDIA Triton™ Inference Server <a id="start-triton-server"></a>

By default services of IBM Z Accelerated for NVIDIA Triton™ Inference Server
docker container listening at the following ports.

| Service Name                   | Port |
| ------------------------------ | :--: |
| HTTP – Triton Inference Server | 8000 |
| GRPC – Triton Inference Server | 8001 |
| HTTP – Metrics                 | 8002 |

IBM Z Accelerated for NVIDIA Triton™ Inference Server can be launched by
running the following command.

```bash
docker run --shm-size 1G --rm \
    -p <EXPOSE_HTTP_PORT_NUM>:8000 \
    -p <EXPOSE_GRPC_PORT_NUM>:8001 \
    -p <EXPOSE_Metrics_PORT_NUM>:8002 \
    -v $PWD/models:/models \
    <triton_inference_server_image> tritonserver \
    --model-repository=/models
```

Use IBM Z Accelerated for NVIDIA Triton™ Inference Server's REST API endpoint
to verify if the server and the models are ready for inferencing. From the host
system use curl to access the HTTP endpoint that provides server status.

```bash
curl -v localhost:8000/v2/health/ready
```

```bash
...
< HTTP/1.1 200 OK
< Content-Length: 0
< Content-Type: text/plain
```

The HTTP request returns status 200 if IBM Z Accelerated for NVIDIA Triton™
Inference Server is ready and non-200 if it is not ready.

# Security and Deployment Guidelines <a id="security-and-deployment-guidelines"></a>

For security and deployment best practices, please visit the common AI Toolkit documentation found [here](https://github.com/IBM/ai-toolkit-for-z-and-linuxone?tab=readme-ov-file#recommended-deployment-guidelines-)

## Triton Inference Server using HTTPS/Secure gRPC

Configuring Triton Inference Server for HTTPS and gRPC Secure ensures secure and
encrypted communication channels, protecting clients and server data
confidentiality, integrity, and authenticity.

### HTTPS

HTTPS (Hypertext Transfer Protocol Secure) is a secure version of the HTTP
protocol used for communication between clients (such as web browsers) and
servers over the internet. It provides encryption and secure data transmission
by using SSL/TLS (Secure Sockets Layer/Transport Layer Security) protocols.

Reverse proxy servers can help secure Triton Inference Server communications
with HTTPS by protecting backend servers and enabling secure communication and
performance required to handle incoming request well.

The HTTPS protocol ensures that the data exchanged between the client and server
is encrypted and protected from eavesdropping or tampering. By configuring
reverse proxy server with HTTPS, you enable secure communication between the
Triton Inference Server and clients, ensuring data confidentiality and
integrity.

### Secure gRPC

Triton Inference Server supports the gRPC protocol, which is a high-performance.
gRPC provides efficient and fast communication between clients and servers,
making it ideal for real-time inferencing scenarios.

Using gRPC with Triton Inference Server offers benefits such as high
performance, bidirectional streaming, support for client and server-side
streaming, and automatic code generation for client and server interfaces.

**SSL/TLS**: gRPC support SSL/TLS and the use of SSL/TLS to authenticate the
server, and to encrypt all the data exchanged between the gRPC client and the
Triton Inference Server. Optional mechanisms are available for clients to
provide certificates for mutual authentication


# IBM Z Accelerated for NVIDIA Triton™ Inference Server Backends <a id="triton-server-backends"></a>

IBM Z Accelerated for NVIDIA Triton™ Inference Server supports four backends as
of today.

- [Python Backend](#python-backend)
- [ONNX-MLIR Backend](#onnx-mlir-backend)
- [Snap ML C++ Backend](#snapml-cpp-backend)
- [PyTorch Backend](#pytorch-backend)

## Python Backend <a id="python-backend"></a>

Triton Inference Server supports the deployment of AI models or any logic written in Python via python backend. 

For more details about triton python backend are documented [here](https://github.com/triton-inference-server/python_backend/tree/r24.07?tab=readme-ov-file#user-documentation)

Format of the python backend model directory looks like below

```text
$ model_1
   |-- 1
   |   |-- model.py
   |   `-- model.txt
   `-- config.pbtxt
```

### Minimal Model Configuration
Every Python Backend model must provide config.pbtxt file describing the model configuration.
Below is a sample ```config.pbtxt``` for Python Backend:

```
max_batch_size: 32
input {
  name: "IN0"
  data_type: TYPE_FP32
  dims: 5
}
output {
  name: "OUT0"
  data_type: TYPE_FP64
  dims: 1
}
backend: "python"
```
#### Configuration Parameters:
Triton Inference Server exposes some flags to control the execution mode of models through parameters section in the model’s config.pbtxt file.

- **Backend** :
   Backend parameter must be provided as “python” while utilising Python Backend. For more details related to backend [here](https://github.com/triton-inference-server/backend/blob/r24.07/README.md#backends)
   
   ```
   backend: "python"
   ```
   
- **Inputs and Outputs:** : 
    Each model input and output must specify a name, datatype, and shape. The name specified for an input or output tensor must match the name expected by the model. For more details on inputs and output tensors check documentation of Triton Inference server [here](https://github.com/triton-inference-server/server/blob/r24.07/docs/user_guide/model_configuration.md#inputs-and-outputs)
    
For more options see [Model Configuration](https://github.com/triton-inference-server/server/blob/r24.07/docs/user_guide/model_configuration.md#model-configuration).
    
Using the Python backend in Triton is especially useful for deploying custom
models or models developed with specific libraries that are not natively
supported by Triton's other backends. It provides a flexible way to bring in
your own AI model and integrate it with the server's inference
capabilities. It provides flexibility by allowing you to use any Python machine learning/deep learning library to define your model's inference logic.

NOTE:

1. model.py should be there in the model repository to use the python backend framework.
2. Multiple versions are supported, only positive values as version model are supported

### SHAP and LIME Explainability (Python Backend)

The Python backend has no built-in SHAP or LIME support. Explainability must be implemented directly in your `model.py` using the `shap` and `lime` Python packages. You have full control over the implementation — how the background dataset is constructed, which explainer settings are used, and how attribution values are assembled into output tensors.

#### Required config.pbtxt changes

Declare `shap_values` and `lime_values` as explicit output tensors alongside your model's primary outputs:

```
backend: "python"
max_batch_size: 32

input [{
  name: "IN0"
  data_type: TYPE_FP32
  dims: [ 30 ]
}]

output [
  { name: "OUT0"       data_type: TYPE_FP64  dims: [ 1  ] },
  { name: "shap_values" data_type: TYPE_FP64  dims: [ 30 ] },
  { name: "lime_values" data_type: TYPE_FP64  dims: [ 30 ] }
]
```

The number of features (`30` above) must match your model's input dimension. The tensor names `shap_values` and `lime_values` are conventions — use any names that are consistent between `config.pbtxt` and your `model.py`.

#### Example model.py structure

```python
import numpy as np
import triton_python_backend_utils as pb_utils

NF = 30  # number of input features

class TritonPythonModel:

    def initialize(self, args):
        # Load your model here (e.g. from a pickle or checkpoint file).
        # Import shap and lime inside initialize() so a missing package
        # produces a clear warning without preventing the model from loading.
        try:
            import shap as _shap
            self._shap = _shap
            self._has_shap = True
        except ImportError:
            self._has_shap = False

        try:
            import lime.lime_tabular as _lime
            self._lime = _lime
            self._has_lime = True
        except ImportError:
            self._has_lime = False

    def execute(self, requests):
        responses = []
        for request in requests:
            x = pb_utils.get_input_tensor_by_name(request, "IN0") \
                        .as_numpy().astype(np.float32)
            n = x.shape[0]

            # --- Primary inference ---
            labels = self._model.predict(x).astype(np.float64)

            # --- Build a predict_fn that returns a scalar score per sample ---
            def predict_fn(x_mat):
                proba = self._model.predict_proba(
                    np.asarray(x_mat, dtype=np.float32))
                return proba[:, 1].astype(np.float64)  # score for class 1

            # --- SHAP ---
            if self._has_shap:
                # Use the inference batch as the background dataset.
                explainer = self._shap.KernelExplainer(predict_fn, x)
                raw = explainer.shap_values(x, nsamples=200, silent=True)
                shap_arr = np.array(
                    raw[1] if isinstance(raw, list) else raw,
                    dtype=np.float64)
            else:
                shap_arr = np.zeros((n, NF), dtype=np.float64)

            # --- LIME ---
            if self._has_lime:
                lime_exp = self._lime.LimeTabularExplainer(
                    training_data=x,
                    mode="classification",
                    discretize_continuous=False,
                )
                lime_arr = np.zeros((n, NF), dtype=np.float64)
                for i in range(n):
                    exp = lime_exp.explain_instance(
                        x[i], predict_fn, num_features=NF, num_samples=1000,
                        labels=(1,))
                    for fi, coef in exp.local_exp[1]:
                        lime_arr[i, fi] = coef
            else:
                lime_arr = np.zeros((n, NF), dtype=np.float64)

            # --- Assemble response ---
            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("OUT0",
                    labels.reshape(n, 1).astype(np.float64)),
                pb_utils.Tensor("shap_values", shap_arr),
                pb_utils.Tensor("lime_values",  lime_arr),
            ]))
        return responses

    def finalize(self):
        pass
```

Key points:

- Import `shap` and `lime` inside `initialize()`, not at module level. A missing package logs a warning and falls back to zero-filled output rather than preventing the model from loading.
- The `predict_fn` must accept a 2-D numpy array `[n_samples, n_features]` and return a 1-D array of scores.
- The inference batch is used as the background dataset for both SHAP and LIME, so no separate training dataset needs to be bundled with the model.
- Per-sample LIME failures are caught inside the per-sample loop; the affected row remains zero while the rest of the batch continues.
- Attribution tensors must be `TYPE_FP64` (`float64`) to match the convention used by the C++ backends.

## ONNX-MLIR Backend <a id="onnx-mlir-backend"></a>

A triton backend which allows the deployment of onnx-mlir or zDLC compiled
models (model.so) with the triton inference server. 

Format of the onnx-mlir backend model directory looks like below

```text
$ model_1
    |-- 1
    |   `-- model.so
    `-- config.pbtxt
```

### Minimal Model Configuration
Every ONNX-MLIR Backend model must provide config.pbtxt file describing the model configuration.
Below is a sample ```config.pbtxt``` for ONNX-MLIR Backend:

```
max_batch_size: 32
input {
  name: "IN0"
  data_type: TYPE_FP32
  dims: 5
  dims: 5
  dims: 1
}
output {
  name: "OUT0"
  data_type: TYPE_FP64
  dims: 1
}
backend: "onnxmlir"
```
#### Configuration Parameters:

- **Backend** :
   Backend parameter must be provided as “onnxmlir” while utilising ONNX-MLIR Backend. For more details related to backend [here](https://github.com/triton-inference-server/backend/blob/r24.07/README.md)
   
   ```
   backend: "onnxmlir"
   ```
   
- **Inputs and Outputs:** : 
    Each model input and output must specify a name, datatype, and shape. The name specified for an input or output tensor must match the name expected by the model. For more details on inputs and output tensors check documentation of Triton Inference server [here](https://github.com/triton-inference-server/server/blob/r24.07/docs/user_guide/model_configuration.md#inputs-and-outputs)
    
For more options see [Model Configuration](https://github.com/triton-inference-server/server/blob/r24.07/docs/user_guide/model_configuration.md).

NOTE: Multiple versions are supported, only positive values as version model are supported

### SHAP and LIME Explainability (ONNX-MLIR Backend)

The ONNX-MLIR backend implements SHAP and LIME explanations directly in C++. Both methods are model-agnostic and work with any model compiled by ONNX-MLIR or IBM zDLC.

#### How it works

When `COMPUTE_SHAP` or `COMPUTE_LIME` is set to `"true"` in `config.pbtxt`, the backend:

1. Runs normal model inference to produce the primary output.
2. Re-executes the model across coalition-masked inputs (SHAP) or Gaussian-noise-perturbed inputs (LIME).
3. Writes the resulting `float64` attribution arrays into additional output tensors alongside the primary model output.

Explanations are computed **synchronously** on the same inference path as the primary prediction. If SHAP or LIME computation fails, the backend logs a warning, zero-fills the attribution output tensor, and still returns the primary prediction. The request is **not** failed.

**Algorithm details:**
- **SHAP**: Coalition-based attribution with full enumeration for ≤ 12 features; randomised paired-complement sampling up to 4096 coalitions for larger feature spaces. The background/reference baseline is the current inference batch itself (N > 1) or a zero-vector (N = 1).
- **LIME**: Local linear surrogate model. Perturbs each input sample with Gaussian noise and fits a weighted linear model. Default: 256 perturbation samples. The perturbation set is data-dependent but reproducible for the same input.

#### XAI output tensor registration

XAI output tensors are **automatically registered** by the backend. You do not need to declare them in the `output` block of `config.pbtxt`. If you do declare them (e.g. to control the tensor name), the backend will patch their dimensions to match the number of input features.

If a disabled XAI tensor name remains declared in `config.pbtxt` but its corresponding flag is not set, it will be stripped from the output tensor list during model loading to avoid a configuration mismatch.

#### Configuration parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `COMPUTE_SHAP` | `string` | `"false"` | Set to `"true"` to compute and return SHAP attributions. Case-sensitive. |
| `COMPUTE_LIME` | `string` | `"false"` | Set to `"true"` to compute and return LIME attributions. Case-sensitive. |
| `SHAP_OUTPUT_TENSOR_NAME` | `string` | `"shap_values"` | Output tensor name for SHAP results. |
| `LIME_OUTPUT_TENSOR_NAME` | `string` | `"lime_values"` | Output tensor name for LIME results. |

#### Output tensor specification

Both XAI output tensors are always `TYPE_FP64`. Their shape is `[-1, num_features]` when `max_batch_size > 0`, or `[num_features]` for non-batched models. `num_features` is the product of all non-batch dimensions of the first declared input tensor.

#### Example config.pbtxt with SHAP and LIME enabled

The backend auto-registers XAI output tensors if they are not declared. Declaring them explicitly (as shown below) is recommended so the shape is visible in the configuration and can be validated. Use `dims: [-1]` — the backend patches the real feature count at first inference.

The tensor names `shap_values` and `lime_values` are the defaults. They are configurable via `SHAP_OUTPUT_TENSOR_NAME` and `LIME_OUTPUT_TENSOR_NAME` — the declared output name must match the parameter value exactly.

```
backend: "onnxmlir"
max_batch_size: 16

input {
  name: "IN0"
  data_type: TYPE_FP32
  dims: [ 30 ]
}

output {
  name: "OUT0"
  data_type: TYPE_FP32
  dims: [ 2 ]
}
output {
  # SHAP attributions — one value per input feature.
  # dims: [-1] is resolved to the actual feature count at first inference.
  name: "shap_values"
  data_type: TYPE_FP64
  dims: [ -1 ]
}
output {
  # LIME coefficients — one value per input feature.
  name: "lime_values"
  data_type: TYPE_FP64
  dims: [ -1 ]
}

parameters {
  key: "COMPUTE_SHAP"
  value { string_value: "true" }
}
parameters {
  key: "COMPUTE_LIME"
  value { string_value: "true" }
}
parameters {
  # Name of the SHAP output tensor (default: "shap_values").
  # Must match the output declaration name above.
  key: "SHAP_OUTPUT_TENSOR_NAME"
  value { string_value: "shap_values" }
}
parameters {
  # Name of the LIME output tensor (default: "lime_values").
  # Must match the output declaration name above.
  key: "LIME_OUTPUT_TENSOR_NAME"
  value { string_value: "lime_values" }
}
```

Clients request XAI tensors by name in the `outputs` block of the inference request. Omitting an XAI tensor name from `outputs` causes Triton to omit that tensor from the response, which reduces transfer overhead when only the primary prediction is needed.

#### Inference request example

```bash
curl -X POST http://localhost:8000/v2/models/my_onnxmlir_model/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "IN0",
        "shape": [1, 30],
        "datatype": "FP32",
        "data": [[0.1, 0.2, 0.3, 0.4, 0.5,
                  0.6, 0.7, 0.8, 0.9, 1.0,
                  1.1, 1.2, 1.3, 1.4, 1.5,
                  1.6, 1.7, 1.8, 1.9, 2.0,
                  2.1, 2.2, 2.3, 2.4, 2.5,
                  2.6, 2.7, 2.8, 2.9, 3.0]]
      }
    ],
    "outputs": [
      {"name": "OUT0"},
      {"name": "shap_values"}
    ]
  }'
```

The response will include both the primary model output and the `shap_values` tensor with shape `[1, 30]`.

#### Limitations

- Only the first input tensor is used as the explainability input. Models with multiple inputs are inferred normally, but explanations are computed against the first input only.
- Input must have data type `TYPE_FP32`. Other input types are not supported for XAI.
- SHAP and LIME are computed over the **flattened** feature space of the first input. Spatial structure (e.g. HWC layout for images) is not preserved.
- Classification output format is assumed to have at least 2 output values per sample. For such models the backend reports the positive-class (index 1) prediction for the SHAP/LIME perturbation loop. For regression (1 output value), the single output is used.
- There is no background/reference dataset cache. Each inference request uses the current batch as its own background. Single-sample batches (N=1) fall back to a zero-vector reference, which may reduce SHAP accuracy.


## Snap ML C++ Backend <a id="snapml-cpp-backend"></a>

Snap Machine Learning (Snap ML in short) is a library for training and scoring traditional machine learning models with end to end inferencing capabilities. As part of the preprocessing pipeline it supports Normalizer, Kbinsdiscritizer, TargetEncoder, OrdinalEncoder, and OneHotEncoder  transformers. 
This backend supports importing tree ensembles models that were trained with other frameworks (e.g., scikit-learn, XGBoost, LightGBM) so one can leverage Integrated On-Chip Accelerator on IBM Z and IBM LinuxONE transparently via Snap ML’s accelerated inference engine. For more information on supported models refer Snap ML documentation [here](https://snapml.readthedocs.io/en/latest/model_import.html).

### Backend Usage Information
In Order to deploy any model on Triton Inference Server, one should have a model repository and Model configuration ready. 

Model Configuration: The model configuration  ```(config.pbtxt)``` in Triton Inference Server defines metadata, optimization settings, and customised  parameters for each model. This configuration ensures models are served with optimal performance and tailored behaviour.For more details [Model Configuration](https://github.com/triton-inference-server/server/blob/r24.07/docs/user_guide/model_configuration.md).

### Models Repository
Like all Triton backends, models deployed via the Snap ML C++ Backend make use of a specially laid-out "model repository" directory containing at least one serialized model and a "config.pbtxt" configuration file.

Typically, a models directory should look like below: 

```
models
└── test_snapml_model
    │   ├── 1
    │       ├── model.pmml
    │       └── pipeline.json
    └── config.pbtxt

```

Given above is sample for ```pmml``` format. User should change the model formats as per the model framework chosen.

pipeline.json: This is optional and required to be provided only when pre-processing is chosen. 

### Model Configuration
Every Snap ML C++ Backend model must provide config.pbtxt file describing the model configuration.
Below is a sample ```config.pbtxt``` for Snap ML C++ Backend:

```
max_batch_size: 32
input {
  name: "IN0"
  data_type: TYPE_FP32
  dims: 5
}
output {
  name: "OUT0"
  data_type: TYPE_FP64
  dims: 1
}
instance_group  {
   count: 2
   kind: KIND_CPU
}
dynamic_batching {
  preferred_batch_size: 32
  max_queue_delay_microseconds: 25000
}
parameters {
  key: "MODEL_FILE_FORMAT"
  value {
    string_value: "pmml"
  }
}
backend: "ibmsnapml"

```
### Configuration Parameters:

- **Backend** :
   Backend parameter must be provided as “ibmsnapml” while utilising Snap ML C++ Backend.
   ```
   backend: "ibmsnapml"
   ```
- **PREPROCESSING_FILE** :
   This configuration file parameter specifies the name of the file to be used for preprocessing. The preprocessing file must be named as ‘pipeline.json’
   when preprocessing is selected for end-to-end inferencing.

   If this parameter is not provided, the backend assumes that preprocessing is not required,even if pipeline.json is present in the model repository.

   ```
   parameters {
   key: "PREPROCESSING_FILE"
   value {
   string_value: "pipeline.json"
   }
   }
   ```
   ***Note*** : The ```‘pipeline.json’``` is serialised preprocessing pipeline captured during training using Snap ML API ```(export_preprocessing_pipeline(pipeline_xgb['preprocessor'],'pipeline.json’)```.For more details visit [here](https://snapml.readthedocs.io/en/latest/pipeline_export.html).
   
- **SNAPML_TARGET_CLASS** :
   This configuration parameter specifies which model needs to be imported by Snap ML C++ Backend as per Snap ML documentation [here](https://snapml.readthedocs.io/en/latest/model_import.html).

   ```
   parameters {
   key: "SNAPML_TARGET_CLASS"
   value {
   string_value: "BoostingMachineClassifier"
   }
   }

   ```
   For example, pre-trained model is ```‘sklearn.ensemble.RandomForestClassifier’ ``` then target  SnapML  class should ```‘snapml.RandomForestClassifier’``` . In case of C++ backend same target class will be referred as ```‘RandomForestClassifier’```  as a value for ```SNAPML_TARGET_CLASS``` in the config.pbtxt.
   
- **MODEL_FILE_FORMAT** :   
   Model File Format provided in the configuration file should be as per the Snap ML supported  models. Refer to Snap ML documentation [here](https://snapml.readthedocs.io/en/latest/model_import.html)

   ```
   parameters {
     key: "MODEL_FILE_FORMAT"
     value {
       string_value: "pmml"
     }
   }
   
   ```   
- **NUM_OF_PREDICT_THREADS** :  
    This defines the CPU threads running for each inference call.
  ```
    parameters {
    key: "NUM_OF_PREDICT_THREADS"
    value {
      string_value: "12"
    }
  }
  
  ``` 
- **PREDICT_PROBABILITY** : 
    This Configuration parameter  controls whether the model’s prediction is to be in terms of probability. 

  If Predict probability is chosen user should add the ```PREDICT_PROBABILITY``` parameter and set the ```output``` dimensions to 2, as it will return probabilities for both classes in a binary classification, as shown below:
  
  Note that  ```PREDICT_PROBABILITY``` is case sensitive and accepts only ```‘true’``` or ```‘false’``` . Any other values apart from these like 
  (True/False/1/0) are provided , by default ```‘false’``` will be assumed and it will not provide the probability.
    
    ```
     output {
      name: "OUT0"
      data_type: TYPE_FP64
      dims: 2
    }
    parameters {
     key: "PREDICT_PROBABILITY"
     value {
      string_value: "true"
     }
    }
    ```
    
- **Inputs and Outputs:** :
    Each model input and output must specify a name, datatype, and shape( for more details on
    inputs and output tensors check documentation of Triton Inference server [here](https://github.com/triton-inference-server/server/blob/r24.07/docs/user_guide/model_configuration.md#model-configuration)).The name specified for an input or output tensor must match the name
    expected by the model. An input shape indicates the shape of an input tensor expected by the model and by Triton inference request. An output
    shape indicates the shape of an output tensor produced by the model and returned by Triton in response to an inference request. Both input and output
    shape must have rank greater-or-equal-to 1, that is, the empty shape [ ] is not allowed.In case of preprocessing, the data type of Input
    must be ```TYPE_STRING```and output tensor is ```TYPE_FP64```. For more details on
    supported tensor data types by Triton Inference server ,vist [here](https://github.com/triton-inference-server/server/blob/r24.07/docs/user_guide/model_configuration.md#datatypes).

### SHAP and LIME Explainability (Snap ML C++ Backend)

The Snap ML C++ backend implements SHAP and LIME explanations natively in C++. Both methods are available for any tree-ensemble model type supported by the backend. For formats where the tree structure can be parsed at load time, exact tree-path attribution is used; for all other formats a coalition-based marginal fallback is used instead.

#### How it works

When `COMPUTE_SHAP` or `COMPUTE_LIME` is set to `"true"` in `config.pbtxt`:

1. The model is loaded and initialized.
2. If `COMPUTE_SHAP` is `"true"`, the backend additionally parses the raw model file (`model.pmml`, `model.json`, or `model.txt`) into an internal tree representation at load time for exact tree-path attribution.
3. On each inference request, after the primary prediction is computed, SHAP and/or LIME attribution is computed.
4. Resulting attribution arrays are appended to the inference response alongside the primary prediction output.

Explanations are computed **synchronously** on the same inference path. If SHAP or LIME computation fails, the backend logs a warning, zero-fills the attribution output tensor, and still returns the primary prediction. The request is **not** failed.

**SHAP algorithm details:**
- For `pmml`, `xgb_json`, and `lightgbm` model formats: exact tree-path attribution is used. The model's tree structure is parsed at load time and the path-dependent algorithm is applied per input row.
- For other model formats (e.g. `onnx`) or when tree parsing fails: a coalition-based marginal masking fallback is used. Coalitions are fully enumerated for ≤ 12 features; for larger feature spaces, a randomised paired-complement sampling schedule of up to 4096 coalitions is used.
- When batch size N > 1, the batch itself serves as the background/reference baseline for the marginal fallback. When N = 1, a zero-vector is used as the reference.

**LIME algorithm details:**
- Kernel-weighted linear surrogate model.
- Default: 1000 perturbation samples, kernel width 0.75. These parameters are internal defaults and are not currently configurable via `config.pbtxt`.
- Absent features are replaced using marginal sampling over the background batch.

If a preprocessing pipeline (`PREPROCESSING_FILE`) is configured, the **preprocessed** feature matrix (post-transform) is passed to both SHAP and LIME. The attribution values therefore correspond to the preprocessed feature space, not the raw input space.

#### XAI output tensor registration

XAI output tensors are **automatically registered** by the backend at model load time. You do **not** need to declare `shap_values` or `lime_values` in the `output` block of `config.pbtxt`. The backend registers them with:
- Data type: `TYPE_FP64`
- Shape: `[-1, n_features]` (with batching) or `[n_features]` (without), where `n_features` is the number of features after any configured preprocessing step.

If you manually declare the XAI output tensors in `config.pbtxt`, the backend detects the matching name and skips auto-registration, leaving your declaration intact.

#### Configuration parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `COMPUTE_SHAP` | `string` | `"false"` | Set to `"true"` to compute and return SHAP attributions. Case-sensitive. |
| `COMPUTE_LIME` | `string` | `"false"` | Set to `"true"` to compute and return LIME attributions. Case-sensitive. |
| `SHAP_OUTPUT_TENSOR_NAME` | `string` | `"shap_values"` | Name of the output tensor that receives SHAP results. |
| `LIME_OUTPUT_TENSOR_NAME` | `string` | `"lime_values"` | Name of the output tensor that receives LIME results. |

All four parameters are optional. Omitting `COMPUTE_SHAP` and `COMPUTE_LIME` (or setting them to any value other than `"true"`) disables XAI entirely.

#### Output tensor specification

Both XAI output tensors have data type `TYPE_FP64`. Their shape per sample is `[n_features]`, where `n_features` is the number of features in the (optionally preprocessed) input space. With batching enabled, the tensor shape is `[batch_size, n_features]`.

#### Example config.pbtxt with SHAP and LIME enabled

The backend auto-registers `shap_values` and `lime_values` if not declared. Declare them explicitly with `dims: [-1]` so the shape is visible and the backend patches the real feature count at first inference. The tensor names are configurable via `SHAP_OUTPUT_TENSOR_NAME` and `LIME_OUTPUT_TENSOR_NAME` — the declared output name must match the parameter value exactly.

```
backend: "ibmsnapml"
max_batch_size: 32

input {
  name: "IN0"
  data_type: TYPE_FP32
  dims: [ 30 ]
}

output {
  name: "OUT0"
  data_type: TYPE_FP64
  dims: [ 1 ]
}
output {
  # SHAP attributions — one value per feature (preprocessed space when
  # PREPROCESSING_FILE is set). dims: [-1] is resolved at first inference.
  name: "shap_values"
  data_type: TYPE_FP64
  dims: [ -1 ]
}
output {
  # LIME coefficients — one value per feature.
  name: "lime_values"
  data_type: TYPE_FP64
  dims: [ -1 ]
}

parameters {
  key: "MODEL_FILE_FORMAT"
  value { string_value: "pmml" }
}
parameters {
  key: "SNAPML_TARGET_CLASS"
  value { string_value: "BoostingMachineClassifier" }
}
parameters {
  key: "COMPUTE_SHAP"
  value { string_value: "true" }
}
parameters {
  key: "COMPUTE_LIME"
  value { string_value: "true" }
}
parameters {
  # Name of the SHAP output tensor (default: "shap_values").
  # Must match the output declaration name above.
  key: "SHAP_OUTPUT_TENSOR_NAME"
  value { string_value: "shap_values" }
}
parameters {
  # Name of the LIME output tensor (default: "lime_values").
  # Must match the output declaration name above.
  key: "LIME_OUTPUT_TENSOR_NAME"
  value { string_value: "lime_values" }
}
```

#### Inference request example

```bash
curl -X POST http://localhost:8000/v2/models/my_snapml_model/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "IN0",
        "shape": [2, 30],
        "datatype": "FP32",
        "data": [[0.1, 0.2, 0.3, 0.4, 0.5,
                  0.6, 0.7, 0.8, 0.9, 1.0,
                  1.1, 1.2, 1.3, 1.4, 1.5,
                  1.6, 1.7, 1.8, 1.9, 2.0,
                  2.1, 2.2, 2.3, 2.4, 2.5,
                  2.6, 2.7, 2.8, 2.9, 3.0],
                 [0.5, 0.5, 0.5, 0.5, 0.5,
                  0.5, 0.5, 0.5, 0.5, 0.5,
                  0.5, 0.5, 0.5, 0.5, 0.5,
                  0.5, 0.5, 0.5, 0.5, 0.5,
                  0.5, 0.5, 0.5, 0.5, 0.5,
                  0.5, 0.5, 0.5, 0.5, 0.5]]
      }
    ],
    "outputs": [
      {"name": "OUT0"},
      {"name": "shap_values"},
      {"name": "lime_values"}
    ]
  }'
```

The response includes `OUT0` (shape `[2, 1]`), `shap_values` (shape `[2, 30]`), and `lime_values` (shape `[2, 30]`), all with `TYPE_FP64`.

#### Supported model formats for XAI

| Format | `MODEL_FILE_FORMAT` value | Model file | SHAP algorithm |
|--------|--------------------------|------------|----------------|
| PMML | `pmml` | `model.pmml` | Exact tree-path attribution |
| XGBoost JSON | `xgb_json` | `model.json` | Exact tree-path attribution |
| LightGBM text | `lightgbm` | `model.txt` | Exact tree-path attribution |
| ONNX | `onnx` | `model.onnx` | Coalition-based marginal fallback (no tree parsing) |

#### Limitations

- XAI requires at least one input tensor to be declared. If no input tensors are defined, model loading fails.
- SHAP and LIME are computed on the **preprocessed** feature space when `PREPROCESSING_FILE` is set. Attributions do not directly correspond to the raw input features in that case.
- There is no background/reference dataset cache between requests. Each request batch serves as its own background.
- The `LIME_OUTPUT_TENSOR_NAME` and `SHAP_OUTPUT_TENSOR_NAME` parameters must not match any primary output tensor name already declared in `config.pbtxt`.

## PyTorch Backend <a id="pytorch-backend"></a>

PyTorch Backend is a custom Python-based backend that enables the execution of PyTorch models on the IBM Z systems using the [ibmz-accelerated-for-pytorch](https://github.com/IBM/ibmz-accelerated-for-pytorch) framework. This backend provides the Python Backend's flexibility, with the ability to transparently leverage new hardware inference acceleration capabilities that target the IBM Integrated Accelerator for AI through PyTorch.

### Backend Usage Information

To deploy any model on Triton Inference Server, one should have a model repository ready.

The following is the model repository structure :

```text
$ model_1
   |-- 1
   |   |-- model.py ( Model file ) 
   |   `-- model.pth ( Model weights ) 
   `-- config.pbtxt ( Model configuration ) 
```

#### Model file: <a id="pytorch-backend-model-file"></a>

The user must provide a model file (default: `model.py` ) containing at least one class that inherits from `torch.nn.Module` to describe the model architecture. 

Sample model.py file : 

```
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(5, 2, bias=False)
        self.linear2 = nn.Linear(2, 1, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x
```

Notes: 

- The model must return the correctly batched output in case of batched inputs. For example, if the input shape is (batch_size, 5), then each output tensor shape must be (batch_size,1).
- The default model file name is `model.py`; however, the user can change it to any other name using `default_model_filename` parameter in config.pbtxt, you can find the details in the config section.
- If the default model file has more than one class that is an extension of `torch.nn.Module`, then the user must provide the name of the main class to be used as the model class. This is done via the `model_class_name` config parameter, you can find the details in the config section.
    

#### Model weights:

The user can provide model weights in supported formats which contains the saved [state_dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference) of the model.

Note : The file name must be either `model.pth` or `model.bin` or `model.pt`.

In case of complex deployments, the model weights can be directly accessed from `model.py` by using the absolute path or relative path of the weights file/folder. 

For example : 

Model weights can also be stored in the model repository, 

```text
$ model_1
   |-- 1
   |   |-- model.py
   |   `-- weights
   |      |-- encoder.bin
   |      `-- classifier.bin
   `-- config.pbtxt
```
In this case, the `model.py` can access the weights in the following way :

```
import os
import torch 
import torch.nn as nn 
from pathlib import Path

class MyCustomWeighsModel(nn.Module):
    def __init__(self):
        super().__init__()
        ...
        ...
        current_file_path = Path(__file__).resolve()
        encoder_weights_path = os.path.join(current_file_path.parent, "weights/encoder.bin")
        classifier_weights_path = os.path.join(current_file_path.parent, "weights/classifier.bin")

        ...

        self.encoder_model.load_state_dict(torch.load(encoder_weights_path))
        self.classifier_model.load_state_dict(torch.load(classifier_weights_path))

        ...

    def forward(self, ...):
        ....
        
```

Note: When providing a different weights path ( as shown in the example above ), please make sure that you do not have any of the default weights file (`model.pth` or `model.bin` or `model.pt` ) in model folder (`model_repository/model_name/model_version/`). 

#### Model Configuration:

Every PyTorch Backend model must provide config.pbtxt file describing the model configuration. The model configuration (config.pbtxt) in Triton Inference Server defines metadata, optimization settings, and customised parameters for each model. This configuration ensures models are served with optimal performance and tailored behaviour. For more details [Model Configuration](https://github.com/triton-inference-server/server/blob/r24.07/docs/user_guide/model_configuration.md).


Below is a sample `config.pbtxt` for Python Backend:

 ```
 max_batch_size: 32
 backend: "pytorch"
 input {
   name: "IN0"
   data_type: TYPE_FP32
   dims: 5
 }
 output {
   name: "OUT0"
   data_type: TYPE_FP64
   dims: 1
 }
 ```

Configuration Parameters:

Triton Inference Server exposes some flags to control the execution mode of models through the parameters section in the model’s config.pbtxt file.

- **Backend** :
   Backend parameter must be provided as “pytorch” while utilising the PyTorch Backend. 
   
   ```
   backend: "pytorch"
   ```
   
- **Inputs and Outputs:** : 
    Each model input and output must specify a name, datatype, and shape. For more details on inputs and output tensors, check the documentation of Triton Inference server [here](https://github.com/triton-inference-server/server/blob/r24.07/docs/user_guide/model_configuration.md#inputs-and-outputs).

  In case of multiple inputs and multiple outputs, the ```forward``` method of the model class provided in `model.py` must take the inputs in the same sequence as specified in `config.pbtxt` and  must return the outputs in the same sequence as specified in `config.pbtxt`.
    
    For example,
    
    If `config.pbtxt` contains :
    
    ```
    input [
      {
        name: "input_1"
        data_type: TYPE_FP32
        dims: [ 128 ]
      },
      {
        name: "input_2"
        data_type: TYPE_FP32
        dims: [ 128 ]
      }
    ]
    output [
      {
        name: "predicted_class"
        data_type: TYPE_FP32
        dims: [ 1 ]
      },
      {
        name: "confidence"
        data_type: TYPE_FP32
        dims: [ 1 ]
      }
    ]
    ```
    
    Then `model.py` should look like : 
    
    ```
    class MyCustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            ...
    
        def forward(self, input_1, input_2):
            ...
            ...
            return predicted_class, confidence
      ```

- **Extra parameters:** :

    - hw_optm
    
        This parameter allows the user to decide the type of hardware optimization for the model inferencing: ```cpu``` or ```nnpa``` ( the IBM Integrated Accelerator for AI ). If not specified, the PyTorch backend will decide the hardware optimization based on the availability, i.e, if NNPA is available, it will use NNPA as the default hardware optimization, else CPU.

        ```
        parameters {
            key: "hw_optm"
            value {
                string_value: "nnpa"
                }
            }
        ```
    
    - model_class_name
 
      If the default model file contains more than one class that is an extension of `torch.nn.Module`, then the user must provide the name of the class that should be taken as the model class by pytorch backend.  
    
        ```
        parameters {
            key: "model_class_name"
            value {
                string_value: "ParentNet"
                }
            }
        ```
    - default_model_filename
 
      This parameter allows the user to change the name of the default model file. If not specified, the default model file name will be set as `model.py`.  The default model file must follow the structure defined in the [model file](#pytorch-backend-model-file) section.  This parameter could be useful if the model source code is modularized across multiple `.py` files.      
    
        ```
        parameters {
            key: "default_model_filename"
            value {
                string_value: "custom_model.py"
                }
            }
        ```

The PyTorch Backend also supports the config parameters available in the open-source Triton-Inference-Server pytorch_backend. Please refer to the customization guide [here](https://github.com/triton-inference-server/pytorch_backend?tab=readme-ov-file#customization)

### PyTorch Backend: Extension to utilize HuggingFace models via checkpoints.

To leverage this feature, AI practitioners and personas has to extend the ibmz-accelerated-for-nvidia-triton-inference-server image by installing transformers via `pip3 install transformers==4.51.1`

This enables the deployment of the model classes available in the `transformers` library ( such as `BertModel`, `RobertaModel` ) with just trained weights. 

In this case, the model repository structure would look like :

```text
$ model_1
   |-- 1
   |   |-- data ( Model weights folder ) 
   |   |   |-- ... ( model files ) 
   `-- config.pbtxt ( Model configuration ) 
```

For example :

```text
$ bert_base_cased_model/
   ├── 1
   │   └── data
   │       ├── bert-base-cased
   │       │   ├── config.json
   │       │   ├── pytorch_model.bin
   |       |   | ...
   │       │   ├── special_tokens_map.json
   │       │   ├── tokenizer_config.json
   │       │   └── vocab.txt
   └── config.pbtxt
```

With this feature, the user can deploy models in two ways: with Tokenizer and without Tokenizer. The tokenizer is a tool that transforms raw text into numerical inputs (token IDs, attention masks, etc.) suitable for transformer models.   

#### With Tokenizer (string inputs): 

When an encoder LLM model is deployed with tokenizer, the Triton server will tokenize the incoming data before sending it to the encoder model. Hence, users must provide string inputs in the inference requests.  

The user must provide tokenizer information in config.pbtxt file, this information includes ( tokenizer_instance_name, tokenizer_weights, and tokenizer_kwargs). Please refer to [Extra config parameters](#pytorch-backend-extra-config-parameters) section for the details of these parameters.  

The weights_folder must contain model weights (such as config.json, pytorch_model.pt) as well as tokenizer weights (such as tokenizer_config.json, vocab.txt/tokenizer.json). 

Note: The models with custom tokenization, such as BertForMultipleChoice, are not currently supported for this feature. 

The supported models for this deployment type are : 

    "BertModel" 
    "BertForPreTraining"
    "BertLMHeadModel"
    "BertForMaskedLM"
    "BertForNextSentencePrediction"
    "BertForSequenceClassification"
    "BertForTokenClassification"
    "BertForQuestionAnswering"

    "RobertaModel"
    "RobertaForCausalLM"
    "RobertaForMaskedLM"
    "RobertaForSequenceClassification"
    "RobertaForTokenClassification"
    "RobertaForQuestionAnswering"

    "AutoModel" 
    "AutoModelForPreTraining"
    "AutoModelForMaskedLM" 
    "AutoModelForCausalLM"
    "AutoModelForNextSentencePrediction"
    "AutoModelForSequenceClassification" 
    "AutoModelForTokenClassification"
    "AutoModelForQuestionAnswering"

    

#### Without Tokenizer (non-string inputs): 

When an encoder model is deployed without tokenizer, triton server will not tokenize the incoming data. Hence, users must provide already tokenized data (“input_ids” and “attention_mask”) in the inputs of the inference request. 

In config.pbtxt, the user must at least provide “input_ids”. 
 

***Extra config parameters*** : <a id="pytorch-backend-extra-config-parameters"></a>

 - is_hf_weights_model
 
     This parameter tells the PyTorch backend that the given model is a transformers based weights model. In this case, PyTorch backend will not look for `model.py` file in the model repository. 

     ```
     parameters {
         key: "is_hf_weights_model"
         value {
             string_value: "true"
             }
         }
     ```
    
 - model_instance_name

   The name of the transformers model instance to be used by the backend. The name provided here must be importable from the transformers library. For example, you can chose name "BertModel" if `python3 -c "from transformers import BertModel"` works. Other examples could be, `AutoModel`, `RobertaModel`, `BertForSequenceClassification`, etc. 
 
     ```
     parameters {
         key: "model_instance_name"
         value {
             string_value: "BertModel"
             }
         }
     ```
 - model_instance_output_fetch_attr

   The name of the attribute to be fetched from the model output. These attributes must be the supported attributes for that particular model instance output. For example, for `BertModel`, the supported attributes are `last_hidden_state`, `pooler_outputs`, etc.  You can refer to [this](https://huggingface.co/docs/transformers/en/main_classes/output) documentation to understand the transformers model output attributes. 
 
     ```
     parameters {
         key: "model_instance_output_fetch_attr"
         value {
             string_value: "last_hidden_state"
             }
         }
     ```

     Note: If the user wants to use multiple outputs, the names must be specified in a comma-separated fashion and the number of model_instance_output_fetch_attributes must match the number of entries in the `outputs` section of the config.pbtxt. 
        
 - weights_path

   The `absolute` or `relative path` ( w.r.t model version folder) to the folder where model weights are stored. For example, if the model version folder path is `/model_repository/bert_base_cased_model/1/`, then your weights_path can be `data` ( relative path ) or `/model_repository/bert_base_cased_model/1/data/`. 
 
     ```
     parameters {
         key: "weights_path"
         value {
             string_value: "data"
             }
         }
     ```

 - model_weights

   The name of the model weights folder present in `weights_path`. For example if your model weights are present at `/model_repository/bert_base_cased_model/1/data/bert-base-cased` then your `weights_path` will be  `/model_repository/bert_base_cased_model/1/data/` and `model_weights` will be `bert-base-cased`. 
 
     ```
     parameters {
         key: "model_weights"
         value {
             string_value: "bert-base-cased"
             }
         }
     ```

 - tokenizer_instance_name

   Similar to  `model_instance_name`, this field should contain the name of the transformers tokenizer instance to be used by the backend. The name provided here must be importable from the transformers library. For example, you can choose the name "BertTokenizer" if `python3 -c "from transformers import BertTokenizer"` works. Other examples could be `AutoTokenizer`, `RobertaTokenizer`, etc. 
 
     ```
     parameters {
         key: "tokenizer_instance_name"
         value {
             string_value: "BertTokenizer"
             }
         }
     ```

Note: If the `tokenizer_instance_name` field is provided in the `config.pbtxt`, then the backend expects the string inputs ( and not tensors ); thus, in this case, the `inputs` field in `config.pbtxt` must be of `TYPE_STRING` data type. 
     

 - tokenizer_weights

   The name of the tokenizer weights folder present in `weights_path`. For example, In case of HuggingFace deployments, if your model weights are present at `/model_repository/bert_base_cased_model/1/data/bert-base-cased` then your `weights_path` will be  `/model_repository/bert_base_cased_model/1/data/` and `tokenizer_weights` will be `bert-base-cased`. As in this case, the same folder (`bert-base-cased`) contains both model and tokenizer weights.    
 
     ```
     parameters {
         key: "tokenizer_weights"
         value {
             string_value: "bert-base-cased"
             }
         }
     ```

 - tokenizer_kwargs

   The tokenizer's keyword arguments. These arguments will be passed to the `from_pretrained` method of the [tokenizer](https://huggingface.co/docs/transformers/en/main_classes/tokenize). The input string value must be JSON serializable, as shown in the example below.
   Note: You can quickly verify if your input string is correct or not by running a Python snippet like this `import json ; json.loads("{\"padding\":\"max_length\", \"max_length\": 128 , \"truncation\": true}")`
 
     ```
     parameters {
         key: "tokenizer_kwargs"
         value {
             string_value: "{\"padding\":\"max_length\", \"max_length\": 128 , "{\"truncation\": true}"
             }
         }
     }
     ```

Note:

- This feature is only for the models available in Transformers package and whose weights are compatible by `from_pretrained` method of transformers models/tokenizers. Please refer to [this](https://huggingface.co/docs/transformers/en/models) document to understand model loading.
- If `is_hf_weights_model` is set to `true` the following config parameters will be ignored: `model_class_name` and `default_model_filename`.
- With this feature, you can either deploy models accepting pre-tokenized inputs ( string inputs ) or post-tokenized inputs ( `input_ids`, `attention_mask`, etc ). This is determined based on whether the tokenization information ( `tokenizer_instance_name`, `tokenizer_weights`, and `tokenizer_kwargs` ) is provided in the `config.pbtxt`.

### SHAP and LIME Explainability (PyTorch Backend)

The PyTorch backend integrates SHAP and LIME explanations via the `shap` and `lime` Python packages (`shap==0.52.0` and `lime==0.2.0.1`, both pre-installed in the container image). If either is absent, the model fails to load.

> **TorchScript models** (`model.pt` only, no `model.py`) are fully supported. The backend loads the TorchScript file directly and applies SHAP and LIME without requiring any user-written Python model code.

#### How it works

When `COMPUTE_SHAP` or `COMPUTE_LIME` is set to `"true"` in `config.pbtxt`, for each inference request the backend:

1. Executes the model normally to produce the primary output tensors.
2. Identifies the target class from the highest-logit class of the primary output (first output tensor, first sample in the batch).
3. Builds a prediction function that re-runs the model for any given input array, returning the scalar logit of the fixed target class. This function is used internally for perturbation scoring.
4. Calls the `shap` library's model-agnostic kernel explainer (SHAP) and/or `lime.lime_tabular.LimeTabularExplainer` (LIME) using that prediction function.
5. Inserts the resulting `float64` attribution arrays into the output tensor list at the positions declared in `config.pbtxt`.

Explanations are computed **synchronously** per inference request.

**Failure handling:** A SHAP computation failure causes the whole Triton request to fail with an error. Per-sample LIME failures are caught and logged; the affected sample's attribution row is left at zero while the rest of the batch continues. Other skip conditions — such as a non-tensor model output or an input that cannot be reshaped to a 2-D feature matrix — also cause the request to fail because the response tensor count would no longer match the declared outputs in `config.pbtxt`.

**SHAP algorithm details:**
- Background: the current batch itself (N > 1), or a zero-vector of the same feature count (N = 1). The explainer uses `nsamples="auto"`.
- The target class is fixed to the argmax of the first sample's primary output. All batch rows are explained against the same class.

**LIME algorithm details:**
- `lime.lime_tabular.LimeTabularExplainer` in `"regression"` mode with `discretize_continuous=False` and `sample_around_instance=True`.
- Background: 500 uniform samples from each feature's `[min, max]` range observed in the current batch. For constant features, the range is expanded to `[min, min + 1.0]`.
- Number of LIME perturbation samples: `max(250, 2 * n_features + 1)`.
- Fixed random seed (42) for both the background sampling and the `LimeTabularExplainer` instance, making the perturbation set reproducible for the same input.
- Attribution values come from `exp.local_exp`, key `1` (class index 1 in regression mode) falling back to key `0`.

The first input tensor (index 0 in `config.pbtxt`) is used as the explainability input. It is flattened to `[batch, n_features]` before being passed to the XAI methods.

#### XAI output tensors must be declared in config.pbtxt

Unlike the C++ backends, the PyTorch backend does **not** auto-register XAI output tensors. You must declare them explicitly in `config.pbtxt`. The backend matches declared output tensor names against the configured XAI names to determine which output slots to fill with attribution arrays.

If `COMPUTE_SHAP=true` is set but no output tensor named `shap_values` (or the custom `SHAP_OUTPUT_TENSOR_NAME`) is declared, the flag is silently cleared and no SHAP computation occurs.

#### Configuration parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `COMPUTE_SHAP` | `string` | `"false"` | Set to `"true"` to compute and return SHAP attributions. Case-sensitive. |
| `COMPUTE_LIME` | `string` | `"false"` | Set to `"true"` to compute and return LIME attributions. Case-sensitive. |
| `SHAP_OUTPUT_TENSOR_NAME` | `string` | `"shap_values"` | Name of the output tensor for SHAP results. Must match a declared output in `config.pbtxt`. |
| `LIME_OUTPUT_TENSOR_NAME` | `string` | `"lime_values"` | Name of the output tensor for LIME results. Must match a declared output in `config.pbtxt`. |

#### Output tensor specification

XAI output tensors must be declared as `TYPE_FP64` in `config.pbtxt`. Their shape is `[-1, n_features]` where `n_features` is the number of elements in the (flattened) first input tensor.

#### Required dependencies

The `shap` (`==0.52.0`) and `lime` (`==0.2.0.1`) packages are pre-installed in the container image. No manual installation is needed. If you are building a custom image, install the same pinned versions:

```bash
pip install shap==0.52.0 lime==0.2.0.1
```

The backend imports both packages at model load time. If either is missing, the model fails to initialize with a clear error message indicating which package is absent.

#### Example config.pbtxt with SHAP and LIME enabled

XAI output tensors **must** be declared explicitly in `config.pbtxt` for the PyTorch backend — unlike the C++ backends, no auto-registration is performed. Use `dims: [-1]` so the feature count is resolved at runtime. The tensor names are configurable via `SHAP_OUTPUT_TENSOR_NAME` and `LIME_OUTPUT_TENSOR_NAME` — the declared output name must match the parameter value exactly.

```
backend: "pytorch"
max_batch_size: 16

input {
  name: "IN0"
  data_type: TYPE_FP32
  dims: [ 30 ]
}

output {
  name: "OUT0"
  data_type: TYPE_FP32
  dims: [ 2 ]
}
output {
  # SHAP attributions — one value per input feature.
  # dims: [-1] is resolved to n_features at runtime.
  name: "shap_values"
  data_type: TYPE_FP64
  dims: [ -1 ]
}
output {
  # LIME coefficients — one value per input feature.
  name: "lime_values"
  data_type: TYPE_FP64
  dims: [ -1 ]
}

parameters {
  key: "COMPUTE_SHAP"
  value { string_value: "true" }
}
parameters {
  key: "COMPUTE_LIME"
  value { string_value: "true" }
}
parameters {
  # Name of the SHAP output tensor (default: "shap_values").
  # Must match the output declaration name above.
  key: "SHAP_OUTPUT_TENSOR_NAME"
  value { string_value: "shap_values" }
}
parameters {
  # Name of the LIME output tensor (default: "lime_values").
  # Must match the output declaration name above.
  key: "LIME_OUTPUT_TENSOR_NAME"
  value { string_value: "lime_values" }
}
```

#### Inference request example

```bash
curl -X POST http://localhost:8000/v2/models/my_pytorch_model/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "IN0",
        "shape": [1, 30],
        "datatype": "FP32",
        "data": [[0.1, 0.2, 0.3, 0.4, 0.5,
                  0.6, 0.7, 0.8, 0.9, 1.0,
                  1.1, 1.2, 1.3, 1.4, 1.5,
                  1.6, 1.7, 1.8, 1.9, 2.0,
                  2.1, 2.2, 2.3, 2.4, 2.5,
                  2.6, 2.7, 2.8, 2.9, 3.0]]
      }
    ],
    "outputs": [
      {"name": "OUT0"},
      {"name": "shap_values"},
      {"name": "lime_values"}
    ]
  }'
```

The response includes `OUT0` (the primary model prediction), `shap_values` (shape `[1, 30]`, `TYPE_FP64`), and `lime_values` (shape `[1, 30]`, `TYPE_FP64`).

#### XAI output tensor ordering

XAI output tensors can appear at any position in `config.pbtxt`'s `output` list. The backend uses the declared name to identify them, not their position. The final response tensor list matches the `config.pbtxt` order exactly, with primary model outputs and XAI arrays merged accordingly.

#### Limitations

- Only numeric input types are supported. If the first input tensor has data type `TYPE_STRING`, both XAI flags are cleared before inference. Because the declared XAI output slots are never filled, the response tensor count would not match `config.pbtxt` and the request fails. Do not set `COMPUTE_SHAP` or `COMPUTE_LIME` for models whose first input is `TYPE_STRING`.
- Only the first input tensor (index 0) is used for XAI perturbations. Multi-input models have all inputs forwarded to the model normally, but only `inputs[0]` is perturbed for attribution.
- For batched inference, the batch is gathered before XAI computation. SHAP uses the full batch as the background dataset; the target class is fixed to the argmax of the **first** sample in the batch.
- XAI is not supported for HuggingFace models with tokenized string inputs (`tokenizer_instance_name` set and `TYPE_STRING` inputs).
- `shap` and `lime` must be pre-installed. The model load fails with a clear error if either package is missing.


# Triton Model Analyzer<a id="triton-model-analyzer"></a>
Triton Model Analyzer is a CLI tool which can help you find a more optimal configuration, on a given piece of hardware,
for single, multiple, ensemble, or BLS models running on a Triton Inference Server. It will also generate reports to
help you better understand the trade-offs of the different configurations along with their compute and memory requirements.

Triton Model Analyzer uses Triton Performance Analyzer to calculate inference server side metrics.

## Triton Performance Analyzer:
Triton Performance Analyzer is CLI tool which can help you optimize the inference performance of models running on Triton
Inference Server by measuring changes in performance as you experiment with different optimization strategies.

Automatically runs multiple configuration combinations provided search (Optuna, Quick, Brute, Manual) in different modes(Remote, Local)

Metrics : Provides multiple metrics Throughput (infer/sec), P99/ p95/ p90/ p50/ Average Latency (ms),  Inference count RAM Usage vs Latency, Server compute output, input, infer, queue time

Detailed and summary reports: Model Analyzer is  able to generate summarized and detailed reports that can help you better understand the trade-offs between different model configurations that can be used for your mode

QoS (Quality of Service) Constraints: Constraints can help you filter out the Model Analyzer results based on your QoS requirements.

NOTE: If your model requires a custom backend when running on the Triton Inference Server please use either local or remote mode. [here](https://github.com/triton-inference-server/model_analyzer/blob/r24.07/docs/launch_modes.md)

NOTE: We support only ```KIND_CPU```. Need to pass ```cpu_only: true``` parameter in config.yaml [here](https://github.com/triton-inference-server/model_analyzer/issues/927)

## Configuring Model Analyzer:

Model Analyzer must provide the yaml file which contains the information about the model profiling:
LOCAL MODE: Local mode is the default mode if no triton-launch-mode is specified.

```
model_repository: /models
profile_models: 
    model1:
    perf_analyzer_flags:
        input-data: input.json(path to the input file/directory)
    cpu_only: true
run_config_search_mode: quick
run_config_search_max_instance_count: 4
run_config_search_max_concurrency: 16
run_config_search_min_model_batch_size: 8
override_output_model_repository: true
```

REMOTE MODE:This mode is beneficial when you want to use an already running Triton Inference Server.
You may provide the URLs for the Triton instance's HTTP or GRPC endpoint depending on your chosen client 
protocol using the --triton-grpc-endpoint, and --triton-http-endpoint flags.Triton Server in this mode needs to be 
launched with --model-control-mode explicit flag to support loading/unloading of the models.
    
```
run_config_search_disable: True
concurrency: [2, 4, 8, 16, 32]
batch_sizes: [8, 16, 32]
profile_models:
    model1:
    perf_analyzer_flags:
        input-data: input.json(path to the input file/directory)
    cpu_only: true
    model_config_parameters:
        dynamic_batching:
        max_queue_delay_microseconds: [200, 400]
        instance_group:
        - kind: KIND_CPU
            count: [1,2]
client_protocol: http
collect_cpu_metrics: True
triton_launch_mode: remote
triton_http_endpoint: <TRITON_SERVER_IP>:<EXPOSED_HTTP_PORT_NUM>
triton_grpc_endpoint: <TRITON_SERVER_IP>:<EXPOSED_GRPC_PORT_NUM>
triton_metrics_url:   <TRITON_SERVER_IP>:<EXPOSED_METRIC_PORT_NUM>/metrics
override_output_model_repository: true
```

To pass the data as real input file json -  the tensors need to be flattened in a row-major format `data.flatten('C')`

Example of input.json file:
```
{"data":
    [
    {
        "IN0":
        {   "content":[
            "92.5596638292661",
            "7.103605819788694",
            "abcd",
            "efgh"
        ],
            "shape": [4]
        }
}]} 
```


Example of directory:
Directory must contains the file with the same name of Input mentioned in config.pbtxt of the model. Example here `IN0`
Text inside `IN0` present in below format
```
92.5596638292661
7.103605819788694
abcd
efgh
```


More info about configuration parameters can be found [here](https://github.com/triton-inference-server/model_analyzer/blob/r24.07/docs/config.md)
More info about performance analyzer metrics [here](https://github.com/triton-inference-server/perf_analyzer/blob/main/README.md)
More info about the real input to be passed for model profiling [here](https://github.com/triton-inference-server/perf_analyzer/blob/main/docs/input_data.md)

  
## Model Analyzer Generated Metadata:

result: A csv file which contains all the all the configuration permutation and combination results.  

reports: Summary and detailed HTML report of best 3 configurations of the model. More Info [here](https://github.ibm.com/zosdev/TIS-model_analyzer/blob/r24.07/docs/report.md)

checkpoints: Model Analyzer will save a checkpoint after all the perf analyzer runs for a given model are complete. More Info [here](https://github.ibm.com/zosdev/TIS-model_analyzer/blob/r24.07/docs/checkpoints.md)

output_model_repository: A directory contains all the optimal configurations that are experimented.

plots: A directory consists of images of plots of different calculated parameters.






# REST APIs <a id="triton-server-restapi"></a>

- [Model Management](#tis-model-mgmt-restapi)
  - [Model Repository](#tis-model-repository-restapi)
  - [Model Configuration](#tis-model-config-restapi)
  - [Model Metadata](#tis-model-metadata-restapi)
- [Health Check](#tis-health-check-restapi)
- [Inference](#tis-inference-restapi)
- [Metrics Collection](#tis-metrics-restapi)
- [Traces](#tis-traces-restapi)
- [Statistics](#tis-statistics-restapi)
- [Server Metadata](#tis-server-metadata-restapi)

## Model Management <a id="tis-model-mgmt-restapi"></a>

Triton Inference Server operates in one of three model control modes: NONE,
EXPLICIT, or POLL. The model control determines how Triton Inference Server
handles changes to the model repository and which protocols and APIs are
available.

More details about model management can be found
[here](https://github.com/triton-inference-server/server/blob/r24.07/docs/user_guide/model_management.md)

### Model Repository <a id="tis-model-repository-restapi"></a>

The model-repository extension allows a client to query and control the one or
more model repositories being served by Triton Inference Server.

- Index API
  - POST `v2/repository/index`
- Load API
  - POST `v2/repository/models/${MODEL_NAME}/load`
- unload API
  - POST `v2/repository/models/${MODEL_NAME}/unload`

For more details about the model repository index, load and unload API calls
please visit the Triton documentation website link
[here](https://github.com/triton-inference-server/server/blob/r24.07/docs/protocol/extension_model_repository.md)

### Model Configuration <a id="tis-model-config-restapi"></a>

The model configuration extension allows Triton Inference Server to return server-specific information.

GET `v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]/config`

For more details about the model configuration API calls please visit the Triton
documentation website link
[here](https://github.com/triton-inference-server/server/blob/r24.07/docs/protocol/extension_model_configuration.md)

### Custom Model Configuration

The Triton Inference Server allows multiple configuration files for the same model within a model repository. This feature is useful when you want to maintain different configurations for the same model without duplicating the model files. You can specify which configuration to use by leveraging the --model-config-name option when launching the server.

Ex: ```--model-config-name=preprocessing```

This option tells Triton to use the configuration named preprocessing (without including the .pbtxt extension). By doing this, you can seamlessly switch between different configurations for the same model, such as configurations for preprocessing, batching, or inference settings, without modifying the model repository itself. For more details check [here](https://github.com/triton-inference-server/server/blob/r24.07/docs/user_guide/model_configuration.md?plain=1#L324)


### Model Metadata <a id="tis-model-metadata-restapi"></a>

Model Metadata per-model endpoint provides the following details

GET `v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]`

- name : The name of the model.
- versions : The model versions that may be explicitly requested via the
  appropriate endpoint. Optional for servers that don’t support versions.
  Optional for models that don’t allow a version to be explicitly requested.
- platform : The framework/backend for the model. See
  [Platforms](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#platforms).
- inputs : The inputs required by the model.
- outputs : The outputs produced by the model.

For more details about the model configuration API calls please visit the Triton
documentation website link
[here](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#model-metadata)

## Health Check <a id="tis-health-check-restapi"></a>

Health Check API provides status of Triton Inference Server, Model etc.

GET `v2/health/live`

GET `v2/health/ready`

GET `v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]/ready`

For more details about the statistics API calls please visit the kserve
documentation website link
[here](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#health)

## Inference <a id="tis-inference-restapi"></a>

An inference request is made with an HTTP POST to an inference endpoint.

POST `v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]/infer`

For more details about the inference API calls please visit the kserve
documentation website link
[here](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#inference)

- Classification
  - `POST /v2/models/<model_name>/infer` –data <JSON_Data>

The classification extension allows Triton Inference Server to return an output
as a classification index and (optional) label instead of returning the output
as raw tensor data. Because this extension is supported, Triton Inference Server
reports “classification” in the extensions field of its Server Metadata.

For more details about the classification API calls please visit the Triton
documentation website link
[here](https://github.com/triton-inference-server/server/blob/r24.07/docs/protocol/extension_classification.md)

- Binary data \* `POST /v2/models/<model_name>/infer`. The binary tensor
  data extension allows Triton Inference Server to support tensor data
  represented in a binary format in the body of an HTTP/REST request.

For more details about the binary data please visit the Triton documentation
website link
[here](https://github.com/triton-inference-server/server/blob/r24.07/docs/protocol/extension_binary_data.md)

## Metrics Collection <a id="tis-metrics-restapi"></a>

Triton Inference Server provides [Prometheus](https://prometheus.io/) metrics
indicating CPU and request statistics.

GET `/metrics`

For more details about the metrics collections please visit the Triton
documentation website link
[here](https://github.com/triton-inference-server/server/blob/r24.07/docs/user_guide/metrics.md)

## Traces <a id="tis-traces-restapi"></a>

Traces extension enables client to fetch or configure trace settings for a given
model while Triton Inference Server is running.

GET `v2[/models/${MODEL_NAME}]/trace/setting`

POST `v2[/models/${MODEL_NAME}]/trace/setting`

For more details about the trace API calls please visit the Triton documentation
website link
[here](https://github.com/triton-inference-server/server/blob/r24.07/docs/protocol/extension_trace.md)

## Statistics <a id="tis-statistics-restapi"></a>

GET `v2/models[/${MODEL_NAME}[/versions/${MODEL_VERSION}]]/stats`

For more details about the statistics API calls please visit the Triton
documentation website link
[here](https://github.com/triton-inference-server/server/blob/r24.07/docs/protocol/extension_statistics.md)

## Server Metadata <a id="tis-server-metadata-restapi"></a>

The server metadata endpoint provides information about the server. A server
metadata request is made with an HTTP GET to a server metadata endpoint.

GET `v2`

For more details about the statistics API calls please visit the kserve
documentation website link
[here](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#server-metadata-response-json-object)

# MLflow Integration <a id="mlflow-integration"></a>

IBM Z Accelerated for NVIDIA Triton Inference Server includes an MLflow integration layer that enables model deployment lifecycle management and inference statistics monitoring through the MLflow Python SDK and CLI. The integration does not replace Triton's native inference path — all model inference is performed by Triton through its standard `/v2/models/<name>/infer` endpoint.

## Architecture Overview

The MLflow integration consists of three components:

1. **mlflow-triton plugin** (`mlflow_triton.deployments.TritonPlugin`) — A Python package installed in the container that implements the MLflow `BaseDeploymentClient` interface. It handles artifact resolution, file transfer to the Triton model repository, and model load/unload operations via the Triton HTTP client.

2. **MLflow adapter model** (`mlflow_model`) — A Triton Python backend model with two versions loaded at server startup. Version 1 handles deployment operations; Version 2 handles monitoring operations. Both receive requests through custom HTTP routes.

3. **Custom HTTP routes** — The Triton HTTP server exposes two additional endpoints registered by the server's C++ layer:
   - `POST /v2/mlflow_deploy` — Routes to `mlflow_model` version 1 (deployment operations)
   - `POST /v2/mlflow_monitor` — Routes to `mlflow_model` version 2 (monitoring operations)

The adapter models use a shared `config.pbtxt` with a single `MLFLOW_REQUEST` input tensor (BYTES, shape [1]) and a single `MLFLOW_RESPONSE` output tensor (BYTES, shape [1]). Each request body is a JSON-encoded operation payload. The HTTP layer strips the KServe v2 envelope and returns only the raw JSON payload, so clients do not need to parse a KServe response wrapper.

### Ports and environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_URL` | `localhost:8000` (deployment plugin); `http://localhost:8000` (monitor adapter) | Triton HTTP endpoint. The deployment plugin strips any `http://` scheme prefix internally; the monitor adapter passes the value as-is to the HTTP client. Set this explicitly to avoid ambiguity. |
| `TRITON_MODEL_REPO` | *(required)* | Filesystem path to the user model repository. Must point to the same directory that was passed as one of the `--model-repository` flags. |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow tracking server URI. Used by both the monitor adapter and the deployment plugin when logging metrics or resolving model URIs. |

## Configuration and Startup

### Prerequisites

- The container image must include the `mlflow-triton` plugin package and the `mlflow` server binary.
- Triton must be started with `--model-control-mode=explicit` so that `load_model` / `unload_model` API calls work.
- The `mlflow_model` adapter directory (with `config.pbtxt`, `1/model.py`, `2/model.py`) must be present in one of the model repositories and loaded at startup via `--load-model=mlflow_model`.
- `/mlflow` inside the container is used by the MLflow tracking server for its SQLite database and artifact storage. Mount a host directory there to persist data across container restarts.
- `TRITON_MODEL_REPO` must point to the user model repository (the directory where deployed models are written). This is the second `--model-repository` path, not the MLflow-internal one.

### Single-container startup (recommended)

Both the MLflow tracking server and Triton run inside the same container. Port `5000` is published so external clients can reach the MLflow UI and tracking API. The `/mlflow/models` repository is managed internally by the MLflow adapter; `/models` is the user model repository where your own models live.

```bash
podman run -d \
  --name triton_mlflow \
  -p 5000:5000 \          # MLflow tracking server UI and API
  -p 8000:8000 \          # Triton HTTP inference endpoint
  -p 8001:8001 \          # Triton gRPC endpoint
  -p 8002:8002 \          # Triton Prometheus metrics endpoint
  -v $PWD/models:/models:z \   # Mount user model repository (SELinux :z relabels for container access)
  -e TRITON_MODEL_REPO=/models \   # Tells the plugin where to write deployed models
  <image_id_or_tag> \
  /bin/bash -c "
    mlflow server \
      --backend-store-uri sqlite:////mlflow/mlflow.db \   # SQLite DB inside the container (persist /mlflow to keep it)
      --default-artifact-root /mlflow/artifacts \        # Artifact storage inside the container
      --host 0.0.0.0 \                                   # Listen on all interfaces so external clients can connect
      --port 5000 &                                      # Run in background; & lets the shell continue to tritonserver
    tritonserver \
      --model-repository=/mlflow/models \    # Internal repo: mlflow_model adapter lives here (managed by the adapter)
      --model-repository=/models \           # User repo: your deployed models are written here by the plugin
      --load-model=mlflow_model \            # Pre-load the MLflow adapter at startup (both version 1 and 2)
      --log-verbose=2 \                      # Verbose logging for troubleshooting (reduce to 0 in production)
      --model-control-mode=explicit          # Required: enables load_model / unload_model API calls
  "
```

**Argument-by-argument explanation:**

| Argument | What it does |
|----------|-------------|
| `-d` | Run the container in detached (background) mode. |
| `--name triton_mlflow` | Assign a name to the container for easier management (`podman logs triton_mlflow`, `podman stop triton_mlflow`). |
| `-p 5000:5000` | Expose the MLflow tracking server so clients outside the container can reach `http://localhost:5000`. |
| `-p 8000:8000` | Expose the Triton HTTP endpoint used for inference and all MLflow deployment/monitoring API calls. |
| `-p 8001:8001` | Expose the Triton gRPC endpoint for standard model inference. |
| `-p 8002:8002` | Expose the Triton Prometheus metrics endpoint. |
| `-v $PWD/models:/models:z` | Bind-mount the host `models/` directory to `/models` inside the container. The `:z` flag relabels the mount for SELinux-enforcing systems; omit it on non-SELinux hosts. |
| `-e TRITON_MODEL_REPO=/models` | Tells the MLflow deployment plugin where to copy model files when a `create` or `update` operation is received. Must match the second `--model-repository` path. |
| `--backend-store-uri sqlite:////mlflow/mlflow.db` | Store MLflow run metadata in a SQLite file at `/mlflow/mlflow.db`. Mount `/mlflow` to a host directory to persist this across restarts. |
| `--default-artifact-root /mlflow/artifacts` | Store model artifacts at `/mlflow/artifacts` inside the container. Also persisted if `/mlflow` is mounted. |
| `--host 0.0.0.0` | Bind the MLflow server to all interfaces so it is reachable at `http://localhost:5000` from outside the container. |
| `--port 5000 &` | Start the MLflow server on port 5000 as a background process so the shell can continue to launch Triton. |
| `--model-repository=/mlflow/models` | First model repository: the MLflow adapter (`mlflow_model`) is stored here. This directory is managed internally by the adapter and should not be used for your own models. |
| `--model-repository=/models` | Second model repository: your user models and any models deployed via the plugin are written here. |
| `--load-model=mlflow_model` | Load the `mlflow_model` adapter at startup. Because `config.pbtxt` sets `version_policy: { all {} }`, both version 1 (deployment) and version 2 (monitoring) are loaded. |
| `--log-verbose=2` | Enable verbose server logging. Useful during initial setup; set to `0` in production to reduce log volume. |
| `--model-control-mode=explicit` | Disables automatic model polling. Models are loaded and unloaded only via explicit API calls, which is required for the deployment plugin to work correctly. |

> **Persistence:** The MLflow database (`mlflow.db`) and artifacts (`/mlflow/artifacts`) exist only inside the container unless `/mlflow` is bind-mounted to a host directory. The `mlflow-meta.json` tracking file written by each deployment lives in `/models/<name>/`, so it is persisted as long as the `/models` volume is mounted.

> **`MLFLOW_TRACKING_URI` inside the container:** Because both processes run in the same container, the monitor adapter can reach the MLflow server at `http://localhost:5000`. This is the default value of `MLFLOW_TRACKING_URI`, so no additional `-e` flag is needed for single-container deployments.
>
> **Changing the MLflow tracking URI or port:** If you start the MLflow server on a different port (e.g. `--port 5001`) or point it to an external tracking server, you **must** pass the updated URI when starting the container:
> ```bash
> -e MLFLOW_TRACKING_URI=http://localhost:5001
> ```
> The monitor adapter reads `MLFLOW_TRACKING_URI` once at process start. Changing the environment variable after the container is running has no effect — you must restart the container with the correct value set.

## Experiment Tracking

The MLflow integration uses standard MLflow tracking APIs. Clients set the tracking URI and log parameters, metrics, and artifacts as they would with any MLflow project. Registered model versions can then be deployed to Triton by passing a `models:/` URI to the deployment operations.

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

with mlflow.start_run(experiment_id=mlflow.create_experiment("my_experiment")):
    mlflow.log_param("model_type", "lgbm")
    mlflow.log_metric("auc", 0.92)

# Register a model from an existing run artifact
mlflow.register_model("runs:/<run_id>/model", "my_lgbm_model")
```

Deploying a registered model version to Triton:

```python
from mlflow.deployments import get_deploy_client

client = get_deploy_client("triton")
client.create_deployment("my_lgbm_model", "models:/my_lgbm_model/1")
```

The plugin resolves the model URI (including version aliases such as `models:/my_lgbm_model@champion`) to an integer version number, downloads the artifact from the MLflow artifact store, copies the model files into the Triton model repository under `<TRITON_MODEL_REPO>/<name>/1/`, and calls `triton_client.load_model(name)`.

## Model Deployment API

Deployment operations are sent to `POST /v2/mlflow_deploy` as a JSON body with an `"operation"` field.

### Supported operations

#### `create`

Deploy a new model from MLflow to Triton.

```bash
curl -X POST http://localhost:8000/v2/mlflow_deploy \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "create",
    "model_name": "my_lgbm_model",
    "model_uri": "models:/my_lgbm_model/1"
  }'
```

Response:
```json
{"status": "ok", "deployment": {"model_name": "my_lgbm_model"}}
```

**Required fields:** `model_name`, `model_uri`.
**Optional fields:** `config` (dict, may contain `"ready_timeout"` in seconds, default 60).

The operation: downloads the artifact from MLflow, copies `config.pbtxt` to `<repo>/<name>/` and all other files to `<repo>/<name>/1/`, calls `load_model(name)`, then polls `/v2/models/<name>/ready` until the model is READY or the timeout expires. If `load_model` fails, the copied files are removed (rollback).

#### `update`

Replace an existing deployment with a new model version.

```bash
curl -X POST http://localhost:8000/v2/mlflow_deploy \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "update",
    "model_name": "my_lgbm_model",
    "model_uri": "models:/my_lgbm_model/2"
  }'
```

Response:
```json
{"status": "ok", "deployment": {"model_name": "my_lgbm_model"}}
```

The operation: unloads the model, copies new artifacts, reloads, and waits for READY. If the file copy or reload fails, the model is left unloaded. The error message directs the caller to retry `update_deployment`.

#### `delete`

Unload and remove a deployed model.

```bash
curl -X POST http://localhost:8000/v2/mlflow_deploy \
  -H "Content-Type: application/json" \
  -d '{"operation": "delete", "model_name": "my_lgbm_model"}'
```

Response:
```json
{"status": "ok", "deleted": "my_lgbm_model", "model_name": "my_lgbm_model"}
```

Calls `unload_model(name)` then removes the entire model directory from the Triton model repository.

#### `list`

List all plugin-managed deployments currently in READY state.

```bash
curl -X POST http://localhost:8000/v2/mlflow_deploy \
  -H "Content-Type: application/json" \
  -d '{"operation": "list"}'
```

Response:
```json
{
  "status": "ok",
  "deployments": [
    {
      "model_name": "my_lgbm_model",
      "version": "1",
      "state": "READY",
      "triton_model_path": "/models/my_lgbm_model",
      "mlflow_model_uri": "models:/my_lgbm_model/1"
    }
  ]
}
```

Only models that have an `mlflow-meta.json` file in their model directory (i.e. models deployed through this plugin) and are currently in `READY` state are listed.

#### `get`

Get information about a specific deployment.

```bash
curl -X POST http://localhost:8000/v2/mlflow_deploy \
  -H "Content-Type: application/json" \
  -d '{"operation": "get", "model_name": "my_lgbm_model"}'
```

Response:
```json
{
  "status": "ok",
  "deployment": {
    "model_name": "my_lgbm_model",
    "version": "1",
    "state": "READY",
    "triton_model_path": "/models/my_lgbm_model",
    "mlflow_model_uri": "models:/my_lgbm_model/1"
  }
}
```

Raises `{"status": "error"}` if the deployment does not exist.

#### `predict`

Run inference on a deployed model through the plugin.

```bash
curl -X POST http://localhost:8000/v2/mlflow_deploy \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "predict",
    "model_name": "my_lgbm_model",
    "input": {
      "inputs": [
        {
          "name": "IN0",
          "shape": [1, 30],
          "datatype": "FP32",
          "data": [[0.1, 0.2, 0.3, 0.4, 0.5,
                    0.6, 0.7, 0.8, 0.9, 1.0,
                    1.1, 1.2, 1.3, 1.4, 1.5,
                    1.6, 1.7, 1.8, 1.9, 2.0,
                    2.1, 2.2, 2.3, 2.4, 2.5,
                    2.6, 2.7, 2.8, 2.9, 3.0]]
        }
      ]
    }
  }'
```

Response:
```json
{
  "status": "ok",
  "outputs": {
    "OUT0": [[0.15, 0.85]]
  }
}
```

The predict operation forwards the request to the deployed model's standard Triton inference endpoint (`/v2/models/<name>/infer`). Inference runs inside Triton against the deployed model — the MLflow adapter is not on the inference path.

**Input formats accepted:**
- KServe v2 format: `{"inputs": [{"name":..., "shape":..., "datatype":..., "data":...}]}`
- Plain dict format: `{"IN0": [1.0, 2.0, ...]}` (values converted to `np.ndarray` automatically)

### Model URI formats

The `create` and `update` operations accept the following `model_uri` formats:

| Format | Example |
|--------|---------|
| Version number | `models:/my_model/1` |
| MLflow stage | `models:/my_model/Production` |
| Alias (new-style) | `models:/my_model@champion` |
| Alias (path-style) | `models:/my_model/champion` |

Aliases and stages are resolved to an integer version number before artifact download.

## Monitoring API

Monitoring operations are sent to `POST /v2/mlflow_monitor`.

The monitor collects Triton inference statistics from `GET /v2/models/stats` (or `GET /v2/models/<name>/stats` for a single model), computes delta values between consecutive collection windows, and logs the resulting metrics to an MLflow experiment run.

### Supported operations

#### `monitor` (single run)

Collect Triton statistics once and log them to MLflow as a single completed run.

```bash
curl -X POST http://localhost:8000/v2/mlflow_monitor \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "monitor",
    "experiment_name": "triton_stats",
    "model_name": "my_lgbm_model",
    "watch": false
  }'
```

Response:
```json
{
  "status": "ok",
  "mode": "single_run",
  "experiment_name": "triton_stats",
  "run_id": "abc123..."
}
```

#### `monitor` (continuous background watch)

Start a background daemon thread that collects and logs metrics at a configurable interval.

```bash
curl -X POST http://localhost:8000/v2/mlflow_monitor \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "monitor",
    "experiment_name": "triton_stats",
    "model_name": "my_lgbm_model",
    "watch": true,
    "interval": 60
  }'
```

Response:
```json
{
  "status": "ok",
  "mode": "watch",
  "experiment_name": "triton_stats",
  "model_name": "my_lgbm_model",
  "interval": 60,
  "message": "Continuous background watch loop started ..."
}
```

Each interval, the worker: fetches current stats, computes deltas vs. the previous snapshot, logs the metrics to a new MLflow run, closes the previous run, then waits for the next interval. The experiment is created automatically if it does not exist.

**Request fields for `monitor`:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `experiment_name` | `string` | *(required)* | MLflow experiment name. Created if absent. |
| `model_name` | `string` | `null` | Filter stats to one model. Omit to collect across all loaded models. |
| `watch` | `bool` | `false` | `true` for continuous background loop; `false` for a single immediate flush. |
| `interval` | `number` | `60` | Flush interval in seconds for watch mode. Must be > 0. |
| `tracking_uri` | `string` | `$MLFLOW_TRACKING_URI` | Override the MLflow tracking URI for this request. |
| `triton_url` | `string` | `$TRITON_URL` | Override the Triton URL for this request. |
| `step` | `int` | `0` | MLflow step number for single-run mode. |

#### `stop`

Stop an active background watch process.

```bash
curl -X POST http://localhost:8000/v2/mlflow_monitor \
  -H "Content-Type: application/json" \
  -d '{"operation": "stop", "experiment_name": "triton_stats"}'
```

To stop all watchers at once, pass `"experiment_name": "*"` or `"experiment_name": "all"`.

Response:
```json
{
  "status": "ok",
  "stopped_experiments": ["triton_stats"]
}
```

#### `list`

List all currently active background watch processes.

```bash
curl -X POST http://localhost:8000/v2/mlflow_monitor \
  -H "Content-Type: application/json" \
  -d '{"operation": "list"}'
```

Response:
```json
{
  "status": "ok",
  "count": 1,
  "watchers": [
    {
      "experiment_name": "triton_stats",
      "model_name": "my_lgbm_model",
      "interval": 60,
      "uptime_seconds": 312.5,
      "alive": true
    }
  ]
}
```

### Monitored metrics

The following metrics are logged to MLflow by each flush window. They are sourced from the Triton `/v2/models/stats` endpoint (`inference_stats` fields) and aggregated across all matched models (or the single named model).

| MLflow metric name | Unit | Source field | Description |
|--------------------|------|-------------|-------------|
| `inference_count` | count | `inference_count` | Total inference requests in the window. |
| `error_count` | count | `fail.count` | Failed inference requests in the window. |
| `inference_latency_mean` | ms | `success.ns / success.count` | Mean end-to-end inference latency (successful requests only). |
| `queue_latency_mean_ms` | ms | `queue.ns / success.count` | Mean time spent in the Triton request queue. |
| `compute_latency_mean_ms` | ms | `compute_infer.ns / success.count` | Mean time spent in the compute kernel. |

The latency metrics are only logged when at least one successful inference occurred in the window (`success_count > 0`). All values represent **deltas** within the current flush window, not cumulative totals.

**Note:** Input drift and prediction drift monitoring are not implemented. The monitor collects Triton inference statistics only. Drift detection requires a separate external pipeline.

### Background watcher lifecycle

- Each background watcher is identified by its `experiment_name`. Only one watcher per experiment name can be active at a time.
- Watchers run as daemon threads inside the `mlflow_model/2` Python process. They do **not** persist across Triton server restarts.
- If the MLflow tracking server is temporarily unavailable, the `log_metrics` call fails with an `[ERROR]` log message and the metrics for that window are lost. The watcher continues running.
- If the Triton stats endpoint is unreachable, the watcher logs a warning and skips the current window. The watcher continues running.

## Troubleshooting

| Issue | Likely cause | Fix |
|-------|-------------|-----|
| `TRITON_MODEL_REPO is not set` | `TRITON_MODEL_REPO` env var missing | `export TRITON_MODEL_REPO=/models` |
| `Cannot reach Triton server` | Triton not running or wrong port | Ensure Triton is up, check `TRITON_URL` |
| `Model did not become READY within Ns` | Model init failed | Check Triton server logs for backend errors |
| `Unable to create deployment … already exists` | Model already deployed | Use `update` instead of `create` |
| `mlflow-triton plugin is not installed` | Package absent | `pip install mlflow-triton` inside the container |
| `experiment_name is required` | Missing field in JSON body | Add `"experiment_name"` to monitor request |
| `[monitor] ERROR: log_metrics failed` | MLflow tracking server unavailable | Start `mlflow server`, check `MLFLOW_TRACKING_URI` |
| `model_uri is required for update_deployment` | Missing `model_uri` | Pass `"model_uri"` in the update request |

## Python SDK usage

The MLflow deployment client can also be used directly from Python:

```python
from mlflow.deployments import get_deploy_client
import os

os.environ["TRITON_MODEL_REPO"] = "/models"
os.environ["TRITON_URL"] = "http://localhost:8000"

client = get_deploy_client("triton")

# Create
client.create_deployment("my_lgbm_model", "models:/my_lgbm_model/1")

# List
deployments = client.list_deployments()

# Get
info = client.get_deployment("my_lgbm_model")

# Update
client.update_deployment("my_lgbm_model", model_uri="models:/my_lgbm_model/2")

# Delete
client.delete_deployment("my_lgbm_model")
```

# Model Validation <a id="model-validation"></a>

Various models that were trained on x86 or IBM Z have demonstrated focused
optimizations that transparently target the IBM Integrated Accelerator for AI
for a number of compute-intensive operations during inferencing.

Note: Models that were trained on x86 ecosystem may throw endianness issues._

# Using the Code Samples <a id="code-samples"></a>

## Python Backend for IBM Snap ML : Random Forest Classifier

The purpose of this section is to provide details on how to deploy a model of
type Random Forest Classifier trained with scikit-learn, but deployable on
triton inference server leveraging snapml and python backend.

1. As per the documentation create python runtime
2. Train scikit learn python Random Forest Classifier. This will generate
   model.pmml file
3. Create model folder structure including relevant file that are needed to be
   deployed using Triton Inference Server
4. Deploy the model For more details about the sample model is documented
   [here](https://github.com/IBM/ai-on-z-triton-is-examples/tree/main/snapml-examples)

## ONNX-MLIR Backend : Convolutional Neural Network (CNN) with MNIST

The purpose of this section is to provide details on how to deploy a model of
type onnx-mlir trained with cntk a deep learning toolkit, but deployable on
triton inference server leveraging onnx-mlir backend. Convolutional Neural
Network with MNIST

1. Jupyter notebook for Convolutional Neural Network with MNIST can be found
   [here](https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_103D_MNIST_ConvolutionalNeuralNetwork.ipynb)
2. Use step 1 to train model on your own or download pre-trained model from
   [here](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist)
3. Use IBM zDLC to transform onnx-mlir model to model.so file
4. Import the compiled model into mode repository.
5. Create model folder structure including relevant files that are needed to be deployed
6. Deploy the model

For more details about the sample model is documented
[here](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist)

## ONNX-MLIR Backend : ONNX Model Zoo

See Verified for the list of models from the
[ONNX Model Zoo](https://github.com/onnx/models) that have been built and
verified with the IBM Z Deep Learning Compiler.

1. Import the compiled model into mode repository.
2. Create the model structure including relevant files that are needed to be
   deployed.
3. Deploy the model on to Triton Inference Server.

# Additional Topics <a id="additional-topics"></a>

## Supported Tensor Data Types

Tensor data types are shown in the following table along with the size of each
type, in bytes are supported.

| Data Type | Size (bytes)                  |
| --------- | ----------------------------- |
| BOOL      | 1                             |
| UINT8     | 1                             |
| UINT16    | 2                             |
| UINT32    | 4                             |
| UINT64    | 8                             |
| INT8      | 1                             |
| INT16     | 2                             |
| INT32     | 4                             |
| INT64     | 8                             |
| FP16      | 2                             |
| FP32      | 4                             |
| FP64      | 8                             |
| BYTES     | Variable (max 2<sup>32</sup>) |

## Triton response cache

The Triton response cache is used by Triton to hold inference results generated
for previous executed inference requests and sent as response if new inference
request hits cache.

For more details about the Triton response cache is documented
[here](https://github.com/triton-inference-server/server/blob/r24.07/docs/user_guide/response_cache.md)

## Repository agent (checksum_repository_agent)

Repository agent allows to introduce code that will perform authentication,
decryption, conversion, or similar operations when a model is loaded.

For more details about the Repository agent(checksum_repository_agent) is
documented
[here](https://github.com/triton-inference-server/server/blob/r24.07/docs/customization_guide/repository_agents.md)

## Version policy

Below details provides how the version policy been applied while model with different versions available in the model repository.

### Snap ML C++ and ONNX_MLIR backends with the -1 version

```text
|-- model_1
|   |-- -1
|   |   `-- model.txt
|   `-- config.pbtxt
`-- model_2
    |-- -1
    |   `-- model.so
    `-- config.pbtxt

---------------------------------------+
| Model     | Version | Status         |
+-----------+---------+-----------------
------------------------------------------------------------------------------------------------------------------------------------+
| model_1|| -1      | UNAVAILABLE: Unavailable: unable to find '/models/model_1/18446744073709551615/model.txt' for model 'model_1' |
| model_2|| -1      | UNAVAILABLE: Unavailable: unable to find '/models/model_2/18446744073709551615/model.so' for model 'model_2' |
+-----------+---------+-------------------------------------------------------------------------------------------------------------+
```

### Python and onnx-mlir backend with the 0 version

```text
|-- model_1
|   |-- 0
|   |   |-- model.py
|   |   `-- model.txt
|   `-- config.pbtxt
`-- model_2
    |-- 0
    |   `-- model.so
    `-- config.pbtxt

I0718 12:07:48.131561 1 server.cc:653
+-----------+---------+--------+
| Model     | Version | Status |
+-----------+---------+--------+
| model_1   | 0       | READY  |
| model_2   | 0       | READY  |
+-----------+---------+--------+
```

### Python and onnx-mlir backend with the 1 version

```text
|-- model_1
|   |-- 1
|   |   |-- model.py
|   |   `-- model.txt
|   `-- config.pbtxt
`-- model_2
    |-- 1
    |   `-- model.so
    `-- config.pbtxt

+-----------+---------+--------+
| Model     | Version | Status |
+-----------+---------+--------+
| model_1   | 1       | READY  |
| model_2   | 1       | READY  |
+-----------+---------+--------+
```

Each model can have one or more versions. The ModelVersionPolicy property of the
model configuration is used to set one of the following policies. 

• All: Load all the versions of the model.All versions of the model that are available in the model repository are available for inferencing. `version_policy: { all: {}} `

• Latest: Only the latest ‘n’versions of the model in the repository are available for inferencing. The latest versions of the model are the numerically greatest version numbers.`version_policy: { latest: { num_versions: 2}}`

• Specific: Only the specifically listed versions of the model are available for inferencing. `version_policy: {specific: { versions: [1,3]}}` If no version policy is specified, then Latest (with n=1) is used as the default, indicating that only the most recent version of the model is made available by Triton. In all cases, the addition or removal of version subdirectories from the model repository can change which model version is used on subsequent inference requests.

### Version Policy check: All

#### Test backends with multiple versions along with -1

```text
|-- model_1
|   |-- -1
|   |   `-- model.txt
|   |-- 4
|   |   |-- model.py
|   |   `-- model.txt
`-- model_2
    |-- -1
    |   `-- model.so
    |-- 3
        `-- model.so
---------------------------------------+
| Model     | Version | Status         |
+-----------+---------+-----------------
------------------------------------------------------------------------------------------------------------------------------------+
| model_1 | -1      | UNAVAILABLE: Unavailable: unable to find '/models/model_1/18446744073709551615/model.txt' for model 'model_1' |
| model_1 | 4       | UNLOADING                                                                                                     |
| model_2 | -1      | UNAVAILABLE: Unavailable: unable to find '/models/model_2/18446744073709551615/model.so' for model 'model_2'  |
| model_2 | 3       | UNAVAILABLE: unloaded                                                                                         |
------------------------------------------------------------------------------------------------------------------------------------+
```

error: creating server: Internal - failed to load all models

### Version Policy check: Latest

`version_policy: { latest: { num_versions: 2}}` to load latest 2 versions of the
model. The default is the higher version of a model.

```text
|-- model_1
|   |-- -1
|   |   |-- model.py
|   |   `-- model.txt
|   |-- 13
|   |   |-- model.py
|   |   `-- model.txt
|   |-- 17
|   |   |-- model.py
|   |   `-- model.txt
`-- model_2
    |-- -1
    |   `-- model.so
    |-- 15
    |   `-- model.so
    |-- 3
        `-- model.so
+-----------+---------+--------+
| Model     | Version | Status |
+-----------+---------+--------+
| model_1   | 13      | READY  |
| model_1   | 17      | READY  |
| model_2   | 9       | READY  |
| model_2   | 15      | READY  |
+-----------+---------+--------+
```

### Version Policy check: Specific

`version_policy: { specific: { versions: [4]}}` to load specific versions of the model. – model_1 

`version_policy: { specific: { versions: [3,9]}}` to load specific versions of the model. – model_2

```text
+-----------+---------+--------+
| Model     | Version | Status |
+-----------+---------+--------+
| model_1   | 4       | READY  |
| model_2   | 3       | READY  |
| model_2   | 9       | READY  |
+-----------+---------+--------+
```

### Model management examples

#### Single-model with Single version

```text
-- model_1
    |-- 1
    |   |-- model.py
    |   |-- model.txt
    `-- config.pbtxt
```

#### Single-model with Multi version

```text
`-- model_1
    |-- 1
    |   |-- model.py
    |   `-- model.txt
    |-- 2
    |   |-- model.py
    |   `-- model.txt
    `-- config.pbtxt
```

#### Multi-model with Single version

```text
|-- model_1
|   |-- 1
|   |   |-- model.py
|   |   `-- model.txt
|   `-- config.pbtxt
|-- model_2
|   |-- 0
|   |   `-- model.so
|   `-- config.pbtxt
```

#### Multi-model with Multi versions

```text
|-- model_1
|   |-- 1
|   |   |-- model.py
|   |   |-- model.txt
|   |-- 2
|   |   |-- model.py
|   |   |-- model.txt
|-- model_3
|   |-- 1
|   |   |-- model.py
|   |   |-- model_rfd10.pmml
|   |   |-- pipeline_rfd10.joblib
|   |-- 2
|   |   |-- model.py
|   |   |-- model_rfd10.pmml
|   |   |-- pipeline_rfd10.joblib
```

# Limitations and Known Issues <a id="limitations-known-issues"></a>

1. Consumer of Triton Inference Server may or may not face an issue when
   utilizing Triton Inference Server with a Python backend and an HTTP endpoint
   on a Big Endian machine, experience errors related to the TYPE_STRING
   datatype. For more details, see
   [link](https://github.com/triton-inference-server/server/issues/5610)

2. Consumer of Triton Inference server may or may not face an issue when running
   Triton Inference Server on a Big Endian machine, specifically related to GRPC
   calls with BYTES input. It appears that the current configuration may not
   fully support GRPC calls with BYTES input. For more details, see
   [link](https://github.com/triton-inference-server/server/issues/5811)

3. In case, user of Triton Inference Server want restrict access the protocols
   on a given endpoint by leveraging the configuration option
   '--grpc-restricted-protocol'. This feature provides fine-grained control over
   access to various endpoints by specifying protocols and associated restricted
   keys and values. Consumer of Triton Inference server may or may not find
   similar capability for restricting endpoint access for the HTTP protocol as
   currently not available. For more details, see
   [link](https://github.com/triton-inference-server/server/issues/5837)

4. Consumer of Triton server may or may not face an issue while having model with
   version -1 or model.py isn't present for python backend. For more details, see
   [link](https://github.com/triton-inference-server/server/issues/7052)

5. Deploying Hugging Face models using zero-filled inference inputs with sequence
   lengths ≥ 64 result in NaN outputs when running on NNPA hardware optimization.
   For more details, see [link](https://github.com/onnx/onnx-mlir/issues/3347)
       

# Versions and Release cadence <a id="versioning"></a>

IBM Z Accelerated for NVIDIA Triton™ Inference Server will follow the semantic
versioning guidelines with a few deviations. Overall IBM Z Accelerated for
NVIDIA Triton™ Inference Server follows a continuous release model with a
cadence of 1-2 minor releases per year. In general, bug fixes will be applied to
the next minor release and not back ported to prior major or minor releases.
Major version changes are not frequent and may include features supporting new
IBM Z hardware as well as major feature changes in Triton Inference Server that
are not likely backward compatible.

## IBM Z Accelerated for NVIDIA Triton™ Inference Server versions

Each release version of IBM Z Accelerated for NVIDIA Triton™ Inference Server
has the form MAJOR.MINOR.PATCH (X.Y.Z). For example, IBM Z Accelerated for
NVIDIA Triton™ Inference Server version 1.2.3 has MAJOR version 1, MINOR
version 2, and PATCH version 3. Changes to each number have the following
meaning:

### MAJOR / VERSION

All releases with the same major version number will have API compatibility.
Major version numbers will remain stable. For instance, 1.Y.Z may last 1 year or
more. It will potentially have backwards incompatible changes. Code and data
that worked with a previous major release will not necessarily work with the new
release.

`Note:` pybind11 PyRuntimes, any other python packages for versions of Python
that have reached end of life can be removed or updated to newer stable version
without a major release increase change.

### MINOR / FEATURE

Minor releases will typically contain new backward compatible features,
improvements, and bug fixes.

### PATCH / MAINTENANCE

Maintenance releases will occur more frequently and depend on specific patches
introduced (e.g. bug fixes) and their urgency. In general, these releases are
designed to patch bugs.

## Release cadence

Feature releases for IBM Z Accelerated for NVIDIA Triton™ Inference Server
occur about every 6 months in general. Hence, IBM Z Accelerated for NVIDIA
Triton™ Inference Server X.3.0 would generally be released about 6 months after
X.2.0. Maintenance releases happen as needed in between feature releases. Major
releases do not happen according to a fixed schedule.

# Frequently Asked Questions <a id="faq"></a>

## Q: Where can I get the IBM Z Accelerated for NVIDIA Triton™ Inference Server container image? <!--markdownlint-disable-line MD013 -->

Please visit this link
[here](https://ibm.github.io/ibm-z-oss-hub/containers/ibmz-accelerated-for-triton-serving.html).
Or read the section titled
[Downloading the IBM Z Accelerated for NVIDIA Triton™ Inference Server container image](#container).<!--markdownlint-disable-line MD013 -->

## Q: Where can I run the IBM Z Accelerated for NVIDIA Triton™ Inference Server container image? <!--markdownlint-disable-line MD013 -->

You may run the IBM Z Accelerated for NVIDIA Triton™ Inference Server container
image on IBM Linux on Z or IBM® z/OS® Container Extensions (IBM zCX).

_Note. The IBM Z Accelerated for NVIDIA Triton™ Inference Server will
transparently target the IBM Integrated Accelerator for AI on IBM z16 and later.
However, if using the IBM Z Accelerated for NVIDIA Triton™ Inference Server on
either an IBM z15® or an IBM z14®, IBM Snap ML or ONNX-MLIR will transparently
target the CPU with no changes to the model._

## Q: What are the different errors that can arise while using Triton Inference Server? <!--markdownlint-disable-line MD013 -->

| Error Type           | Description                                                                                                                                                                                                                                                                                                                                                                                                                       |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model load errors    | These errors occur when the server fails to load the machine learning model. Possible reasons could be incorrect model configuration, incompatible model format, or missing model files Backend errors Triton supports multiple backends for running models, such as Python, ONNX-MLIR. Errors can occur if there are issues with the backend itself, such as version compatibility problems or unsupported features. |
| Input data errors    | When sending requests to the Triton server, issues might arise with the input data provided by the client. This could include incorrect data types, shape mismatches, or missing required inputs. Any valid request batched with invalid request might lead to either inaccurate or invalid response by inference server to the entire batch.                                                                                              |
| Inference errors     | Errors during the inference process can happen due to problems with the model's architecture or issues within the model's code.                                                                                                                                                                                                                                                                                                   |
| Resource errors      | Triton uses system resources like CPU and memory to perform inference. Errors can occur if there are resource allocation problems or resource constraints are not handled properly.                                                                                                                                                                                                                                         |
| Networking errors    | Triton is a server that communicates with clients over the network. Network-related issues such as timeouts, connection problems, or firewall restrictions can lead to errors.                                                                                                                                                                                                                                                    |
| Configuration errors | Misconfigurations in the Triton server settings or environment variables could result in unexpected behavior or failures.                                                                                                                                                                                                                                                                                                         |
| Scaling errors       | When deploying Triton in a distributed or multi-instance setup, errors can occur due to load balancing issues, communication problems between instances, or synchronization failures.                                                                                                                                                                                                                                             |

# Technical Support <a id="contact"></a>

Information regarding technical support can be found
[here](https://github.com/IBM/ai-toolkit-for-z-and-linuxone).

# Licenses <a id="licenses"></a>

The International License Agreement for Non-Warranted Programs (ILAN) agreement
can be found
[here](https://www.ibm.com/support/customer/csol/terms/?id=L-CMSD-YF5T8W&lc=en) 

The registered trademark Linux® is used pursuant to a sublicense from the Linux
Foundation, the exclusive licensee of Linus Torvalds, owner of the mark on a
worldwide basis.

NVIDIA and Triton are trademarks and/or registered trademarks of NVIDIA
Corporation in the U.S. and/or other countries. 

Docker and the Docker logo are trademarks or registered trademarks of Docker,
Inc. in the United States and/or other countries. Docker, Inc. and other parties
may also have trademark rights in other terms used herein.

IBM, the IBM logo, and ibm.com, IBM z16, IBM z15, IBM z14 are trademarks or
registered trademarks of International Business Machines Corp., registered in
many jurisdictions worldwide. Other product and service names might be
trademarks of IBM or other companies. The current list of IBM trademarks can be
found [here](https://www.ibm.com/legal/copyright-trademark).
