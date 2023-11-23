<!-- markdownlint-disable MD033 -->

# Using the IBM Z Accelerated for NVIDIA Triton™ Inference Server Container Image

# Table of contents

- [Overview](#overview)
- [Downloading the IBM Z Accelerated for NVIDIA Triton™ Inference server Container Image](#container)
- [Container Image Contents](#contents)
- [IBM Z Accelerated for NVIDIA Triton™ Inference Server container image usage](#launch-container)
- [Security and Deployment Guidelines](#security-and-deployment-guidelines)
- [IBM Z Accelerated for NVIDIA Triton™ Inference Server Backends](#triton-server-backends)
- [REST APIs](#triton-server-restapi)
- [Model Validation](#model-validation)
- [Using the Code Samples](#code-samples)
- [Additional Topics](#additional-topics)
- [Limitations and Known Issues](#limitations-known-issues)
- [Versions and Release Cadence](#versioning)
- [Frequently Asked Questions](#faq)
- [Technical Support](#contact)
- [Licenses](#licenses)

# Overview <a id="overview"></a>

[Triton Inference Server](https://github.com/triton-inference-server/server) is
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

A Triton backend API is provided by Triton Inference Server exposes C API and
backend libraries and frameworks in an optimized manner. This enables ONNX-MLIR
support on IBM Z.

Triton Inference Server python backend enables pre-processing, post-processing
and server models written in python programming language. This enables IBM Snap
ML support on IBM Z.

The models being served by Triton Inference Server can be queried and controlled
by a dedicated model management API that is available by HTTP/REST or GRPC
protocol, or by the C API.

From above architecture diagram, The model repository is a file-system based
repository of the models that Triton Inference Server will make available for a
deployment.

On IBM® z16™ and later (running Linux on IBM Z or IBM® z/OS® Container
Extensions (IBM zCX)), With Triton Inference server 2.33.0 python backend for IBM
Snap ML or custom backend like ONNX-MLIR will leverage new inference
acceleration capabilities that transparently target the IBM Integrated
Accelerator for AI through the
[IBM z Deep Neural Network (zDNN)](https://github.com/IBM/zDNN) library. The IBM
zDNN library contains a set of primitives that support Deep Neural Networks.
These primitives transparently target the IBM Integrated Accelerator for AI on
IBM z16 and later. No changes to the original model are needed to take advantage
of the new inference acceleration capabilities.

Please visit the section
[Downloading theIBM Z Accelerated for NVIDIA Triton™ Inference Server container image](#container)<!--markdownlint-disable-line MD013 -->
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
[IBM Z and LinuxONE Container Registry](https://ibm.github.io/ibm-z-oss-hub/containers/ibmz-accelerated-for-triton.html).

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

For documentation how serving models with Triton Inference Server please visit
the official
[Open Source Triton Inference Server documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html).

For brief examples on deploying models with Triton Inference Server, please
visit our [samples section](#code-samples)

## Launch IBM Z Accelerated for NVIDIA Triton™ Inference Server container

Launching and maintaining IBM Z Accelerated Triton™ Inference Server revolves
around official
[user guide](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton_inference_server_1140/user-guide/docs/run.html)
and
[quick start](https://github.com/triton-inference-server/server/blob/r23.04/docs/getting_started/quickstart.md)
tutorial.

This documentation will cover :

- [Creating a Model Repository](#create-model-repository)
- [Launching IBM Z Accelerated for NVIDIA Triton™ Inference Server](#start-triton-server)
- [Deploying the model](#security-and-deployment-guidelines)

## Creating a Model Repository <a id="create-model-repository"></a>

User can follow the steps describe at
[model repository](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md)
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
docker run --shm-size 1G --rm
    -p <EXPOSE_HTTP_PORT_NUM>:8000
    -p <EXPOSE_GRPC_PORT_NUM>:8001
    -p <EXPOSE_Metrics_PORT_NUM>:8002
    -v $PWD/models:/models <triton_inference_server_image> tritonserver
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

Once the model been available either on a system or in a
[model repository](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md),
model can be deployed automatically by specifying the path to the model location
while launching the Triton Inference Server.

Command to start the Triton Inference Server

```bash
docker run --shm-size 1G --rm \
    -p <TRITONSERVER_HTTP_PORT_NUM>:8000 \
    -p <TRITONSERVER_GRPC_PORT_NUM>:8001 \
    -p <TRITONSERVER_METRICS_PORT_NUM>:8002 \
    -v $PWD/models:/models <triton_inference_server_image>   tritonserver \
    --model-repository=/models --log-verbose 1
```

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

- For security and deployment best practices, please visit the common AI Toolkit
  documentation found
  [here](https://github.com/IBM/ai-toolkit-for-z-and-linuxone).

# IBM Z Accelerated for NVIDIA Triton™ Inference Server Backends <a id="triton-server-backends"></a>

IBM Z Accelerated for NVIDIA Triton™ Inference Server supports two backends as
of today.

- [Python Backend](#python-backend)
- [ONNX-MLIR Backend](#onnx-mlir-backend)

## Python Backend <a id="python-backend"></a>

Python backend Triton Inference Server has a Python backend that allows you to
deploy machine learning models written in Python for inference. This backend is
known as the "Python backend" or "Python script backend."

For more details about triton python backend are documented here

Format of the python backend looks like below

```text
$ model_1
   |-- -1
   |   |-- model.py
   |   `-- model.txt
   `-- config.pbtxt
```

Using the Python backend in Triton is especially useful for deploying custom
models or models developed with specific libraries that are not natively
supported by Triton's other backends. It provides a flexible way to bring in
your own machine learning code and integrate it with the server's inference
capabilities. It provides flexibility by allowing you to use any Python machine
learning library to define your model's inference logic.

Here's a high-level overview of how the Python backend works in the Triton
Inference Server:

1. Model Definition: You define your own machine learning model using a Python
   library of your choice. For example, you might use Scikit-learn, Keras, or
   any other Python machine learning library that can load and execute your
   model.

2. Triton Python Backend: The Triton Inference Server includes a Python backend
   that allows you to specify a custom Python script to handle inference for
   your model.

3. Inference Logic: In the Python script, you implement the inference logic that
   takes input data, performs the necessary computations using your model, and
   produces the output in a format that can be returned to the client.

4. Triton Server Configuration: When starting the Triton server, you configure
   it to use the Python backend and provide the path to your custom Python
   script.

5. Model Deployment: You package your model and the custom Python script
   together, so when you deploy the model to the Triton server, it knows how to
   use your Python script for inference.

6. Inference Requests: Clients can then send inference requests to the Triton
   server, specifying the model they want to use. The server will pass the input
   data to your custom Python script, which will handle the inference and return
   the results.

Using the Python backend in Triton is especially useful for deploying custom
models or models developed with specific libraries that are not natively
supported by Triton's other backends. It provides a flexible way to bring in
your own machine learning code and integrate it with the server's inference
capabilities.

NOTE:

1. model.py should be there in the model repository to use the python backend framework.
2. Mention the backend key value as python in the file config.pbtxt a. backend:
   "python"
3. Multiple versions are supported, only positive values as version model are supported

## ONNX-MLIR Backend <a id="onnx-mlir-backend"></a>

A triton backend which allows the deployment of onnx-mlir or zDLC compiled
models (model.so) with the triton inference server. For more details about the
onnx-mlir backend are documented
[here](https://github.com/IBM/onnxmlir-triton-backend) Format of the python
backend looks like below

```text
$ model_1
    |-- -1
    |   `-- model.so
    `-- config.pbtxt
```

# REST APIs <a id="triton-server-restapi"></a>

- [Model Management](#tis-model-mgmt-restapi)
  - [Model Repository](#tis-model-repository-restapi)
  - [Model Configuration](#tis-model-config-restapi)
  - [Model Metadata](#tis-model-metadata-restapi)
- [Health Check](#tis-health-check-restapi)
- [Inference](#tis-inference-restapi)
- [Logging](#tis-logging-restapi)
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
[here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_management.html#model-management)

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
[here](https://github.com/triton-inference-server/server/blob/r23.04/docs/protocol/extension_model_repository.md)

### Model Configuration <a id="tis-model-config-restapi"></a>

The model configuration extension allows Triton Inference Server to return server-specific information.

GET `v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]/config`

For more details about the model configuration API calls please visit the Triton
documentation website link
[here](https://github.com/triton-inference-server/server/blob/r23.04/docs/protocol/extension_model_configuration.md)

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
[here](https://github.com/triton-inference-server/server/blob/r23.04/docs/protocol/extension_classification.md)

- Binary data \* `POST /v2/models/<model_name>/infer`. The binary tensor
  data extension allows Triton Inference Server to support tensor data
  represented in a binary format in the body of an HTTP/REST request.

For more details about the binary data please visit the Triton documentation
website link
[here](https://github.com/triton-inference-server/server/blob/r23.04/docs/protocol/extension_binary_data.md)

## Logging <a id="tis-logging-restapi"></a>

Managing and troubleshooting machine learning models on Triton Inference Server can be effectively accomplished by configuring the logging settings and monitoring the logs.

Explore more command line options:
```
docker run <triton_inference_server_image>  tritonserver [options]
```
GET `v2/logging`

POST `v2/logging`
  ```json
   {
     "logging": {
       "log-verbose": false,
       "log-info": true,
       "log-warning": true,
       "log-error": true,
       "log-file:triton.log"
     },
   }
   ```
View the logs:

```bash
   tail -f /path/to/log/directory/triton.log
```
Logs will be written/overwritten into the file mentioned during the server runtime.

NOTE: Triton Inference Server allows creation of 40 log files in total. 20 for each protocol (http and grpc)

For more details about the logging API calls please visit the Triton
documentation website link
[here](https://github.com/triton-inference-server/server/blob/r23.04/docs/protocol/extension_logging.md)

## Metrics Collection <a id="tis-metrics-restapi"></a>

Triton Inference Server provides [Prometheus](https://prometheus.io/) metrics
indicating CPU and request statistics.

GET `/metrics`

For more details about the metrics collections please visit the Triton
documentation website link
[here](https://github.com/triton-inference-server/server/blob/r23.04/docs/user_guide/metrics.md)

## Traces <a id="tis-traces-restapi"></a>

Traces extension enables client to fetch or configure trace settings for a given
model while Triton Inference Server is running.

GET `v2[/models/${MODEL_NAME}]/trace/setting`

POST `v2[/models/${MODEL_NAME}]/trace/setting`

For more details about the trace API calls please visit the Triton documentation
website link
[here](https://github.com/triton-inference-server/server/blob/r23.04/docs/protocol/extension_trace.md)

## Statistics <a id="tis-statistics-restapi"></a>

GET `v2/models[/${MODEL_NAME}[/versions/${MODEL_VERSION}]]/stats`

For more details about the statistics API calls please visit the Triton
documentation website link
[here](https://github.com/triton-inference-server/server/blob/r23.04/docs/protocol/extension_statistics.md)

## Server Metadata <a id="tis-server-metadata-restapi"></a>

The server metadata endpoint provides information about the server. A server
metadata request is made with an HTTP GET to a server metadata endpoint.

GET `v2`

For more details about the statistics API calls please visit the kserve
documentation website link
[here](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#server-metadata-response-json-object)

# Model Validation <a id="model-validation"></a>

Various models that were trained on x86 or IBM Z have demonstrated focused
optimizations that transparently target the IBM Integrated Accelerator for AI
for a number of compute-intensive operations during inferencing.

_Note. Models that were trained on x86 ecosystem may throw endianness issues._

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
   [here](https://github.com/onnx/models/tree/main/vision/classification/mnist/model)
3. Use IBM zDLC to transform onnx-mlir model to model.so file
4. Import the compiled model into mode repository.
5. Create model folder structure including relevant files that are needed to be deployed
6. Deploy the model

For more details about the sample model is documented
[here](https://github.com/onnx/models/tree/main/vision/classification/mnist)

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
[here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/response_cache.html#triton-response-cache)

## Repository agent (checksum_repository_agent)

Repository agent allows to introduce code that will perform authentication,
decryption, conversion, or similar operations when a model is loaded.

For more details about the Repository agent(checksum_repository_agent) is
documented
[here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/customization_guide/repository_agents.html#repository-agent)

## Version policy

Below details provides how the version policy been applied while model with
different versions available in the model repository.

### Python and onnx-mlir backend with the -1 version

```text
|-- model_1
|   |-- -1
|   |   |-- model.py
|   |   `-- model.txt
|   `-- config.pbtxt
`-- model_2
    |-- -1
    |   `-- model.so
    `-- config.pbtxt

---------------------------------------+
| Model     | Version | Status         |
+-----------+---------+-----------------
-------------------------------------------------------------------------------------------------------------+
| model_1|| -1      | UNAVAILABLE: Internal: model.py does not exist in the model
 repository path: /models/ model_1/18446744073709551615/model.py |
| model_2|| -1      | UNAVAILABLE: Unavailable: unable to find '/models/ model_2
/18446744073709551615/model.so' for model model_2|
+-----------+---------+------------------------------------------------------------------------------------------------------------------------------+
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
| model_1 | 0       | READY  |
| model_2| 0       | READY  |
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
| model_1 | 1       | READY  |
| model_2  | 1       | READY  |
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
|   |   |-- model.py
|   |   `-- model.txt
|   |-- 4
|   |   |-- model.py
|   |   `-- model.txt
`-- model_2
    |-- -1
    |   `-- model.so
    |-- 3
    |   `-- model.so
| model_1 | -1      | UNAVAILABLE: Internal: model.py does not exist in the model
 repository path: /models/gbm_model/18446744073709551615/model.py |
| model_1 | 4       | UNLOADING                                   |
| model_2  | -1      | UNAVAILABLE: Unavailable: unable to find
'/models/mnist-12/18446744073709551615/model.so' for model 'mnist-12'        |
| model_2  | 3       | UNAVAILABLE: unloaded
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
    |   `-- model.so
+-----------+---------+--------+
| Model     | Version | Status |
+-----------+---------+--------+
| model_1  | 13      | READY  |
| model_1  | 17      | READY  |
| model_2  | 9       | READY  |
| model_2  | 15      | READY  |
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

4. Consumer of Triton server will only be able to create up to 40 log files in
   total out of which 20 for protocol http and 20 for protocol grpc. For more
   details, see
   [link](https://github.com/triton-inference-server/server/issues/6152)

5. Consumer of Triton server may or may not face an issue while changing the
   trace settings like the trace_rate, log_rate and log_frequency has value that
   exceeds more than 10 digits. For more details, see
   [link](https://github.com/triton-inference-server/server/issues/6153)

6. Trace settings would work even if the model has been unloaded. The reported
   issue here is that users can still access/change trace settings of a
   previously existing model that was unloaded to keep around certain
   model/version metadata even for models that are unloaded for the sake of
   reporting in the API /repository/index. For more details, see
   [link](https://github.com/triton-inference-server/server/issues/6154)

# Versions and Release cadence <a id="versioning"></a>

IBM Z Accelerated for NVIDIA Triton™ Inference Server will follow the semantic
versioning guidelines with a few deviations. Overall IBM Z Accelerated for
NVIDIA Triton™ Inference Server follows a continuous release model with a
cadence of 1-2 minor releases per year. In general, bug fixes will be applied to
the next minor release and not back ported to prior major or minor releases.
Major version changes are not frequent and may include features supporting new
IBM Z hardware as well as major feature changes in Triton Inference Server that
are not likely backward compatible.

## IBM Z Accelerated for for NVIDIA Triton™ Inference Server versions

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

## Q: What are the difference Errors that can arise while using Triton Inference Server? <!--markdownlint-disable-line MD013 -->

| Error Type           | Description                                                                                                                                                                                                                                                                                                                                                                                                                       |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model load errors    | These errors occur when the server fails to load the machine learning model. Possible reasons could be incorrect model configuration, incompatible model format, or missing model files Backend errors Triton supports multiple backends for running models, such as Python, ONNX-MLIR. Errors can occur if there are issues with the backend itself, such as version compatibility problems or unsupported features. |
| Input data errors    | When sending requests to the Triton server, issues might arise with the input data provided by the client. This could include incorrect data types, shape mismatches, or missing required inputs.                                                                                                                                                                                                                                 |
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
[here](https://www14.software.ibm.com/cgi-bin/weblap/lap.pl?li_formnum=L-VADS-9E8LT7)

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
