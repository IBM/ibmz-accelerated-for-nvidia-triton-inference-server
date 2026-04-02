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


# Triton Model Analyzer<a id="triton-model-analyzer"></a>
Triton Model Analyzer is a CLI tool which can help you find a more optimal configuration, on a given piece of hardware,
for single, multiple, ensemble, or BLS models running on a Triton Inference Server. It will also generate reports to
help you better understand the trade-offs of the different configurations along with their compute and memory requirements.

Triton Model Analyzer uses Triton Performance Analyzer to calculate inefernce server side metrics.

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
[here](https://www.ibm.com/support/customer/csol/terms/?id=L-JXNY-4LLR96&lc=en) 

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
