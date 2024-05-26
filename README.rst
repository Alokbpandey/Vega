```rst
# Vega Framework

Vega Framework is a generative AI framework built for researchers and PyTorch developers working on large language models (LLMs), multimodal models (MM), automatic speech recognition (ASR), and text-to-speech synthesis (TTS). The primary objective of Vega is to provide a scalable framework for researchers and developers from industry and academia to more easily implement and design new generative AI models by being able to leverage existing code and pretrained models.

For technical documentation, please see the Vega Framework User Guide.

All Vega models are trained with Lightning and training is automatically scalable to 1000s of GPUs.

When applicable, Vega models take advantage of the latest possible distributed training techniques, including parallelism strategies such as

- data parallelism
- tensor parallelism
- pipeline model parallelism
- fully sharded data parallelism (FSDP)
- sequence parallelism
- context parallelism
- mixture-of-experts (MoE)
- and mixed precision training recipes with bfloat16 and FP8 training.

Vega's Transformer based LLM and Multimodal models leverage NVIDIA Transformer Engine for FP8 training on NVIDIA Hopper GPUs and leverage NVIDIA Megatron Core for scaling transformer model training.

Vega LLMs can be aligned with state of the art methods such as SteerLM, DPO and Reinforcement Learning from Human Feedback (RLHF), see Vega Aligner for more details.

Vega LLM and Multimodal models can be deployed and optimized with NVIDIA Inference Microservices (Early Access).

Vega ASR and TTS models can be optimized for inference and deployed for production use-cases with NVIDIA Riva.

For scaling Vega LLM and Multimodal training on Slurm clusters or public clouds, please see the Vega Framework Launcher. The Vega Framework launcher has extensive recipes, scripts, utilities, and documentation for training Vega LLMs and Multimodal models and also has an Autoconfigurator which can be used to find the optimal model parallel configuration for training on a specific cluster. To get started quickly with the Vega Framework Launcher, please see the Vega Framework Playbooks. The Vega Framework Launcher does not currently support ASR and TTS training but will soon.

Getting started with Vega is simple. State of the Art pretrained Vega models are freely available on HuggingFace Hub and NVIDIA NGC. These models can be used to generate text or images, transcribe audio, and synthesize speech in just a few lines of code.

We have extensive tutorials that can be run on Google Colab or with our NGC Vega Framework Container. and we have playbooks for users that want to train Vega models with the Vega Framework Launcher.

For advanced users that want to train Vega models from scratch or finetune existing Vega models we have a full suite of example scripts that support multi-GPU/multi-node training.

## Key Features
- Large Language Models
- Multimodal
- Automatic Speech Recognition
- Text to Speech
- Computer Vision

## Requirements
- Python 3.10 or above
- Pytorch 1.13.1 or above
- NVIDIA GPU, if you intend to do model training

## Developer Documentation
| Version | Status | Description |
|---------|--------|-------------|
| Latest  | Documentation Status | Documentation of the latest (i.e. main) branch. |
| Stable  | Documentation Status | Documentation of the stable (i.e. most recent release) branch. |

## Getting help with Vega
FAQ can be found on Vega's Discussions board. You are welcome to ask questions or start discussions there.

## Installation
The Vega Framework can be installed in a variety of ways, depending on your needs. Depending on the domain, you may find one of the following installation methods more suitable.

### Conda / Pip
Refer to the Conda and Pip sections for installation instructions. This is recommended for Automatic Speech Recognition (ASR) and Text-to-Speech (TTS) domains.

When using a Nvidia PyTorch container as the base, this is the recommended installation method for all domains.

### Docker Containers
Refer to the Docker containers section for installation instructions. This is recommended for Large Language Models (LLM), Multimodal and Vision domains.

- Vega LLM & Multimodal Container: nvcr.io/nvidia/vega:24.03.framework
- Vega Speech Container: nvcr.io/nvidia/vega:24.01.speech

### LLM and Multimodal Dependencies
It's highly recommended to start with a base NVIDIA PyTorch container: nvcr.io/nvidia/pytorch:24.02-py3

### Conda
We recommend installing Vega in a fresh Conda environment.

```bash
conda create --name vega python==3.10.12
conda activate vega
```

Install PyTorch using their configurator.

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

The command used to install PyTorch may depend on your system. Please use the configurator linked above to find the right command for your system.

### Pip
Use this installation mode if you want the latest released version.

```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython
pip install vega_toolkit['all']
```

Depending on the shell used, you may need to use "vega_toolkit[all]" instead in the above command.

### Pip (Domain Specific)
To install only a specific domain of Vega, use the following commands. Note: It is required to install the above pre-requisites before installing a specific domain of Vega.

```bash
pip install vega_toolkit['asr']
pip install vega_toolkit['nlp']
pip install vega_toolkit['tts']
pip install vega_toolkit['vision']
pip install vega_toolkit['multimodal']
```

### Pip from source
Use this installation mode if you want the version from a particular GitHub branch (e.g main).

```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython
python -m pip install git+https://github.com/username/Vega.git@{BRANCH}#egg=vega_toolkit[all]
```

### From source
Use this installation mode if you are contributing to Vega.

```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
git clone https://github.com/username/Vega
cd Vega
./reinstall.sh
```

If you only want the toolkit without additional conda-based dependencies, you may replace `reinstall.sh` with `pip install -e .` when your PWD is the root of the Vega repository.

### Mac computers with Apple silicon
To install Vega on Mac with Apple M-Series GPU:

1. Create a new Conda environment.
2. Install PyTorch 2.0 or higher.
3. Run the following code:

```bash
# [optional] install mecab using Homebrew, to use sacrebleu for NLP collection
# you can install Homebrew here: https://brew.sh
brew install mecab

# [optional] install pynini using Conda, to use text normalization
conda install -c conda-forge pynini

# install Cython manually
pip install cython

# clone the repo and install in development mode
git clone https://github.com/username/Vega
cd Vega
pip install 'vega_toolkit[all]'

# Note that only the ASR toolkit is guaranteed to work on MacBook - so for MacBook use pip install 'vega_toolkit[asr]'
```

### Windows Computers
One of the options is using Windows Subsystem for Linux (WSL).

To install WSL:

1. In PowerShell, run the following code:

```powershell
wsl --install
# [note] If you run wsl --install and see the WSL help text, it means WSL is already installed.
```

Learn more about installing WSL at Microsoft's official documentation.

After Installing your Linux distribution with WSL:

Option 1: Open the distribution (Ubuntu by default) from the Start menu and follow the instructions.

Option 2: Launch the Terminal application. Download it from Microsoft's Windows Terminal page if not installed.

Next, follow the instructions for Linux systems, as provided above. For example:

```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
git clone https://github.com/username/Vega
cd Vega
./reinstall.sh
```

### LLM and Multimodal Dependencies
The LLM and Multimodal domains require three additional dependencies: NVIDIA Apex, NVIDIA Transformer Engine, and NVIDIA Megatron Core.

When working with the main branch these dependencies may require a recent commit. The most recent working versions of these dependencies are:

```bash
export apex_commit=810ffae374a2b9cb4b5c5e28eaeca7d7998fca0c
export te_commit=bfe21c3d68b0a9951e5716fb520045db53419c5e
export mcore_commit=fbb375d4b5e88ce52f5f7125053068caff47f93f
export nv_pytorch_tag=24.02-py3
```

When using a released version of Vega, please refer to the Software Component Versions for the correct versions.

If starting with a base NVIDIA PyTorch container first launch the container:

```

bash
docker run \
  --gpus all \
  -it \
  --rm \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  nvcr.io/nvidia/pytorch:$nv_pytorch_tag
```

Then install the dependencies:

### Apex
Vega LLM Multimodal Domains require that NVIDIA Apex to be installed. Apex comes installed in the NVIDIA PyTorch container but it's possible that Vega LLM and Multimodal may need to be updated to a newer version.

To install Apex, run:

```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout $apex_commit
pip install . -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam --group_norm"
```

While installing Apex outside of the NVIDIA PyTorch container, it may raise an error if the CUDA version on your system does not match the CUDA version torch was compiled with. This raise can be avoided by commenting it here: https://github.com/NVIDIA/apex/blob/master/setup.py#L32

`cuda-nvprof` is needed to install Apex. The version should match the CUDA version that you are using:

```bash
conda install -c nvidia cuda-nvprof=11.8
```

`packaging` is also needed:

```bash
pip install packaging
```

With the latest versions of Apex, the `pyproject.toml` file in Apex may need to be deleted in order to install locally.

### Transformer Engine
The Vega LLM Multimodal Domains require that NVIDIA Transformer Engine to be installed. Transformer Engine comes installed in the NVIDIA PyTorch container but it's possible that Vega LLM and Multimodal may need Transformer Engine to be updated to a newer version.

Transformer Engine enables FP8 training on NVIDIA Hopper GPUs and many performance optimizations for transformer-based model training. Documentation for installing Transformer Engine can be found here.

```bash
git clone https://github.com/NVIDIA/TransformerEngine.git && \
cd TransformerEngine && \
git checkout $te_commit && \
git submodule init && git submodule update && \
NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install .
```

Transformer Engine requires PyTorch to be built with at least CUDA 11.8.

### Megatron Core
The Vega LLM Multimodal Domains require that NVIDIA Megatron Core to be installed. Megatron core is a library for scaling large transformer-based models. Vega LLM and Multimodal models leverage Megatron Core for model parallelism, transformer architectures, and optimized PyTorch datasets.

Vega LLM and Multimodal may need Megatron Core to be updated to a recent version.

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git && \
cd Megatron-LM && \
git checkout $mcore_commit && \
pip install . && \
cd megatron/core/datasets && \
make
```

### Vega Text Processing
Vega Text Processing, specifically (Inverse) Text Normalization, is now a separate repository https://github.com/NVIDIA/Vega-text-processing.

### Docker containers
We release Vega containers alongside Vega releases. For example, Vega r1.23.0 comes with container vega:24.01.speech, you may find more details about released containers in the releases page.

To use a pre-built container, please run:

```bash
docker pull nvcr.io/nvidia/vega:24.01.speech
```

To build a Vega container with Dockerfile from a branch, please run:

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile -t vega:latest .
```

If you choose to work with the main branch, we recommend using NVIDIA's PyTorch container version 23.10-py3 and then installing from GitHub.

```bash
docker run --gpus all -it --rm -v <vega_github_folder>:/Vega --shm-size=8g \
-p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit \
stack=67108864 --device=/dev/snd nvcr.io/nvidia/pytorch:23.10-py3
```

## Examples
Many examples can be found under the "Examples" folder.

## Contributing
We welcome community contributions! Please refer to CONTRIBUTING.md for the process.

## Publications
We provide an ever-growing list of publications that utilize the Vega Framework.

If you would like to add your own article to the list, you are welcome to do so via a pull request to this repository's gh-pages-src branch. Please refer to the instructions in the README of that branch.

## License
Vega is released under an Apache 2.0 license.
```
