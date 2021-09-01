# Machine Learning on Apple MacBook Pro M1

A repository for running machine learning models on Apple MacBook Pro M1.

## TensorFlow with tensorflow-metal PluggableDevice

With the tensorflow-metal PluggableDevice provided by Apple for TensorFlow,
TensorFlow is able to be accelerated with Metal on Apple silicon Mac GPUs.

## PyTorch

At the time of this writing, PyTorch is not yet able to be accelerated
using Apple silicon Mac GPUs.

See https://github.com/pytorch/pytorch/issues/47702 for current status.

## Notebooks

In this repository you will find some notebooks for TF and PyTorch.

```text
.
├── pytorch_notebooks
└── tensorflow_notebooks
```

After you've completed the setup below, you will be able to run these
notebooks on your MacBook Pro M1.

## Setup

```zsh
brew install miniforge

conda create -n conda-ml-py39 python=3.9
conda activate conda-ml-py39

# https://developer.apple.com/metal/tensorflow-plugin/
conda install -c apple tensorflow-deps
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal

# https://stackoverflow.com/a/53546634
conda install ipykernel
ipython kernel install --user --name=setterpels39
```

```zsh
mkdir -p ~/ml-py39
cd ~/ml-py39/

git clone https://github.com/openai/CLIP
git clone https://github.com/dribnet/clipit
git clone https://github.com/CompVis/taming-transformers.git
```

```zsh
conda install numpy~=1.19.2
conda install -c pytorch pytorch==1.9.0 torchvision==0.10.0
conda install braceexpand omegaconf torch-optimizer \
              tqdm ftfy regex kornia pytorch-lightning \
              einops scikit-image cssutils wrapt opt_einsum \
              gast astunparse termcolor pandas ipywidgets
pip install git+https://github.com/pvigier/perlin-numpy
pip install torch-tools
```

```zsh
mkdir -p ~/src/github.com/BachiLi
cd ~/src/github.com/BachiLi/

git clone https://github.com/BachiLi/diffvg
cd diffvg

git submodule update --init --recursive

DIFFVG_CUDA=0 python setup.py install
```

## Running

```zsh
conda activate conda-ml-py39
cd ~/ml-py39/ && jupyter lab
```
