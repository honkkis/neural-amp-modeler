# NAM HD: Neural Amp Modeler on Steroids (96kHz and longer time scale support)

This repository is a fork of the NAM trainer, but it supports high-definition training using the GUI trainer. (No 96kHz training support for Colab yet, just the GUI). See advanced options to train in 96kHz and with an HD model.

The main idea is that you provide the normal 48kHz training files (input and output) and this modified GUI trainer resamples them in memory during training. The workflow remains the same as with the normal GUI trainer, also batch training is supported.

NOTE: the models trained with this will automatically work in the normal version of the NAM plugin without any changes. https://www.neuralampmodeler.com/ . Note that you need to run the whole project in 96kHz or then use e.g., Reaper 2x effects oversampling. The oversampling in the standard plugin does not include antialias filter. Another alternative is to use this fork of the plugin: https://github.com/honkkis/NeuralAmpModelerPluginHD

![Alt text](img/hd_96000_advanced_options.png?raw=true "Title")

Recommended Advanced Options:

| Resample rate (Hz) | Recommended model | Notes                                               |
|--------------------|-------------------|-----------------------------------------------------|
| 48000              | standard          | no resampling, same as normal NAM trainer           |
| 96000              | hd                | hd: comparable receptive field (2x) to standard_48k |
| 192000             | ultrahd           | ultrahd: comparable receptive field (4x) to standard_48k |



Training 96kHz models reduce alias quite dramatically as can be seen from the spectrogram of this sine sweep 
![Alt text](img/alias.jpg?raw=true "Title")

Then when training in 96kHz, one can use the HD model, which reduces ESR:

![Alt text](img/ESR.png?raw=true "Title")


This repository handles training, reamping, and exporting the weights of a model.
For playing trained models in real time in a standalone application or plugin, see the partner repo,
[NeuralAmpModelerPlugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin).

* [How to use](https://github.com/sdatkinson/neural-amp-modeler/tree/main#how-to-use)
  * [Google Colab](https://github.com/sdatkinson/neural-amp-modeler/tree/main#google-colab)
  * [GUI](https://github.com/sdatkinson/neural-amp-modeler/tree/main#gui)
  * [The command line trainer (all features)](https://github.com/sdatkinson/neural-amp-modeler/tree/main#the-command-line-trainer-all-features)
* [Standardized reamping files](https://github.com/sdatkinson/neural-amp-modeler/tree/main#standardized-reamping-files)
* [Other utilities](https://github.com/sdatkinson/neural-amp-modeler/tree/main#other-utilities)



## How to install the NAM HD version

Using PIP

```bash   
pip install --upgrade https://github.com/honkkis/neural-amp-modeler-HD/archive/refs/heads/main.zip
```

Run NAMHD

```bash   
nam
```
Victory!


## How to install the NAM HD version (OPTION 2, using Conda)


1. Install Conda and Download the NAM HD code to your computer
```bash   
wget https://github.com/honkkis/neural-amp-modeler-HD/archive/refs/heads/main.zip 
```
3. unzip the code
```bash   
unzip main.zip
```
   
3. cd to the dir

```bash   
cd neural-amp-modeler-HD-main
```

4. install (make sure you have conda installed)

```bash   

conda env create -f environment_gpu.yml --name namhd

(OR conda env create -f environment_cpu.yml --name namhd)
```

5. change to the newly create conda env

```bash   
conda activate namhd
```

6. run the UI

```bash   
python -m nam.train.gui
```

7. Victory!


## How to use
There are three main ways to use the NAM trainer. There are two simplified trainers available (1) in your browser via Google Colab and (2) Locally via a GUI. There is also a full-featured trainer for power users than can be run from the command line.

### Google Colab

If you don't have a good computer for training ML models, you use Google Colab to train
in the cloud using the pre-made notebooks under `bin\train`.

For the very easiest experience, open 
[`easy_colab.ipynb` on Google Colab](https://colab.research.google.com/github/sdatkinson/neural-amp-modeler/blob/48353508431a62a17bf5e35deee862f83f730f6c/bin/train/easy_colab.ipynb) 
and follow the steps!

### GUI

After installing the Python package, a GUI can be accessed by running `nam` in the command line.

### The command line trainer (all features)

Alternatively, you can clone this repo to your computer and use it locally.

#### Installation

Installation uses [Anaconda](https://www.anaconda.com/) for package management.

For computers with a CUDA-capable GPU (recommended):

```bash
conda env create -f environment_gpu.yml
```
_Note: you may need to modify the CUDA version if your GPU is older. Have a look at [nVIDIA's documentation](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) if you're not sure._

Otherwise, for a CPU-only install (will train much more slowly):

```bash
conda env create -f environment_cpu.yml
```

_Note: if Anaconda takes a long time "`Solving environment...`", then you can speed up installing the environment by using the mamba experimental sovler with `--experimental-solver=libmamba`._

Then activate the environment you've created with

```bash
conda activate nam
```

#### Train models (GUI)
After installing, you can open a GUI trainer by running

```bash
nam
```

from the terminal.

#### Train models (Python script)
For users looking to get more fine-grained control over the modeling process, 
NAM includes a training script that can be run from the terminal. In order to run it
#### Download audio files
Download the [v1_1_1.wav](https://drive.google.com/file/d/1CMj2uv_x8GIs-3X1reo7squHOVfkOa6s/view?usp=drive_link) and [output.wav](https://drive.google.com/file/d/1e0pDzsWgtqBU87NGqa-4FbriDCkccg3q/view?usp=drive_link) to a folder of your choice 

##### Update data configuration 
Edit `bin/train/data/single_pair.json` to point to relevant audio files: 
```json
    "common": {
        "x_path": "C:\\path\\to\\v1_1_1.wav",
        "y_path": "C:\\path\\to\\output.wav",
        "delay": 0
    }
```

##### Run training script
Open up a terminal. Activate your nam environment and call the training with
```bash
python bin/train/main.py \
bin/train/inputs/data/single_pair.json \
bin/train/inputs/models/demonet.json \
bin/train/inputs/learning/demo.json \
bin/train/outputs/MyAmp
```

`data/single_pair.json` contains the information about the data you're training
on   
`models/demonet.json` contains information about the model architecture that
is being trained. The example used here uses a `feather` configured `wavenet`.  
`learning/demo.json` contains information about the training run itself (e.g. number of epochs).

The configuration above runs a short (demo) training. For a real training you may prefer to run something like,

```bash
python bin/train/main.py \
bin/train/inputs/data/single_pair.json \
bin/train/inputs/models/wavenet.json \
bin/train/inputs/learning/default.json \
bin/train/outputs/MyAmp
```

As a side note, NAM uses [PyTorch Lightning](https://lightning.ai/pages/open-source/) 
under the hood as a modeling framework, and you can control many of the Pytorch Lightning configuration options from `bin/train/inputs/learning/default.json`

#### Export a model (to use with [the plugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin))
Exporting the trained model to a `.nam` file for use with the plugin can be done
with:

```bash
python bin/export.py \
path/to/config_model.json \
path/to/checkpoints/epoch=123_val_loss=0.000010.ckpt \
path/to/exported_models/MyAmp
```

Then, point the plugin at the exported `model.nam` file and you're good to go!

## Standardized reamping files

NAM can train using any paired audio files, but the simplified trainers (Colab and GUI) can use some pre-made audio files for you to reamp through your gear.

You can use any of the following files:

* [v3_0_0.wav](https://drive.google.com/file/d/1Pgf8PdE0rKB1TD4TRPKbpNo1ByR3IOm9/view?usp=drive_link) (preferred)
* [v2_0_0.wav](https://drive.google.com/file/d/1xnyJP_IZ7NuyDSTJfn-Jmc5lw0IE7nfu/view?usp=drive_link)
* [v1_1_1.wav](https://drive.google.com/file/d/1CMj2uv_x8GIs-3X1reo7squHOVfkOa6s/view?usp=drive_link)
* [v1.wav](https://drive.google.com/file/d/1jxwTHOCx3Zf03DggAsuDTcVqsgokNyhm/view?usp=drive_link)

## Other utilities

#### Run a model on an input signal ("reamping")

Handy if you want to just check it out without needing to use the plugin:

```bash
python bin/run.py \
path/to/source.wav \
path/to/config_model.json \
path/to/checkpoints/epoch=123_val_loss=0.000010.ckpt \
path/to/output.wav
```
