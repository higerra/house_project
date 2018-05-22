# Semantic segmentation for houses
## Overview
The code performs semantic segmentation for house exteriors. The algorithm is based on [dilated residual network](https://arxiv.org/abs/1705.09914).

## Requirements
* Python3
* Pytorch: tested with 0.4.0. CUDA support required.
* numpy, opencv-python, pillow, pydensecrf

## Dataset format
The dataset folder should have the following structure:

```
-- dataset1280
      |-- info.json
      |-- train
            |-- dataset.json
            |-- *****.jpg
            |-- *****_anno.npy
      |-- validation
            |-- dataset.json
            |-- *****.jpg
            |-- *****_anno.npy
      |-- test
            |-- dataset.json
            |-- *****.jpg
            |-- *****_anno.npy
```

The `info.json` file contains the mean and standard devitation of all pixels in all samples. See `code/data_util.py` for details.

Each sample in each of the `train`, `validation`, `test` folder is representated as a jpg file `*****.jpg` and the ground truth
annotation `*****_anno.npy`. Each subfolder also contains a `dataset.json` file. See `code/data_util.py` for the format.

## Run
* To run training:
```
python3 code/run_drn.py --mode train --data_dir <path-to-dataset> --model_dir <path-to-output-model-dir>
```
See `code/run_drn.py` for additional arguments.

* To run testing:
```
python3 code/run_drn.py --mode test --data_dir <path-to-dataset> --model_dir <path-to-trained-model-dir>
```
