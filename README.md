Python implementation code for the paper titled,

Title: Microstructure reconstruction of porous materials from a single three-dimensional training volume using a conditional denoising diffusion probabilistic model

Authors: Ting Zhang1, Xinzheng Huang1, Xiangyu Chen1, Yi Du2, *

1.College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China

2.College of Engineering, Shanghai Polytechnic University, Shanghai 201209, China

(*corresponding author, E-mail: duyi0701@126.com. Tel.: 86 - 21- 50214252. Fax: 86 - 21- 50214252. )

# CSinDDPM

CSinDDPM trains a diffusion model from a single 3D binary TIFF and generates porous-media samples. High-intensity TIFF values are treated as pores by default.

## Installation

Python 3.10 or later is recommended. Install a PyTorch build compatible with your CUDA environment, then run:

```bash
python -m pip install -r requirements.txt
```

## Check the Code and Data

The `data/` directory contains three example TIFF files. Run:

```bash
python -m pytest -q
python scripts/verify_best_3data.py --stage data
```

## Training and Sampling

Train on the three example datasets:

```bash
bash scripts/train_best_3data.sh
```

Generate samples after training:

```bash
bash scripts/sample_best_3data.sh
```

The training scripts require CUDA. On Windows, run the shell scripts in Git Bash or WSL. Results are saved to `revision_outputs/` by default.

## Use Your Own TIFF

```bash
python demo_train.py --data_dir /path/to/input.tif --output_dir revision_outputs/my_run

python demo_sample_num.py --model_path /path/to/model.pt --data_dir /path/to/input.tif --target_porosity 0.20 --output_dir revision_outputs/my_samples
```

View all available options:

```bash
python demo_train.py --help
python demo_sample_num.py --help
```
