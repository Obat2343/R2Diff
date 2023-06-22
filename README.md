# R2Diff
## Install

```sh
git clone https://github.com/Obat2343/R2Diff.git
mkdir git
cd git
```

Install followings in the git directory.

- Pyrep (<https://github.com/stepjam/PyRep>)
- CoppeliaSim (<https://www.coppeliarobotics.com/downloads.html>) # Please check the Pyrep repository to confirm the version of CoppeliaSim
- RLBench (<https://github.com/Obat2343/RLBench>)
- RotationConinuity (<https://github.com/papagina/RotationContinuity>)

Next, Install requirements

```sh
pip install -r requirements.txt
```
***

## Prepare Dataset

To create the dataset for training and testing, please run the following command.

```sh
python create_dataset.py --task_list TaskA TaskB
```

***
## Download Pre-trained weights

```sh
mkdir result
cd result
```

Please download and unzip the file from https://drive.google.com/file/d/1ECP7Vsz7HkC7dbgYmnI7gG1zVZAlwaXM/view?usp=share_link

***
## Train

```sh
cd main
python Train_Diffusion.py --config_file ../config/RLBench_Diffusion.yaml
```

***
## Test

```sh
cd main
python Evaluate_Diffusion_on_sim.py --config_path ../config/Test_config.yaml --diffusion_path ../weights/RLBench/PickUpCup/Diffusion_frame_100_mode_6d_step_1000_start_1e-05_auto_rank_1/model/model_iter50000.pth --tasks PickUpCup --inf_method_list retrieve_from_SPE
```