# Generating T1 and T2 Mapping Cardiac MRI Using Semantic Diffusion Model for Data Augmentation

Source code for our paper "*Generating T1 and T2 Mapping Cardiac MRI Using Semantic Diffusion Model for Data Augmentation*" published at SAMI 2026.

The segmentation model's source code can be found: https://github.com/BME-SmartLab/CardiacMappingSeg 
The original Semantic Diffusion Model repository can be found: https://github.com/WeilunWang/semantic-diffusion-model


## Project description

The project is about generating synthetic MRI images using a diffusion model. The model is conditioned on the type of the image (T1 or T2), the target resolution, and the mask of the ventricle and the myocardium. The model is trained on a dataset of real MRI images from Semmelweis University. The goal is to generate realistic synthetic images that can be used for training deep learning models for example in image segmentation tasks.

## Related works

Our work relies on the works of Kalapos et al. [1] and Stojanovski et al. [2]. The dataset is described in Kalapos et al. [1], and the deep learning model is based on the work of Stojanovski et al. [2]. The dataset is not publicly available, but it can be requested from the authors of Kalapos et al. Some of the data reading and processing code is based on the code provided by Kalapos et al. 

The base concept of my work comes from Stojanovski et al. [2], where they used a diffusion model to generate synthetic ultrasound images. I use a similar approach to generate synthetic MRI images building on their publicly available github repository. Though I extend the capabilities of the model to not only condition on the mask of the ventricle and the myocardium, but to also condition on the type of the image (T1 or T2) and the target resolution to acquire the most realistic images possible.

The model is the Semantic Diffusion Model (SDM) [3]. This framework directly feeds the semantic layout and noisy image in to the model to generate high-quality and diverse images. This approach has demonstrated superior performance compared to traditional GAN-based methods, particularly in terms of image quality and diversity.


[1]
A. Kalapos et al., “Automated T1 and T2 mapping segmentation on cardiovascular magnetic resonance imaging using deep learning,” Frontiers in Cardiovascular Medicine, vol. 10, p. 1147581, 2023.

[2]
D. Stojanovski, U. Hermida, P. Lamata, A. Beqiri, and A. Gomez, “Echo from noise: synthetic ultrasound image generation using diffusion models for real image segmentation,” in International Workshop on Advances in Simplifying Medical Ultrasound, 2023, pp. 34–43.

[3]
W. Wang et al., “Semantic image synthesis via diffusion models,” arXiv preprint arXiv:2207.00050, 2022.
<https://github.com/WeilunWang/semantic-diffusion-model>

## Usage

### Docker

To build the docker image, run the following command in the root directory of the repository:
`docker build -t sdm .`

Run the docker container with the following command:
`docker run -it --rm sdm`

Inside the container, you should clone the repository.

Note: for the training of the segmentation models for the evaluation, a seperate docker image is built, because the requirements are different for that project. That image can be built in the same way from the `Cardiac-Mapping` project.

### Conda environment

In this case you need cuda already installed on your machine if you want to use the GPU.
I used cuda 11.3.

Alternatively, you can also create the environment from the `requirements.txt` file:
1. `conda create --name sdm python=3.8`
2. `conda activate sdm`
3. `pip install -r requirements.txt`
4. `conda install mpi4py`

### Running the solution

The dataset should also be mounted into the container, adjust the path in the notebook accordingly.
(The same way the mounting of the code is possible, then cloning is not necessary.)

For mounting use:
`docker run -it --rm -v /path/to/dataset:/path/in/container sdm`

### Training and inference commands

`wandb login '' && python3 semantic_diffusion_model/image_train.py --datadir ./se/ --savedir ./output --batch_size_train 6 --lr 0.00007 --is_train True --save_interval 75000 --lr_anneal_steps 75000 --deterministic_train True --image_log_interval 2000 --dataset_mode se --random_flip False --num_classes 3 --use_kl False --use_wandb --distributed_data_parallel True --use_fp16 True --img_size 256 --grayscale True --num_res_blocks 2 --noise_schedule cosine --rescale_learned_sigmas True --rescale_timesteps True --type_labeling True --diffusion_steps 1000 --resize False`

Although it is important to note that the dataset is not publicly available, so the training command is not going to work without the dataset, if it is available, the path should be adjusted in the command and the wandb entity in the code.

The inference command is the following:

`python3 semantic_diffusion_model/image_sample_se.py --datadir ./se/ --is_train False --resume_checkpoint ./output/2024-06-08_21-15-55/model075000.pt --dataset_mode se --random_flip False --num_classes 3 --results_dir ./results/2024-06-08_21-15-55/test/ --num_samples 945 --distributed_data_parallel True --use_fp16 True --num_res_blocks 2 --inference_on_train False --grayscale True --img_size 256 --deterministic_test True --batch_size_test 10 --s 1.0 --rescale_timesteps True --rescale_learned_sigmas True --type_labeling True --resize False --split test`