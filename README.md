# 环境安装

    conda create -n 543 python=3.8
    conda activate 543
GPU:

    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
CPU:

    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

其他软件

    conda install tqdm scikit-image -y
    pip install moviepy hydra-core opencv-python tensorboard --upgrade

