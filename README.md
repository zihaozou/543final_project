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
    

# 注意事项：
    如需运行predict.py，需先在config文件下的predict_config.yaml改动model.path
    把model.path改成你训练完后保存的模型路径
    
