# Env install

    conda create -n 543 python=3.8
    conda activate 543
GPU:

    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
CPU:

    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

Other software:

    conda install tqdm scikit-image -y
    pip install moviepy hydra-core opencv-python tensorboard --upgrade
    

# RUN：
To run training, change the image path under conf/config.yaml
    
To run prediction, change the model path and image path in the conf/predict_config.ymal

Detail config information is in the config file under conf.
    
