python: 3.11.x
env name: dl_env
Commands:
    conda create -n dl_env python=3.11
    conda activate dl_env
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
    pip3 install torchaudio --index-url https://download.pytorch.org/whl/cu130   # if you installed it
    pip install ipykernel
    python -m ipykernel install --user --name dl_env --display-name "Python (dl_env)"


Sanity check: 
    import torch

    print("Torch:", torch.__version__, "| CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))




Task: 
    - Learn dataset
    - Learn the reccomended pipeline