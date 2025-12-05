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

NOTE UNTUK PROMPT 5-12-2025:
we are doing a final project with the problems in the codebase files #codebase #file:soal.txt  and we decided to do a deepfake detection. #file:pipeline.md  #file:dataset_explanation.md  we preprocessed the dataset with the script #file:processing_detect_n_crop.ipynb  and now the code pipeline is finished #file:full_pipeline_ffpp_yolo_faces.ipynb but unfortunately the computer supports tensorflow cuda. you are an ai engineer with 5 yrs + of experience and you are our mentor who is going to help us complete this project. can you help to change this code, same pipeline for tensorflow instead? is that possible?? create a new ipynb. 
