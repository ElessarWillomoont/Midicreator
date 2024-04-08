
# Talk with piano

this is a project that bridges the gap between AI and music education by offering an interactive platform where learners can engage in a dialogue with a digital piano, fostering an intuitive understanding of music composition. Embracing the transformative power of AI, the project aims to democratize music learning, making it more accessible and enjoyable, particularly in under-resourced areas. While it stands on the frontier of educational innovation, the project is an ongoing exploration, seeking to refine the synergy between human musical creativity and machine intelligence.

**Before Use:**

normaly, if you need to change the parameter of the program, only need to adjust the config.py

make sure to install packages or set up a virtual enviorment according to requirements.txt

due to the complexity to upload big file into github, if you want to generate melodis directly without training one yourself, you should following the step below:

download the check point through the link below:
    
        https://drive.google.com/file/d/1v3GVClKDoLs4aSeJGOvs4v2jrSA1l2lS/view?usp=sharing
        
put the ckpt_pretrained.pt into the folder shared/ckpt, there should be a file named here-should-have-ckpts.
    
The generator part is designed to interact with Yamaha Disklavier piano, if you use another deivce, you may need to adjust it based on your deivce

If you want to train the model yourself, feel free to adjust the parameter realated in config.py, but rember to adjust the PROJECT_NAME and ENTITY_NAME to your own wandb project, or if you don't want to use it, jus delete every line realated to waandb in model_train.py

**How to Use:**

If you want to train a model yourself, first, prepare your own dataset(in form of midi file), and then, put all of your midi file in a folder in root folder of the project, and then, adjust the line below in the shared/configure.py, change maestro to the dataset folder of yourself.

    DATASET_LOC = "maestro" #folder of the dataset

Then, run pre_process/tokenlize_dataset.py, all of midifiles will be tokenlized into token ids, stored in the folder of dataset/dataset_json.

Then, run the pre_process/randomly_pick.py, the dataset will be splited into three part: train, validation, evaluation, you can adjust the ratio of the files through changing lines below in shared/configure.py

    PICK_RATIO = 0.1  # Ratio of files to pick for training/validation/evaluation

    SIZE_OF_TRAIN = 0.95  # Proportion of the dataset to use for training

    SIZE_OF_VAL = 0.04  # Proportion of the dataset to use for validation

    SIZE_OF_EVAL = 0.01  # Proportion of the dataset to use for evaluation

Then, run the model_training/model_training.py. the training process will begin.

feel free to adjust other prameters in shared/configure.py to fit your training device.

If you want just directly generate melodies, simply plug in the piano, and run the program_generating/generator.py
    
