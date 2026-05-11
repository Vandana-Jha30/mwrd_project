Dataset used are UIEB and EUVP

This is the path for accessing the dataset it is in train.py

# if '__file__' in globals():
#     ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# else:
#     ROOT_DIR = os.getcwd()


# default_train_input = os.path.join(ROOT_DIR, 'Dataset/train/input')
# default_train_target = os.path.join(ROOT_DIR, 'Dataset/train/target')
# default_val_input = os.path.join(ROOT_DIR, 'Dataset/val/input')
# default_val_target = os.path.join(ROOT_DIR, 'Dataset/val/target')
# default_ckpt_dir = os.path.join(ROOT_DIR, 'checkpoints')
# default_out_dir = os.path.join(ROOT_DIR, 'results')

To install mamba-ssm, use this to install

!pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
!pip install causal-conv1d==1.4.0 && pip install mamba-ssm==2.2.2