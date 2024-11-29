# WELCOME TO TINY-ASSISTANCE

## INTRODUCTION

TINY-ASSISTANCE is a virtual assistance that runs on some of the ML models like text-to-speech, LLMs, and speech-to-text. 
The core purpose of this project is to build a brain for a smart house (Iot) that can generate human-like renponses, control household appliances, and interact naturally with users.

## GETTING STARTED

### Installation

To test this model, feel free to clone it: 

        $ git clone https://github.com/Long1555-ffd/virtual-assistance.git

or you can clone our fix-bug versions to see some of the interesting works we've done

        $ git clone https://github.com/Long1555-ffd/virtual-assistance.git -b ENG

After cloning the repo:

```sh
cd virtual-assistance
code .
pip install -r requirements.txt
```

### TESTING WITH GPU 

To check if your computer has cuda support, run this command:

```python
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version in PyTorch:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "None")
```
or:

```sh
nvidia-smi
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Read here for more information if you have any problems with CUDA or GPU installation

### TESTING WITH THE LLM MODELS

Here are some of the LLMs that you can play around with:

- [X] GPT-NEOX-1.3B

```sh
python model/gpt-neo.py
```

- [X] LARGE GPT-2

```sh
python model/large-gpt-2.py
```

- [X] LLAMA BY META

```sh
python model/meta-llama.py
```

- [X] GPT-J-6B

```sh
python model/gpt-j-6b.py
```

## WHAT'S NEXT FOR TINY-ASSISTANCE?

Our main mission is to build up a virtual-assistance that is user-friendly and it can control, minotor your house appliances. Essentially, it will be the big brain behind your smart house. 
This is currently still a challenge as we'll have try to integrate Iot and other ML models into projects. But through testing with LLMs, we have a high hope that we can make this virtual assistance a reality. Our models can now generate human-like texts, and in the future, with the integration of text-to-speech model, we can make this project become more interesting.

