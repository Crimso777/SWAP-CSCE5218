The provided code was tested on a NVIDIA GPU instance on VastAI.  The total run time of the training script is 24 hours on an RTX 5090.

The program was tested with all packages up to date with the latest version of Python 11.  This can be installed on linux with the following command:
``` Bash
sudo apt-get install python3.11
```

Pip can be installed with the following command:
``` Bash
sudo apt-get install python3-pip 
``` 

The installation instructions for pytorch varies widely depending on your operating system, cuda version, and preferred package manager.  Details can be found at:
[https://pytorch.org/get-started/locally/]
The command for Linux using a 5090 GPU are below for convenience:
``` Bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```
  All remaining required packages can be installed in the root directory of the project with the following command:
``` Bash
pip install -r requirements.txt -U
```

Finally, the code can be ran with the following command:
``` Bash
python3 train.py
```

The code can be ran within 2 hours by switching the num_epochs parameter in the first line of the file to 1.  This will not result in high accuracy, but works as a proof of concept.

The model I used to get the 81% accuracy is the 14b model.  To get this maximum accuracy comment out the model_name line, and uncomment the line above it.