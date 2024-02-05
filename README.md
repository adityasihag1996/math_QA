# math_QA
This repository contains a Huggingface code to finetune any LLM, using QLoRA, on your custom dataset.


## Table of Contents

- [Installation](#Installation)
- [Dataset](#Dataset)
- [Training](#Training)
- [Inference](#Inference)
- [Results](#Results)
- [To-Do](#to-do)
- [Contributing](#contributing)

## Installation
To use this implementation, you will need to have Python >= 3.10 installed on your system, as well as the following Python libraries:

```
git clone https://github.com/adityasihag1996/math_QA.git
cd math_QA
pip install -r requirements.txt
```

## Dataset
Dataset needs to be in chatML format.
```
{"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "..."}]}
```

## Training
To start the finetuning process, adjust the parameters in `finetuning_config.py` to your needs. This includes the dataset path, hyperparameters such as learning rate and batch size, and training settings such as the number of epochs.

Once you have configured the training parameters, you can start the training process by running:

```
python finetune.py
```

The script finetune.py will handle the training loop, checkpoint saving, and logging.

## Inference
For inference, use the `inference.py` script. This script will load a QLoRA model.
Adjust base_model name, and adapter path in the script.

```
python inference.py
```

## Results
refer here:- [HUgginface Model Card](https://huggingface.co/adityasihag/math_QA-Mistral-7B-QLoRA-adapter)


## Contributing
Contributions to improve the project are welcome. Please follow these steps to contribute:

Fork the repository.\
Create a new branch for each feature or improvement.\
Submit a pull request with a comprehensive description of changes.