## HF model and data params
base_model = "mistralai/Mistral-7B-v0.1"   # base model HF path
project_name = ""   # save model as name
dataset_name = ""  # training dataset HF path
dataset_text_field = None   # key of data from dataset
hf_token = ""   # HF key for gated models / tokenisers


## QLoRA parameters
lora_r = 128   # LoRA attention dimension
lora_alpha = 256   # Alpha parameter for LoRA scaling
lora_dropout = 0.05   # Dropout probability for LoRA layers
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


## bitsandbytes parameters
use_4bit = True   # Activate 4-bit precision base model loading
use_8bit = False   # Activate 8-bit precision base model loading
bnb_4bit_compute_dtype = "float16"   # Compute dtype for 4-bit base models
bnb_4bit_quant_type = "nf4"   # Quantization type (fp4 or nf4)
use_nested_quant = False   # Activate nested quantization for 4-bit base models (double quantization)


## TrainingArguments parameters
output_dir = f"./results_llm/{project_name}/"   # Output directory where the model predictions and checkpoints will be stored
num_train_epochs = 3   # Number of training epochs
fp16 = True   # Enable fp16/bf16 training
bf16 = False   # Enable fp16/bf16 training (set bf16 to True with an A100)
per_device_train_batch_size = 2   # Batch size per GPU for training
gradient_accumulation_steps = 128   # Number of update steps to accumulate the gradients for
gradient_checkpointing = False   # Enable gradient checkpointing
max_grad_norm = 1.0   # Maximum gradient normal (gradient clipping)
learning_rate = 5e-5   # Initial learning rate (AdamW optimizer)
weight_decay = 0.001   # Weight decay to apply to all layers except bias/LayerNorm weights
optim = "paged_adamw_32bit"   # Optimizer to use
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_epsilon = 1e-8
lr_scheduler_type = "cosine"   # Learning rate schedule
max_steps = -1   # Number of training steps (overrides num_train_epochs)
warmup_ratio = 0.03   # Ratio of steps for a linear warmup (from 0 to learning rate)
group_by_length = False   # Group sequences into batches with same length (Saves memory and speeds up training considerably)
save_steps = 100   # Save checkpoint every X updates steps
logging_steps = 5   # Log every X updates steps  


# prompt template to be used for finetuning
IGNORE_INDEX = -100
template = {
    "instruction": "####Question: {instruction}",
    "response": " ####Answer: {response} ",
}


## SFT parameters
max_seq_length = 2048   # Maximum sequence length to use
packing = False   # Pack multiple short examples in the same input sequence to increase efficiency
device_map = {"": 0}   # Load the entire model on the GPU 0
NUM_CORES = 10

## CODE SNIPPET FOR WHEN USING NON-CHAT DATA (WILL ADD SUPPORT LATER)
# user_prompt_data_key = "question"
# assisstant_prompt_data_key = "answer"
# def formatting_func_packed(example, tokenizer):
#     text = FINETUNING_PROMPT.format(
#         instruction_template = template["instruction"],
#         instruction = example[user_prompt_data_key],
#         response_template = template["response"],
#         response = example[assisstant_prompt_data_key],
#     ) + tokenizer.eos_token
#     return text
# def formatting_func_unpacked(examples, tokenizer):
#     texts = [FINETUNING_PROMPT.format(
#                 instruction_template = template["instruction"],
#                 instruction = qs,
#                 response_template = template["response"],
#                 response = st,
#             ) + tokenizer.eos_token for qs, st in zip(examples[user_prompt_data_key], examples[assisstant_prompt_data_key]) ]
#     return texts
