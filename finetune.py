# Getting this error sometimes, set correct env variables
import os
os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from datetime import datetime
from functools import partial

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model

# finetuning config args
import finetuning_config

# preprocessing step for dataset
def preprocess_supervised_dataset(examples, max_length, train_on_prompt):  
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}  

    # iterating through every conversation
    for chat in examples["messages"]:
        input_ids, attention_mask, label_ids = [], [], []

        # Iterate over each message in the dataset
        for i, msg in enumerate(chat):
            if i % 2 == 0:
                msg_chatml = finetuning_config.template["instruction"].format(instruction = msg)
            else:
                msg_chatml = finetuning_config.template["response"].format(response = msg)
    
            msg_tokenized = tokenizer(
                                msg_chatml,
                                add_special_tokens = False,
                            )
    
            input_ids += msg_tokenized["input_ids"]
            attention_mask += msg_tokenized["attention_mask"]
    
            if train_on_prompt:
                label_ids += msg_tokenized["input_ids"]
            else:
                if i % 2 == 0:
                    label_ids += [finetuning_config.IGNORE_INDEX] * len(msg_tokenized["input_ids"])
                else:
                    label_ids += msg_tokenized["input_ids"]
  
        # add eos_token to all
        input_ids.append(tokenizer.eos_token_id)
        label_ids.append(tokenizer.eos_token_id)
        attention_mask.append(1)

        # only accept samples within max_length range
        if len(input_ids) <= max_length:
            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(label_ids)
            model_inputs["attention_mask"].append(attention_mask)

    return model_inputs


if __name__ == "__main__":

    # load dataset
    dataset = load_dataset(
                finetuning_config.dataset_name,
                split = "train",
                token = finetuning_config.hf_token,
            ).shuffle(seed = 42)
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
                    finetuning_config.base_model,
                    trust_remote_code = True,
                    token = finetuning_config.hf_token,
                )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # tokenize the dataset
    dataset = dataset.map(  
                    partial(preprocess_supervised_dataset, max_length = finetuning_config.max_seq_length, train_on_prompt = False),
                    batched = True,
                    remove_columns = dataset.column_names,
                    num_proc = finetuning_config.NUM_CORES,
                )

    # BitsAndBytes config ( model quantisation config )
    bnb_config = BitsAndBytesConfig(
                    load_in_8bit = finetuning_config.use_8bit,
                    load_in_4bit = finetuning_config.use_4bit,
                    bnb_4bit_quant_type = finetuning_config.bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype = finetuning_config.bnb_4bit_compute_dtype,
                    bnb_4bit_use_double_quant = finetuning_config.use_nested_quant,
                )
    
    # load base model
    model = AutoModelForCausalLM.from_pretrained(
                finetuning_config.base_model,
                quantization_config = bnb_config,
                device_map = finetuning_config.device_map,
                attn_implementation = "flash_attention_2",
                low_cpu_mem_usage = True,
                token = finetuning_config.hf_token,
            )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False

    # load LoRA configuration
    lora_config = LoraConfig(
                    lora_alpha = finetuning_config.lora_r * 2,
                    lora_dropout = finetuning_config.lora_dropout,
                    r = finetuning_config.lora_r,
                    bias = "none",
                    task_type = "CAUSAL_LM",
                    target_modules = finetuning_config.lora_target_modules,
                )
    
    # attach LoRA adapter to model
    model = get_peft_model(model, lora_config)

    # for logging purposes
    current_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M')
    working_dir = f"{finetuning_config.output_dir}{current_datetime}"

    # set training parameters
    training_arguments = Seq2SeqTrainingArguments(
                            output_dir = working_dir,
                            num_train_epochs = finetuning_config.num_train_epochs,
                            per_device_train_batch_size = finetuning_config.per_device_train_batch_size,
                            gradient_accumulation_steps = finetuning_config.gradient_accumulation_steps,
                            optim = finetuning_config.optim,
                            save_steps = finetuning_config.save_steps,
                            logging_steps = finetuning_config.logging_steps,
                            learning_rate = finetuning_config.learning_rate,
                            weight_decay = finetuning_config.weight_decay,
                            fp16 = finetuning_config.fp16,
                            bf16 = finetuning_config.bf16,
                            max_grad_norm = finetuning_config.max_grad_norm,
                            max_steps = finetuning_config.max_steps,
                            warmup_ratio = finetuning_config.warmup_ratio,
                            group_by_length = finetuning_config.group_by_length,
                            lr_scheduler_type = finetuning_config.lr_scheduler_type,
                            adam_beta1 = finetuning_config.adam_beta1,
                            adam_beta2 = finetuning_config.adam_beta2,
                            adam_epsilon = finetuning_config.adam_epsilon,
                            logging_dir = f"{working_dir}/logs",
                            report_to = "tensorboard",
                        )
    
    # data collator definiton
    data_collator = DataCollatorForSeq2Seq(
                        tokenizer = tokenizer,
                        pad_to_multiple_of = 8,
                        label_pad_token_id = finetuning_config.IGNORE_INDEX,
                    )

    # init LLM trainer
    trainer = Trainer(
                model = model,
                train_dataset = dataset,
                tokenizer = tokenizer,
                args = training_arguments,
                data_collator = data_collator,
            )

    # train model
    trainer.train()

    # save trained model
    trainer.model.save_pretrained(f"{working_dir}/checkpoint-final")

