import os 
import torch.nn as nn 
import torch 
from torch.utils.data import DataLoader
from torch import optim
import tqdm
import json
import numpy as np
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
from accelerate import infer_auto_device_map, init_empty_weights
from utils.dataset import *
from utils.promptclip import *
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils.model import *
import random
import yaml
from datetime import datetime
import sys
from transformers import CLIPImageProcessor
from transformers.utils import logging


def add_new_tokens(llm, tokenizer, new_tokens):
    new_tokens = list(set(new_tokens) - set(tokenizer.vocab.keys()))
    n_new_tokens = tokenizer.add_tokens(new_tokens)
    print(f"{n_new_tokens} tokens added to tokenizer.")
    llm.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        input_embeddings_avg = llm.model.embed_tokens.weight[:-n_new_tokens].mean(axis=0, keepdim=True)
        llm.model.embed_tokens.weight[-n_new_tokens:] = input_embeddings_avg


def train(configs, exp_name, g):
    # device
    device = f'cuda:{configs["cuda"]}' # for inputs and model if not device_map
    new_tokens = ['<tact_start>', '<tact_end>']

    # load tokenizer and LLM weights
    if configs["model_type"] == "vicuna-7b":
        tokenizer_path = "lmsys/vicuna-7b-v1.5"
        model_path = "lmsys/vicuna-7b-v1.5"
    elif configs["model_type"] == "vicuna-13b":
        tokenizer_path = "lmsys/vicuna-13b-v1.5"
        model_path = "lmsys/vicuna-13b-v1.5"
    
    # model GPU and tokenizer setup
    os.makedirs(configs["offload_dir"], exist_ok=True)
    if configs["quantized"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    if configs["gpu_config"] is not None:
        if configs["tokenizer_path"] is not None:
            tokenizer_path = configs["tokenizer_path"]
        if not configs["lora_trained"]:
            if configs["llm_path"] is not None:
                model_path = configs["llm_path"]
        with init_empty_weights():
            config = AutoConfig.from_pretrained(model_path)
            auto_model = AutoModelForCausalLM.from_config(config)
        f = open(configs["gpu_config"])
        data = json.load(f)
        gpu_max_mem_config = {}
        for k, v in data.items():
            gpu_max_mem_config[int(k)] = v
        device_map = infer_auto_device_map(
            auto_model, max_memory = gpu_max_mem_config, no_split_module_classes=["LLaMADecoderLayer", "LlamaDecoderLayer"]
        )
        if configs["lora_trained"]:
            if configs["quantized"]:
                llm = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, offload_folder=configs["offload_dir"], quantization_config=bnb_config)
            else:
                llm = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, offload_folder=configs["offload_dir"])
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_auth_token=True, padding_side="left")
            # reference: https://jaotheboss.medium.com/domain-training-your-llm-6c77f53e3e27
            add_new_tokens(llm, tokenizer, new_tokens)
            if configs["quantized"]:
                llm = PeftModel.from_pretrained(model=llm, model_id=configs["llm_path"], is_trainable=False, device_map="auto", max_memory=gpu_max_mem_config, quantization_config=bnb_config)
            else:
                llm = PeftModel.from_pretrained(model=llm, model_id=configs["llm_path"], is_trainable=False, device_map="auto", max_memory=gpu_max_mem_config)
        else:
            if configs["quantized"]:
                llm = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, offload_folder=configs["offload_dir"], quantization_config=bnb_config)
            else:
                llm = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, offload_folder=configs["offload_dir"])
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")

    # add new tokens
    if configs["tokenizer_path"] is None:
        # reference: https://jaotheboss.medium.com/domain-training-your-llm-6c77f53e3e27
        new_tokens = ['<tact_start>', '<tact_end>']
        add_new_tokens(llm, tokenizer, new_tokens)

    # load datasets
    if configs["use_clip"]:
        image_processor = CLIPImageProcessor.from_pretrained(configs["use_clip"])
    if configs["train"]:
        train_dataset = TactileLLMDataset(image_processor, configs["train_files"], split_name="train", tokenizer=tokenizer, flip_p=configs["flip_p"])
        train_loader = DataLoader(train_dataset, batch_size=configs["per_device_train_batch_size"], shuffle=True, worker_init_fn=seed_worker, generator=g)
    if configs["val"]:
        val_dataset = TactileLLMDataset(image_processor, configs["val_files"], split_name="val", tokenizer=tokenizer, flip_p=configs["flip_p"])
        val_loader = DataLoader(val_dataset, batch_size=configs["per_device_val_batch_size"], shuffle=False, worker_init_fn=seed_worker, generator=g)
    if configs["test"]:
        test_dataset = TactileLLMDataset(image_processor, configs["test_files"], split_name="test", tokenizer=tokenizer, flip_p=configs["flip_p"])
        test_loader = DataLoader(test_dataset, batch_size=configs["per_device_val_batch_size"], shuffle=False, worker_init_fn=seed_worker, generator=g)

    # model instantiation
    if configs["lora_trained"]:
        model = MultimodalLLMForCausalLM(clip_model=configs["use_clip"], encoder_output_size=configs["encoder_output_size"], tokenizer=tokenizer, cutoff_len=configs["cutoff_len"], llm=llm.model, use_vqvae=configs["use_vqvae"], device=device)
    else:
        model = MultimodalLLMForCausalLM(clip_model=configs["use_clip"], encoder_output_size=configs["encoder_output_size"], tokenizer=tokenizer, cutoff_len=configs["cutoff_len"], llm=llm, use_vqvae=configs["use_vqvae"], device=device)
    model.to(device)

    # 1) LLM setup
    if configs["use_lora"]:
        ## LoRA
        peft_config = LoraConfig(
            r=configs["r"],
            lora_alpha=configs["lora_alpha"],
            lora_dropout=configs["lora_dropout"],
            target_modules=configs["target_modules"],
            bias=configs["bias"],
            inference_mode=False,
            task_type="CAUSAL_LM",
            modules_to_save=configs["modules_to_save"]
        )
        llm_weights_path = f"{configs['exps_path']}/{exp_name}/llm_weights"
        if not os.path.exists(llm_weights_path):
            os.makedirs(llm_weights_path)
            llm_peft = get_peft_model(llm, peft_config)
            llm_peft.save_pretrained(llm_weights_path)
            llm_peft = None
        if configs["quantized"]:
            llm = PeftModel.from_pretrained(model=llm, model_id=llm_weights_path, is_trainable=True, device_map="auto", max_memory=gpu_max_mem_config, quantization_config=bnb_config)
        else:
            llm = PeftModel.from_pretrained(model=llm, model_id=llm_weights_path, is_trainable=True, device_map="auto", max_memory=gpu_max_mem_config)
        model.llm = llm
    else:
        model.llm = llm
    if configs["train"]:
        ## LLM optimizer
        llm_params = []
        if not configs["use_lora"]:
            for name, param in model.llm.named_parameters():
                # NOTE: no lm_head here since they are not tied to word embeddings in LLaMA and no new tokens for generation
                if "embed_tokens" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                if param.requires_grad:
                    llm_params.append(param)
        else:
            for name, param in model.llm.named_parameters():
                if param.requires_grad:
                    llm_params.append(param)
        if len(llm_params) > 0:
            optimizer_llm = torch.optim.AdamW(llm_params, lr=configs["llm_lr"])
            num_steps = int(len(train_loader) / configs["llm_gradient_accumulation_steps"])
            scheduler_llm = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_llm, T_max=num_steps)

    # 2) encoder setup
    if configs["use_vqvae"]:
        model.encoder.load_state_dict(torch.load("encoders/vqvae/encoder.pth"))
        model.vector_quantization.load_state_dict(torch.load("encoders/vqvae/vector_quantization.pth"))
    elif configs["encoder_path"] is not None:
        try:
            model.encoder.load_state_dict(torch.load(configs["encoder_path"]))
        except RuntimeError:
            clip = PromptLearningCLIPModel.from_pretrained(configs["use_clip"], configs).to(device)
            model.encoder.model.vision_model = clip.vision_model
            model.encoder.load_state_dict(torch.load(configs["encoder_path"]), strict=False)
    if configs["freeze_encoder"]:
        for name, param in model.encoder.named_parameters():
            param.requires_grad = False
    else:
        for name, param in model.encoder.named_parameters():
            param.requires_grad = True
        encoder_params = model.encoder.parameters()
        optimizer_encoder = torch.optim.SGD(encoder_params, lr=configs["encoder_lr"])

    # 3) projection setup
    if configs["projection_path"] is not None:
        projection_dict = torch.load(configs["projection_path"])
        model.project.load_state_dict(projection_dict)
    if configs["freeze_projection"]:
        for name, param in model.project.named_parameters():
            param.requires_grad = False
    else:
        for name, param in model.project.named_parameters():
            param.requires_grad = True
        project_params = model.project.parameters()
        optimizer_project = torch.optim.AdamW(project_params, lr=configs["projection_lr"])

    # training
    if configs["train"]:
        # get trainable/non-trainable model parameter stats
        model.train()
        trainable_model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        trainable_params = sum([np.prod(p.size()) for p in trainable_model_parameters])
        all_params = sum([np.prod(p.size()) for p in model.parameters()])
        if configs["max_train_steps"] < len(train_loader):
            print(f"\nFinetuning LLM for {configs['max_train_steps']} samples and {int(configs['max_train_steps'] / configs['llm_gradient_accumulation_steps'])} gradient updates...")
        else:
            print(f"\nFinetuning LLM for {len(train_loader)} samples and {int(len(train_loader) / configs['llm_gradient_accumulation_steps'])} gradient updates...")
        print('Trainable params: {} ({:.2f}%)'.format(trainable_params, trainable_params / all_params * 100,))
        # total_train_loss = 0
        # NOTE: do not calculate stats during training to save time
        for train_sample_step, batch in enumerate(t:=tqdm.tqdm(train_loader)):
            question, answer_tokens, tactile_frames, tactile, question_type, question_step, all_indices = batch
            answer_tokens = answer_tokens.to(device)
            outputs, _ = model(question=question, tactile_frames=tactile_frames, answer_tokens=answer_tokens, all_indices=all_indices)
            train_loss = outputs.loss.detach().float()
            t.set_description(f"Train loss: {train_loss}")
            # total_train_loss += train_loss # NOTE: hardcoded for batch size of 1
            loss = outputs.loss / configs["llm_gradient_accumulation_steps"]
            loss.backward()
            if (train_sample_step + 1) % configs["llm_gradient_accumulation_steps"] == 0:
                # optimizer updates
                if not configs["freeze_encoder"]:
                    optimizer_encoder.step()
                    optimizer_encoder.zero_grad()
                if not configs["freeze_projection"]:
                    optimizer_project.step()
                    optimizer_project.zero_grad()
                if len(llm_params) > 0:
                    optimizer_llm.step()
                    scheduler_llm.step()
                    optimizer_llm.zero_grad()
            if configs["save_freq"] is not None:
                if train_sample_step != 0 and (train_sample_step + 1) % configs["save_freq"] == 0:
                     # save models
                    print("Saving tokenizer and models...")
                    tokenizer.save_pretrained(f"{configs['exps_path']}/{exp_name}/tokenizer_{train_sample_step + 1}")
                    model.llm.save_pretrained(f"{configs['exps_path']}/{exp_name}/llm_weights_{train_sample_step + 1}")
                    # if configs["newton"] is False:
                    torch.save(model.encoder.state_dict(), f"{configs['exps_path']}/{exp_name}/encoder_{train_sample_step + 1}.pt")
                    torch.save(model.project.state_dict(), f"{configs['exps_path']}/{exp_name}/project_{train_sample_step + 1}.pt")
            if (train_sample_step + 1) >= configs["max_train_steps"]:
                break
        if configs["save_freq"] is None:
            # save models
            print("Saving tokenizer and models...")
            tokenizer.save_pretrained(f"{configs['exps_path']}/{exp_name}/tokenizer")
            model.llm.generation_config.temperature = None
            model.llm.generation_config.top_p = None
            model.llm.save_pretrained(f"{configs['exps_path']}/{exp_name}/llm_weights")
            # if configs["newton"] is False:
            torch.save(model.encoder.state_dict(), f"{configs['exps_path']}/{exp_name}/encoder.pt")
            torch.save(model.project.state_dict(), f"{configs['exps_path']}/{exp_name}/project.pt")
            print(f"LLM training done!")

    # validation
    if configs["val"]:
        print(f"\nEvaluating LLM on the validation set...")
        model.eval()
        preds = []
        with torch.no_grad():
            for val_sample_step, batch in enumerate(tqdm.tqdm(val_loader)):
                # NOTE: hardcoded for batch size of 1
                question, answer_tokens, tactile_frames, tactile, question_type, question_step, all_indices = batch
                answer_tokens = answer_tokens.to(device)
                outputs, question_embeds = model(question=question, tactile_frames=tactile_frames, answer_tokens=answer_tokens, all_indices=all_indices)
                max_new_tokens = configs["max_new_tokens"][question_type[0]]
                generation_tokens = model.llm.generate(inputs_embeds=question_embeds, max_new_tokens=max_new_tokens, temperature=None)
                generation = tokenizer.decode(generation_tokens[0], skip_special_tokens=True).strip() # https://huggingface.co/docs/transformers/main/llm_tutorial
                generation = generation.strip().split("</s>")[0].strip()
                if "</s>" not in generation:
                    generation += "</s>"
                answer_tokens = answer_tokens[0].cpu().numpy()
                answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
                preds.append({
                    "question": "".join([i[0] for i in question]),
                    "question_type": question_type[0],
                    "question_step": question_step.item(),
                    "sample_paths": [i[0] for i in tactile],
                    "answer": answer,
                    "generation": generation
                })
            with open(f'{configs["exps_path"]}/{exp_name}/val_preds.json', 'w') as f:
                json.dump(preds, f, indent=4)
                f.close()
        print(f"LLM validation done!")

    # test
    if configs["test"]:
        print(f"\nTesting LLM on the test set...")
        model.eval()
        preds = []
        with torch.no_grad():
            for test_sample_step, batch in enumerate(tqdm.tqdm(test_loader)):
                # NOTE: hardcoded for batch size of 1
                question, answer_tokens, tactile_frames, tactile, question_type, question_step, all_indices = batch
                answer_tokens = answer_tokens.to(device)
                outputs, question_embeds = model(question=question, tactile_frames=tactile_frames, answer_tokens=answer_tokens, all_indices=all_indices)
                max_new_tokens = configs["max_new_tokens"][question_type[0]]
                generation_tokens = model.llm.generate(inputs_embeds=question_embeds, max_new_tokens=max_new_tokens, temperature=None)
                generation = tokenizer.decode(generation_tokens[0], skip_special_tokens=True).strip() # https://huggingface.co/docs/transformers/main/llm_tutorial
                answer_tokens = answer_tokens[0].cpu().numpy()
                answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
                generation = generation.strip().split("</s>")[0].strip()
                if "</s>" not in generation:
                    generation += "</s>"
                preds.append({
                    "question": "".join([i[0] for i in question]),
                    "question_type": question_type[0],
                    "question_step": question_step.item(),
                    "sample_paths": [i[0] for i in tactile],
                    "answer": answer,
                    "generation": generation
                })
            with open(f'{configs["exps_path"]}/{exp_name}/test_preds.json', 'w') as f:
                json.dump(preds, f, indent=4)
                f.close()
        print(f"LLM test done!")


if __name__ == "__main__":
    exp_type = f"train_llm"
    config_path = f'configs/{exp_type}_config.yaml'
    # get configs
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    if configs["train"]:
        exp_type = exp_type + "_train"
    if configs["val"]:
        exp_type = exp_type + "_val"
    if configs["test"]:
        exp_type = exp_type + "_test"
    if configs["use_lora"]:
        exp_type = exp_type + f"_lora_{configs['lora_alpha']}_{configs['r']}"
    exp_type = exp_type + f"_{configs['model_type']}"
    if configs["train"]:
        exp_type += f"_{configs['max_train_steps']}"
    exp_id = input("Identifier for experiment: ")
    if len(exp_id) > 0:
        exp_id = exp_type + f"_{exp_id}"
    else:
        exp_id = exp_type

    # make stats and weights folders
    now = datetime.now()
    exp_name = now.strftime("%Y_%m_%d_%H_%M_%S")
    exp_name = exp_name + "_" + exp_id
    print(f"\n{exp_name}\n")
    os.makedirs(f"{configs['exps_path']}", exist_ok=True)
    os.makedirs(f"{configs['exps_path']}/{exp_name}", exist_ok=True)
    with open(f"{configs['exps_path']}/{exp_name}/{exp_type}_config.yaml", 'w') as file:
        documents = yaml.dump(configs, file)
        file.close()

    # log outputs
    sys.stdout = open(f"{configs['exps_path']}/{exp_name}/log.txt", 'w')
    logging.set_verbosity_error()

    # seed
    torch.manual_seed(configs["seed"])
    torch.random.manual_seed(configs["seed"])
    torch.cuda.manual_seed(configs["seed"])
    torch.cuda.manual_seed_all(configs["seed"])
    # torch.use_deterministic_algorithms(True)
    random.seed(configs["seed"])
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(configs["seed"])

    train(configs, exp_name, g)