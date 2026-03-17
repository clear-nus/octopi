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
from transformers import CLIPImageProcessor, get_cosine_schedule_with_warmup
from transformers.utils import logging
from evaluate_llm import LLMEvaluator, random_scores


def add_new_tokens(llm, tokenizer, new_tokens):
    new_tokens = list(set(new_tokens) - set(tokenizer.vocab.keys()))
    if len(new_tokens) == 0:
        return
    n_new_tokens = tokenizer.add_tokens(new_tokens)
    print(f"{n_new_tokens} tokens added to tokenizer.")
    llm.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        input_embeddings_avg = llm.model.embed_tokens.weight[:-n_new_tokens].mean(axis=0, keepdim=True)
        llm.model.embed_tokens.weight[-n_new_tokens:] = input_embeddings_avg


def evaluate_metrics(model, val_loader, device, tokenizer, configs):
    evaluator = LLMEvaluator()
    val_subset_size = configs.get("val_subset_size", None)
    if val_subset_size is not None:
        print(f"\nEvaluating metrics on validation subset ({val_subset_size} samples)...")
    else:
        print(f"\nEvaluating metrics on validation set...")
    model.eval()
    # Merge LoRA adapters for faster inference (especially important for DoRA)
    did_merge = False
    if configs.get("use_lora", False):
        if hasattr(model.llm, "merge_adapter"):
            try:
                print("Merging LoRA adapters for validation...")
                model.llm.merge_adapter()
                did_merge = True
            except Exception as e:
                print(f"Warning: Could not merge adapters: {e}")
        else:
            print("Warning: model.llm does not have merge_adapter method. Inference might be slow.")
    with torch.no_grad():
        val_loss_total = 0.0
        val_steps = 0
        for i, batch in enumerate(tqdm.tqdm(val_loader, desc="Evaluating metrics")):
            if val_subset_size is not None and i >= val_subset_size:
                break
            question, answer_tokens, tactile_frames, tactile, question_type, question_step, all_indices = batch
            answer_tokens = answer_tokens.to(device)
            outputs, question_embeds = model(question=question, tactile_frames=tactile_frames, answer_tokens=answer_tokens, all_indices=all_indices)
            val_loss_total += outputs.loss.item()
            val_steps += 1
            # Generation dynamically removed to speed up validation since we save on loss
    if did_merge:
        print("Unmerging LoRA adapters to resume training...")
        try:
            model.llm.unmerge_adapter()
        except Exception as e:
            print(f"Warning: Could not unmerge adapters: {e}")
    val_loss = val_loss_total / max(val_steps, 1)
    model.train()
    return val_loss


def run_evaluation(model, test_loader, device, tokenizer, configs, exp_name, file_suffix):
    print(f"\nEvaluating LLM on the test set ({file_suffix})...")
    model.eval()
    # Merge LoRA adapters for faster inference
    did_merge = False
    if configs.get("use_lora", False):
        if hasattr(model.llm, "merge_adapter"):
            try:
                print("Merging LoRA adapters for testing...")
                model.llm.merge_adapter()
                did_merge = True
            except Exception as e:
                print(f"Warning: Could not merge adapters: {e}")
        else:
            # If model.llm is not a PeftModel but we are in use_lora mode (e.g. merge_and_unload was called)
            # then we don't need to do anything.
            pass
    preds = []
    evaluator = LLMEvaluator()
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
            evaluator.evaluate(question="".join([i[0] for i in question]), generation=generation, answer=answer, question_type=question_type[0], question_step=question_step.item(), show_opd=False, show_pc=False, show_pss=False, show_pom=False, show_question=False)
    if did_merge:
        print("Unmerging LoRA adapters after testing...")
        try:
            model.llm.unmerge_adapter()
        except Exception as e:
            print(f"Warning: Could not unmerge adapters: {e}")
    with open(f'{configs["exps_path"]}/{exp_name}/{file_suffix}_preds.json', 'w') as f:
        json.dump(preds, f, indent=4)
        f.close()
    results = evaluator.get_results()
    with open(f'{configs["exps_path"]}/{exp_name}/{file_suffix}_results.txt', 'w') as f:
        for task, stats in results.items():
            f.write(f"{task}:\n")
            for stat, value in stats.items():
                if task in random_scores and stat in random_scores[task]:
                    f.write(f"\t{stat}: {value} ({random_scores[task][stat]})\n")
                else:
                    f.write(f"\t{stat}: {value}\n")
        f.close()
    print(f"Evaluation ({file_suffix}) done!")
    return results

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
                if configs["llm_path"].endswith(".pt"):
                    # if llm_path is a .pt file, we load the base model first
                    pass
                else:
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
            print("Loading LoRA trained model...")
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
            
            if len(tokenizer) > llm.get_input_embeddings().weight.shape[0]:
                llm.resize_token_embeddings(len(tokenizer))

            if configs["tokenizer_path"] is None:
                new_tokens = ['<tact_start>', '<tact_end>']
                add_new_tokens(llm, tokenizer, new_tokens)

            if configs["llm_path"] is not None and configs["llm_path"].endswith(".pt"):
                print(f"Loading LLM weights from {configs['llm_path']}...")
                llm_weights = torch.load(configs["llm_path"], map_location="cpu", weights_only=True)
                llm.load_state_dict(llm_weights, strict=False)

    # Fix generation config warnings
    if hasattr(llm, "generation_config"):
        llm.generation_config.do_sample = False
        llm.generation_config.temperature = None
        llm.generation_config.top_p = None

    # add new tokens
    if configs["tokenizer_path"] is None:
        # reference: https://jaotheboss.medium.com/domain-training-your-llm-6c77f53e3e27
        new_tokens = ['<tact_start>', '<tact_end>']
        add_new_tokens(llm, tokenizer, new_tokens)

    # load datasets
    if configs["use_clip"]:
        try:
            image_processor = CLIPImageProcessor.from_pretrained(configs["use_clip"])
        except OSError:
            print(f"Warning: Could not load CLIP processor from {configs['use_clip']} in offline mode. Trying without it or check your internet connection/cache.")
            # Depending on logic, we might need to crash or set to None
            # But the dataset needs it. Retrying with local_files_only=True might be redundant if mode is offline.
            raise
        except Exception as e:
            print(f"Error loading CLIP processor: {e}")
            raise
    if configs["train"]:
        train_dataset = TactileLLMDataset(image_processor, configs["train_files"], split_name="train", tokenizer=tokenizer, flip_p=configs["flip_p"])
        train_loader = DataLoader(train_dataset, batch_size=configs["per_device_train_batch_size"], shuffle=True, worker_init_fn=seed_worker, generator=g)
    if configs["val"]:
        val_dataset = TactileLLMDataset(image_processor, configs["val_files"], split_name="val", tokenizer=tokenizer, flip_p=configs["flip_p"])
        val_loader = DataLoader(val_dataset, batch_size=configs["per_device_val_batch_size"], shuffle=True, worker_init_fn=seed_worker, generator=g)
    if configs["test"]:
        test_dataset = TactileLLMDataset(image_processor, configs["test_files"], split_name="test", tokenizer=tokenizer, flip_p=configs["flip_p"])
        test_loader = DataLoader(test_dataset, batch_size=configs["per_device_val_batch_size"], shuffle=False, worker_init_fn=seed_worker, generator=g)

    # model instantiation
    if configs["lora_trained"]:
        model = MultimodalLLMForCausalLM(clip_model=configs["use_clip"], encoder_output_size=configs["encoder_output_size"], tokenizer=tokenizer, cutoff_len=configs["cutoff_len"], llm=llm.model, use_vqvae=configs["use_vqvae"], device=device)
    else:
        model = MultimodalLLMForCausalLM(clip_model=configs["use_clip"], encoder_output_size=configs["encoder_output_size"], tokenizer=tokenizer, cutoff_len=configs["cutoff_len"], llm=llm, use_vqvae=configs["use_vqvae"], device=device)
    
    # If using device_map (gpu_config present), we should not move the whole model to device
    # because the LLM parts are already placed. We only move the other modules.
    if configs["gpu_config"] is not None:
        model.encoder.to(device)
        model.project.to(device)
    else:
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
            modules_to_save=configs["modules_to_save"],
            use_dora=True
        )
        llm_weights_path = f"{configs['exps_path']}/{exp_name}/llm_weights"
        if not os.path.exists(llm_weights_path):
            # os.makedirs(llm_weights_path)
            # llm_peft = get_peft_model(llm, peft_config)
            # llm_peft.save_pretrained(llm_weights_path)
            # llm_peft = None
            
            # Instead just apply the config in-memory directly
            llm = get_peft_model(llm, peft_config)
            
        if configs["quantized"]:
            # If we are not loading from disk, ‘llm’ is already the PeftModel from above
            pass
            # llm = PeftModel.from_pretrained(model=llm, model_id=llm_weights_path, is_trainable=True, device_map="auto", max_memory=gpu_max_mem_config, quantization_config=bnb_config)
        else:
            # llm = PeftModel.from_pretrained(model=llm, model_id=llm_weights_path, is_trainable=True, device_map="auto", max_memory=gpu_max_mem_config)
             pass
        model.llm = llm
    else:
        model.llm = llm

    # 2) projection setup
    if configs["projection_path"] is not None:
        projection_dict = torch.load(configs["projection_path"], map_location='cpu', weights_only=True)
        model.project.load_state_dict(projection_dict)
    
    project_params = []
    if configs["freeze_projection"]:
        for name, param in model.project.named_parameters():
            param.requires_grad = False
    else:
        for name, param in model.project.named_parameters():
            param.requires_grad = True
        project_params = list(model.project.parameters())

    if configs["train"]:
        ## LLM optimizer
        llm_params = []
        if not configs["use_lora"]:
            for name, param in model.llm.named_parameters():
                # NOTE: no lm_head here since they are not tied to word embeddings in LLaMA and no new tokens for generation
                if "embed_tokens" in name:
                    param.requires_grad = True
                    # Only train new tokens
                    def make_hook(n_old_tokens):
                        def hook(grad):
                            # Zero out gradients for the old tokens
                            grad[:n_old_tokens] = 0
                            return grad
                        return hook
                    # Register the hook
                    n_old_tokens = len(tokenizer) - len(new_tokens)
                    param.register_hook(make_hook(n_old_tokens))
                else:
                    param.requires_grad = False
                if param.requires_grad:
                    llm_params.append(param)
        else:
            for name, param in model.llm.named_parameters():
                # NOTE: no lm_head here since they are not tied to word embeddings in LLaMA and no new tokens for generation
                if "embed_tokens" in name:
                    param.requires_grad = True
                    # Only train new tokens
                    def make_hook(n_old_tokens):
                        def hook(grad):
                            # Zero out gradients for the old tokens
                            grad[:n_old_tokens] = 0
                            return grad
                        return hook
                    # Register the hook
                    n_old_tokens = len(tokenizer) - len(new_tokens)
                    param.register_hook(make_hook(n_old_tokens))
                if param.requires_grad:
                    llm_params.append(param)
        print(f"len(llm_params): {len(llm_params)}")
        print(f"len(project_params): {len(project_params)}")
        
        optimizer_grouped_parameters = []
        if len(llm_params) > 0:
            optimizer_grouped_parameters.append({
                "params": llm_params,
                "lr": configs["llm_lr"]
            })
        if len(project_params) > 0:
            optimizer_grouped_parameters.append({
                "params": project_params,
                "lr": configs["projection_lr"]
            })

        if len(optimizer_grouped_parameters) > 0:
            optimizer_llm = torch.optim.AdamW(optimizer_grouped_parameters)
            if configs["max_train_steps"] < len(train_loader):
                num_training_steps = int(configs["max_train_steps"] / configs["llm_gradient_accumulation_steps"])
            else:
                num_training_steps = int(len(train_loader) / configs["llm_gradient_accumulation_steps"])
            if configs["warmup_steps"] < 1:
                num_warmup_steps = int(num_training_steps * configs["warmup_steps"])
            else:
                num_warmup_steps = int(configs["warmup_steps"])
            scheduler_llm = get_cosine_schedule_with_warmup(
                optimizer_llm, 
                num_warmup_steps=num_warmup_steps, 
                num_training_steps=num_training_steps
            )

    # 3) encoder setup
    if configs["use_vqvae"]:
        model.encoder.load_state_dict(torch.load("encoders/vqvae/encoder.pth", map_location='cpu', weights_only=True))
        model.vector_quantization.load_state_dict(torch.load("encoders/vqvae/vector_quantization.pth", map_location='cpu', weights_only=True))
    elif configs["encoder_path"] is not None:
        try:
            model.encoder.load_state_dict(torch.load(configs["encoder_path"], map_location='cpu', weights_only=True))
        except RuntimeError:
            clip = PromptLearningCLIPModel.from_pretrained(configs["use_clip"], configs).to(device)
            model.encoder.model.vision_model = clip.vision_model
            model.encoder.load_state_dict(torch.load(configs["encoder_path"], map_location='cpu', weights_only=True), strict=True)
    if configs["freeze_encoder"]:
        for name, param in model.encoder.named_parameters():
            param.requires_grad = False
    else:
        for name, param in model.encoder.named_parameters():
            param.requires_grad = True
        encoder_params = model.encoder.parameters()
        optimizer_encoder = torch.optim.SGD(encoder_params, lr=configs["encoder_lr"])

    # training
    if configs["train"]:
        best_val_loss = float('inf')
        patience_counter = 0
        # get trainable/non-trainable model parameter stats
        model.train()
        if configs["freeze_encoder"]:
            model.encoder.eval()
        if configs["freeze_projection"]:
            model.project.eval()
        trainable_model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        trainable_params = sum([np.prod(p.size()) for p in trainable_model_parameters])
        all_params = sum([np.prod(p.size()) for p in model.parameters()])
        # # NOTE: Print trainable parameter names
        # print("\nTrainable Parameters:")
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"- {name}: {param.shape}")
        # print("-" * 50)

        if configs["max_train_steps"] < len(train_loader):
            print(f"\nFinetuning LLM for {configs['max_train_steps']} samples and {int(configs['max_train_steps'] / configs['llm_gradient_accumulation_steps'])} gradient updates...")
        else:
            print(f"\nFinetuning LLM for {len(train_loader)} samples and {int(len(train_loader) / configs['llm_gradient_accumulation_steps'])} gradient updates...")
        print('Trainable params: {} ({:.2f}%)'.format(trainable_params, trainable_params / all_params * 100,))
        # NOTE: Cache original embeddings to prevent AdamW momentum drift on frozen tokens
        n_old_tokens = len(tokenizer) - len(new_tokens)
        original_embeddings = model.llm.get_input_embeddings().weight.data.clone().detach()
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
                    torch.nn.utils.clip_grad_norm_(encoder_params, max_norm=1.0)
                    optimizer_encoder.step()
                    optimizer_encoder.zero_grad()
                if len(optimizer_grouped_parameters) > 0:
                    torch.nn.utils.clip_grad_norm_(llm_params + project_params, max_norm=1.0)
                    optimizer_llm.step()
                    scheduler_llm.step()
                    optimizer_llm.zero_grad()
                    # Ensure frozen token embeddings are not corrupted by AdamW weight decay or variance tracking
                    if any("embed_tokens" in n for n, p in model.llm.named_parameters() if p.requires_grad):
                        with torch.no_grad():
                            model.llm.get_input_embeddings().weight.data[:n_old_tokens] = original_embeddings[:n_old_tokens]

            # validation
            if configs.get("val_freq") is not None and (train_sample_step + 1) % configs["val_freq"] == 0:
                if configs["val"]:
                    val_loss = evaluate_metrics(model, val_loader, device, tokenizer, configs)
                    if configs["freeze_encoder"]:
                        model.encoder.eval()
                    if configs["freeze_projection"]:
                        model.project.eval()
                    print(f"Validation Loss: {val_loss}")
                    
                    # Automate saving at every val step for post-training full evaluation
                    current_step = train_sample_step + 1
                    print(f"Saving checkpoint at step {current_step}...")
                    model.llm.generation_config.temperature = None
                    model.llm.generation_config.top_p = None
                    if len(llm_params) > 0:
                        if configs["use_lora"]:
                            model.llm.save_pretrained(f"{configs['exps_path']}/{exp_name}/step_{current_step}_llm_weights")
                        else:
                            torch.save({n: p for n, p in model.llm.named_parameters() if p.requires_grad}, f"{configs['exps_path']}/{exp_name}/step_{current_step}_llm_weights.pt")
                    torch.save(model.project.state_dict(), f"{configs['exps_path']}/{exp_name}/step_{current_step}_project.pt")
                    if not configs["freeze_encoder"]:
                        torch.save(model.encoder.state_dict(), f"{configs['exps_path']}/{exp_name}/step_{current_step}_encoder.pt")
                    tokenizer.save_pretrained(f"{configs['exps_path']}/{exp_name}/tokenizer")
            if (train_sample_step + 1) >= configs["max_train_steps"]:
                break
        # if not configs["val"]:
        #     # Save models
        #     print(f"\nNo validation during training. Saving final tokenizer and models...")
        #     tokenizer.save_pretrained(f"{configs['exps_path']}/{exp_name}/tokenizer")
        #     model.llm.generation_config.temperature = None
        #     model.llm.generation_config.top_p = None
        #     if len(llm_params) > 0:
        #         if configs["use_lora"]:
        #             model.llm.save_pretrained(f"{configs['exps_path']}/{exp_name}/llm_weights")
        #         else:
        #             torch.save({n: p for n, p in model.llm.named_parameters() if p.requires_grad}, f"{configs['exps_path']}/{exp_name}/llm_weights.pt")
        #     if not configs["freeze_encoder"]:
        #         torch.save(model.encoder.state_dict(), f"{configs['exps_path']}/{exp_name}/encoder.pt")
        #     torch.save(model.project.state_dict(), f"{configs['exps_path']}/{exp_name}/project.pt")
        print(f"LLM training done!")

    # test
    if configs["test"]:
        if configs["train"] and configs["val"] and configs.get("val_freq") is not None:
            print(f"\n=========================================")
            print(f"Post-Training Automated Checkpoint Sweep")
            print(f"=========================================")
            
            import glob
            import shutil
            import gc
            
            checkpoints = glob.glob(f"{configs['exps_path']}/{exp_name}/step_*_project.pt")
            steps = sorted([int(p.split('step_')[1].split('_project.pt')[0]) for p in checkpoints])
            print(f"Found {len(steps)} checkpoints to evaluate: {steps}")
            
            best_avg_acc = -1
            best_step = -1
            
            for step in steps:
                print(f"\n--- Testing Checkpoint Step {step} ---")
                
                # Reload Projection
                step_proj_path = f"{configs['exps_path']}/{exp_name}/step_{step}_project.pt"
                model.project.load_state_dict(torch.load(step_proj_path, map_location=device, weights_only=True))

                # Reload Encoder
                if not configs["freeze_encoder"]:
                    step_enc_path = f"{configs['exps_path']}/{exp_name}/step_{step}_encoder.pt"
                    model.encoder.load_state_dict(torch.load(step_enc_path, map_location=device))

                # Reload LLM
                if configs["use_lora"]:
                    step_llm_path = f"{configs['exps_path']}/{exp_name}/step_{step}_llm_weights"
                    try:
                        from peft import PeftModel
                        # If the model is already wrapped in PEFT, just load the weights
                        if hasattr(model.llm, "load_adapter"):
                            model.llm.load_adapter(step_llm_path, adapter_name="default")
                            model.llm.set_adapter("default")
                        else:
                            model.llm = PeftModel.from_pretrained(model.llm, step_llm_path)
                    except Exception as e:
                        print(f"Fallback Load: {e}")
                        model.llm.load_state_dict(torch.load(step_llm_path+'/adapter_model.bin', map_location=device), strict=False)
                else:
                    if len(llm_params) > 0:
                        step_llm_path = f"{configs['exps_path']}/{exp_name}/step_{step}_llm_weights.pt"
                        model.llm.load_state_dict(torch.load(step_llm_path, map_location=device), strict=False)
                
                # We do NOT merge the adapter here so we can swap it out easily for the next step loop
                
                # NOTE: We evaluate on the validation set to pick the best checkpoint to avoid test-set test data leakage
                results = run_evaluation(model, val_loader, device, tokenizer, configs, exp_name, f"val_step_{step}")
                
                # Extract combined accuracy
                tasks_to_average = []
                for k, v in results.items():
                    if "accuracy" in v:
                        tasks_to_average.append(v["accuracy"])
                    elif "combined_accuracy" in v:
                        tasks_to_average.append(v["combined_accuracy"])
                
                if len(tasks_to_average) > 0:
                    avg_acc = sum(tasks_to_average) / len(tasks_to_average)
                    print(f"--- Step {step} Average Accuracy: {avg_acc:.4f} ---")
                    if avg_acc > best_avg_acc:
                        best_avg_acc = avg_acc
                        best_step = step
                        
            print(f"\n=========================================")
            print(f"🏆 BEST CHECKPOINT: Step {best_step} with Avg Accuracy {best_avg_acc:.4f}")
            print(f"=========================================")
            
            # Save the winning checkpoint as the 'best_model'
            import shutil
            if best_step != -1:
                shutil.copy(f"{configs['exps_path']}/{exp_name}/step_{best_step}_project.pt", f"{configs['exps_path']}/{exp_name}/best_project.pt")
                if configs["use_lora"]:
                    os.makedirs(f"{configs['exps_path']}/{exp_name}/best_llm_weights", exist_ok=True)
                    shutil.copytree(f"{configs['exps_path']}/{exp_name}/step_{best_step}_llm_weights", f"{configs['exps_path']}/{exp_name}/best_llm_weights", dirs_exist_ok=True)
                else:
                    if len(llm_params) > 0:
                        shutil.copy(f"{configs['exps_path']}/{exp_name}/step_{best_step}_llm_weights.pt", f"{configs['exps_path']}/{exp_name}/best_llm_weights.pt")
                if not configs["freeze_encoder"]:
                    shutil.copy(f"{configs['exps_path']}/{exp_name}/step_{best_step}_encoder.pt", f"{configs['exps_path']}/{exp_name}/best_encoder.pt")
            
            print(f"\n--- Running Final Evaluation against Test Set using Step {best_step} ---")
            # We already have the best model loaded in memory from the end of the loop, or we can reload it:
            # Let's cleanly run it
            model.eval()
            run_evaluation(model, test_loader, device, tokenizer, configs, exp_name, "test_best")
        
        else:
            run_evaluation(model, test_loader, device, tokenizer, configs, exp_name, "test_best")
        
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
    
    if "EXP_ID" in os.environ:
        exp_id = os.environ["EXP_ID"]
    else:
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
    sys.stdout = open(f"{configs['exps_path']}/{exp_name}/log.txt", 'w', buffering=1)
    logging.set_verbosity_error()

    # seed
    torch.manual_seed(configs["seed"])
    torch.random.manual_seed(configs["seed"])
    torch.cuda.manual_seed(configs["seed"])
    torch.cuda.manual_seed_all(configs["seed"])
    # torch.use_deterministic_algorithms(True)
    np.random.seed(configs["seed"])
    random.seed(configs["seed"])
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(configs["seed"])

    train(configs, exp_name, g)