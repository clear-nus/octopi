import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Sampler
from torch import optim
import tqdm
import json
import numpy as np
from collections import defaultdict, Counter
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, set_peft_model_state_dict
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


def llm_task_group(question_type):
    if question_type.endswith("_object_property_description"):
        return "opd"
    if question_type.endswith("_property_object_match"):
        return "pom"
    if question_type.endswith("_property_comparison"):
        return "pc"
    if "property_superlative_selection" in question_type:
        return "pss"
    return "other"


def get_conclusion_loss_weight(configs, question_type):
    global_weight = configs.get("conclusion_loss_weight", 0.0)
    pom_weight = configs.get("pom_conclusion_loss_weight", None)
    task_groups = configs.get("conclusion_loss_task_groups", None)
    task_group = llm_task_group(question_type)
    if task_groups is not None and task_group not in task_groups:
        return 0.0
    if pom_weight is not None and task_group == "pom":
        return pom_weight
    return global_weight


class TaskBalancedSampler(Sampler):
    def __init__(self, dataset, accumulation_steps, seed=0):
        self.dataset = dataset
        self.accumulation_steps = accumulation_steps
        self.seed = seed
        self.groups = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            self.groups[llm_task_group(sample[0]["question_type"])].append(idx)
        self.group_names = sorted(self.groups.keys())
        self.per_group = max(1, accumulation_steps // max(len(self.group_names), 1))
        self.total_size = len(dataset)

    def __iter__(self):
        rng = random.Random(self.seed)
        groups = {name: indices[:] for name, indices in self.groups.items()}
        cursors = {name: 0 for name in self.group_names}
        for indices in groups.values():
            rng.shuffle(indices)
        yielded = 0
        while yielded < self.total_size:
            window = []
            for name in self.group_names:
                indices = groups[name]
                if cursors[name] + self.per_group > len(indices):
                    rng.shuffle(indices)
                    cursors[name] = 0
                start = cursors[name]
                end = start + self.per_group
                window.extend(indices[start:end])
                cursors[name] = end
            rng.shuffle(window)
            for idx in window:
                if yielded >= self.total_size:
                    break
                yielded += 1
                yield idx

    def __len__(self):
        return self.total_size


def write_llm_results(results, path):
    with open(path, 'w') as f:
        for task, stats in results.items():
            f.write(f"{task}:\n")
            for stat, value in stats.items():
                if task in random_scores and stat in random_scores[task]:
                    f.write(f"\t{stat}: {value} ({random_scores[task][stat]})\n")
                else:
                    f.write(f"\t{stat}: {value}\n")


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
        task_loss_totals = {}
        task_loss_counts = {}
        for i, batch in enumerate(tqdm.tqdm(val_loader, desc="Evaluating metrics")):
            if val_subset_size is not None and i >= val_subset_size:
                break
            question, answer_tokens, tactile_frames, tactile, question_type, question_step, all_indices, conclusion_start, property_labels = batch
            answer_tokens = answer_tokens.to(device)
            conclusion_weight = get_conclusion_loss_weight(configs, question_type[0])
            cs = conclusion_start if (configs.get("conclusion_only_loss", False) or conclusion_weight > 0) else None
            outputs, question_embeds = model(question=question, tactile_frames=tactile_frames, answer_tokens=answer_tokens, all_indices=all_indices,
                                             conclusion_start=cs, conclusion_only=configs.get("conclusion_only_loss", False),
                                             conclusion_loss_weight=conclusion_weight)
            val_loss_total += outputs.loss.item()
            val_steps += 1
            task = question_type[0]
            task_loss_totals[task] = task_loss_totals.get(task, 0.0) + outputs.loss.item()
            task_loss_counts[task] = task_loss_counts.get(task, 0) + 1
            # Generation dynamically removed to speed up validation since we save on loss
    if did_merge:
        print("Unmerging LoRA adapters to resume training...")
        try:
            model.llm.unmerge_adapter()
        except Exception as e:
            print(f"Warning: Could not unmerge adapters: {e}")
    val_loss = val_loss_total / max(val_steps, 1)
    balanced_val_loss = None
    if task_loss_totals:
        print("Validation Loss by task:")
        for task in sorted(task_loss_totals):
            task_loss = task_loss_totals[task] / max(task_loss_counts[task], 1)
            print(f"  {task}: {task_loss:.4f} ({task_loss_counts[task]} samples)")
        balanced_val_loss = np.mean([
            task_loss_totals[task] / max(task_loss_counts[task], 1)
            for task in task_loss_totals
        ])
        print(f"Balanced validation loss: {balanced_val_loss:.4f}")
    model.train()
    return val_loss, balanced_val_loss


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
    exp_dir = f'{configs["exps_path"]}/{exp_name}'
    partial_jsonl_path = f'{exp_dir}/{file_suffix}_partial_preds.jsonl'
    partial_json_path = f'{exp_dir}/{file_suffix}_partial_preds.json'
    partial_results_path = f'{exp_dir}/{file_suffix}_partial_results.txt'
    save_freq = configs.get("eval_save_freq", 10)
    with torch.no_grad():
        partial_f = open(partial_jsonl_path, 'w')
        for test_sample_step, batch in enumerate(tqdm.tqdm(test_loader)):
            if configs.get("val_subset_size") is not None and test_sample_step >= configs["val_subset_size"]:
                break
            # NOTE: hardcoded for batch size of 1
            question, answer_tokens, tactile_frames, tactile, question_type, question_step, all_indices, conclusion_start, property_labels = batch
            answer_tokens = answer_tokens.to(device)
            conclusion_weight = get_conclusion_loss_weight(configs, question_type[0])
            cs = conclusion_start if (configs.get("conclusion_only_loss", False) or conclusion_weight > 0) else None
            outputs, question_embeds = model(question=question, tactile_frames=tactile_frames, answer_tokens=answer_tokens, all_indices=all_indices,
                                             conclusion_start=cs, conclusion_only=configs.get("conclusion_only_loss", False),
                                             conclusion_loss_weight=conclusion_weight)
            max_new_tokens = configs["max_new_tokens"][question_type[0]]
            llm_dtype = model.llm.get_input_embeddings().weight.dtype
            generation_tokens = model.llm.generate(inputs_embeds=question_embeds.to(llm_dtype), max_new_tokens=max_new_tokens, temperature=None)
            generation = tokenizer.decode(generation_tokens[0], skip_special_tokens=True).strip() # https://huggingface.co/docs/transformers/main/llm_tutorial
            answer_tokens = answer_tokens[0].cpu().numpy()
            answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
            generation = generation.strip().split("</s>")[0].strip()
            if "</s>" not in generation:
                generation += "</s>"
            pred = {
                "question": "".join([i[0] for i in question]),
                "question_type": question_type[0],
                "question_step": question_step.item(),
                "sample_paths": [i[0] for i in tactile],
                "answer": answer,
                "generation": generation
            }
            preds.append(pred)
            evaluator.evaluate(question="".join([i[0] for i in question]), generation=generation, answer=answer, question_type=question_type[0], question_step=question_step.item(), show_opd=False, show_pc=False, show_pss=False, show_pom=False, show_question=False)
            partial_f.write(json.dumps(pred) + "\n")
            partial_f.flush()
            if save_freq and (test_sample_step + 1) % save_freq == 0:
                with open(partial_json_path, 'w') as f:
                    json.dump(preds, f, indent=4)
                write_llm_results(evaluator.get_results(), partial_results_path)
        partial_f.close()
    if did_merge:
        print("Unmerging LoRA adapters after testing...")
        try:
            model.llm.unmerge_adapter()
        except Exception as e:
            print(f"Warning: Could not unmerge adapters: {e}")
    with open(f'{exp_dir}/{file_suffix}_preds.json', 'w') as f:
        json.dump(preds, f, indent=4)
        f.close()
    results = evaluator.get_results()
    write_llm_results(results, f'{exp_dir}/{file_suffix}_results.txt')
    print(f"Evaluation ({file_suffix}) done!")
    return results

def _extract_key(generation, question_type):
    """Extract the scoreable portion of a generation for majority voting."""
    gen = generation.split("</s>")[0].strip()
    if question_type == "eval_object_property_description":
        parts = gen.split("presents")
        return parts[-1].strip() if len(parts) > 1 else gen
    parts = gen.split("Conclusion: ")
    return parts[-1].strip() if len(parts) > 1 else gen


def run_evaluation_tta(model, test_files, image_processor, device, tokenizer, configs, exp_name, file_suffix):
    n_passes = configs.get("tta_passes", 1)
    print(f"\nRunning TTA evaluation ({n_passes} passes) on {file_suffix}...")
    did_merge = False
    if configs.get("use_lora", False) and hasattr(model.llm, "merge_adapter"):
        try:
            model.llm.merge_adapter()
            did_merge = True
        except Exception as e:
            print(f"Warning: Could not merge adapters: {e}")
    model.eval()

    all_generations = []  # n_passes x n_samples
    all_meta = None

    for pass_idx in range(n_passes):
        dataset = TactileLLMDataset(
            image_processor, test_files, split_name="test",
            tokenizer=tokenizer, flip_p=0, random_frames=(pass_idx > 0),
            max_frames=configs.get("max_frames", 5)
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        pass_gens = []
        pass_meta = []
        with torch.no_grad():
            for batch in tqdm.tqdm(loader, desc=f"TTA pass {pass_idx + 1}/{n_passes}"):
                question, answer_tokens, tactile_frames, tactile, question_type, question_step, all_indices, conclusion_start, property_labels = batch
                answer_tokens = answer_tokens.to(device)
                conclusion_weight = get_conclusion_loss_weight(configs, question_type[0])
                cs = conclusion_start if (configs.get("conclusion_only_loss", False) or conclusion_weight > 0) else None
                _, question_embeds = model(question=question, tactile_frames=tactile_frames, answer_tokens=answer_tokens, all_indices=all_indices,
                                           conclusion_start=cs, conclusion_only=configs.get("conclusion_only_loss", False),
                                           conclusion_loss_weight=conclusion_weight)
                max_new_tokens = configs["max_new_tokens"][question_type[0]]
                llm_dtype = model.llm.get_input_embeddings().weight.dtype
                generation_tokens = model.llm.generate(inputs_embeds=question_embeds.to(llm_dtype), max_new_tokens=max_new_tokens, temperature=None)
                generation = tokenizer.decode(generation_tokens[0], skip_special_tokens=True).strip()
                generation = generation.split("</s>")[0].strip()
                if "</s>" not in generation:
                    generation += "</s>"
                pass_gens.append(generation)
                if pass_idx == 0:
                    answer = tokenizer.decode(answer_tokens[0].cpu().numpy(), skip_special_tokens=True).strip()
                    pass_meta.append({
                        "question": "".join([i[0] for i in question]),
                        "question_type": question_type[0],
                        "question_step": question_step.item(),
                        "sample_paths": [i[0] for i in tactile],
                        "answer": answer,
                    })
        all_generations.append(pass_gens)
        if pass_idx == 0:
            all_meta = pass_meta

    if did_merge:
        try:
            model.llm.unmerge_adapter()
        except Exception as e:
            print(f"Warning: Could not unmerge adapters: {e}")

    from collections import Counter
    evaluator = LLMEvaluator()
    preds = []
    for i, meta in enumerate(all_meta):
        qt = meta["question_type"]
        keys = [_extract_key(all_generations[p][i], qt) for p in range(n_passes)]
        majority_key = Counter(keys).most_common(1)[0][0]
        base_gen = all_generations[0][i].split("</s>")[0].strip()
        if qt == "eval_object_property_description" and "presents" in base_gen:
            idx = base_gen.rfind("presents")
            merged_gen = base_gen[:idx + len("presents")] + " " + majority_key
        elif "Conclusion: " in base_gen:
            merged_gen = base_gen.split("Conclusion: ")[0] + "Conclusion: " + majority_key
        else:
            merged_gen = majority_key
        merged_gen += "</s>"
        pred = dict(meta)
        pred["generation"] = merged_gen
        pred["tta_keys"] = keys
        preds.append(pred)
        evaluator.evaluate(
            question=meta["question"], generation=merged_gen, answer=meta["answer"],
            question_type=qt, question_step=meta["question_step"],
            show_opd=False, show_pc=False, show_pss=False, show_pom=False, show_question=False
        )

    with open(f'{configs["exps_path"]}/{exp_name}/{file_suffix}_tta_preds.json', 'w') as f:
        json.dump(preds, f, indent=4)
    results = evaluator.get_results()
    write_llm_results(results, f'{configs["exps_path"]}/{exp_name}/{file_suffix}_tta_results.txt')
    print(f"TTA evaluation ({file_suffix}) done!")


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
            tokenizer_path = os.path.abspath(configs["tokenizer_path"])
        if not configs["lora_trained"]:
            if configs["llm_path"] is not None:
                if configs["llm_path"].endswith(".pt"):
                    # if llm_path is a .pt file, we load the base model first
                    pass
                elif os.path.isfile(os.path.join(configs["llm_path"], "adapter_config.json")):
                    # LoRA adapter dir - keep model_path as base model (e.g. vicuna-7b-v1.5)
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
                llm = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, offload_folder=configs["offload_dir"], torch_dtype=torch.bfloat16)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_auth_token=True, padding_side="left")
            # Match base embedding size to tokenizer vocab BEFORE loading LoRA adapter,
            # since the saved adapter's embed_tokens may have extra rows (e.g. <tact_start>/<tact_end>).
            if len(tokenizer) > llm.get_input_embeddings().weight.shape[0]:
                llm.resize_token_embeddings(len(tokenizer))
            # reference: https://jaotheboss.medium.com/domain-training-your-llm-6c77f53e3e27
            add_new_tokens(llm, tokenizer, new_tokens)
            if configs["quantized"]:
                llm = PeftModel.from_pretrained(model=llm, model_id=configs["llm_path"], is_trainable=False, device_map="auto", max_memory=gpu_max_mem_config, quantization_config=bnb_config)
            else:
                llm = PeftModel.from_pretrained(model=llm, model_id=configs["llm_path"], is_trainable=False, device_map="auto", max_memory=gpu_max_mem_config)
            # PeftModel loads LoRA params in fp32; cast to base weight dtype (bf16)
            for module in llm.modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and hasattr(module, 'weight'):
                    td = module.weight.device
                    dt = module.weight.dtype
                    module.lora_A.to(device=td, dtype=dt)
                    module.lora_B.to(device=td, dtype=dt)
        else:
            if configs["quantized"]:
                llm = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, offload_folder=configs["offload_dir"], quantization_config=bnb_config)
            else:
                llm = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, offload_folder=configs["offload_dir"], torch_dtype=torch.bfloat16)
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
    tokenizer.save_pretrained(f"{configs['exps_path']}/{exp_name}/tokenizer")

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
    max_frames = configs.get("max_frames", 5)
    if configs["train"]:
        train_dataset = TactileLLMDataset(image_processor, configs["train_files"], split_name="train", tokenizer=tokenizer, flip_p=configs["flip_p"], max_frames=max_frames,
            rotation_degrees=configs.get("rotation_degrees", 0), color_jitter=configs.get("color_jitter", 0.0), gaussian_blur=configs.get("gaussian_blur", False),
            pom_candidate_permutation_augmentation=configs.get("pom_candidate_permutation_augmentation", 0),
            multi_object_slot_permutation_augmentation=configs.get("multi_object_slot_permutation_augmentation", 0),
            multi_object_label_order_permutation_augmentation=configs.get("multi_object_label_order_permutation_augmentation", 0),
            pom_option_id_format=configs.get("pom_option_id_format", False))
        if configs.get("task_balanced_training", False):
            if configs["per_device_train_batch_size"] != 1:
                raise ValueError("task_balanced_training currently expects per_device_train_batch_size=1")
            task_sampler = TaskBalancedSampler(
                train_dataset,
                accumulation_steps=configs["llm_gradient_accumulation_steps"],
                seed=configs.get("seed", 0),
            )
            group_counts = {name: len(indices) for name, indices in task_sampler.groups.items()}
            print(f"Task-balanced training enabled: groups={group_counts}, per_group_per_update={task_sampler.per_group}")
            train_loader = DataLoader(train_dataset, batch_size=configs["per_device_train_batch_size"], sampler=task_sampler, worker_init_fn=seed_worker, generator=g)
        else:
            train_loader = DataLoader(train_dataset, batch_size=configs["per_device_train_batch_size"], shuffle=True, worker_init_fn=seed_worker, generator=g)
    if configs["val"]:
        val_dataset = TactileLLMDataset(image_processor, configs["val_files"], split_name="val", tokenizer=tokenizer, flip_p=configs["flip_p"], max_frames=max_frames,
            pom_option_id_format=configs.get("pom_option_id_format", False))
        val_loader = DataLoader(val_dataset, batch_size=configs["per_device_val_batch_size"], shuffle=False, worker_init_fn=seed_worker, generator=g)
    if configs["test"]:
        test_dataset = TactileLLMDataset(image_processor, configs["test_files"], split_name="test", tokenizer=tokenizer, flip_p=configs["flip_p"], max_frames=max_frames,
            pom_option_id_format=configs.get("pom_option_id_format", False))
        test_loader = DataLoader(test_dataset, batch_size=configs["per_device_val_batch_size"], shuffle=False, worker_init_fn=seed_worker, generator=g)

    # model instantiation
    if configs["lora_trained"]:
        model = MultimodalLLMForCausalLM(clip_model=configs["use_clip"], encoder_output_size=configs["encoder_output_size"], tokenizer=tokenizer, cutoff_len=configs["cutoff_len"], llm=llm.model, device=device)
    else:
        model = MultimodalLLMForCausalLM(clip_model=configs["use_clip"], encoder_output_size=configs["encoder_output_size"], tokenizer=tokenizer, cutoff_len=configs["cutoff_len"], llm=llm, device=device)
    
    # If using device_map (gpu_config present), we should not move the whole model to device
    # because the LLM parts are already placed. We only move the other modules.
    if configs["gpu_config"] is not None:
        model.encoder.to(device)
        model.project.to(device)
        model.consistency_hardness.to(device)
        model.consistency_roughness.to(device)
        model.consistency_texture.to(device)
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
        )
        llm_weights_path = f"{configs['exps_path']}/{exp_name}/llm_weights"
        if not os.path.exists(llm_weights_path):
            # os.makedirs(llm_weights_path)
            # llm_peft = get_peft_model(llm, peft_config)
            # llm_peft.save_pretrained(llm_weights_path)
            # llm_peft = None
            
            # Instead just apply the config in-memory directly
            llm = get_peft_model(llm, peft_config)
            # LoRA params and modules_to_save copies are initialized on CPU with float32.
            # Move them to match the device AND dtype of their corresponding base weights.
            for module in llm.modules():
                # LoRA linear layers
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and hasattr(module, 'weight'):
                    target_device = module.weight.device
                    target_dtype = module.weight.dtype
                    module.lora_A.to(device=target_device, dtype=target_dtype)
                    module.lora_B.to(device=target_device, dtype=target_dtype)
                # modules_to_save copies (e.g. embed_tokens adapter copy)
                if type(module).__name__ == 'ModulesToSaveWrapper' and hasattr(module, 'original_module'):
                    try:
                        ref = next(module.original_module.parameters())
                        for adapter_copy in module.modules_to_save.values():
                            adapter_copy.to(device=ref.device, dtype=ref.dtype)
                    except StopIteration:
                        pass

        if configs["quantized"]:
            pass
        else:
            pass
        # Load pre-trained LoRA weights for continued training (e.g., Stage 3 from Stage 2 checkpoint)
        if configs["llm_path"] is not None and not configs["llm_path"].endswith(".pt"):
            adapter_bin = os.path.join(configs["llm_path"], "adapter_model.bin")
            if os.path.isfile(adapter_bin):
                print(f"Loading Stage 2 LoRA weights from {configs['llm_path']}...")
                lora_state = torch.load(adapter_bin, map_location="cpu", weights_only=True)
                set_peft_model_state_dict(llm, lora_state)
                # Cast loaded LoRA weights to match base weight dtype/device
                for module in llm.modules():
                    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and hasattr(module, 'weight'):
                        module.lora_A.to(device=module.weight.device, dtype=module.weight.dtype)
                        module.lora_B.to(device=module.weight.device, dtype=module.weight.dtype)
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

    # 3) encoder setup — must happen before optimizer build so encoder params can join the grouped optimizer
    if configs["encoder_path"] is not None:
        try:
            model.encoder.load_state_dict(torch.load(configs["encoder_path"], map_location='cpu', weights_only=True))
        except RuntimeError:
            clip = PromptLearningCLIPModel.from_pretrained(configs["use_clip"], configs).to(device)
            model.encoder.model.vision_model = clip.vision_model
            model.encoder.load_state_dict(torch.load(configs["encoder_path"], map_location='cpu', weights_only=True), strict=True)
    encoder_params = []
    if configs["freeze_encoder"]:
        for name, param in model.encoder.named_parameters():
            param.requires_grad = False
    else:
        for name, param in model.encoder.named_parameters():
            param.requires_grad = True
        encoder_params = list(model.encoder.parameters())

    if configs["train"]:
        consistency_params = []
        consistency_enabled = configs.get("opd_consistency_loss_weight", 0.0) > 0
        for module in [model.consistency_hardness, model.consistency_roughness, model.consistency_texture]:
            for param in module.parameters():
                param.requires_grad = consistency_enabled
                if consistency_enabled:
                    consistency_params.append(param)
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
        print(f"len(encoder_params): {len(encoder_params)}")
        print(f"len(consistency_params): {len(consistency_params)}")

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
        if len(encoder_params) > 0:
            optimizer_grouped_parameters.append({
                "params": encoder_params,
                "lr": configs["encoder_lr"]
            })
        if len(consistency_params) > 0:
            optimizer_grouped_parameters.append({
                "params": consistency_params,
                "lr": configs.get("opd_consistency_lr", configs["llm_lr"])
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
            question, answer_tokens, tactile_frames, tactile, question_type, question_step, all_indices, conclusion_start, property_labels = batch
            answer_tokens = answer_tokens.to(device)
            conclusion_weight = get_conclusion_loss_weight(configs, question_type[0])
            cs = conclusion_start if (configs.get("conclusion_only_loss", False) or conclusion_weight > 0) else None
            consistency_weight = configs.get("opd_consistency_loss_weight", 0.0)
            if not question_type[0].endswith("_object_property_description"):
                consistency_weight = 0.0
            outputs, _ = model(
                question=question,
                tactile_frames=tactile_frames,
                answer_tokens=answer_tokens,
                all_indices=all_indices,
                conclusion_start=cs,
                conclusion_only=configs.get("conclusion_only_loss", False),
                conclusion_loss_weight=conclusion_weight,
                property_labels=property_labels,
                consistency_loss_weight=consistency_weight,
            )
            train_loss = outputs.loss.detach().float()
            t.set_description(f"Train loss: {train_loss}")
            # total_train_loss += train_loss # NOTE: hardcoded for batch size of 1
            loss = outputs.loss / configs["llm_gradient_accumulation_steps"]
            loss.backward()
            if (train_sample_step + 1) % configs["llm_gradient_accumulation_steps"] == 0:
                if len(optimizer_grouped_parameters) > 0:
                    # Clip each parameter group separately so one group's large gradients
                    # don't suppress another group's learning signal
                    llm_gn = torch.nn.utils.clip_grad_norm_(llm_params, max_norm=1.0) if llm_params else 0.0
                    proj_gn = torch.nn.utils.clip_grad_norm_(project_params, max_norm=1.0) if project_params else 0.0
                    enc_gn = torch.nn.utils.clip_grad_norm_(encoder_params, max_norm=1.0) if encoder_params else 0.0
                    optimizer_llm.step()
                    scheduler_llm.step()
                    optimizer_llm.zero_grad()
                    t.set_postfix(loss=train_loss.item(),
                                  llm_gn=llm_gn.item() if torch.is_tensor(llm_gn) else llm_gn,
                                  proj_gn=proj_gn.item() if torch.is_tensor(proj_gn) else proj_gn)
                    # Ensure frozen token embeddings are not corrupted by AdamW weight decay or variance tracking
                    if any("embed_tokens" in n for n, p in model.llm.named_parameters() if p.requires_grad):
                        with torch.no_grad():
                            model.llm.get_input_embeddings().weight.data[:n_old_tokens] = original_embeddings[:n_old_tokens]

            # validation
            if configs.get("val_freq") is not None and (train_sample_step + 1) % configs["val_freq"] == 0:
                if configs["val"]:
                    val_loss, balanced_val_loss = evaluate_metrics(model, val_loader, device, tokenizer, configs)
                    if configs["freeze_encoder"]:
                        model.encoder.eval()
                    if configs["freeze_projection"]:
                        model.project.eval()
                    print(f"Validation Loss: {val_loss}")
                    checkpoint_metric = val_loss
                    checkpoint_metric_name = "validation loss"
                    if configs.get("val_checkpoint_metric", "loss") == "balanced_task_loss":
                        if balanced_val_loss is None:
                            print("Balanced validation loss unavailable; falling back to sample-weighted validation loss.")
                        else:
                            checkpoint_metric = balanced_val_loss
                            checkpoint_metric_name = "balanced validation loss"
                    
                    if checkpoint_metric < best_val_loss:
                        print(f"New best {checkpoint_metric_name}: {checkpoint_metric:.4f} (previous: {best_val_loss:.4f})")
                        best_val_loss = checkpoint_metric
                        current_step = train_sample_step + 1
                        print(f"Saving BEST checkpoint at step {current_step}...")
                        model.llm.generation_config.temperature = None
                        model.llm.generation_config.top_p = None
                        if len(llm_params) > 0:
                            if configs["use_lora"]:
                                model.llm.save_pretrained(f"{configs['exps_path']}/{exp_name}/best_llm_weights")
                            else:
                                torch.save({n: p for n, p in model.llm.named_parameters() if p.requires_grad}, f"{configs['exps_path']}/{exp_name}/best_llm_weights.pt")
                        torch.save(model.project.state_dict(), f"{configs['exps_path']}/{exp_name}/best_project.pt")
                        if not configs["freeze_encoder"]:
                            torch.save(model.encoder.state_dict(), f"{configs['exps_path']}/{exp_name}/best_encoder.pt")
                        tokenizer.save_pretrained(f"{configs['exps_path']}/{exp_name}/tokenizer")
            if (train_sample_step + 1) >= configs["max_train_steps"]:
                break
        print(f"\nSaving FINAL checkpoint after training...")
        tokenizer.save_pretrained(f"{configs['exps_path']}/{exp_name}/tokenizer")
        model.llm.generation_config.temperature = None
        model.llm.generation_config.top_p = None
        if len(llm_params) > 0:
            if configs["use_lora"]:
                model.llm.save_pretrained(f"{configs['exps_path']}/{exp_name}/final_llm_weights")
            else:
                torch.save({n: p for n, p in model.llm.named_parameters() if p.requires_grad}, f"{configs['exps_path']}/{exp_name}/final_llm_weights.pt")
        torch.save(model.project.state_dict(), f"{configs['exps_path']}/{exp_name}/final_project.pt")
        if not configs["freeze_encoder"]:
            torch.save(model.encoder.state_dict(), f"{configs['exps_path']}/{exp_name}/final_encoder.pt")
        print(f"LLM training done!")

    # test
    if configs["test"]:
        if configs["train"]:
            print(f"\n=========================================")
            print(f"Evaluating FINAL checkpoint for testing...")
            print(f"=========================================")
            model.eval()
            run_evaluation(model, test_loader, device, tokenizer, configs, exp_name, "test_final")
            if configs.get("use_lora", False) and configs.get("tta_passes", 1) > 1:
                run_evaluation_tta(model, configs["test_files"], image_processor, device, tokenizer, configs, exp_name, "test_final")

        if configs["train"] and configs["val"] and configs.get("val_freq") is not None:
            print(f"\n=========================================")
            print(f"Loading best checkpoint for testing...")
            print(f"=========================================")
            
            # Reload Projection
            best_proj_path = f"{configs['exps_path']}/{exp_name}/best_project.pt"
            if os.path.exists(best_proj_path):
                model.project.load_state_dict(torch.load(best_proj_path, map_location=device, weights_only=True))

            # Reload Encoder
            if not configs["freeze_encoder"]:
                best_enc_path = f"{configs['exps_path']}/{exp_name}/best_encoder.pt"
                if os.path.exists(best_enc_path):
                    model.encoder.load_state_dict(torch.load(best_enc_path, map_location=device))

            # Reload LLM
            if configs["use_lora"]:
                best_llm_path = f"{configs['exps_path']}/{exp_name}/best_llm_weights"
                adapter_bin = os.path.join(best_llm_path, 'adapter_model.bin')
                if os.path.exists(adapter_bin):
                    # Use load_state_dict directly — PeftModel.from_pretrained would double-wrap
                    # an existing PeftModel and corrupt the model. strict=False loads LoRA
                    # weights and embed_tokens (modules_to_save) while skipping base weights.
                    print(f"Loading best LoRA adapter from {adapter_bin}...")
                    adapter_weights = torch.load(adapter_bin, map_location='cpu', weights_only=True)
                    model.llm.load_state_dict(adapter_weights, strict=False)
                    # Cast reloaded LoRA params to bf16 to match base model dtype
                    for module in model.llm.modules():
                        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and hasattr(module, 'weight'):
                            td = module.weight.device
                            dt = module.weight.dtype
                            module.lora_A.to(device=td, dtype=dt)
                            module.lora_B.to(device=td, dtype=dt)
            else:
                if len(llm_params) > 0:
                    best_llm_path = f"{configs['exps_path']}/{exp_name}/best_llm_weights.pt"
                    if os.path.exists(best_llm_path):
                        model.llm.load_state_dict(torch.load(best_llm_path, map_location=device), strict=False)
                
            model.eval()
            run_evaluation(model, test_loader, device, tokenizer, configs, exp_name, "test_best")
            if configs.get("use_lora", False) and configs.get("tta_passes", 1) > 1:
                run_evaluation_tta(model, configs["test_files"], image_processor, device, tokenizer, configs, exp_name, "test_best")

        else:
            run_evaluation(model, test_loader, device, tokenizer, configs, exp_name, "test_best")
            if configs.get("use_lora", False) and configs.get("tta_passes", 1) > 1:
                run_evaluation_tta(model, configs["test_files"], image_processor, device, tokenizer, configs, exp_name, "test_best")

        print(f"LLM test done!")


if __name__ == "__main__":
    exp_type = f"train_llm"
    config_path = os.environ.get("TRAIN_LLM_CONFIG", f'configs/{exp_type}_config.yaml')
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
    log_file = open(f"{configs['exps_path']}/{exp_name}/log.txt", 'w', buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file
    logging.set_verbosity_error()

    # seed
    torch.manual_seed(configs["seed"])
    torch.random.manual_seed(configs["seed"])
    torch.cuda.manual_seed(configs["seed"])
    torch.cuda.manual_seed_all(configs["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    np.random.seed(configs["seed"])
    random.seed(configs["seed"])
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(configs["seed"])

    train(configs, exp_name, g)
