import pickle 
import torch 
from torch.utils.data import Dataset
import numpy as np
import os
import ast
import csv
import natsort
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import json
import copy
import itertools
import re
from utils.constants import *


def get_property_labels_from_tactile_path(tactile_path):
    sample_name = os.path.basename(tactile_path)
    object_name = sample_name.rsplit("_", 1)[0]
    if object_name not in RANKS["hardness"]:
        return torch.tensor([-100, -100, -100], dtype=torch.long)
    return torch.tensor([
        RANKS["hardness"][object_name],
        RANKS["roughness"][object_name],
        RANKS["texture"][object_name],
    ], dtype=torch.long)


def _parse_candidate_chunk(chunk):
    match = re.match(r"^\s*\d\)\s*(.*?)(,\s*|\.\s*)?$", chunk)
    if match is None:
        return None
    return match.group(1).strip()


def _build_candidate_chunk(position, object_name, is_last):
    suffix = "." if is_last else ", "
    return f"{position}) {object_name}{suffix}"


def _normalize_object_name(name):
    name = name.replace("</s>", "").strip().strip(".").lower()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"^(the|a|an) ", "", name)
    return name


def _get_pom_candidate_map(sample):
    user_turn_index = None
    for idx in range(1, len(sample) - 1):
        if sample[idx].get("role") == "USER":
            user_turn_index = idx
            break
    if user_turn_index is None:
        return None, None
    content = sample[user_turn_index].get("content", [])
    candidate_map = {}
    for chunk in content:
        object_name = _parse_candidate_chunk(chunk)
        if object_name is not None:
            candidate_map[_normalize_object_name(object_name)] = len(candidate_map) + 1
    if len(candidate_map) != 3:
        return None, None
    return user_turn_index, candidate_map


def _parse_pom_assignment(conclusion):
    assignments = {}
    pattern = re.compile(
        r"([abc])\)\s*(?:is\s*)?(.*?)(?=(?:,\s*[abc]\)|\s+and\s+[abc]\)|\.?\s*$))",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(conclusion.strip()):
        assignments[match.group(1).lower()] = match.group(2).strip(" ,.")
    if any(label not in assignments for label in "abc"):
        return None
    return [assignments[label] for label in "abc"]


def rewrite_pom_to_option_id_format(samples):
    rewritten = []
    for sample in samples:
        if "property_object_match" not in sample[0].get("question_type", ""):
            rewritten.append(sample)
            continue
        _, candidate_map = _get_pom_candidate_map(sample)
        if candidate_map is None:
            rewritten.append(sample)
            continue
        answer = "".join(sample[-1].get("content", []))
        if "Conclusion: " not in answer:
            rewritten.append(sample)
            continue
        answer_prefix, conclusion = answer.split("Conclusion: ", 1)
        assignments = _parse_pom_assignment(conclusion)
        if assignments is None:
            rewritten.append(sample)
            continue
        option_ids = []
        for object_name in assignments:
            option_id = candidate_map.get(_normalize_object_name(object_name))
            if option_id is None:
                option_ids = []
                break
            option_ids.append(option_id)
        if len(option_ids) != 3:
            rewritten.append(sample)
            continue
        sample_rewritten = copy.deepcopy(sample)
        sample_rewritten[-1]["content"] = [
            answer_prefix
            + "Conclusion: "
            + f"a) is option {option_ids[0]}, b) is option {option_ids[1]} and c) is option {option_ids[2]}."
        ]
        rewritten.append(sample_rewritten)
    return rewritten


def augment_pom_candidate_order(samples, max_extra_per_sample):
    if max_extra_per_sample <= 0:
        return samples
    augmented = []
    for sample in samples:
        augmented.append(sample)
        if sample[0].get("question_type") != "train_property_object_match":
            continue
        user_turn_index = None
        for idx in range(1, len(sample) - 1):
            if sample[idx].get("role") == "USER":
                user_turn_index = idx
                break
        if user_turn_index is None:
            continue
        content = sample[user_turn_index].get("content", [])
        candidate_indices = []
        candidate_objects = []
        for idx, chunk in enumerate(content):
            object_name = _parse_candidate_chunk(chunk)
            if object_name is not None:
                candidate_indices.append(idx)
                candidate_objects.append(object_name)
        if len(candidate_indices) != 3:
            continue
        original_order = tuple(candidate_objects)
        extra_added = 0
        for perm in itertools.permutations(candidate_objects):
            if perm == original_order:
                continue
            sample_aug = copy.deepcopy(sample)
            aug_content = sample_aug[user_turn_index]["content"]
            for pos, idx in enumerate(candidate_indices):
                aug_content[idx] = _build_candidate_chunk(pos + 1, perm[pos], pos == len(candidate_indices) - 1)
            augmented.append(sample_aug)
            extra_added += 1
            if extra_added >= max_extra_per_sample:
                break
    return augmented


def _parse_abc_answer(answer):
    match = re.match(
        r"^a\)\s*(.*?)\s+b\)\s*(.*?)\s+c\)\s*(.*?)\s+Conclusion:\s*(.*)$",
        answer.strip(),
        flags=re.DOTALL,
    )
    if match is None:
        return None
    descriptions = [match.group(i).strip() for i in range(1, 4)]
    conclusion = match.group(4).strip()
    return descriptions, conclusion


def _build_abc_answer(descriptions, conclusion):
    return (
        f"a) {descriptions[0]} "
        f"b) {descriptions[1]} "
        f"c) {descriptions[2]} "
        f"Conclusion: {conclusion}"
    )


def _build_label_order_answer(label_order, description_by_label, conclusion):
    parts = [f"{label}) {description_by_label[label]}" for label in label_order]
    return " ".join(parts) + f" Conclusion: {conclusion}"


def _parse_video_prompt_parts(content):
    text = "".join(content)
    pattern = (
        r"^(.*?)"
        r"a\)\s*<tact_start><img_tokens><tact_end>,\s*"
        r"b\)\s*<tact_start><img_tokens><tact_end>,\s*"
        r"c\)\s*<tact_start><img_tokens><tact_end>"
        r"(\..*)$"
    )
    match = re.match(pattern, text, flags=re.DOTALL)
    if match is None:
        return None
    return match.group(1), match.group(2)


def _build_label_order_question_content(prefix, suffix, label_order):
    content = []
    for idx, label in enumerate(label_order):
        if idx == 0:
            content.append(f"{prefix}{label}) <tact_start>")
        else:
            content.append(f"{label}) <tact_start>")
        content.append("<img_tokens>")
        if idx + 1 == len(label_order):
            content.append(f"<tact_end>{suffix}")
        else:
            content.append("<tact_end>, ")
    return content


def _rewrite_pom_label_order_conclusion(conclusion, label_order):
    assignments = _parse_pom_assignment(conclusion)
    if assignments is None:
        return None
    assignment_by_label = {label: assignments[idx] for idx, label in enumerate("abc")}
    parts = [f"{label}) is {assignment_by_label[label]}" for label in label_order]
    return f"{parts[0]}, {parts[1]} and {parts[2]}."


def _rewrite_pom_slot_conclusion(conclusion, perm):
    assignments = _parse_pom_assignment(conclusion)
    if assignments is None:
        return None
    assignments = [assignments[old_idx] for old_idx in perm]
    return f"a) is {assignments[0]}, b) is {assignments[1]} and c) is {assignments[2]}."


def _rewrite_pss_slot_conclusion(conclusion, perm):
    match = re.match(r"^\s*([abc])\)\s+is\s+(.*)$", conclusion.strip(), flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    old_winner = "abc".index(match.group(1).lower())
    new_winner = perm.index(old_winner)
    return f"{'abc'[new_winner]}) is {match.group(2).strip()}"


def augment_multi_object_slot_order(samples, max_extra_per_sample):
    if max_extra_per_sample <= 0:
        return samples
    augmented = []
    for sample in samples:
        augmented.append(sample)
        question_type = sample[0].get("question_type", "")
        if question_type not in ("train_property_object_match", "train_property_superlative_selection"):
            continue
        user_turn_index = None
        for idx in range(1, len(sample) - 1):
            if sample[idx].get("role") == "USER":
                user_turn_index = idx
                break
        if user_turn_index is None or len(sample[user_turn_index].get("tactile", [])) != 3:
            continue
        answer = "".join(sample[-1].get("content", []))
        parsed = _parse_abc_answer(answer)
        if parsed is None:
            continue
        descriptions, conclusion = parsed
        original_order = (0, 1, 2)
        extra_added = 0
        for perm in itertools.permutations(original_order):
            if perm == original_order:
                continue
            if question_type == "train_property_object_match":
                new_conclusion = _rewrite_pom_slot_conclusion(conclusion, perm)
            else:
                new_conclusion = _rewrite_pss_slot_conclusion(conclusion, perm)
            if new_conclusion is None:
                continue
            sample_aug = copy.deepcopy(sample)
            sample_aug[user_turn_index]["tactile"] = [
                sample[user_turn_index]["tactile"][old_idx] for old_idx in perm
            ]
            new_descriptions = [descriptions[old_idx] for old_idx in perm]
            sample_aug[-1]["content"] = [_build_abc_answer(new_descriptions, new_conclusion)]
            augmented.append(sample_aug)
            extra_added += 1
            if extra_added >= max_extra_per_sample:
                break
    return augmented


def augment_multi_object_label_order(samples, max_extra_per_sample):
    if max_extra_per_sample <= 0:
        return samples
    augmented = []
    original_order = ("a", "b", "c")
    for sample in samples:
        augmented.append(sample)
        question_type = sample[0].get("question_type", "")
        if question_type not in ("train_property_object_match", "train_property_superlative_selection"):
            continue
        user_turn_index = None
        for idx in range(1, len(sample) - 1):
            if sample[idx].get("role") == "USER":
                user_turn_index = idx
                break
        if user_turn_index is None or len(sample[user_turn_index].get("tactile", [])) != 3:
            continue
        prompt_parts = _parse_video_prompt_parts(sample[user_turn_index].get("content", []))
        if prompt_parts is None:
            continue
        answer = "".join(sample[-1].get("content", []))
        parsed = _parse_abc_answer(answer)
        if parsed is None:
            continue
        descriptions, conclusion = parsed
        description_by_label = {label: descriptions[idx] for idx, label in enumerate(original_order)}
        tactile_by_label = {label: sample[user_turn_index]["tactile"][idx] for idx, label in enumerate(original_order)}
        prefix, suffix = prompt_parts
        extra_added = 0
        for label_order in itertools.permutations(original_order):
            if label_order == original_order:
                continue
            if question_type == "train_property_object_match":
                new_conclusion = _rewrite_pom_label_order_conclusion(conclusion, label_order)
            else:
                new_conclusion = conclusion
            if new_conclusion is None:
                continue
            sample_aug = copy.deepcopy(sample)
            sample_aug[user_turn_index]["content"] = _build_label_order_question_content(prefix, suffix, label_order)
            sample_aug[user_turn_index]["tactile"] = [tactile_by_label[label] for label in label_order]
            sample_aug[-1]["content"] = [
                _build_label_order_answer(label_order, description_by_label, new_conclusion)
            ]
            augmented.append(sample_aug)
            extra_added += 1
            if extra_added >= max_extra_per_sample:
                break
    return augmented


def get_frames(frames_path, image_processor, transforms_image, max_length=5, skip=True, return_indices=False):
    # get relevant object(s) and their frames
    tactile_tensors = []
    all_obj_sample_frames = natsort.natsorted(os.path.join(frames_path, i) for i in os.listdir(frames_path))
    num_frames = len(all_obj_sample_frames)
    if num_frames > max_length:
        if skip:
            all_obj_sample_frames = [all_obj_sample_frames[int(num_frames * i/ max_length)] for i in range(0, max_length)]
        else:
            all_obj_sample_frames = natsort.natsorted(random.sample(all_obj_sample_frames, k=max_length))
    
    for frame in all_obj_sample_frames:
        if image_processor is not None:
            img = Image.open(frame).convert('RGB')
            if transforms_image is not None:
                img = transforms_image(img)
            tactile_tensors.append(image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0])
    
    tactile_tensors = torch.stack(tactile_tensors, dim=0) # (l, c, h, w)

    if return_indices:
        frame_indices = [int(i.split("/")[-1].split(".jpg")[0]) for i in all_obj_sample_frames]
        return tactile_tensors, frame_indices
    return tactile_tensors
    

class CLIPPropertyUniqueDataset(Dataset):
    def __init__(self, image_processor, data_path, split_name, flip_p=0, max_frames=5,
                 rotation_degrees=0, color_jitter=0.0, gaussian_blur=False):
        super().__init__()
        self.rotation_degrees = rotation_degrees
        self.color_jitter = color_jitter
        self.gaussian_blur = gaussian_blur
        self.split_name = split_name
        self.flip_p = flip_p
        self.max_frames = max_frames
        self.image_processor = image_processor
        self.properties = ["hardness", "roughness", "texture"]
        json_path = [os.path.join(data_path, f"{self.split_name}_samples.json")]
        for i in range(len(json_path)):
            if i == 0:
                with open(json_path[i]) as json_file:
                    self.samples = json.load(json_file)
                    json_file.close()
            else:
                with open(json_path[i]) as json_file:
                    samples_temp = json.load(json_file)
                    json_file.close()
                for k, v in samples_temp.items():
                    if k in self.samples.keys():
                        self.samples[k] += v
                    else:
                        self.samples[k] = v

        self.objects = []
        self.all_samples = []
        for k in sorted(self.samples.keys()):
            if k not in TRAIN_OBJECTS + VAL_OBJECTS + TEST_OBJECTS:
                continue
            for v in sorted(self.samples[k]):
                self.objects.append(k)
                self.all_samples.append(v)

    def get_frames_and_label(self, index, transforms_image):
        # get frames
        objects = self.objects[index]
        video = self.all_samples[index]
        objects_tactile_frames = []
        all_indices = []
        if self.split_name == "train":
            frames, indices = get_frames(video, self.image_processor, transforms_image, max_length=self.max_frames, skip=False, return_indices=True)
        else:
            frames, indices = get_frames(video, self.image_processor, transforms_image, max_length=self.max_frames, return_indices=True)
        objects_tactile_frames.append(frames) # [(l, c, h, w)]
        all_indices.append(indices)
        # get label
        hardness_label = RANKS["hardness"][objects]
        roughness_label = RANKS["roughness"][objects]
        texture_label = RANKS["texture"][objects]
        return objects_tactile_frames, hardness_label, roughness_label, texture_label, all_indices
    
    def __len__(self): 
        return len(self.objects)

    def __getitem__(self, index):
        # load tactile info
        transform_list = []
        if self.split_name == "train":
            if random.random() < self.flip_p:
                transform_list.append(transforms.RandomHorizontalFlip(1))
            if random.random() < self.flip_p:
                transform_list.append(transforms.RandomVerticalFlip(1))
            if self.rotation_degrees > 0:
                transform_list.append(transforms.RandomRotation(self.rotation_degrees))
            if self.color_jitter > 0:
                transform_list.append(transforms.ColorJitter(
                    brightness=self.color_jitter, contrast=self.color_jitter,
                    saturation=self.color_jitter * 0.5, hue=0.0
                ))
            if self.gaussian_blur and random.random() < 0.2:
                transform_list.append(transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 1.5)))
            transforms_image = transforms.Compose(transform_list)
        else:
            transforms_image = None
        objects_tactile_frames, hardness_label, roughness_label, texture_label, all_indices = self.get_frames_and_label(index, transforms_image=transforms_image)
        return objects_tactile_frames, hardness_label, roughness_label, texture_label, all_indices


class TactileLLMDataset(Dataset):
    def __init__(self, image_processor, files, split_name, tokenizer, flip_p, random_frames=False, max_frames=5,
                 rotation_degrees=0, color_jitter=0.0, gaussian_blur=False, pom_candidate_permutation_augmentation=0,
                 multi_object_slot_permutation_augmentation=0,
                 multi_object_label_order_permutation_augmentation=0, pom_option_id_format=False):
        super().__init__()
        self.rotation_degrees = rotation_degrees
        self.color_jitter = color_jitter
        self.gaussian_blur = gaussian_blur
        self.split_name = split_name
        self.tokenizer = tokenizer
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.pad_token = tokenizer.pad_token
        self.eos_token_number = self.tokenizer.encode(self.eos_token)
        self.flip_p = flip_p
        self.random_frames = random_frames
        self.max_frames = max_frames
        self.image_processor = image_processor
        self.samples = None
        for f in files:
            with open(f) as json_file:
                if self.samples is None:
                    self.samples = json.load(json_file)
                else:
                    self.samples += json.load(json_file)
                json_file.close()
        if self.split_name == "train":
            self.samples = augment_pom_candidate_order(self.samples, pom_candidate_permutation_augmentation)
            self.samples = augment_multi_object_slot_order(self.samples, multi_object_slot_permutation_augmentation)
            self.samples = augment_multi_object_label_order(self.samples, multi_object_label_order_permutation_augmentation)
        if pom_option_id_format:
            self.samples = rewrite_pom_to_option_id_format(self.samples)
    
    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, index):
        # 1) sample get questions, answers and tactile paths
        # NOTE: ignore BOS tokens
        transform_list = []
        if self.split_name == "train":
            if random.random() < self.flip_p:
                transform_list.append(transforms.RandomHorizontalFlip(1))
            if random.random() < self.flip_p:
                transform_list.append(transforms.RandomVerticalFlip(1))
            if self.rotation_degrees > 0:
                transform_list.append(transforms.RandomRotation(self.rotation_degrees))
            if self.color_jitter > 0:
                transform_list.append(transforms.ColorJitter(
                    brightness=self.color_jitter, contrast=self.color_jitter,
                    saturation=self.color_jitter * 0.5, hue=0.0
                ))
            if self.gaussian_blur and random.random() < 0.2:
                transform_list.append(transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 1.5)))
            transforms_image = transforms.Compose(transform_list)
        else:
            transforms_image = None
        sample = self.samples[index]
        question_type = sample[0]["question_type"]
        question_step = sample[0]["question_steps"]
        question = []
        tactile = []
        for s in sample[1:-1]:
            if s["role"] == "ASSISTANT":
                question += [s["role"]] + [": "] + s["content"] + [f"{self.eos_token}"]
            elif s["role"] == "USER":
                question += [s["role"]] + [": "] + s["content"] + [" "]
            elif s["role"] == "SYSTEM":
                question += [s["content"] + " "]
            tactile += s["tactile"]
        question += ["ASSISTANT: "]
        # Merge adjacent text chunks to avoid tokenization artifacts
        merged_question = []
        current_text = ""
        for item in question:
            if "img_tokens" in item:
                if current_text:
                    merged_question.append(current_text)
                    current_text = ""
                merged_question.append(item)
            else:
                current_text += item
        if current_text:
            merged_question.append(current_text)
        question = merged_question
        answer = "".join(sample[-1]["content"])
        # 2) get tokens
        answer_tokens = torch.tensor(self.tokenizer.encode(answer + f'{self.eos_token}'), dtype=torch.int64)[1:]
        # Compute where "Conclusion: " starts in answer_tokens (0 if absent, e.g. OPD)
        conclusion_start = 0
        if "Conclusion: " in answer:
            desc_part = answer[:answer.index("Conclusion: ")]
            # encode() includes BOS; [1:] mirrors the [1:] slice on answer_tokens
            conclusion_start = max(0, len(self.tokenizer.encode(desc_part)) - 1)
        # 3) get frame tensors
        all_tactile_frames = []
        all_indices = []
        for t in tactile:
            if self.split_name == "train" or self.random_frames:
                frames, indices = get_frames(t, self.image_processor, transforms_image, max_length=self.max_frames, skip=False, return_indices=True)
            else:
                frames, indices = get_frames(t, self.image_processor, transforms_image, max_length=self.max_frames, return_indices=True)
            all_tactile_frames.append(frames)
            all_indices.append(indices)
        property_labels = torch.tensor([-100, -100, -100], dtype=torch.long)
        if question_type.endswith("_object_property_description") and len(tactile) > 0:
            property_labels = get_property_labels_from_tactile_path(tactile[0])
        return question, answer_tokens, all_tactile_frames, tactile, question_type, question_step, all_indices, conclusion_start, property_labels
