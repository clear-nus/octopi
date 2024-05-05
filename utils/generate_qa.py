import os
import random
from constants import *
import json
import argparse


def get_sample_description(sample, properties, use_unstructured):
    if use_unstructured:
        description = DESCRIPTIONS[sample][0] + " Overall, it presents a"
    else:
        description = "It presents a"
    assert len(properties) >= 1
    if "hardness" in properties:
        description += f" {HARDNESS_MAP[HARDNESS_RANK[sample]]}"
    if "hardness" in properties and "roughness" in properties:
        description += f" and"
    if "roughness" in properties:
        description += f" {ROUGHNESS_MAP[ROUGHNESS_RANK[sample]]}"
    description += " surface"
    if "texture" in properties:
        description += f" with {TEXTURE_MAP[TEXTURE_RANK[sample]]}"
    description += "."
    return description


def generate_one_step_qa(start_prompt, json_path, data_path, split, num_samples, use_unstructured, use_properties):
    properties = ["hardness", "roughness", "texture"]

    property_names = {
        "hardness": "hardness",
        "roughness": "roughness",
        "texture": "the size of the bumps present"
    }

    # prompt setup
    property_comparisons = {
        "hardness": {
            "<more_property>": "harder",
            "<less_property>": "softer",
            "<most_property>": "hardest",
            "<least_property>": "softest"
        },
        "roughness": {
            "<more_property>": "rougher",
            "<less_property>": "smoother",
            "<most_property>": "roughest",
            "<least_property>": "smoothest"
        },
        "texture": {
            "<more_property>": "covered with bigger bumps",
            "<less_property>": "covered with smaller bumps",
            "<most_property>": "one covered with the biggest bumps",
            "<least_property>": "one covered with the smallest bumps"
        }
    }
    object_property_description = [{
        "object_property_description_0": ["Describe the physical properties of <tact_start>", "<img_tokens>", "<tact_end>."],
        "object_property_description_1": ["How does this tactile video <tact_start>", "<img_tokens>", "<tact_end> feel?"],
    }]
    property_comparison = [{
        "property_comparison_more_0": ["Is the object in the tactile video <tact_start>", "<img_tokens>", "<tact_end> ", "<more_property>", " than the one in <tact_start>", "<img_tokens>", "<tact_end>?", " Describe both objects before answering."],
        "property_comparison_more_1": ["Is the object in <tact_start>", "<img_tokens>", "<tact_end> ", "<more_property>", " than the object in <tact_start>", "<img_tokens>", "<tact_end>?", " Describe both objects before answering."],
        "property_comparison_less_0": ["Is the object in the tactile video <tact_start>", "<img_tokens>", "<tact_end> ",  "<less_property>", " than the one in <tact_start>", "<img_tokens>", "<tact_end>?", " Describe both objects before answering."],
        "property_comparison_less_1": ["Is the object in <tact_start>", "<img_tokens>", "<tact_end> ", "<less_property>", " than the object in <tact_start>", "<img_tokens>", "<tact_end>?", " Describe both objects before answering."],
    }]
    property_superlative_selection = [{
        "property_superlative_selection_most_0": ["Given three tactile videos: a) <tact_start>", "<img_tokens>", "<tact_end>, b) <tact_start>", "<img_tokens>", "<tact_end>, c) <tact_start>", "<img_tokens>", "<tact_end>.", " Describe each object and then select the ", "<most_property>", "."],
        "property_superlative_selection_most_1": ["You have tactile videos of one object each: a) <tact_start>", "<img_tokens>", "<tact_end>, b) <tact_start>", "<img_tokens>", "<tact_end>, c) <tact_start>", "<img_tokens>", "<tact_end>.", " Describe each object and then select the ", "<most_property>", " object."],
        "property_superlative_selection_least_0": ["Given these tactile videos: a) <tact_start>", "<img_tokens>", "<tact_end>, b) <tact_start>", "<img_tokens>", "<tact_end>, c) <tact_start>", "<img_tokens>", "<tact_end>.", " Describe each object and then select the ", "<least_property>", "."],
        "property_superlative_selection_least_1": ["You have tactile videos of one object each: a) <tact_start>", "<img_tokens>", "<tact_end>, b) <tact_start>", "<img_tokens>", "<tact_end>, c) <tact_start>", "<img_tokens>", "<tact_end>.", " Describe each object and then select the ", "<least_property>", " object."],
    }]
    property_object_match = [{
        "property_object_match_0": ["Given three tactile videos: a) <tact_start>", "<img_tokens>", "<tact_end>, b) <tact_start>", "<img_tokens>", "<tact_end>, c) <tact_start>", "<img_tokens>", "<tact_end>.", " Describe the object in each video, then match each video to one of the following objects in alphabetical order: "],
        "property_object_match_1": ["You have tactile videos of one object each: a) <tact_start>", "<img_tokens>", "<tact_end>, b) <tact_start>", "<img_tokens>", "<tact_end>, c) <tact_start>", "<img_tokens>", "<tact_end>.", " Describe the object in each video, then match each video to one of the following objects in alphabetical order: "],
    }]
    object_description = [{
        "object_description_0": ["What object is <tact_start>", "<img_tokens>", "<tact_end>?"],
        "object_description_1": ["What object does <tact_start>", "<img_tokens>", "<tact_end> show?"],
    }]
    if split == "train":
        property_questions = {
            "train_property_comparison": property_comparison,
            "train_property_superlative_selection": property_superlative_selection,
            "train_property_object_match": property_object_match,
        }
        if use_properties:
            property_questions["train_object_property_description"] = object_property_description
    elif split == "eval":
        property_questions = {
            "eval_property_comparison": property_comparison,
            "eval_property_superlative_selection": property_superlative_selection,
            "eval_property_object_match": property_object_match,
        }

    # load samples
    for i in range(len(json_path)):
        if i == 0:
            with open(json_path[i]) as json_file:
                samples = json.load(json_file)
                json_file.close()
        else:
            with open(json_path[i]) as json_file:
                samples_temp = json.load(json_file)
                json_file.close()
            for k, v in samples_temp.items():
                if k in samples.keys():
                    samples[k] += v
                else:
                    samples[k] = v

    # data
    all_data = []

    if split == "eval":
        existing = {
            "eval_property_comparison": [],
            "eval_property_superlative_selection": [],
            "eval_property_object_match": []
        }
    
    for i in range(num_samples):
        if split == "eval":
            exist = False
        question_type = random.choice(list(property_questions.keys()))
        question_steps =  random.randint(1, len(property_questions[question_type]))
        data = [{
            "question_type": question_type,
            "question_steps": question_steps
        }]
        if question_type == f"{split}_object_property_description":
            for qs in range(question_steps):
                question_key = random.choice(list(property_questions[question_type][qs].keys()))
                question = property_questions[question_type][qs][question_key].copy()
                num_tactile = question.count("<img_tokens>")
                # get relevant object(s) and their frames
                sample = random.sample(samples.keys(), k=num_tactile)[0]
                tactile = [random.choice(samples[sample])]
                answer = get_sample_description(sample, properties, use_unstructured)
                if qs == 0:
                    question.insert(0, start_prompt)
                data.append({
                        "role": "USER",
                        "content": question,
                        "tactile": tactile
                    })
                data.append({
                        "role": "ASSISTANT",
                        "content": [answer],
                        "tactile": []
                    })
        elif question_type == f"{split}_property_comparison":
            num_tactile = 2
            # get relevant object(s) and their frames
            all_samples = random.sample(samples.keys(), k=num_tactile)
            prop = random.choice(properties)
            for qs in range(question_steps):
                question_key = random.choice(list(property_questions[question_type][qs].keys()))
                question = property_questions[question_type][qs][question_key].copy()
                if "property_comparison_more" in question_key:
                    tactile = [random.choice(samples[i]) for i in all_samples]
                    if split == "eval":
                        if (tactile[0], tactile[1], prop) in existing[question_type]:
                            exist = True
                            break
                        else:
                            existing[question_type].append((tactile[0], tactile[1], prop))
                    for question_index, chunk in enumerate(question):
                        if "<" in chunk and ">" in chunk and "property" in chunk:
                            question[question_index] = property_comparisons[prop][chunk]
                            prop_comparison = property_comparisons[prop][chunk]
                    rank = RANKS[prop]
                    if not use_properties:
                        question = question[:-1]
                    if rank[all_samples[0]] > rank[all_samples[1]]:
                        if use_properties:
                            answer = f"First object: {get_sample_description(all_samples[0], properties, use_unstructured)}" + " " + f"Second object: {get_sample_description(all_samples[1], properties, use_unstructured)}" + f" Conclusion: Yes, the first object is {prop_comparison}."
                        else:
                            answer = f"Yes, the first object is {prop_comparison}."
                    elif rank[all_samples[0]] < rank[all_samples[1]]:
                        if use_properties:
                            answer = f"First object: {get_sample_description(all_samples[0], properties, use_unstructured)}" + " " + f"Second object: {get_sample_description(all_samples[1], properties, use_unstructured)}" + f" Conclusion: No, the second object is {prop_comparison}."
                        else:
                            answer = f"No, the second object is {prop_comparison}."
                    else:
                        if use_properties:
                            answer = f"First object: {get_sample_description(all_samples[0], properties, use_unstructured)}" + " " + f"Second object: {get_sample_description(all_samples[1], properties, use_unstructured)}" + f" Conclusion: Both objects are similar in terms of {property_names[prop]}."
                        else:
                            answer = f"Both objects are similar in terms of {property_names[prop]}."
                elif "property_comparison_less" in question_key:
                    tactile = [random.choice(samples[i]) for i in all_samples]
                    for question_index, chunk in enumerate(question):
                        if "<" in chunk and ">" in chunk and "property" in chunk:
                            question[question_index] = property_comparisons[prop][chunk]
                            prop_comparison = property_comparisons[prop][chunk]
                    rank = RANKS[prop]
                    if not use_properties:
                        question = question[:-1]
                    if rank[all_samples[0]] < rank[all_samples[1]]:
                        if use_properties:
                            answer = f"First object: {get_sample_description(all_samples[0], properties, use_unstructured)}" + " " + f"Second object: {get_sample_description(all_samples[1], properties, use_unstructured)}" + f" Conclusion: Yes, the first object is {prop_comparison}."
                        else:
                            answer = f"Yes, the first object is {prop_comparison}."
                    elif rank[all_samples[0]] > rank[all_samples[1]]:
                        if use_properties:
                            answer = f"First object: {get_sample_description(all_samples[0], properties, use_unstructured)}" + " " + f"Second object: {get_sample_description(all_samples[1], properties, use_unstructured)}" + f" Conclusion: No, the second object is {prop_comparison}."
                        else:
                            answer = f"No, the second object is {prop_comparison}."
                    else:
                        if use_properties:
                            answer = f"First object: {get_sample_description(all_samples[0], properties, use_unstructured)}" + " " + f"Second object: {get_sample_description(all_samples[1], properties, use_unstructured)}" + f" Conclusion: Both objects are similar in terms of {property_names[prop]}."
                        else:
                            answer = f"Both objects are similar in terms of {property_names[prop]}."
                if qs == 0:
                    question.insert(0, start_prompt)
                data.append({
                        "role": "USER",
                        "content": question,
                        "tactile": tactile
                    })
                data.append({
                        "role": "ASSISTANT",
                        "content": [answer],
                        "tactile": []
                    })
        elif question_type == f"{split}_property_superlative_selection":
            for qs in range(question_steps):
                question_key = random.choice(list(property_questions[question_type][qs].keys()))
                question = property_questions[question_type][qs][question_key].copy()
                num_tactile = question.count("<img_tokens>")
                # get relevant object(s) and their frames
                prop = random.choice(properties)
                if not use_properties:
                    if "property_superlative_selection_most" in question_key:
                        question = question[:-3]
                        question += [" Select the ", "<most_property>", "."]
                    elif "property_superlative_selection_least" in question_key:
                        question = question[:-3]
                        question += [" Select the ", "<least_property>", "."]
                for question_index, chunk in enumerate(question):
                    if "<" in chunk and ">" in chunk and "property" in chunk:
                        question[question_index] = property_comparisons[prop][chunk]
                        prop_description = property_comparisons[prop][chunk]
                rank = RANKS[prop]
                options = {0: "a)", 1: "b)", 2: "c)"}
                if "property_superlative_selection_most" in question_key:
                    max_rank = max(rank.values())
                    other_samples = random.sample([i for i in samples.keys() if rank[i] < max_rank], k=2)
                    target_sample = random.choice([i for i in samples.keys() if rank[i] == max_rank])
                elif "property_superlative_selection_least" in question_key:
                    min_rank = min(rank.values())
                    other_samples = random.sample([i for i in samples.keys() if rank[i] > min_rank], k=2)
                    target_sample = random.choice([i for i in samples.keys() if rank[i] == min_rank])
                all_samples = [target_sample] + other_samples
                all_samples_shuffled_index = [i for i in range(num_tactile)]
                random.shuffle(all_samples_shuffled_index)
                target_idx = all_samples_shuffled_index.index(0)
                answer = ""
                if use_properties:
                    for i, shuffled_index in enumerate(all_samples_shuffled_index):
                        sample = all_samples[shuffled_index]
                        answer += f"{options[i]} {get_sample_description(sample, properties, use_unstructured)} "
                    answer += "Conclusion: "
                answer += f"{options[target_idx]} is the {prop_description}."
                tactile = [random.choice(samples[all_samples[i]]) for i in all_samples_shuffled_index]
                if split == "eval":
                    if (tactile[0], tactile[1], tactile[2], prop_description) in existing[question_type]:
                        exist = True
                        break
                    else:
                        existing[question_type].append((tactile[0], tactile[1], tactile[2], prop_description))
                if qs == 0:
                    question.insert(0, start_prompt)
                data.append({
                        "role": "USER",
                        "content": question,
                        "tactile": tactile
                    })
                data.append({
                        "role": "ASSISTANT",
                        "content": [answer],
                        "tactile": []
                    })
        elif question_type == f"{split}_property_object_match":
            for qs in range(question_steps):
                question_key = random.choice(list(property_questions[question_type][qs].keys()))
                question = property_questions[question_type][qs][question_key].copy()
                num_tactile = question.count("<img_tokens>")
                if not use_properties:
                    question = question[:-1]
                    question += [" Match each video to one of the following objects in alphabetical order: "]
                # get relevant object(s) and their frames
                all_samples = []
                while len(all_samples) < num_tactile:
                    sample = random.choice(list(samples.keys()))
                    if len(all_samples) == 0:
                        all_samples.append(sample)
                    else:
                        not_exist = True
                        for s in all_samples:
                            if RANKS["hardness"][sample] == RANKS["hardness"][s] and RANKS["roughness"][sample] == RANKS["roughness"][s] and RANKS["texture"][sample] == RANKS["texture"][s]:
                                not_exist = False
                                break
                        if not_exist:
                            all_samples.append(sample)
                all_samples_shuffled_index = [i for i in range(num_tactile)]
                random.shuffle(all_samples_shuffled_index)
                obj_letter = {
                    0: "1)",
                    1: "2)",
                    2: "3)"
                }
                for i, shuffled_index in enumerate(all_samples_shuffled_index):
                    if i + 1 == len(all_samples_shuffled_index):
                        question += [f"{obj_letter[i]} {OBJECTS[all_samples[shuffled_index]]}."]
                    else:
                        question += [f"{obj_letter[i]} {OBJECTS[all_samples[shuffled_index]]}, "]
                tactile = [random.choice(samples[i]) for i in all_samples]
                if split == "eval":
                    if (tactile[0], tactile[1], tactile[2]) in existing[question_type]:
                        exist = True
                        break
                    else:
                        existing[question_type].append((tactile[0], tactile[1], tactile[2]))
                answer = ""
                obj_index = {
                    0: "a)",
                    1: "b)",
                    2: "c)"
                }
                if use_properties:
                    for sample_index, sample in enumerate(all_samples):
                        answer += f"{obj_index[sample_index]} {get_sample_description(sample, properties, use_unstructured)} "
                    answer += "Conclusion: "
                answer += f"a) is {OBJECTS[all_samples[0]]}, "
                answer += f"b) is {OBJECTS[all_samples[1]]} and "
                answer += f"c) is {OBJECTS[all_samples[2]]}."
                if qs == 0:
                    question.insert(0, start_prompt)
                data.append({
                        "role": "USER",
                        "content": question,
                        "tactile": tactile
                    })
                data.append({
                        "role": "ASSISTANT",
                        "content": [answer],
                        "tactile": []
                    })
        if split == "eval":
            if not exist:
                all_data.append(data)
        else:
            all_data.append(data)

    # save all data
    if split == "eval":
        file_name = f"test_qa"
    else:
        file_name = f"{split}_qa"
    if not use_properties:
        file_name += "_no_properties"
    if not use_unstructured:
        file_name += "_no_unstructured"
    data_file = open(os.path.join(data_path, f"{file_name}.json"), "w")
    json.dump(all_data, data_file, indent=4) 
    data_file.close()


def generate_opd_evaluation_qa(start_prompt, json_path, data_path, split, use_unstructured):
    properties = ["hardness", "roughness", "texture"]

    # load samples
    with open(json_path) as json_file:
        samples = json.load(json_file)
        json_file.close()
    all_samples = []
    for k in samples.keys():
        for v in samples[k]:
            all_samples.append((k, v))

    all_data = []
    for i in all_samples:
        question_type = "eval_object_property_description"
        question_steps =  1
        data = [{
            "question_type": question_type,
            "question_steps": question_steps
        }]
        for qs in range(question_steps):
            question = ["Describe the physical properties of <tact_start>", "<img_tokens>", "<tact_end>."]
            # get relevant object(s) and their frames
            sample = i[0]
            tactile = i[1]
            answer = get_sample_description(sample, properties, use_unstructured)
            if qs == 0:
                question.insert(0, start_prompt)
            data.append({
                    "role": "USER",
                    "content": question,
                    "tactile": [tactile]
                })
            data.append({
                    "role": "ASSISTANT",
                    "content": [answer],
                    "tactile": []
                })
        all_data.append(data)

    # save all data
    file_name = f"{split}_opd_qa"
    if not use_unstructured:
        file_name += "_no_unstructured"
    data_file = open(os.path.join(data_path, f"{file_name}.json"), "w")
    json.dump(all_data, data_file, indent=4) 
    data_file.close()


def generate_psr_evaluation_qa(start_prompt, json_path, data_path, num_samples, use_unstructured, use_tactile):
    properties = ["hardness", "roughness", "texture"]

    # load samples
    for i in range(len(json_path)):
        if i == 0:
            with open(json_path[i]) as json_file:
                samples = json.load(json_file)
                json_file.close()
        else:
            with open(json_path[i]) as json_file:
                samples_temp = json.load(json_file)
                json_file.close()
            for k, v in samples_temp.items():
                if k in samples.keys():
                    samples[k] += v
                else:
                    samples[k] = v
    all_samples = []
    for k in samples.keys():
        for v in samples[k]:
            all_samples.append((k, v))

    if use_tactile:
        property_scenario_reasoning = [{
            "property_scenario_reasoning_opd_0": ["Describe these two tactile videos: a) <tact_start>", "<img_tokens>", "<tact_end>, b) <tact_start>", "<img_tokens>", "<tact_end>."]
        }, {
            "property_scenario_reasoning_property_0": ["<scenario_question>", " Select only one most appropriate object for this scenario based on physical property descriptions of the two objects. Use the format 'The most suitable object is x), because xxx'"],
        }]
        property_questions = {
            "eval_property_scenario_reasoning": property_scenario_reasoning
        }
    else:
        property_scenario_reasoning = [{
            "property_scenario_reasoning_opd_0": ["Given these descriptions of two objects: a) ", "<description>", " b) ", "<description>", " ", "<scenario_question>", " Select the most appropriate object in the format 'The most suitable object is x).'"]
        }]
        property_questions = {
            "eval_property_scenario_reasoning": property_scenario_reasoning
        }

    # data
    all_data = []
    existing = {
        "eval_property_scenario_reasoning": [],
    }
    for _ in range(num_samples):
        exist = False
        question_type = random.choice(list(property_questions.keys()))
        if use_tactile:
            question_steps = 2
        else:
            question_steps = 1
        data = [{
            "question_type": question_type,
            "question_steps": question_steps
        }]
        obj_letter = {
            0: "a)",
            1: "b)",
            2: "c)"
        }
        # NOTE: 3 options in total
        question_type == "eval_property_scenario_reasoning"
        scenario = random.choice(SCENARIOS)
        scenario_question = scenario["question"]
        OBJECTS = TEST_OBJECTS
        scenario_other_properties = scenario["other_properties"]
        if "target_properties" in scenario.keys():
            scenario_target_properties = scenario["target_properties"]
            scenario_target_objects = []
            for i in OBJECTS:
                add = True
                if scenario_target_properties[0] != -1:
                    if HARDNESS_RANK[i] not in scenario_target_properties[0]:
                        add = False
                if scenario_target_properties[1] != -1:
                    if ROUGHNESS_RANK[i] not in scenario_target_properties[1]:
                        add = False
                if scenario_target_properties[2] != -1:
                    if TEXTURE_RANK[i] not in scenario_target_properties[2]:
                        add = False
                if add:
                    scenario_target_objects.append(i)
            target_sample = [random.choice(scenario_target_objects)]
            scenario_other_objects = []
            for i in OBJECTS:
                add = True
                if i not in scenario_target_objects:
                    if scenario_other_properties[0] != -1:
                        if HARDNESS_RANK[i] not in scenario_other_properties[0]:
                            add = False
                    if scenario_other_properties[1] != -1:
                        if ROUGHNESS_RANK[i] not in scenario_other_properties[1]:
                            add = False
                    if scenario_other_properties[2] != -1:
                        if TEXTURE_RANK[i] not in scenario_other_properties[2]:
                            add = False
                else:
                    add = False
                if add:
                    scenario_other_objects.append(i)
        elif "target_objects" in scenario.keys():
            scenario_target_objects = scenario["target_objects"]
            target_sample = [random.choice(scenario_target_objects)]
            scenario_other_objects = []
            for i in OBJECTS:
                add = True
                if i not in scenario_target_objects:
                    if scenario_other_properties[0] != -1:
                        if HARDNESS_RANK[i] not in scenario_other_properties[0]:
                            add = False
                    if scenario_other_properties[1] != -1:
                        if ROUGHNESS_RANK[i] not in scenario_other_properties[1]:
                            add = False
                    if scenario_other_properties[2] != -1:
                        if TEXTURE_RANK[i] not in scenario_other_properties[2]:
                            add = False
                else:
                    add = False
                if add:
                    scenario_other_objects.append(i)
        other_samples = random.sample(scenario_other_objects, k=1)
        all_samples = []
        for qs in range(question_steps):
            question_key = random.choice(list(property_questions[question_type][qs].keys()))
            question = property_questions[question_type][qs][question_key].copy()
            num_tactile = question.count("<img_tokens>")
            if qs == 0:
                if use_tactile:
                    target_sample_idx = None
                    all_samples = target_sample + other_samples
                    all_samples_shuffled_index = [i for i in range(num_tactile)]
                    random.shuffle(all_samples_shuffled_index)
                    target_sample_idx = all_samples_shuffled_index.index(0)
                    tactile = [random.choice(samples[all_samples[i]]) for i in all_samples_shuffled_index]
                    if (tactile[0], tactile[1], scenario_question) in existing[question_type]:
                        exist = True
                        break
                    else:
                        existing[question_type].append((tactile[0], tactile[1], scenario_question))
                    answer = ""
                    for sample_index, shuffled_sample_index in enumerate(all_samples_shuffled_index):
                        sample = all_samples[shuffled_sample_index]
                        if sample_index != len(all_samples) - 1:
                            answer += f"{obj_letter[sample_index]} {get_sample_description(sample, properties, use_unstructured)} "
                        else:
                            answer += f"{obj_letter[sample_index]} {get_sample_description(sample, properties, use_unstructured)}"
                else:
                    target_sample_idx = None
                    all_samples = target_sample + other_samples
                    all_samples_shuffled_index = [i for i in range(num_tactile)]
                    random.shuffle(all_samples_shuffled_index)
                    target_sample_idx = all_samples_shuffled_index.index(0)
                    tactile = []
                    description_idx = 0
                    for question_index, chunk in enumerate(question):
                        if chunk == "<description>":
                            sample = all_samples[all_samples_shuffled_index[description_idx]]
                            question[question_index] = get_sample_description(sample, properties, use_unstructured)
                            description_idx += 1
                        elif chunk == "<scenario_question>":
                            question[question_index] = scenario_question
                    answer = f"The most suitable object is {obj_letter[target_sample_idx]}."
            elif qs == 1:
                tactile = []
                answer = f"The most suitable object is {obj_letter[target_sample_idx]}."
                for question_index, chunk in enumerate(question):
                    if chunk == "<scenario_question>":
                        question[question_index] = scenario_question
            if qs == 0:
                question.insert(0, start_prompt)
            data.append({
                    "role": "USER",
                    "content": question,
                    "tactile": tactile
                })
            data.append({
                    "role": "ASSISTANT",
                    "content": [answer],
                    "tactile": []
                })
        if not exist:
            all_data.append(data)
 
    # save all data
    file_name = "psr_qa"
    # if not use_properties:
    #     file_name += "_no_properties"
    if not use_tactile:
        file_name += "_no_tactile"
    if not use_unstructured:
        file_name += "_no_unstructured"
    data_file = open(os.path.join(data_path, f"{file_name}.json"), "w")
    json.dump(all_data, data_file, indent=4)
    data_file.close()


def generate_avocado_evaluation_qa(start_prompt, json_path, data_path, num_samples, use_tactile):
    properties = ["hardness", "roughness", "texture"]

    # load samples
    for i in range(len(json_path)):
        if i == 0:
            with open(json_path[i]) as json_file:
                samples = json.load(json_file)
                json_file.close()
        else:
            with open(json_path[i]) as json_file:
                samples_temp = json.load(json_file)
                json_file.close()
            for k, v in samples_temp.items():
                if k in samples.keys():
                    samples[k] += v
                else:
                    samples[k] = v
    all_samples = []
    for k in samples.keys():
        for v in samples[k]:
            all_samples.append((k, v))

    if use_tactile:
        property_scenario_reasoning = [{
            "property_scenario_reasoning_opd_0": ["Describe these two tactile videos: a) <tact_start>", "<img_tokens>", "<tact_end>, b) <tact_start>", "<img_tokens>", "<tact_end>."]
        }, {
            "property_scenario_reasoning_property_riper_0": ["Both describe different avocados. Which avocado is more likely to be riper?"],
            "property_scenario_reasoning_property_less_0": ["Both describe different avocados. Which avocado is more likely to be less ripe?"],
        }]
        property_questions = {
            "eval_property_scenario_reasoning": property_scenario_reasoning
        }
    else:
        property_scenario_reasoning = [{
            "property_scenario_reasoning_property_riper_0": ["Given the descriptions of two tactile videos: a) ", "<description>", " b) ", "<description>", " Which avocado is more likely to be riper?"],
            "property_scenario_reasoning_property_less_0": ["Given the descriptions of two tactile videos: a) ", "<description>", " b) ", "<description>", " Which avocado is more likely to be less ripe?"],
        }]
        property_questions = {
            "eval_property_scenario_reasoning": property_scenario_reasoning
        }

    # data
    all_data = []
    existing = {
        "eval_property_scenario_reasoning": [],
    }
    for _ in range(num_samples):
        exist = False
        obj_letter = {
            0: "a)",
            1: "b)",
            2: "c)"
        }
        # NOTE: 3 options in total
        question_type = "eval_property_scenario_reasoning"
        scenario = random.choice(AVOCADO_SCENARIOS)
        scenario_question = scenario["question"]
        if use_tactile:
            question_steps = 2
        else:
            question_steps = 1
        data = [{
            "question_type": question_type,
            "question_steps": question_steps
        }]
        scenario_other_properties = scenario["other_properties"]
        if "target_properties" in scenario.keys():
            scenario_target_properties = scenario["target_properties"]
            scenario_target_objects = []
            for i in AVOCADO_OBJECTS:
                add = True
                if scenario_target_properties[0] != -1:
                    if HARDNESS_RANK[i] not in scenario_target_properties[0]:
                        add = False
                if scenario_target_properties[1] != -1:
                    if ROUGHNESS_RANK[i] not in scenario_target_properties[1]:
                        add = False
                if scenario_target_properties[2] != -1:
                    if TEXTURE_RANK[i] not in scenario_target_properties[2]:
                        add = False
                if add:
                    scenario_target_objects.append(i)
            target_sample = [random.choice(scenario_target_objects)]
            scenario_other_objects = []
            for i in AVOCADO_OBJECTS:
                add = True
                if i not in scenario_target_objects:
                    if scenario_other_properties[0] != -1:
                        if HARDNESS_RANK[i] not in scenario_other_properties[0]:
                            add = False
                    if scenario_other_properties[1] != -1:
                        if ROUGHNESS_RANK[i] not in scenario_other_properties[1]:
                            add = False
                    if scenario_other_properties[2] != -1:
                        if TEXTURE_RANK[i] not in scenario_other_properties[2]:
                            add = False
                else:
                    add = False
                if add:
                    scenario_other_objects.append(i)
        other_samples = random.sample(scenario_other_objects, k=1)
        all_samples = []
        for qs in range(question_steps):
            question_key = random.choice(list(property_questions[question_type][qs].keys()))
            question = property_questions[question_type][qs][question_key].copy()
            num_tactile = question.count("<img_tokens>")
            if qs == 0:
                if use_tactile:
                    target_sample_idx = None
                    all_samples = target_sample + other_samples
                    all_samples_shuffled_index = [i for i in range(num_tactile)]
                    random.shuffle(all_samples_shuffled_index)
                    target_sample_idx = all_samples_shuffled_index.index(0)
                    tactile = [random.choice(samples[all_samples[i]]) for i in all_samples_shuffled_index]
                    if (tactile[0], tactile[1], scenario_question) in existing[question_type]:
                        exist = True
                        break
                    else:
                        existing[question_type].append((tactile[0], tactile[1], scenario_question))
                    answer = ""
                    for sample_index, shuffled_sample_index in enumerate(all_samples_shuffled_index):
                        sample = all_samples[shuffled_sample_index]
                        if sample_index != len(all_samples) - 1:
                            answer += f"{obj_letter[sample_index]} {get_sample_description(sample, properties, use_unstructured)} "
                        else:
                            answer += f"{obj_letter[sample_index]} {get_sample_description(sample, properties, use_unstructured)}"
                else:
                    if "riper" in scenario_question:
                        question_key = "property_scenario_reasoning_property_riper_0"
                    else:
                        question_key = "property_scenario_reasoning_property_less_0"
                    question = property_questions[question_type][qs][question_key].copy()
                    target_sample_idx = None
                    all_samples = target_sample + other_samples
                    all_samples_shuffled_index = [i for i in range(2)]
                    random.shuffle(all_samples_shuffled_index)
                    target_sample_idx = all_samples_shuffled_index.index(0)
                    tactile = []
                    description_idx = 0
                    for question_index, chunk in enumerate(question):
                        if chunk == "<description>":
                            sample = all_samples[all_samples_shuffled_index[description_idx]]
                            question[question_index] = get_sample_description(sample, properties, use_unstructured)
                            description_idx += 1
                        elif chunk == "<scenario_question>":
                            question[question_index] = scenario_question
                    answer = f"The most suitable avocado is {obj_letter[target_sample_idx]}."
            elif qs == 1:
                tactile = []
                answer = f"The most suitable avocado is {obj_letter[target_sample_idx]}."
                for question_index, chunk in enumerate(question):
                    if chunk == "<scenario_question>":
                        question[question_index] = scenario_question
            if qs == 0:
                question.insert(0, start_prompt)
            data.append({
                    "role": "USER",
                    "content": question,
                    "tactile": tactile
                })
            data.append({
                    "role": "ASSISTANT",
                    "content": [answer],
                    "tactile": []
                })
        if not exist:
            all_data.append(data)

 
    # save all data
    file_name = "avocado_qa"
    data_file = open(os.path.join(data_path, f"{file_name}.json"), "w")
    json.dump(all_data, data_file, indent=4)
    data_file.close()


def generate_object_evaluation_qa(start_prompt, json_path, data_path, split, num_samples, use_unstructured, use_properties):
    properties = ["hardness", "roughness", "texture"]

    property_names = {
        "hardness": "hardness",
        "roughness": "roughness",
        "texture": "the size of the bumps present"
    }

    # prompt setup
    property_comparisons = {
        "hardness": {
            "<more_property>": "harder",
            "<less_property>": "softer",
            "<most_property>": "hardest",
            "<least_property>": "softest"
        },
        "roughness": {
            "<more_property>": "rougher",
            "<less_property>": "smoother",
            "<most_property>": "roughest",
            "<least_property>": "smoothest"
        },
        "texture": {
            "<more_property>": "covered with bigger bumps",
            "<less_property>": "covered with smaller bumps",
            "<most_property>": "one covered with the biggest bumps",
            "<least_property>": "one covered with the smallest bumps"
        }
    }
    object_property_description = [{
        "object_property_description_0": ["What object is <tact_start>", "<img_tokens>", "<tact_end>?"],
    }]
    property_comparison = [{
        "property_comparison_more_0": ["Is the object in the tactile video <tact_start>", "<img_tokens>", "<tact_end> ", "<more_property>", " than the one in <tact_start>", "<img_tokens>", "<tact_end>?", " Describe both objects before answering."],
        "property_comparison_more_1": ["Is the object in <tact_start>", "<img_tokens>", "<tact_end> ", "<more_property>", " than the object in <tact_start>", "<img_tokens>", "<tact_end>?", " Describe both objects before answering."],
        "property_comparison_less_0": ["Is the object in the tactile video <tact_start>", "<img_tokens>", "<tact_end> ",  "<less_property>", " than the one in <tact_start>", "<img_tokens>", "<tact_end>?", " Describe both objects before answering."],
        "property_comparison_less_1": ["Is the object in <tact_start>", "<img_tokens>", "<tact_end> ", "<less_property>", " than the object in <tact_start>", "<img_tokens>", "<tact_end>?", " Describe both objects before answering."],
    }]
    property_superlative_selection = [{
        "property_superlative_selection_most_0": ["Given three tactile videos: a) <tact_start>", "<img_tokens>", "<tact_end>, b) <tact_start>", "<img_tokens>", "<tact_end>, c) <tact_start>", "<img_tokens>", "<tact_end>.", " Describe each object and then select the ", "<most_property>", "."],
        "property_superlative_selection_most_1": ["You have tactile videos of one object each: a) <tact_start>", "<img_tokens>", "<tact_end>, b) <tact_start>", "<img_tokens>", "<tact_end>, c) <tact_start>", "<img_tokens>", "<tact_end>.", " Describe each object and then select the ", "<most_property>", " object."],
        "property_superlative_selection_least_0": ["Given these tactile videos: a) <tact_start>", "<img_tokens>", "<tact_end>, b) <tact_start>", "<img_tokens>", "<tact_end>, c) <tact_start>", "<img_tokens>", "<tact_end>.", " Describe each object and then select the ", "<least_property>", "."],
        "property_superlative_selection_least_1": ["You have tactile videos of one object each: a) <tact_start>", "<img_tokens>", "<tact_end>, b) <tact_start>", "<img_tokens>", "<tact_end>, c) <tact_start>", "<img_tokens>", "<tact_end>.", " Describe each object and then select the ", "<least_property>", " object."],
    }]
    property_object_match = [{
        "property_object_match_0": ["Given three tactile videos: a) <tact_start>", "<img_tokens>", "<tact_end>, b) <tact_start>", "<img_tokens>", "<tact_end>, c) <tact_start>", "<img_tokens>", "<tact_end>.", " Describe the object in each video, then match each video to one of the following objects in alphabetical order: "],
        "property_object_match_1": ["You have tactile videos of one object each: a) <tact_start>", "<img_tokens>", "<tact_end>, b) <tact_start>", "<img_tokens>", "<tact_end>, c) <tact_start>", "<img_tokens>", "<tact_end>.", " Describe the object in each video, then match each video to one of the following objects in alphabetical order: "],
    }]
    object_description = [{
        "object_description_0": ["What object is <tact_start>", "<img_tokens>", "<tact_end>?"],
        "object_description_1": ["What object does <tact_start>", "<img_tokens>", "<tact_end> show?"],
    }]
    if split == "train":
        property_questions = {
            "train_property_comparison": property_comparison,
            "train_property_superlative_selection": property_superlative_selection,
            "train_property_object_match": property_object_match,
        }
        if use_properties:
            property_questions["train_object_property_description"] = object_property_description
    elif split == "eval":
        property_questions = {
            "eval_property_comparison": property_comparison,
            "eval_property_superlative_selection": property_superlative_selection,
            "eval_property_object_match": property_object_match,
        }

    # load samples
    for i in range(len(json_path)):
        if i == 0:
            with open(json_path[i]) as json_file:
                samples = json.load(json_file)
                json_file.close()
        else:
            with open(json_path[i]) as json_file:
                samples_temp = json.load(json_file)
                json_file.close()
            for k, v in samples_temp.items():
                if k in samples.keys():
                    samples[k] += v
                else:
                    samples[k] = v

    # data
    all_data = []

    if split == "eval":
        existing = {
            "eval_property_comparison": [],
            "eval_property_superlative_selection": [],
            "eval_property_object_match": []
        }
    
    for i in range(num_samples):
        if split == "eval":
            exist = False
        question_type = random.choice(list(property_questions.keys()))
        question_steps =  random.randint(1, len(property_questions[question_type]))
        data = [{
            "question_type": question_type,
            "question_steps": question_steps
        }]
        if question_type == f"{split}_object_property_description":
            for qs in range(question_steps):
                question_key = random.choice(list(property_questions[question_type][qs].keys()))
                question = property_questions[question_type][qs][question_key].copy()
                num_tactile = question.count("<img_tokens>")
                # get relevant object(s) and their frames
                sample = random.sample(samples.keys(), k=num_tactile)[0]
                tactile = [random.choice(samples[sample])]
                answer = get_sample_description(sample, properties, use_unstructured)
                if qs == 0:
                    question.insert(0, start_prompt)
                data.append({
                        "role": "USER",
                        "content": question,
                        "tactile": tactile
                    })
                data.append({
                        "role": "ASSISTANT",
                        "content": [answer],
                        "tactile": []
                    })
        elif question_type == f"{split}_property_comparison":
            num_tactile = 2
            # get relevant object(s) and their frames
            all_samples = random.sample(samples.keys(), k=num_tactile)
            prop = random.choice(properties)
            for qs in range(question_steps):
                question_key = random.choice(list(property_questions[question_type][qs].keys()))
                question = property_questions[question_type][qs][question_key].copy()
                if "property_comparison_more" in question_key:
                    tactile = [random.choice(samples[i]) for i in all_samples]
                    if split == "eval":
                        if (tactile[0], tactile[1], prop) in existing[question_type]:
                            exist = True
                            break
                        else:
                            existing[question_type].append((tactile[0], tactile[1], prop))
                    for question_index, chunk in enumerate(question):
                        if "<" in chunk and ">" in chunk and "property" in chunk:
                            question[question_index] = property_comparisons[prop][chunk]
                            prop_comparison = property_comparisons[prop][chunk]
                    rank = RANKS[prop]
                    if not use_properties:
                        question = question[:-1]
                    if rank[all_samples[0]] > rank[all_samples[1]]:
                        if use_properties:
                            answer = f"First object: {get_sample_description(all_samples[0], properties, use_unstructured)}" + " " + f"Second object: {get_sample_description(all_samples[1], properties, use_unstructured)}" + f" Conclusion: Yes, the first object is {prop_comparison}."
                        else:
                            answer = f"Yes, the first object is {prop_comparison}."
                    elif rank[all_samples[0]] < rank[all_samples[1]]:
                        if use_properties:
                            answer = f"First object: {get_sample_description(all_samples[0], properties, use_unstructured)}" + " " + f"Second object: {get_sample_description(all_samples[1], properties, use_unstructured)}" + f" Conclusion: No, the second object is {prop_comparison}."
                        else:
                            answer = f"No, the second object is {prop_comparison}."
                    else:
                        if use_properties:
                            answer = f"First object: {get_sample_description(all_samples[0], properties, use_unstructured)}" + " " + f"Second object: {get_sample_description(all_samples[1], properties, use_unstructured)}" + f" Conclusion: Both objects are similar in terms of {property_names[prop]}."
                        else:
                            answer = f"Both objects are similar in terms of {property_names[prop]}."
                elif "property_comparison_less" in question_key:
                    tactile = [random.choice(samples[i]) for i in all_samples]
                    for question_index, chunk in enumerate(question):
                        if "<" in chunk and ">" in chunk and "property" in chunk:
                            question[question_index] = property_comparisons[prop][chunk]
                            prop_comparison = property_comparisons[prop][chunk]
                    rank = RANKS[prop]
                    if not use_properties:
                        question = question[:-1]
                    if rank[all_samples[0]] < rank[all_samples[1]]:
                        if use_properties:
                            answer = f"First object: {get_sample_description(all_samples[0], properties, use_unstructured)}" + " " + f"Second object: {get_sample_description(all_samples[1], properties, use_unstructured)}" + f" Conclusion: Yes, the first object is {prop_comparison}."
                        else:
                            answer = f"Yes, the first object is {prop_comparison}."
                    elif rank[all_samples[0]] > rank[all_samples[1]]:
                        if use_properties:
                            answer = f"First object: {get_sample_description(all_samples[0], properties, use_unstructured)}" + " " + f"Second object: {get_sample_description(all_samples[1], properties, use_unstructured)}" + f" Conclusion: No, the second object is {prop_comparison}."
                        else:
                            answer = f"No, the second object is {prop_comparison}."
                    else:
                        if use_properties:
                            answer = f"First object: {get_sample_description(all_samples[0], properties, use_unstructured)}" + " " + f"Second object: {get_sample_description(all_samples[1], properties, use_unstructured)}" + f" Conclusion: Both objects are similar in terms of {property_names[prop]}."
                        else:
                            answer = f"Both objects are similar in terms of {property_names[prop]}."
                if qs == 0:
                    question.insert(0, start_prompt)
                data.append({
                        "role": "USER",
                        "content": question,
                        "tactile": tactile
                    })
                data.append({
                        "role": "ASSISTANT",
                        "content": [answer],
                        "tactile": []
                    })
        elif question_type == f"{split}_property_superlative_selection":
            for qs in range(question_steps):
                question_key = random.choice(list(property_questions[question_type][qs].keys()))
                question = property_questions[question_type][qs][question_key].copy()
                num_tactile = question.count("<img_tokens>")
                # get relevant object(s) and their frames
                prop = random.choice(properties)
                if not use_properties:
                    if "property_superlative_selection_most" in question_key:
                        question = question[:-3]
                        question += [" Select the ", "<most_property>", "."]
                    elif "property_superlative_selection_least" in question_key:
                        question = question[:-3]
                        question += [" Select the ", "<least_property>", "."]
                for question_index, chunk in enumerate(question):
                    if "<" in chunk and ">" in chunk and "property" in chunk:
                        question[question_index] = property_comparisons[prop][chunk]
                        prop_description = property_comparisons[prop][chunk]
                rank = RANKS[prop]
                options = {0: "a)", 1: "b)", 2: "c)"}
                if "property_superlative_selection_most" in question_key:
                    max_rank = max(rank.values())
                    other_samples = random.sample([i for i in samples.keys() if rank[i] < max_rank], k=2)
                    target_sample = random.choice([i for i in samples.keys() if rank[i] == max_rank])
                elif "property_superlative_selection_least" in question_key:
                    min_rank = min(rank.values())
                    other_samples = random.sample([i for i in samples.keys() if rank[i] > min_rank], k=2)
                    target_sample = random.choice([i for i in samples.keys() if rank[i] == min_rank])
                all_samples = [target_sample] + other_samples
                all_samples_shuffled_index = [i for i in range(num_tactile)]
                random.shuffle(all_samples_shuffled_index)
                target_idx = all_samples_shuffled_index.index(0)
                answer = ""
                if use_properties:
                    for i, shuffled_index in enumerate(all_samples_shuffled_index):
                        sample = all_samples[shuffled_index]
                        answer += f"{options[i]} {get_sample_description(sample, properties, use_unstructured)} "
                    answer += "Conclusion: "
                answer += f"{options[target_idx]} is the {prop_description}."
                tactile = [random.choice(samples[all_samples[i]]) for i in all_samples_shuffled_index]
                if split == "eval":
                    if (tactile[0], tactile[1], tactile[2], prop_description) in existing[question_type]:
                        exist = True
                        break
                    else:
                        existing[question_type].append((tactile[0], tactile[1], tactile[2], prop_description))
                if qs == 0:
                    question.insert(0, start_prompt)
                data.append({
                        "role": "USER",
                        "content": question,
                        "tactile": tactile
                    })
                data.append({
                        "role": "ASSISTANT",
                        "content": [answer],
                        "tactile": []
                    })
        elif question_type == f"{split}_property_object_match":
            for qs in range(question_steps):
                question_key = random.choice(list(property_questions[question_type][qs].keys()))
                question = property_questions[question_type][qs][question_key].copy()
                num_tactile = question.count("<img_tokens>")
                if not use_properties:
                    question = question[:-1]
                    question += [" Match each video to one of the following objects in alphabetical order: "]
                # get relevant object(s) and their frames
                all_samples = []
                while len(all_samples) < num_tactile:
                    sample = random.choice(list(samples.keys()))
                    if len(all_samples) == 0:
                        all_samples.append(sample)
                    else:
                        not_exist = True
                        for s in all_samples:
                            if RANKS["hardness"][sample] == RANKS["hardness"][s] and RANKS["roughness"][sample] == RANKS["roughness"][s] and RANKS["texture"][sample] == RANKS["texture"][s]:
                                not_exist = False
                                break
                        if not_exist:
                            all_samples.append(sample)
                all_samples_shuffled_index = [i for i in range(num_tactile)]
                random.shuffle(all_samples_shuffled_index)
                obj_letter = {
                    0: "1)",
                    1: "2)",
                    2: "3)"
                }
                for i, shuffled_index in enumerate(all_samples_shuffled_index):
                    if i + 1 == len(all_samples_shuffled_index):
                        question += [f"{obj_letter[i]} {OBJECTS[all_samples[shuffled_index]]}."]
                    else:
                        question += [f"{obj_letter[i]} {OBJECTS[all_samples[shuffled_index]]}, "]
                tactile = [random.choice(samples[i]) for i in all_samples]
                if split == "eval":
                    if (tactile[0], tactile[1], tactile[2]) in existing[question_type]:
                        exist = True
                        break
                    else:
                        existing[question_type].append((tactile[0], tactile[1], tactile[2]))
                answer = ""
                obj_index = {
                    0: "a)",
                    1: "b)",
                    2: "c)"
                }
                if use_properties:
                    for sample_index, sample in enumerate(all_samples):
                        answer += f"{obj_index[sample_index]} {get_sample_description(sample, properties, use_unstructured)} "
                    answer += "Conclusion: "
                answer += f"a) is {OBJECTS[all_samples[0]]}, "
                answer += f"b) is {OBJECTS[all_samples[1]]} and "
                answer += f"c) is {OBJECTS[all_samples[2]]}."
                if qs == 0:
                    question.insert(0, start_prompt)
                data.append({
                        "role": "USER",
                        "content": question,
                        "tactile": tactile
                    })
                data.append({
                        "role": "ASSISTANT",
                        "content": [answer],
                        "tactile": []
                    })
        if split == "eval":
            if not exist:
                all_data.append(data)
        else:
            all_data.append(data)

    # save all data
    if split == "eval":
        file_name = f"test_qa"
    else:
        file_name = f"{split}_qa"
    if not use_properties:
        file_name += "_no_properties"
    if not use_unstructured:
        file_name += "_no_unstructured"
    data_file = open(os.path.join(data_path, f"{file_name}.json"), "w")
    json.dump(all_data, data_file, indent=4) 
    data_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='directory to save processed frames and sample files')
    args = parser.parse_args()

    use_unstructured = True
    use_tactile = True
    use_properties = True
    # create question-answer pairs for each split
    start_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
    train_json_path = os.path.join(args.data_path, "train_samples.json")
    val_json_path = os.path.join(args.data_path, "val_samples.json")
    test_json_path = os.path.join(args.data_path, "test_samples.json")
    # avocado_json_path = "/home/users/samson/tactile-sensing-llm/robot/avocado_samples.json"
    print("Generating QA...")
    # 1) training
    generate_one_step_qa(start_prompt, [train_json_path], args.data_path, "train", 10000, use_unstructured, use_properties)
    # 2) evaluation
    generate_opd_evaluation_qa(start_prompt, val_json_path, args.data_path, "val", use_unstructured)
    generate_opd_evaluation_qa(start_prompt, test_json_path, args.data_path, "test", use_unstructured)
    generate_one_step_qa(start_prompt, [test_json_path], args.data_path, "eval", 500, use_unstructured, use_properties)
    generate_psr_evaluation_qa(start_prompt, [test_json_path], args.data_path, 50, use_unstructured, use_tactile)
    # 3) avocados
    # generate_opd_evaluation_qa(start_prompt, avocado_json_path, "/home/users/samson/tactile-sensing-llm/robot/avocado_frames", "avocado", use_unstructured)
    # generate_avocado_evaluation_qa(start_prompt, [avocado_json_path], "/home/users/samson/tactile-sensing-llm/robot/avocado_frames", 100, use_tactile)
    print("Done!")