import json
import argparse
import re
from difflib import SequenceMatcher


class LLMEvaluator:
    def __init__(self):
        self.results = {}

    def reset(self):
        self.results = {}

    def get_results(self):
        self.final_results = {}
        for k in self.results.keys():
            self.final_results[k] = {}
            for v in self.results[k].keys():
                if v != "num":
                    self.final_results[k][v] = 0
        for task, stats in self.results.items():
            for stat in stats.keys():
                if stat == "num":
                    continue
                if stats["num"] == 0:
                    self.final_results[task][stat] = 0
                else:
                    self.final_results[task][stat] = stats[stat] / stats["num"]
        return self.final_results

    def evaluate(self, question, generation, answer, question_type, question_step, show_opd, show_pc, show_pss, show_pom, show_question):
        if question_type not in self.results.keys():
            if question_type == "eval_object_property_description":
                self.results[question_type] = {
                    "num": 0,
                    "hardness_accuracy": 0,
                    "roughness_accuracy": 0,
                    "texture_accuracy": 0,
                    "combined_accuracy": 0
                }
            elif "eval_property_superlative_selection" in question_type:
                if "eval_property_superlative_selection" not in self.results.keys():
                    self.results["eval_property_superlative_selection"] = {
                        "num": 0,
                        "accuracy": 0,
                        "slot_accuracy": 0
                    }
            elif question_type == "eval_property_object_match":
                self.results[question_type] = {
                    "num": 0,
                    "accuracy": 0,
                    "slot_accuracy": 0,
                    "canonical_accuracy": 0,
                    "canonical_slot_accuracy": 0
                }
            else:
                self.results[question_type] = {
                    "num": 0,
                    "accuracy": 0
                }
        result = None
        if question_type == "eval_object_property_description":
            if show_opd and show_question:
                print("\n\n" + question)
            result = self.evaluate_opd(generation, answer, show_opd)
        elif question_type == "eval_property_comparison":
            if show_pc and show_question:
                print("\n\n" + question)
            result = self.evaluate_pc(generation, answer, show_pc)
        elif "eval_property_superlative_selection" in question_type:
            if show_pss and show_question:
                print("\n\n" + question)
            question_type = "eval_property_superlative_selection"
            result = self.evaluate_pss(generation, answer, show_pss)
        elif question_type == "eval_property_object_match":
            if show_pom and show_question:
                print("\n\n" + question)
            result = self.evaluate_pom(question, generation, answer, show_pom)
        if result is not None:
            if question_type == "eval_object_property_description":
                self.results[question_type]["hardness_accuracy"] += result[0]
                self.results[question_type]["roughness_accuracy"] += result[1]
                self.results[question_type]["texture_accuracy"] += result[2]
                self.results[question_type]["combined_accuracy"] += result[3]
            elif question_type == "eval_property_object_match" and isinstance(result, tuple):
                self.results[question_type]["accuracy"] += result[0]
                self.results[question_type]["slot_accuracy"] += result[1]
                self.results[question_type]["canonical_accuracy"] += result[2]
                self.results[question_type]["canonical_slot_accuracy"] += result[3]
            elif question_type == "eval_property_superlative_selection" and isinstance(result, tuple):
                self.results[question_type]["accuracy"] += result[0]
                self.results[question_type]["slot_accuracy"] += result[1]
            else:
                self.results[question_type]["accuracy"] += result
            self.results[question_type]["num"] += 1
        
    def evaluate_opd(self, generation, answer, show):
        # evaluate each property separately
        hard_answer = answer.split("presents")[-1].strip().split("and")[0].strip()
        rough_answer = answer.split("presents")[-1].strip().split("and")[1].strip().split("with")[0].strip()
        texture_answer = answer.split("presents")[-1].strip().split("and")[1].strip().split("with")[1].strip()
        if show:
            print("\nOPD:", generation, "||", answer)
        try:
            hard_generation = generation.split("presents")[-1].strip().split("and")[0].strip()
            rough_generation = generation.split("presents")[-1].strip().split("and")[1].strip().split("with")[0].strip()
            texture_generation = generation.split("presents")[-1].strip().split("and")[1].strip().split("with")[1].strip()
        except IndexError:
            return [0, 0, 0, 0]
        correct = [0, 0, 0, 0]
        if hard_answer == hard_generation:
            correct[0] = 1
        if rough_answer == rough_generation:
            correct[1] = 1
        if texture_generation[:len(texture_answer)] == texture_answer:
            correct[2] = 1
        final_answer = answer.split("presents")[-1]
        final_gen = generation.split("presents")[-1]
        if final_gen[:len(final_answer)] == final_answer:
            correct[3] = 1
        return correct
            
    def evaluate_pc(self, generation, answer, show):
        answer = answer.split("Conclusion: ")[-1]
        answer_len = len(answer)
        if show:
            print("\nPC:", generation, "||", answer)
        if generation.split("Conclusion: ")[-1][:answer_len] == answer:
            return 1
        else:
            return 0
    
    def evaluate_pss(self, generation, answer, show):
        answer = answer.split("Conclusion: ")[-1]
        generation = generation.split("Conclusion: ")[-1]
        answer_len = len(answer)
        if show:
            print("\nPSS:", generation, "||", answer)
        full_correct = 1 if generation[:answer_len] == answer else 0
        # slot-level: did the model pick the right option letter, ignoring phrasing?
        answer_letter = self.parse_pss_letter(answer)
        generation_letter = self.parse_pss_letter(generation)
        slot_correct = 1 if (answer_letter is not None and answer_letter == generation_letter) else 0
        return full_correct, slot_correct

    def parse_pss_letter(self, text):
        text = text.replace("</s>", "").strip()
        match = re.search(r"\b([abc])\)", text, flags=re.IGNORECASE)
        if match is not None:
            return match.group(1).lower()
        return None
    
    def evaluate_pom(self, question, generation, answer, show):
        answer = answer.split("Conclusion: ")[-1]
        answer_len = len(answer)
        if show:
            print("\nPOM:", generation, "||", answer)
        generation = generation.split("Conclusion: ")[-1]
        full_correct = 1 if generation[:answer_len] == answer else 0
        slot_correct = 0
        canonical_full_correct = 0
        canonical_slot_correct = 0
        answer_slots = self.parse_pom_slots(answer)
        generation_slots = self.parse_pom_slots(generation)
        if answer_slots is not None and generation_slots is not None:
            slot_correct = sum(1 for key in answer_slots if answer_slots[key] == generation_slots.get(key)) / 3
            candidate_names = self.parse_pom_candidates(question)
            if candidate_names:
                answer_canonical = {
                    key: self.canonicalize_pom_object(value, candidate_names)
                    for key, value in answer_slots.items()
                }
                generation_canonical = {
                    key: self.canonicalize_pom_object(value, candidate_names)
                    for key, value in generation_slots.items()
                }
                if all(answer_canonical.get(key) is not None for key in ["a", "b", "c"]):
                    canonical_slot_correct = sum(
                        1 for key in answer_canonical
                        if answer_canonical[key] == generation_canonical.get(key)
                    ) / 3
                    canonical_full_correct = 1 if canonical_slot_correct == 1 else 0
        return full_correct, slot_correct, canonical_full_correct, canonical_slot_correct

    def parse_pom_slots(self, text):
        text = text.replace("</s>", "").strip()
        pattern = re.compile(
            r"([abc])\)\s*(?:is\s*)?(.*?)(?=(?:,\s*[abc]\)|\s+and\s+[abc]\)|\.?\s*$))",
            flags=re.IGNORECASE | re.DOTALL,
        )
        slots = {}
        for match in pattern.finditer(text):
            slots[match.group(1).lower()] = self.clean_pom_slot_value(match.group(2))
        if any(label not in slots for label in ["a", "b", "c"]):
            return None
        return slots

    def clean_pom_slot_value(self, text):
        text = text.replace("</s>", "").strip(" ,.")
        text = re.sub(r"^\s*is\s+", "", text, flags=re.IGNORECASE)
        return text.strip(" ,.")

    def normalize_object_name(self, text):
        text = text.replace("</s>", "").lower()
        text = re.sub(r"[_\-/]+", " ", text)
        text = re.sub(r"[^a-z0-9 ]+", " ", text)
        text = re.sub(r"\b(the|a|an|object|option)\b", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def parse_pom_candidates(self, question):
        if isinstance(question, list):
            question = "".join([q[0] if isinstance(q, tuple) else str(q) for q in question])
        candidates = []
        for match in re.finditer(r"(?:^|[\s:])([123])\)\s*([^,\.]+)", question):
            candidate = match.group(2).strip()
            if candidate:
                candidates.append(candidate)
        return candidates if len(candidates) == 3 else []

    def canonicalize_pom_object(self, text, candidates):
        option_match = re.search(r"\b(?:option\s*)?([123])\b", text.lower())
        if option_match is not None:
            option_idx = int(option_match.group(1)) - 1
            if 0 <= option_idx < len(candidates):
                return candidates[option_idx]
        text_norm = self.normalize_object_name(text)
        if not text_norm:
            return None
        candidate_norms = [self.normalize_object_name(candidate) for candidate in candidates]
        for candidate, candidate_norm in zip(candidates, candidate_norms):
            if text_norm == candidate_norm:
                return candidate
        for candidate, candidate_norm in zip(candidates, candidate_norms):
            if candidate_norm and (candidate_norm in text_norm or text_norm in candidate_norm):
                return candidate
        scores = [
            SequenceMatcher(None, text_norm, candidate_norm).ratio()
            for candidate_norm in candidate_norms
        ]
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        sorted_scores = sorted(scores, reverse=True)
        if sorted_scores[0] >= 0.86 and sorted_scores[0] - sorted_scores[1] >= 0.08:
            return candidates[best_idx]
        return None


random_scores = {
    "eval_property_comparison": {
        "accuracy": 0.333
    },
    "eval_property_object_match": {
        "accuracy": 0.167,
        "slot_accuracy": 0.333,
        "canonical_accuracy": 0.167,
        "canonical_slot_accuracy": 0.333
    },
    "eval_property_superlative_selection": {
        "accuracy": 0.333,
        "slot_accuracy": 0.333
    },
    "eval_object_property_description": {
        "hardness_accuracy": 0.33,
        "roughness_accuracy": 0.33,
        "texture_accuracy": 0.33,
        "combined_accuracy": 0.037
    }
}


def print_stats(json_path, show_opd, show_pc, show_pss, show_pom, show_question):
    with open(json_path, "r") as f:
        data = json.load(f)
        f.close()
    evaluator = LLMEvaluator()
    for d in data:
        evaluator.evaluate(d["question"], d["generation"], d["answer"], d["question_type"], d["question_step"], show_opd, show_pc, show_pss, show_pom, show_question)
    results = evaluator.get_results()
    print("\n")
    for t in results.keys():
        if t not in random_scores.keys():
            continue
        print(t)
        for k, v in results[t].items():
            print(f"\t{k} -----> {v} ({random_scores[t][k]})")
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_preds_path', help='predictions file to evaluate')
    args = parser.parse_args()

    # print generations/answers or not
    show_opd = False
    show_pc = False
    show_pss = False
    show_pom = False
    show_question = False

    print_stats(args.test_preds_path, show_opd, show_pc, show_pss, show_pom, show_question)
