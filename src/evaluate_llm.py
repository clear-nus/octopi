import json
import argparse


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
                        "accuracy": 0
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
            result = self.evaluate_pom(generation, answer, show_pom)
        if result is not None:
            if question_type == "eval_object_property_description":
                self.results[question_type]["hardness_accuracy"] += result[0]
                self.results[question_type]["roughness_accuracy"] += result[1]
                self.results[question_type]["texture_accuracy"] += result[2]
                self.results[question_type]["combined_accuracy"] += result[3]
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
        if generation[:answer_len] == answer:
            return 1
        else:
            return 0
    
    def evaluate_pom(self, generation, answer, show):
        answer = answer.split("Conclusion: ")[-1]
        answer_len = len(answer)
        if show:
            print("\nPOM:", generation, "||", answer)
        if generation.split("Conclusion: ")[-1][:answer_len] == answer:
            return 1
        else:
            return 0


random_scores = {
    "eval_property_comparison": {
        "accuracy": 0.333
    },
    "eval_property_object_match": {
        "accuracy": 0.167
    },
    "eval_property_superlative_selection": {
        "accuracy": 0.333
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


9
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