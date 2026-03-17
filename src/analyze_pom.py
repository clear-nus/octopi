import json

def check_pom_accuracy(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    pom_data = [d for d in data if d['question_type'] == 'eval_property_object_match']
    
    total = 0
    correct = 0
    format_errors = 0
    wrong_answers = 0

    print(f"Found {len(pom_data)} POM examples.")

    for i, item in enumerate(pom_data):
        total += 1
        gen = item['generation']
        ans = item['answer']

        # Extract conclusion
        if "Conclusion: " not in ans:
            print(f"[{i}] Warning: Answer missing 'Conclusion: '")
            continue
        
        ans_conclusion = ans.split("Conclusion: ")[-1].strip()
        
        if "Conclusion: " not in gen:
            format_errors += 1
            # print(f"[{i}] Format Error: 'Conclusion:' missing in generation.\nGen: {gen}\n")
            continue

        gen_conclusion_full = gen.split("Conclusion: ")[-1]
        # Based on evaluate_llm.py logic:
        # if generation.split("Conclusion: ")[-1][:answer_len] == answer:
        
        # We need to act exactly like evaluate_llm.py to reproduce the score
        # Note: evaluate_llm logic uses the RAW answer string after split, 
        # but ans_conclusion above is stripped. 
        # Let's reproduce exactly:
        
        raw_ans_suffix = ans.split("Conclusion: ")[-1]
        ans_len = len(raw_ans_suffix)
        
        raw_gen_suffix = gen.split("Conclusion: ")[-1]
        
        is_correct = False
        if len(raw_gen_suffix) >= ans_len:
             if raw_gen_suffix[:ans_len] == raw_ans_suffix:
                 is_correct = True
        
        if is_correct:
            correct += 1
        else:
            wrong_answers += 1
            # Print first 5 failures to see what's wrong
            if wrong_answers <= 5:
                print(f"[{i}] Wrong Answer:")
                print(f"  Ref: '{raw_ans_suffix}'")
                print(f"  Gen: '{raw_gen_suffix[:ans_len]}...'")
                print(f"  Full Gen: '{raw_gen_suffix}'")
                # Check if it was just a period or space issue
                if raw_gen_suffix.strip() == raw_ans_suffix.strip():
                    print("  -> MATCHES if stripped!")
                elif raw_gen_suffix.strip().rstrip('.') == raw_ans_suffix.strip().rstrip('.'):
                     print("  -> MATCHES if stripped and period removed!")
                print("-" * 20)

    print(f"\nTotal: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct/total if total else 0:.4f}")
    print(f"Format Errors (missing 'Conclusion:'): {format_errors}")
    print(f"Wrong Answers (eval string mismatch): {wrong_answers}")

if __name__ == "__main__":
    check_pom_accuracy("/home/samson/octopi/exps/2026_02_05_11_04_46_train_llm_train_val_test_lora_256_128_vicuna-7b_3000_full_pipeline_0_lora/test_final_preds.json")
