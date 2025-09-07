import sys
import json
import os
from collections import Counter
import numpy as np

def analyze_uats_results(results_dir: str):
    majority_correct = 0
    best_of_n_correct = 0
    total_questions = 0
    all_tokens = []
    
    # Iterate through all question directories
    for folder_name in os.listdir(results_dir):
        if folder_name.startswith("question_"):
            results_path = os.path.join(results_dir, folder_name, "results.json")
            
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    data = json.load(f)
                
                total_questions += 1
                
                # Get all tokens for this question
                all_tokens.append(data["total_tokens_generated"])
                
                # Check majority vote correctness
                if data["correct_count"] > len(data["all_answers"]) / 2:
                    majority_correct += 1
                
                # Check best of n correctness (highest value score)
                best_branch = max(data["branches"], key=lambda x: x["value"])
                best_answer = best_branch["extracted_answer"]
                
                # Extract numerical values for comparison
                correct_answer = data["correct_answer"]
                if best_answer == correct_answer:
                    best_of_n_correct += 1
                else:
                    # Try to extract numbers for comparison
                    try:
                        best_num = ''.join(filter(str.isdigit, best_answer))
                        correct_num = ''.join(filter(str.isdigit, correct_answer))
                        if best_num and correct_num and best_num == correct_num:
                            best_of_n_correct += 1
                    except:
                        pass
    
    if total_questions == 0:
        print("No question directories found!")
        return
    
    # Calculate statistics
    majority_vote_accuracy = (majority_correct / total_questions) * 100
    best_of_n_accuracy = (best_of_n_correct / total_questions) * 100
    mean_tokens = np.mean(all_tokens)
    percentile_95_tokens = np.percentile(all_tokens, 95)
    
    print(f"Dataset Statistics:")
    print(f"1. Majority vote accuracy: {majority_vote_accuracy:.2f}%")
    print(f"2. Best of n accuracy: {best_of_n_accuracy:.2f}%")
    print(f"3. Mean tokens used: {mean_tokens:.2f}")
    print(f"4. 95th percentile tokens used: {percentile_95_tokens:.2f}")
    print(f"\nTotal questions analyzed: {total_questions}")

if __name__ == "__main__":
    # add simple parser
    results_dir = sys.argv[1]
    analyze_uats_results(results_dir)