import json
import argparse
from nltk.translate.bleu_score import sentence_bleu
from rouge_metric import PyRouge
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from query_data import query_rag  # Assuming this is the function to query the RAG model
from tqdm import tqdm

# Load ground truth data
def load_ground_truth(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

# Compare answers using NLP metrics
def evaluate_answers(ground_truth, predicted):
    scores = {}

    # BLEU score
    reference = [ground_truth.split()]
    hypothesis = predicted.split()
    scores['BLEU'] = sentence_bleu(reference, hypothesis)

    # ROUGE score
    rouge = PyRouge(rouge_n=(1, 2), rouge_l=True, rouge_w=False, rouge_s=False, rouge_su=False)
    rouge_scores = rouge.evaluate([ground_truth], [predicted])
    scores.update({
        'ROUGE-1': rouge_scores['rouge-1']['f'],
        'ROUGE-2': rouge_scores['rouge-2']['f'],
        'ROUGE-L': rouge_scores['rouge-l']['f'],
    })

    # Cosine similarity
    model = SentenceTransformer('all-MiniLM-L6-v2')
    ground_truth_emb = model.encode(ground_truth, convert_to_tensor=True)
    predicted_emb = model.encode(predicted, convert_to_tensor=True)
    cosine_similarity = util.pytorch_cos_sim(ground_truth_emb, predicted_emb).item()
    scores['Cosine Similarity'] = cosine_similarity

    return scores


# Evaluate RAG model
def evaluate_rag(json_path):
    # Load ground truth data
    data = load_ground_truth(json_path)

    results = []
    overall_scores = {'BLEU': [], 'ROUGE-1': [], 'ROUGE-2': [], 'ROUGE-L': [], 'Cosine Similarity': []}

    # Iterate through each data point
    for entry in tqdm(data, desc="Evaluating RAG"):
        question = entry['question']
        ground_truth_answer = entry['answer']

        # Get RAG's response
        predicted_answer = query_rag(question)

        # Evaluate
        scores = evaluate_answers(ground_truth_answer, predicted_answer)
        results.append({
            'question': question,
            'ground_truth': ground_truth_answer,
            'predicted': predicted_answer,
            'scores': scores
        })

        # Aggregate scores
        for key in overall_scores:
            overall_scores[key].append(scores[key])

    # Compute average scores
    average_scores = {key: sum(values) / len(values) for key, values in overall_scores.items()}

    return results, average_scores

# Visualize results
def visualize_results(average_scores):
    plt.figure(figsize=(10, 6))
    metrics = list(average_scores.keys())
    values = list(average_scores.values())
    
    plt.bar(metrics, values)
    plt.title("RAG Model Evaluation Metrics")
    plt.ylabel("Scores")
    plt.xlabel("Metrics")
    plt.ylim(0, 1)  # All metrics are normalized between 0 and 1
    plt.show()

# Main function
def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Model with Ground Truth JSON")
    parser.add_argument("json_path", type=str, help="Path to the JSON file containing ground truth data")
    args = parser.parse_args()

    # Evaluate the RAG model
    print("Evaluating RAG model...")
    results, average_scores = evaluate_rag(args.json_path)

    # Save detailed results
    detailed_results_path = "detailed_results.json"
    with open(detailed_results_path, 'w') as file:
        json.dump(results, file, indent=4)
    print(f"Detailed results saved to {detailed_results_path}")

    # Visualize average scores
    print("Visualizing results...")
    visualize_results(average_scores)

    # Print average scores
    print("Average Scores:")
    for metric, score in average_scores.items():
        print(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    main()
