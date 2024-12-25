import os
import json
import argparse
from nltk.translate.bleu_score import sentence_bleu
from rouge_metric import PyRouge
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from query_data import query_rag
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
    data = load_ground_truth(json_path)

    results = []
    overall_scores = {'BLEU': [], 'ROUGE-1': [], 'ROUGE-2': [], 'ROUGE-L': [], 'Cosine Similarity': []}

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

    return results, overall_scores, average_scores

# Save detailed results and metrics
def save_results(results, overall_scores, average_scores, output_path):
    metric_output_path = f"{output_path}/metric_output.json"
    with open(metric_output_path, 'w') as file:
        json.dump({
            "results": results,
            "overall_scores": overall_scores,
            "average_scores": average_scores
        }, file, indent=4)
    print(f"Detailed results and metrics saved to {metric_output_path}")

# Visualize metrics
def visualize_results(overall_scores, average_scores, output_path):
    # Bar chart for average scores
    plt.figure(figsize=(10, 6))
    metrics = list(average_scores.keys())
    avg_values = list(average_scores.values())
    plt.bar(metrics, avg_values, color='skyblue')
    plt.title("Average Metrics for RAG Model", fontsize=16)
    plt.ylabel("Scores", fontsize=12)
    plt.xlabel("Metrics", fontsize=12)
    plt.ylim(0, 1)
    plt.savefig(f"{output_path}/average_metrics_bar_chart.png")
    plt.close()
    
    # Boxplot for individual scores
    plt.figure(figsize=(12, 8))
    boxplot_data = [overall_scores[metric] for metric in metrics]
    plt.boxplot(boxplot_data, labels=metrics, vert=True, patch_artist=True)
    plt.title("Distribution of Scores Across Metrics", fontsize=16)
    plt.ylabel("Scores", fontsize=12)
    plt.xlabel("Metrics", fontsize=12)
    plt.savefig(f"{output_path}/metrics_boxplot.png")
    plt.close()
    
    # Line plot for trend visualization (example visualization)
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        plt.plot(overall_scores[metric], label=metric, marker='o')
    plt.title("Trend of Scores Across Data Points", fontsize=16)
    plt.ylabel("Scores", fontsize=12)
    plt.xlabel("Data Points", fontsize=12)
    plt.legend()
    plt.savefig(f"{output_path}/metrics_trend_line_plot.png")
    plt.close()

# Main function
def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Model with Ground Truth JSON")
    parser.add_argument("json_path", type=str, help="Path to the JSON file containing ground truth data")
    parser.add_argument("output_path", type=str, help="Directory to save outputs and visualizations")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Evaluate the RAG model
    print("Evaluating RAG model...")
    results, overall_scores, average_scores = evaluate_rag(args.json_path)

    # Save detailed results and metrics
    save_results(results, overall_scores, average_scores, args.output_path)

    # Visualize metrics
    print("Visualizing results...")
    visualize_results(overall_scores, average_scores, args.output_path)

    # Print average scores
    print("Average Scores:")
    for metric, score in average_scores.items():
        print(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    main()
