from query_data import query_rag
from langchain_community.llms.ollama import Ollama
import nltk
from nltk.translate.bleu_score import sentence_bleu

# Download NLTK punkt tokenizer
nltk.download('punkt')

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

def test_monopoly_rules():
    print("Testing Monopoly Rules...")
    return query_and_validate(
        question="How much total money does a player start with in Monopoly? (Answer with the number only)",
        expected_response="$1500",
    )


def test_ticket_to_ride_rules():
    print("Testing Ticket to Ride Rules...")
    return query_and_validate(
        question="How many points does the longest continuous train get in Ticket to Ride? (Answer with the number only)",
        expected_response="10 points",
    )


def calculate_bleu(expected_response: str, actual_response: str):
    # Tokenize expected and actual responses
    reference = [nltk.word_tokenize(expected_response)]
    hypothesis = nltk.word_tokenize(actual_response)
    bleu_score = sentence_bleu(reference, hypothesis)
    return bleu_score


def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)

    # BLEU score calculation
    bleu_score = calculate_bleu(expected_response, response_text)
    print(f"BLEU Score for '{question}': {bleu_score:.2f}")

    # Validation through LLM
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model="llama3.2", num_thread=8)
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )

if __name__ == "__main__":
    # Run tests
    all_tests_passed = True

    try:
        if not test_monopoly_rules():
            all_tests_passed = False
    except Exception as e:
        print(f"Error during Monopoly test: {e}")
        all_tests_passed = False

    try:
        if not test_ticket_to_ride_rules():
            all_tests_passed = False
    except Exception as e:
        print(f"Error during Ticket to Ride test: {e}")
        all_tests_passed = False

    if all_tests_passed:
        print("\033[92mAll tests passed successfully!\033[0m")
    else:
        print("\033[91mSome tests failed. Please check the errors above.\033[0m")


