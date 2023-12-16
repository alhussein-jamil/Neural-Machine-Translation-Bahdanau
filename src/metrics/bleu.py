import collections
import torch

def calculate_bleu(reference_list, candidate_list, n=3):
    """
    Calculate the BLEU score between references and candidates for each pair in the same order.
    """
    bleu_scores = []
    for reference_sequence, candidate_sequence in zip(reference_list, candidate_list):
        # Calculate n-grams for reference and candidate
        reference_ngrams = [tuple(reference_sequence[i : i + n]) for i in range(len(reference_sequence) - n + 1)]
        candidate_ngrams = [tuple(candidate_sequence[i : i + n]) for i in range(len(candidate_sequence) - n + 1)]

        # Count n-grams in the reference
        reference_ngram_counts = collections.Counter(reference_ngrams)

        # Count matching n-grams in the candidate that match the reference
        matching_ngram_counts = sum(min(reference_ngram_counts[ngram], candidate_ngrams.count(ngram)) for ngram in set(candidate_ngrams))

        # Calculate BLEU score
        BLEUscore = matching_ngram_counts / (len(candidate_ngrams) if len(candidate_ngrams) > 0 else 1)
 
        bleu_scores.append(BLEUscore)

    return torch.tensor(bleu_scores)

def bleu_tensor(reference_tensor, candidate_tensor, n=3):
    """
    Calculate the BLEU score between references and candidates for each pair in the same order.
    """
    reference_list = [reference_sequence.tolist() for reference_sequence in reference_tensor]
    candidate_list = [candidate_sequence.tolist() for candidate_sequence in candidate_tensor]
    return calculate_bleu(reference_list, candidate_list, n)

def bleu_seq(reference_sequences, candidate_sequences, n=3):
    """
    Calculate the BLEU score between references and candidates for each pair in the same order.
    """
    reference_list = [reference_sequence.split(" ") for reference_sequence in reference_sequences]
    candidate_list = [candidate_sequence.split(" ") for candidate_sequence in candidate_sequences]
    return calculate_bleu(reference_list, candidate_list, n)
