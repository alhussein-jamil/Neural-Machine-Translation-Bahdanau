import collections
import torch

def bleu_tensor(reference_tensor, candidate_tensor, n=3):
    """
    Calculate the BLEU score between references and candidates for each pair in the same order.
    """
    bleu_scores = []
    for reference_sequence, candidate_sequence in zip(reference_tensor, candidate_tensor):
        reference_list = reference_sequence.tolist()
        candidate_list = candidate_sequence.tolist()

        # Calculate n-grams for reference and candidate
        reference_ngrams = [tuple(reference_list[i : i + n]) for i in range(len(reference_list) - n + 1)]
        candidate_ngrams = [tuple(candidate_list[i : i + n]) for i in range(len(candidate_list) - n + 1)]

        # Count n-grams in the reference
        reference_ngram_counts = collections.Counter(reference_ngrams)

        # Count matching n-grams in the candidate that match the reference
        matching_ngram_counts = sum(min(reference_ngram_counts[ngram], candidate_ngrams.count(ngram)) for ngram in set(candidate_ngrams))

        # Calculate BLEU score
        BLEUscore = matching_ngram_counts / len(candidate_ngrams)

        bleu_scores.append(BLEUscore)

    return torch.tensor(bleu_scores)


def bleu_seq(reference_sequences, candidate_sequences, n=3):
    """
    Calculate the BLEU score between references and candidates for each pair in the same order.
    """
    bleu_scores = []
    for reference_sequence, candidate_sequence in zip(reference_sequences, candidate_sequences):
        reference_list = reference_sequence.split(" ")
        candidate_list = candidate_sequence.split(" ")

        # Calculate n-grams for reference and candidate
        reference_ngrams = [tuple(reference_list[i : i + n]) for i in range(len(reference_list) - n + 1)]
        candidate_ngrams = [tuple(candidate_list[i : i + n]) for i in range(len(candidate_list) - n + 1)]

        # Count n-grams in the reference
        reference_ngram_counts = collections.Counter(reference_ngrams)

        # Count matching n-grams in the candidate that match the reference
        matching_ngram_counts = sum(min(reference_ngram_counts[ngram], candidate_ngrams.count(ngram)) for ngram in set(candidate_ngrams))

        # Calculate BLEU score
        BLEUscore = matching_ngram_counts / (len(candidate_ngrams) if len(candidate_ngrams) > 0 else 1)
 
        bleu_scores.append(BLEUscore)
    return torch.tensor(bleu_scores)