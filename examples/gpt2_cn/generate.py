"""
TopK for text generation
"""
import torch
import pdb
from tqdm import tqdm
import argparse
import re
import torch.nn as nn
import numpy as np
from fairseq.models.huggingface import HuggingFaceGPT2LanguageModel
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--max-prediction', type=int, default=200)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--min_tokens_to_keep',type=int,default=1)
parser.add_argument('--top_p',type=float,default=0.9)
args = parser.parse_args()


gpt_model = HuggingFaceGPT2LanguageModel.from_pretrained(
    'checkpoints',
    checkpoint_file='checkpoint_last.pt',
    data_name_or_path='/mnt/lustre/liuzexiang/Data/gptdata_bin/'
)

label_fn = lambda label: gpt_model.task.label_dictionary.string(
 [label + gpt_model.task.label_dictionary.nspecial]
)

gpt_model.cuda()
gpt_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_list(tensor):
    return list(tensor.cpu().numpy())

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.device_count()>0:
        torch.cuda.manual_seed_all(args.seed)

def _init_sequence_length_for_generation(
    input_ids, max_length
):
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    sequence_lengths = input_ids.new(input_ids.shape[0]).fill_(max_length)

    cur_len = input_ids.shape[-1]
    return sequence_lengths, unfinished_sequences, cur_len


def _update_seq_length_for_generation(
    sequence_lengths,
    unfinished_sequences,
    cur_len,
    is_eos_in_next_token
):
    # check if sentence is not finished yet
    is_sent_unfinished = unfinished_sequences.mul(is_eos_in_next_token.long()).bool()

    # update sentence length
    sequence_lengths = sequence_lengths.masked_fill(is_sent_unfinished, cur_len)
    unfinished_sequences = unfinished_sequences.mul((~is_eos_in_next_token).long())
    return sequence_lengths, unfinished_sequences

def predict(line_encoded, max_predictions,eos_token_id=9,pad_token_id=9):
    """Give continuation of the line with at most max_predictions BPE tokens. Returns line extended with predictions of
     the model."""
    filter_value = -float("Inf")
    max_length = max_predictions+len(line_encoded)
    line_encoded = line_encoded.unsqueeze_(0)  # batch of size 1
    line_encoded = line_encoded.to(device)
    input_ids = line_encoded
    sequence_lengths, unfinished_sequences, cur_len = _init_sequence_length_for_generation(
        input_ids, max_length
    )
    while cur_len < max_length:
        logits, outputs = gpt_model.model(input_ids)
        next_token_logits = logits[:, -1, :]
        scores = next_token_logits
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > args.top_p
        if args.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, filter_value)
        #        predicted = argmax(logits[0,-1,:])
        next_token_scores = scores
        probs = F.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        if eos_token_id is not None:
            assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # add token and increase length by one
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        # update sequence length
        if eos_token_id is not None:
            sequence_lengths, unfinished_sequences = _update_seq_length_for_generation(
                sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
            )
        # [[idx1, idx2, ...]]
        '''
        probs, line_encoded_candidates = torch.topk(logits[:, -1, :],k=1, dim=-1)

        line_encoded_candidates = line_encoded_candidates.squeeze()
        probs = probs.squeeze()
        p = probs.cpu().detach().numpy()
        line_encoded_candidates = line_encoded_candidates.cpu().numpy()
        if not p.size==1:
            p = p / sum(p)
            target_index = np.random.choice(len(p), p=p)
            predicted = line_encoded_candidates[target_index]
        else:
            predicted = int(line_encoded_candidates)
        # determine which candidates are stopwords by decoding them and
        # comparing against NLTK stopword list
        assert predicted is not None
        line_predicted = torch.tensor([[predicted]]).to(device)
        line_encoded = torch.cat([line_encoded,line_predicted],dim=1)
        line_encoded_list.append(predicted)
        '''
        cur_len = cur_len + 1
    for idx,sequence in enumerate(input_ids):
        sequence = sequence.tolist()
    return gpt_model.decode(sequence)
def main():
    set_seed(args)
    while True:
        context = input('inputs:')
        tokens = gpt_model.encode(context)
        prediction = predict(tokens, args.max_prediction)
        print(prediction, flush=True)
        print('***********************\n')


if __name__ == '__main__':
    main()


