import torch
import pdb
from tqdm import tqdm
import argparse
import re
import torch.nn as nn

from fairseq.models.huggingface import HuggingFaceGPT2LanguageModel


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='/mnt/lustre/liuzexiang/Data/LAMBADA/lambada_test.jsonl',
                    help='location of lambada dataset')
parser.add_argument('--word-eval', action='store_true', help="whether to do evaluation on words rather than BPE "
                                                             "tokens.")
parser.add_argument('--beam-width', type=int, default=128, help='predict this many results before stopword filtering')

args = parser.parse_args()


gpt_model = HuggingFaceGPT2LanguageModel.from_pretrained(
    '../gpt2_small_noshuffle/checkpoints/',
    checkpoint_file='checkpoint_4_140000.pt',
    data_name_or_path='data-bin/wikitext-103/'
)
label_fn = lambda label: gpt_model.task.label_dictionary.string(
 [label + gpt_model.task.label_dictionary.nspecial]
)

from plugins.transformers import BasicTokenizer
tokenizer = BasicTokenizer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpt_model.cuda()
gpt_model.eval()

stopwords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
             'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',
             'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the',
             'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were',
             'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to',
             'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have',
             'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can',
             'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
             'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by',
             'doing', 'it', 'how', 'further', 'was', 'here', 'than'}

def remove_last_word(line):
    line = line.strip()
    toks = tokenizer.tokenize(line)
    length_of_word = len(toks[-1])
    assert length_of_word > 0
    return line[:-length_of_word].strip(), toks[-1]

def to_list(tensor):
    return list(tensor.cpu().numpy())


def predict(line_encoded, max_predictions=6):
    """Give continuation of the line with at most max_predictions BPE tokens. Returns line extended with predictions of
     the model."""
    line_encoded = line_encoded.unsqueeze_(0)  # batch of size 1
    line_encoded_list = list(line_encoded[0].numpy())
    line_encoded = line_encoded.to(device)
    state = {'past':None}

    for i in range(max_predictions):

        logits, outputs = gpt_model.model(line_encoded, incremental_state=state)
        state['past'] = outputs[1]

        #        predicted = argmax(logits[0,-1,:])

        # [[idx1, idx2, ...]]
        _, line_encoded_candidates = torch.topk(logits[:, -1, :], k=args.beam_width, dim=-1)

        # determine which candidates are stopwords by decoding them and
        # comparing against NLTK stopword list

        line_encoded_candidates = to_list(line_encoded_candidates[0])
        is_stopword = []
        for s in line_encoded_candidates:
            is_stopword.append(gpt_model.decode([s.item()]).strip() in stopwords)

        # find first prediction which is not a stopword
        predicted = None
        for (idx, candidate) in enumerate(line_encoded_candidates):
            if is_stopword[idx]:
                #                print('skipping stopword ', idx)
                continue
            else:
                predicted = candidate
                break
        assert predicted is not None
        line_encoded = torch.tensor([[predicted]]).to(device)
        line_encoded_list.append(predicted)

    return gpt_model.decode(line_encoded_list)

def main():
    ncorrect, nsamples = 0, 0
    ds_raw = open(f'{args.path}').read()
    regular_lines = ds_raw.strip().split('\n')
    json_lines = []
    if args.path.endswith('.jsonl'):
        # special handling for file from Jeff
        for line in regular_lines:
            # {"text": "In my"} => "In my"
            candidate = line[len('{"text": "'):-len('"}')]
            # wrap to handle quotes inside and
            candidate = f'''"""{candidate}"""'''
            #import pdb;pdb.set_trace()
            json_lines.append(eval(candidate).replace('\n\n',' '))

            #            json_lines.append(eval(line)['text'])
        lines = json_lines
    else:
        lines = regular_lines
    if args.word_eval:
        ####### acc: word level
        print("word level acc test:")
        for line in tqdm(lines):
            line = line.strip()
            context, last_word = remove_last_word(line)
            #gpt_model.model.decoder.init_incremental_state()
            tokens = gpt_model.encode(context)
            length = len(tokens)
            if length == 0:
                continue
            prediction = predict(tokens)
            predicted_part = prediction[len(context):].strip()
            predicted_word = tokenizer.tokenize(predicted_part)[0]
            is_error = predicted_word.lower() != last_word.lower()
            if not is_error:
              ncorrect += 1
            nsamples+=1
        print('| Accuracy: ', float(ncorrect)/float(nsamples))
    else:
        #### acc:token level
        print("token level acc test:")
        for line in tqdm(lines):
            line = line.strip()
            tokens = gpt_model.encode(line)
            length = len(tokens)
            if length == 0:
                continue
            logits, _ = gpt_model.extract_features(tokens)
            target = torch.tensor(tokens[-1]).cuda()
            pred = logits[0, len(tokens)-2, :].argmax(-1)
            if target == pred:
              ncorrect += 1
            nsamples += 1
        print('| Accuracy: ', float(ncorrect)/float(nsamples))

if __name__ == '__main__':
    main()


