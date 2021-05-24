from fairseq.models.huggingface import HuggingFaceGPT2LanguageModel
from sklearn.metrics import f1_score, matthews_corrcoef
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import re
import pdb
import torch
import torch.nn as nn
import argparse
from datasets import load_dataset
from plugins.transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm
from torch.nn import CrossEntropyLoss, MSELoss

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='/mnt/lustre/liuzexiang/Data/LAMBADA/lambada_test.jsonl',
                    help='location of lambada dataset')
args = parser.parse_args()
args.hg_dataset = False
args.tokenize_by_line = False
args.preprocess = True

gpt_model = HuggingFaceGPT2LanguageModel.from_pretrained(
    '../gpt2_small_noshuffle/checkpoints',
    checkpoint_file='checkpoint_4_140000.pt',
    data_name_or_path='/mnt/lustre/liuzexiang/Data/wikitext-103/data-bin/wikitext-103/'
)

gpt_model.cuda()
gpt_model.eval()

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2',local_files_only=True)
max_length = model.decoder.model.config.n_positions
stride = 1024
device = 'cuda'
lls = []

if args.hg_dataset:
    test = load_dataset('ptb_text_only', split='test')
    encodings = gpt_model.encode(' '.join(test['sentence']))
else:
    with open(args.path,'r') as f:
        data = f.readlines()
    if args.preprocess:
        encodings = []
        data_tmp = []
        for line in data:
            if line.strip()=='':
                continue
            line = line.strip()
            line = line.replace(' ,',',')
            line = line.replace(' .', '.')
            line = line.replace(' ,', ',')
            line = line.replace(' ;', ';')
            line = line.replace('``','"')
            line = line.replace("''",'"')
            #line = line.replace('<unk>', '')
            #line = line.replace(' \'', '\'')
            data_tmp.append(line)
        encodings = gpt_model.encode('\n'.join(data_tmp))
if 'jsonl' in args.path:
    ds_raw = open(args.path).read()
    regular_lines = ds_raw.strip().split('\n')
    json_lines = []
    # special handling for file from Jeff
    for line in regular_lines:
        #import pdb;pdb.set_trace()
        # {"text": "In my"} => "In my"
        candidate = line[len('{"text": "'):-len('"}')]
        candidate = candidate.replace('\n\n',' ')
        # wrap to handle quotes inside and
        candidate = f'''"""{candidate}"""'''
        json_lines.append(eval(candidate))
        data = json_lines
    encodings = gpt_model.encode('\n'.join(data))
if encodings.dim() == 1:
    encodings = encodings.unsqueeze(0)

if args.tokenize_by_line:
    n = 0
    for line in tqdm(data):
        if line.strip()=='':
            continue
        encoding = gpt_model.encode(line)
        l = encoding.size(0)
        if encoding.dim() == 1:
            encoding = encoding.unsqueeze(0)
        input_ids = encoding.to(device)
        target_ids = input_ids.clone()
        labels = target_ids
        with torch.no_grad():
            outputs = gpt_model.model(input_ids)
            lm_logits = outputs[0]
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            log_likelihood = loss * (l-1)
            n+=(l-1)

        lls.append(log_likelihood)
    ppl = torch.exp(torch.stack(lls).sum() / n)
    print(f'ppl :{ppl}')

else:
    for i in tqdm(range(0, encodings.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.size(1))
        trg_len = end_loc - i    # may be different from stride on last loop
        input_ids = encodings[:,begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100
        labels = target_ids
        with torch.no_grad():
            outputs = gpt_model.model(input_ids)
            lm_logits = outputs[0]
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            log_likelihood = loss * trg_len
        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    print(f'ppl :{ppl}')
