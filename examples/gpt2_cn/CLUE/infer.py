from fairseq.models.huggingface import HuggingFaceGPT2LanguageModel
from sklearn.metrics import f1_score, matthews_corrcoef
import json
from tqdm import tqdm
for i in range(1,11):
    gpt_model = HuggingFaceGPT2LanguageModel.from_pretrained(
        'checkpoints/',
        checkpoint_file=f'checkpoint{i}.pt',
        data_name_or_path='/mnt/lustre/liuzexiang/Code/fairseq/exp/cn_small_clue_test/clue_data/afqmc-bin/'
    )

    label_fn = lambda label: gpt_model.task.label_dictionary.string(
        [label + gpt_model.task.label_dictionary.nspecial]
    )
    ncorrect, nsamples = 0, 0
    gpt_model.cuda()
    gpt_model.eval()
    preds = []
    labels = []
    with open('/mnt/lustre/liuzexiang/Code/fairseq/exp/cn_small_clue_test/clue_data/afqmc/dev.json') as fin:
        ds_raw = fin.read()
        regular_lines = ds_raw.strip().split('\n')
        for line in tqdm(regular_lines):
            line = json.loads(line)
            sent1 = line['sentence1']
            sent2 = line['sentence2']
            target = line['label']
            tokens = gpt_model.encode(sent1,sent2)
            prediction = gpt_model.predict(tokens).argmax().item()
            prediction_label = label_fn(prediction)
            labels.append(target)
            preds.append(prediction_label)
            ncorrect += int(prediction_label == target)
            nsamples += 1
    print('| Accuracy: ', float(ncorrect)/float(nsamples))
    #print('| F1: ', f1_score(labels, preds,pos_label='1'))
    #print("matthews_correlation:", matthews_corrcoef(labels, preds))
