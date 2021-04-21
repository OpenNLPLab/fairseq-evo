from fairseq.models.huggingface import HuggingFaceGPT2LanguageModel
gpt_model = HuggingFaceGPT2LanguageModel.from_pretrained(
    'checkpoints/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='CoLA-bin'
)

label_fn = lambda label: gpt_model.task.label_dictionary.string(
    [label + gpt_model.task.label_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
gpt_model.cuda()
gpt_model.eval()
with open('glue_data/CoLA/dev.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        target, sent1, sent2 = tokens[1], tokens[2], tokens[3]
        tokens = gpt_model.encode(sent2)
        prediction = gpt_model.predict(tokens).argmax().item()
        prediction_label = label_fn(prediction)
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))