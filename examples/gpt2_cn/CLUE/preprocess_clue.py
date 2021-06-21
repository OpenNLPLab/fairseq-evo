import os
import argparse
import json
def parse_argument():
    parser = argparse.ArgumentParser(description='preprocess clue data')
    parser.add_argument('--task',type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    return args
args = parse_argument()

splits = ['train','dev','test']
os.makedirs(os.path.join(args.data_dir, args.output_dir),exist_ok=True)
if args.task == 'afqmc' or args.task == 'cmnli':
    for split in splits:
        if split != 'test':
            with open(os.path.join(args.data_dir, split+'.json'), 'r') as f:
                ds_raw = f.read()
                regular_lines = ds_raw.strip().split('\n')
            with open(os.path.join(args.data_dir, args.output_dir,f'{split}.raw.input0'), 'w') as f0,  open(os.path.join(args.data_dir, args.output_dir,f'{split}.raw.input1'), 'w') as f1, \
                open(os.path.join(args.data_dir, args.output_dir, f'{split}.label'), 'w') as fl:
                for line in regular_lines:
                    line = json.loads(line)
                    sentence1 = line['sentence1']
                    sentence2 = line['sentence2']
                    label = line['label']
                    f0.write(sentence1+ '\n')
                    f1.write(sentence2 + '\n')
                    fl.write(label + '\n')
        else:
            with open(os.path.join(args.data_dir, split+'.json'), 'r') as f:
                ds_raw = f.read()
                regular_lines = ds_raw.strip().split('\n')
            with open(os.path.join(args.data_dir, args.output_dir,f'{split}.raw.input0'), 'w') as f0,  open(os.path.join(args.data_dir, args.output_dir,f'{split}.raw.input1'), 'w') as f1:
                for line in regular_lines:
                    line = json.loads(line)
                    sentence1 = line['sentence1']
                    sentence2 = line['sentence2']
                    f0.write(sentence1+ '\n')
                    f1.write(sentence2 + '\n')
elif args.task == 'csl':
    for split in splits:
        if split != 'test':
            with open(os.path.join(args.data_dir, split+'.json'), 'r') as f:
                ds_raw = f.read()
                regular_lines = ds_raw.strip().split('\n')
            with open(os.path.join(args.data_dir, args.output_dir,f'{split}.raw.input0'), 'w') as f0,  open(os.path.join(args.data_dir, args.output_dir,f'{split}.raw.input1'), 'w') as f1, \
                open(os.path.join(args.data_dir, args.output_dir, f'{split}.label'), 'w') as fl:
                for line in regular_lines:
                    line = json.loads(line)
                    sentence1 = line['abst']
                    sentence2 = ','.join(line['keyword'])
                    label = line['label']
                    f0.write(sentence1+ '\n')
                    f1.write(sentence2 + '\n')
                    fl.write(label + '\n')
        else:
            with open(os.path.join(args.data_dir, split+'.json'), 'r') as f:
                ds_raw = f.read()
                regular_lines = ds_raw.strip().split('\n')
            with open(os.path.join(args.data_dir, args.output_dir,f'{split}.raw.input0'), 'w') as f0,  open(os.path.join(args.data_dir, args.output_dir,f'{split}.raw.input1'), 'w') as f1:
                for line in regular_lines:
                    line = json.loads(line)
                    sentence1 = line['abst']
                    sentence2 = ','.join(line['keyword'])
                    f0.write(sentence1+ '\n')
                    f1.write(sentence2 + '\n')
elif args.task == 'wsc':
    for split in splits:
        if split != 'test':
            with open(os.path.join(args.data_dir, split+'.json'), 'r') as f:
                ds_raw = f.read()
                regular_lines = ds_raw.strip().split('\n')
            with open(os.path.join(args.data_dir, args.output_dir,f'{split}.raw.input0'), 'w') as f0,  open(os.path.join(args.data_dir, args.output_dir,f'{split}.raw.input1'), 'w') as f1, \
                open(os.path.join(args.data_dir, args.output_dir, f'{split}.label'), 'w') as fl:
                for line in regular_lines:
                    line = json.loads(line)
                    sentence1 = line['text']
                    sentence2 = ','.join([line['target']['span1_text'],line['target']['span2_text']])
                    label = line['label']
                    f0.write(sentence1+ '\n')
                    f1.write(sentence2 + '\n')
                    fl.write(label + '\n')
        else:
            with open(os.path.join(args.data_dir, split+'.json'), 'r') as f:
                ds_raw = f.read()
                regular_lines = ds_raw.strip().split('\n')
            with open(os.path.join(args.data_dir, args.output_dir,f'{split}.raw.input0'), 'w') as f0,  open(os.path.join(args.data_dir, args.output_dir,f'{split}.raw.input1'), 'w') as f1:
                for line in regular_lines:
                    line = json.loads(line)
                    sentence1 = line['text']
                    sentence2 = ','.join([line['target']['span1_text'],line['target']['span2_text']])
                    f0.write(sentence1+ '\n')
                    f1.write(sentence2 + '\n')
else:
    for split in splits:
        if split != 'test':
            with open(os.path.join(args.data_dir, split+'.json'), 'r') as f:
                ds_raw = f.read()
                regular_lines = ds_raw.strip().split('\n')
            with open(os.path.join(args.data_dir, args.output_dir,f'{split}.raw.input0'), 'w') as f0,  \
                open(os.path.join(args.data_dir, args.output_dir, f'{split}.label'), 'w') as fl:
                for line in regular_lines:
                    line = json.loads(line)
                    sentence1 = line['sentence']
                    label = line['label']
                    f0.write(sentence1+ '\n')
                    fl.write(label + '\n')
        else:
            with open(os.path.join(args.data_dir, split+'.json'), 'r') as f:
                ds_raw = f.read()
                regular_lines = ds_raw.strip().split('\n')
            with open(os.path.join(args.data_dir, args.output_dir,f'{split}.raw.input0'), 'w') as f0:
                for line in regular_lines:
                    line = json.loads(line)
                    sentence1 = line['sentence']
                    f0.write(sentence1+ '\n')


