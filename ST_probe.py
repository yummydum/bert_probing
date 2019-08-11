"""
For Hydrogen;
%load_ext autoreload
%autoreload 2

Probe BERT hidden representations for POS tagging and dependency parsing (Linguistic correlation analysis)
"""

import argparse
import logging
import json
from pathlib import Path
import numpy as np
from joblib import dump, load
import torch
from pytorch_transformers import BertConfig,BertModel, BertTokenizer
from sklearn.linear_model import SGDClassifier

# fix seed
np.random.seed(0)

# Load model
model_type = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_type)
config = BertConfig.from_pretrained(model_type)
config.output_hidden_states = True
config.output_attentions = True
model = BertModel(config)
model.eval()

def train(args):
    # Read the datas
    train_path = Path(f"probing_data/ST/ST-train.json")
    with train_path.open(mode="r") as f:
        train_data = json.load(f)

    # Get class num
    if args.target == "position" or args.target == "head":
        class_num = range(train_data["max_sentence_len"])
    elif args.target == "pos":
        class_num = range(len(train_data["all_pos"]))

    # Init probing model for the ith layer
    probing_models = dict()
    for i in range(13):  # Num layer == 13
        probing_models[i] = SGDClassifier(penalty="elasticnet")

    # Train loop
    for epoch in range(args.epoch_num):
        for example_num,example in train_data["data"].items(): # example = train_data["data"]['1']
            example_num = int(example_num)
            # Report progress
            if example_num % 1000 == 0:
                logging.info(f"Now at {example_num}th example")
            # Forward
            sentence = "[CLS] " + " ".join(example["word"]) + " [SEP]"
            tokenized = tokenizer.tokenize(sentence)
            encoded = tokenizer.encode(sentence)
            tensor = torch.LongTensor(encoded).reshape(1,len(encoded))
            last_hid,pooler,all_hid,all_attention = model(tensor)
            all_hid_list = [all_hid[i][0].detach().numpy() for i in range(len(all_hid))]

            # Fit model for ith model
            for i in range(len(all_hid_list)): # i = 1
                X,y = extract_X_y(args,
                                  tokenized,
                                  sentence.split(" "),
                                  example,
                                  all_hid_list[i])
                # preprocess y
                if args.target == "pos":
                    y = np.array([train_data["all_pos"].index(l[0]) for l in y])
                else:
                    y = y.squeeze()
                probing_models[i].partial_fit(X,y,classes=class_num)
            # Stop here if debug mode
            if args.debug and example_num > 3:
                break
        # Evaluate accuracy on dev data
        acc = evaluate(args,probing_models,train_data["all_pos"],is_dev=True)
        logging.info(f"Current accuracy is {acc}")
        # Save the model for this epoch
        for i in range(13):  # Num layer == 13
            save_path = f"probing_data/ST/layer{i}_epoch{epoch}_target{args.target}.joblib"
            dump(probing_models[i],save_path)

# all_pos = train_data["all_pos"]
def evaluate(args,probing_models,all_pos,is_dev):
    logging.info("Now starting evaluation...")
    correct = 0
    total = 0
    # Read the datas
    if is_dev:
        data_path = Path(f"probing_data/ST/ST-dev.json")
    else:
        data_path = Path(f"probing_data/ST/ST-test.json")
    with data_path.open(mode="r") as f:
        data = json.load(f)

    for example_num,example in data["data"].items(): # example = train_data["data"]['1']
        example_num = int(example_num)
        # Report progress
        if example_num % 1000 == 0:
            logging.info(f"Now at {example_num}th example")
        # Forward
        sentence = "[CLS] " + " ".join(example["word"]) + " [SEP]"
        tokenized = tokenizer.tokenize(sentence)
        encoded = tokenizer.encode(sentence)
        tensor = torch.LongTensor(encoded).reshape(1,len(encoded))
        last_hid,pooler,all_hid,all_attention = model(tensor)
        all_hid_list = [all_hid[i][0].detach().numpy() for i in range(len(all_hid))]

        # Fit model for ith model
        for i in range(len(all_hid_list)): # i = 1
            X,y = extract_X_y(args,
                              tokenized,
                              sentence.split(" "),
                              example,
                              all_hid_list[i])
            y_hat = probing_models[i].predict(X)
            # postprocess y
            if args.target == "pos":
                y_hat =[all_pos[l] for l in y_hat]
            correct += np.sum(y_hat == y.T)
            total += len(y_hat)
        # Stop here if debug mode
        if args.debug and example_num > 3:
            break

    return correct / total

# original_tokenized = sentence.split(" "); hidden_i = all_hid_list[i]
def extract_X_y(args,tokenized,original_tokenized,example,hidden_i):
    """
    Resolve the inconsistency between simple tokenization and BPE
    """
    # Start loop
    target = example[args.target]
    skip_step = 0
    cum_skip_step = 0
    X,y = [],[]
    for j in range(len(tokenized)):  # j = 1
        # If there was a concatenation in the prev loop, skip
        if skip_step > 0:
            skip_step -= 1
            cum_skip_step += 1
            continue
        # Special token
        if tokenized[j] in {"[CLS]","[SEP]"}:
            continue
        else:
            # -cum_skip_step for number o0f skips
            # logging.debug(f"BPE tokenized:{tokenized[j]}")
            # logging.debug(f"Original token:{original_tokenized[j-cum_skip_step]}")

            # y
            t = target[j-cum_skip_step-1]
            y.append([t])

            # If the BPE and original token is consistent, write
            if tokenized[j] == original_tokenized[j-cum_skip_step]:
                X.append(hidden_i[j])
            # Else restore the original token
            else:
                # logging.debug("Inconsistency found")
                temp_token_list = [tokenized[j]]
                temp_hid_rep_list = hidden_i[j]
                for k in range(j+1,len(tokenized)):  # k = 0
                    temp_token_list.append(tokenized[k])
                    temp_hid_rep_list += hidden_i[k]
                    skip_step += 1
                    # If the concatenated BPE matches the original token, write the result
                    temp = "".join(temp_token_list).replace("##","")
                    # logging.debug(f"concatenated:{temp}")
                    if original_tokenized[j-cum_skip_step] == temp:
                        # logging.debug(f"Match! {temp}")
                        X.append(temp_hid_rep_list / len(temp_token_list))
                        break

    return np.array(X),np.array(y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",action='store_true',
                        help="Debug mode if flagged")  # pos class num = 46
    parser.add_argument("--target", default="pos",
                        type=str,
                        choices = ["position","pos","head"],
                        help="The target variable to be predicted")  # pos class num = 46
    parser.add_argument("--epoch_num",type=int,default=10,
                        help="Number of epoch")  # pos class num = 46
    args = parser.parse_args()

    # Set logger
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level,
                        format='%(process)d-%(asctime)s-%(levelname)s-%(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    train(args)

    # args = parser.parse_args(["--debug"])
