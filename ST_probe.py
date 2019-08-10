"""
For Hydrogen;
%load_ext autoreload
%autoreload 2

Probe BERT hidden representations for POS tagging and dependency parsing (Linguistic correlation analysis)
"""
import argparse
import logging
import csv
import numpy as np
from joblib import dump, load
from sklearn.linear_model import SGDClassifier
from pathlib import Path
import torch
from pytorch_transformers import BertConfig,BertModel, BertTokenizer

logging.basicConfig(level=logging.INFO,
                    format='%(process)d-%(asctime)s-%(levelname)s-%(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

def clean_sentence(sentence):
    # TODO Delete this part
    sentence = sentence.replace("NUM","0")
    sentence = sentence.replace('*root* ','')
    return sentence.replace("<quote>",'"')

def train_probing_classifier(args):

    with torch.no_grad():
        # Load model
        model_type = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_type)
        config = BertConfig.from_pretrained(model_type)
        config.output_hidden_states = True
        config.output_attentions = True
        model = BertModel(config)
        model.eval()

        # Convert the sentence in ST to BERT hidden rep
        st_path = Path("probing_data/ST/train.tsv")
        probing_models = dict()
        for i in range(13):  # Num layer == 13
            probing_models[i] = SGDClassifier(penalty="elasticnet")

        # with st_path.open(mode="r") as f:
        #     next(f)
        #     row = next(f).split("\t")

        with st_path.open(mode="r") as f:
            next(f)
            for row_num,row in enumerate(csv.reader(f,delimiter="\t")):  # sentence = "[CLS] I love you [SEP]"
                # Reformat the data
                cleaned_sentence = clean_sentence(row[0])
                sentence = f"[CLS] {cleaned_sentence} [SEP]"
                original_token_list = cleaned_sentence.split(" ")

                # Forward
                tokenized = tokenizer.tokenize(sentence)
                encoded = tokenizer.encode(sentence)
                tensor = torch.LongTensor(encoded).reshape(1,len(encoded))
                last_hid,pooler,all_hid,all_attention = model(tensor)

                # Make feature for each layer each token
                all_hid_list = [all_hid[i][0].numpy() for i in range(len(all_hid))]
                for i in range(len(all_hid_list)): # i = 1
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
                            logging.debug(f"BPE tokenized:{tokenized[j]}")
                            logging.debug(f"Original token:{original_token_list[j-cum_skip_step-1]}")

                            # Determine the target variable
                            if args.target == "position":
                                target = j
                            elif args.target == "pos":
                                target = row[1].split(" ")[j-cum_skip_step]
                            elif args.target == "head":
                                target == row[2].split(" ")[j-cum_skip_step]
                            y.append(pos_tag.index(target))

                            # If the BPE and original token is consistent, write
                            if tokenized[j] == original_token_list[j-cum_skip_step-1]:
                                X.append(all_hid_list[i][j])
                            # Else restore the original token
                            else:
                                logging.debug("Inconsistency found")
                                temp_token_list = [tokenized[j]]
                                temp_hid_rep_list = all_hid_list[i][j]
                                for k in range(j+1,len(tokenized)):
                                    temp_token_list.append(tokenized[k])
                                    temp_hid_rep_list += all_hid_list[i][k]
                                    skip_step += 1
                                    # If the concatenated BPE matches the original token, write the result
                                    temp = "".join(temp_token_list).replace("##","")
                                    logging.debug("concatenated:",temp)
                                    if original_token_list[j-cum_skip_step-1] == temp:
                                        logging.debug(f"Match! {temp}")
                                        X.append(temp_hid_rep_list / len(temp_token_list))
                                        break

                    # Update param for this sentence
                    if i == 0:
                        logging.info(f"Now at {row_num}th sentence")
                    X = np.array(X)  # X.shape
                    y = np.array(y)  # y.shape
                    probing_models[i].partial_fit(X,y,classes=range(len(pos_tag)))
                    if args.test_mode and row_num > 10:
                        for i in range(13):  # Num layer == 13
                            dump(probing_models[i],f"probing_data/ST/{i}.joblib")
                        return

    # Finally save the results
    for i in range(13):  # Num layer == 13
        dump(probing_models[i],f"probing_data/ST/{i}.joblib")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mode",action='store_true',
                        help="Test mode if flagged")  # pos class num = 46
    parser.add_argument("--target", default="pos", type=str,
                        help="Word position,pos,head")  # pos class num = 46
    args = parser.parse_args()  # %tb
    # import sys
    # sys.argv[0] = ""
    # del sys


    pos_tag = [ 'NN',
                 'IN',
                 'NNP',
                 'DT',
                 'JJ',
                 'NNS',
                 ',',
                 '.',
                 'CD',
                 'RB',
                 'VBD',
                 'VB',
                 'CC',
                 'VBZ',
                 'VBN',
                 'PRP',
                 'VBG',
                 'TO',
                 'VBP',
                 'MD',
                 'POS',
                 'PRP$',
                 '$',
                 '``',
                 "''",
                 ':',
                 'WDT',
                 'JJR',
                 'NNPS',
                 'RP',
                 'WP',
                 'WRB',
                 'JJS',
                 'RBR',
                 ')',
                 '(',
                 'EX',
                 'RBS',
                 'PDT',
                 'FW',
                 'WP$',
                 '#',
                 'UH',
                 'SYM',
                 'LS']
    train_probing_classifier(args)
