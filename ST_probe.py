"""
For Hydrogen;
%load_ext autoreload
%autoreload 2

Probe BERT hidden representations for POS tagging and dependency parsing (Linguistic correlation analysis)
"""
import logging
import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import torch
from pytorch_transformers import BertConfig,BertModel, BertTokenizer

logging.basicConfig(level=logging.DEBUG,
                    format='%(process)d-%(asctime)s-%(levelname)s-%(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

def clean_sentence(sentence):
    # TODO Delete this part
    sentence = sentence.replace("NUM","0")
    sentence = sentence.replace('*root* ','')
    return sentence.replace("<quote>",'"')

def make_BERT_feature():

    # Load model
    model_type = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_type)
    config = BertConfig.from_pretrained(model_type)
    config.output_hidden_states = True
    config.output_attentions = True
    model = BertModel(config)

    # Convert the sentence in ST to BERT hidden rep
    st_path = Path("probing_data/ST/train.tsv")
    hid_rep_files = dict()
    for i in range(13):  # Num layer == 13
        i_path = Path(f"probing_data/ST/bert_reps_{i}.csv")
        if i_path.exists():
            logging.debug("The file already exists. Exit")
            return
        f = i_path.open(mode="w")
        hid_rep_files[i] = csv.writer(f)
        hid_rep_files[i].writerow(["Token","POS","Head"])

    with st_path.open(mode="r") as f:
        next(f)
        for row in csv.reader(f,delimiter="\t"):  # sentence = "[CLS] I love you [SEP]"
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
            all_hid_list = [all_hid[i][0].tolist() for i in range(len(all_hid))]
            for i in range(len(all_hid_list)):
                skip_step = 0
                cum_skip_step = 0
                for j in range(len(tokenized)):
                    # If there was a concatenation in the prev loop, skip
                    if skip_step > 0:
                        skip_step -= 1
                        cum_skip_step += 1
                        continue
                    # Special token
                    if tokenized[j] in {"[CLS]","[SEP]"}:
                        write_this = [tokenized[j],None,None] + all_hid_list[i][j]
                    else:
                        logging.debug("BPE tokenized:",tokenized[j])
                        logging.debug("Original token",original_token_list[j-cum_skip_step-1])
                        # If the BPE and original token is consistent, write
                        if tokenized[j] == original_token_list[j-cum_skip_step-1]:
                            pos = row[1].split(" ")[j-cum_skip_step]
                            head = row[2].split(" ")[j-cum_skip_step]
                            write_this = [tokenized[j],pos,head] + all_hid_list[i][j]
                        # Else restore the original token
                        else:
                            logging.debug("Inconsistency found")
                            temp_token_list = [tokenized[j]]
                            temp_hid_rep_list = [all_hid_list[i][j]]
                            for k in range(j+1,len(tokenized)):
                                temp_token_list.append(tokenized[k])
                                temp_hid_rep_list.append(all_hid_list[i][k])
                                skip_step += 1
                                # If the concatenated BPE matches the original token, write the result
                                temp = "".join(temp_token_list).replace("##","")
                                logging.debug("concatenated:",temp)
                                if original_token_list[j-cum_skip_step-1] == temp:
                                    logging.debug(f"Match! {temp}")
                                    # Calc mean of the hid rep
                                    mean_hid_rep = np.array(temp_hid_rep_list[0])
                                    for l in range(1,len(temp_hid_rep_list)):
                                        mean_hid_rep += temp_hid_rep_list[l]
                                    mean_hid_rep = mean_hid_rep / len(temp_hid_rep_list)
                                    # Store the result for this row and break
                                    pos = row[1].split(" ")[j-cum_skip_step]
                                    head = row[2].split(" ")[j-cum_skip_step]
                                    write_this = ["".join(temp_token_list),row[1],row[2]] + mean_hid_rep.tolist()
                                    break
                    # Write the result for this loop
                    hid_rep_files[i].writerow(write_this)
                # Empty row to seperate sentences
                hid_rep_files[i].writerow(["\n"])

if __name__ == '__main__':
    # make_BERT_feature()
