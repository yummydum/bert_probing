import argparse
import csv
from pathlib import Path
from typing import List
import numpy as np
import torch
from pytorch_transformers import BertConfig, BertModel, BertTokenizer
from util import set_logger

# Globals
logger = set_logger(__name__)
model_type = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_type)
config = BertConfig.from_pretrained(model_type)
config.output_hidden_states = True
config.output_attentions = True
model = BertModel(config)
model.eval()


def pad(encoded_list):
    # Get the max len
    max_len = 0
    for i in encoded_list:
        if len(i) > max_len:
            max_len = len(i)

    # Return the padded result
    padded = []
    mask = []
    for i in encoded_list:
        pad_len = max_len - len(i)
        padded.append(i + [0] * pad_len)
        mask.append([1] * len(i) + [0] * pad_len)
    return padded, mask


def load_data(args):
    """ Data should be in tsv format. """
    tokenized_list = []
    wordpiece_list = []
    encoded_list = []
    data_path = Path(f"data/ST/train.tsv")  # !ls data/ST
    with data_path.open("r") as f:
        tsv_reader = csv.reader(f, delimiter="\t")
        col = next(tsv_reader)
        colname2ind = {k: v for v, k in enumerate(col)}
        for i, data_i in enumerate(tsv_reader):  # data_i = data["data"]["1"]
            original_sentence = data_i[colname2ind["word"]]
            sentence = "[CLS] " + original_sentence + " [SEP]"
            tokenized_list.append(original_sentence.split(" "))
            wordpiece_list.append(tokenizer.tokenize(sentence))
            encoded_list.append(tokenizer.encode(sentence))
            if i % 3000 == 0:
                logger.info(f"Now loading data: at {i}th row")
            if args.debug and i == 4:
                logger.debug("Break loop (debug mode)")
                break
    return tokenized_list, wordpiece_list, encoded_list


def align_rep_by_tokenization(tokenized: List, word_piece: List, all_hid):
    """
    tokenized: list of tokenized sentence
    word_piece: list of word_piece seperated sentence
    all_hid: Tuple (len=13) holding the hidden representation tendor of each layer
    """

    for i in range(len(tokenized)):  # i = 0
        restored_hid = []
        skips = 0
        cum_skips = 0
        for j in range(len(word_piece[i])):  # j = 0  len(tokenized[i])

            # If there was a concatenation in the prev loop, skip
            if skips > 0:
                skips -= 1
                cum_skips += 1
                continue

            # Special token
            if word_piece[i][j] in {"[CLS]", "[SEP]"}:
                continue
            else:
                # j - cum_skips - 1: the index of the head of tokenized list
                # If the BPE and original token is consistent, write
                if word_piece[i][j] == tokenized[i][j - cum_skips - 1]:
                    temp = np.concatenate([
                        x.squeeze().numpy()[i][j] for x in all_hid
                    ])  # x=all_hid[0];x.squeeze().numpy().shape
                    restored_hid.append(temp)

                # Else restore the original token
                else:
                    # logging.debug("Inconsistency found")
                    temp_token_list = [word_piece[i][j]]
                    temp_hid_rep_list = np.concatenate(
                        [x.squeeze()[i][j].numpy() for x in all_hid])
                    for k in range(j + 1, len(word_piece[i])):  # k = 0
                        temp_token_list.append(word_piece[i][k])
                        temp_hid_rep_list += np.concatenate(
                            [x.squeeze()[i][k].numpy() for x in all_hid])
                        skips += 1
                        # If the concatenated BPE matches the original token, write the result
                        temp = "".join(temp_token_list).replace("##", "")
                        if tokenized[i][j - cum_skips - 1] == temp:
                            # print(f"Match:{temp}")
                            restored_hid.append(temp_hid_rep_list /
                                                len(temp_token_list))
                            break
        # Result for this sentence
        all_hid_array = np.array(restored_hid)  # all_hid_array.shape =
        if i == 0:
            aligned_rep = all_hid_array
        else:
            aligned_rep = np.concatenate((aligned_rep, all_hid_array), axis=0)
        if i % 1000 == 0:
            logger.info(f"Now aligning the all_hid: {i}th example")

    return aligned_rep  # aligned_rep.shape = ()


def main(args):
    # Path("data/BERT").mkdir()
    output_data_path = Path(f"data/BERT/ST_neuron.npy")
    if output_data_path.exists():
        logger.info("The result already exists. Exit")
        return

    # Load the data in tokenized form and BERT encoded list
    tokenized, word_piece, encoded_list = load_data(args)
    padded_encoded_list, mask = pad(encoded_list)

    # Forward
    logger.info("Now forwarding the data...")
    with torch.no_grad():
        result_tensor = torch.LongTensor(padded_encoded_list)
        mask = torch.LongTensor(mask)
        logger.info(f"The shape of tensor is {result_tensor.shape}")
        last_hid, pooler, all_hid, all_attention = model(result_tensor,
                                                         attention_mask=mask)

    # Save the result
    logger.info("Start aligning the hid rep")
    aligned_rep = align_rep_by_tokenization(tokenized, word_piece, all_hid)
    np.save(output_data_path, aligned_rep)
    return


def make_pos_y():
    pos_path = Path(f"probing_data/BERT/ST_y.npy")
    if pos_path.exists():
        return

    data_path = Path(f"probing_data/ST/ST-dev.json")
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cum_count = 0
    pos2ind = {k: v for v, k in enumerate(data["all_pos"])}
    y = np.zeros(40115)
    for i, data_i in data["data"].items():  # data_i = data["data"]["1"]
        pos_i = data_i["pos"]
        for j in range(cum_count, cum_count + len(pos_i)):  # j=0
            y[j] = pos2ind[pos_i[j - cum_count]]
        cum_count += len(pos_i)

    # Save the pos cat
    np.save(pos_path, y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",
                        action='store_true',
                        help="Debug mode if flagged")  # pos class num = 46
    args = parser.parse_args()
    # args = parser.parse_args(["--debug"])
    main(args)
