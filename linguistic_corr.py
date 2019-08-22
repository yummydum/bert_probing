"""
For Hydrogen;
%load_ext autoreload
%autoreload 2

Probe BERT hidden representations for POS tagging and dependency parsing (Linguistic correlation analysis)
"""

import argparse
import logging
from logging import getLogger, StreamHandler, FileHandler
import json
from pathlib import Path
import numpy as np
import seaborn as sns
from joblib import dump, load
import torch
from pytorch_transformers import BertConfig, BertModel, BertTokenizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split

# fix seed
np.random.seed(0)
sns.set()

model_type = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_type)
config = BertConfig.from_pretrained(model_type)
config.output_hidden_states = True
config.output_attentions = True
model = BertModel(config)
model.eval()


def calc_hid_rep():

    output_data_path = Path(f"probing_data/BERT/ST_neuron.npy")
    if output_data_path.exists():
        return

    data_path = Path(f"probing_data/ST/ST-dev.json")
    # Activate the neuron
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for i, data_i in data["data"].items():  # data_i = data["data"]["1"]
        original_sentence = " ".join(data_i["word"])
        original_tokenized = original_sentence.split(" ")
        sentence = "[CLS] " + original_sentence + " [SEP]"
        tokenized = tokenizer.tokenize(sentence)
        encoded = tokenizer.encode(sentence)
        tens = torch.LongTensor(encoded).view(1, -1)
        last_hid, pooler, all_hid, all_attention = model(tens)

        # all_hid[0].shape
        # Restore the original representations
        # len(original_sentence.split(" "))  len(tokenized)
        restored_hid = []
        skip_step = 0
        cum_skip_step = 0
        for j in range(len(tokenized)):  # j = 0
            # If there was a concatenation in the prev loop, skip
            if skip_step > 0:
                skip_step -= 1
                cum_skip_step += 1
                continue
            # Special token
            if tokenized[j] in {"[CLS]", "[SEP]"}:
                continue
            else:
                # If the BPE and original token is consistent, write
                if tokenized[j] == original_tokenized[j - cum_skip_step - 1]:
                    restored_hid.append(
                        np.concatenate([
                            x.squeeze()[j].detach().numpy() for x in all_hid
                        ]))

                # Else restore the original token
                else:
                    # logging.debug("Inconsistency found")
                    temp_token_list = [tokenized[j]]
                    temp_hid_rep_list = np.concatenate(
                        [x.squeeze()[j].detach().numpy() for x in all_hid])
                    for k in range(j + 1, len(tokenized)):  # k = 0
                        temp_token_list.append(tokenized[k])
                        temp_hid_rep_list += np.concatenate(
                            [x.squeeze()[k].detach().numpy() for x in all_hid])
                        skip_step += 1
                        # If the concatenated BPE matches the original token, write the result
                        temp = "".join(temp_token_list).replace("##", "")
                        if original_tokenized[j - cum_skip_step - 1] == temp:
                            # print(f"Match:{temp}")
                            restored_hid.append(temp_hid_rep_list /
                                                len(temp_token_list))
                            break

        # Result for this sentence
        all_hid_array = np.array(restored_hid)
        # all_hid_array.shape
        # Concatenate the result for this sentence to the
        if int(i) == 0:
            result_array = all_hid_array
        else:
            result_array = np.concatenate((result_array, all_hid_array),
                                          axis=0)
        if int(i) % 100 == 0:
            print(f"Now at {i}th example")

    # Save array
    np.save(output_data_path, result_array)
    return


def calc_pos_corr():
    data_path = Path(f"probing_data/ST/ST-dev.json")
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cum_count = 0
    pos2ind = {k: v for v, k in enumerate(data["all_pos"])}
    result = np.zeros((len(data["all_pos"]), 40115))
    for i, data_i in data["data"].items():  # data_i = data["data"]["1"]
        pos_i = data_i["pos"]
        for j in range(cum_count, cum_count + len(pos_i)):  # j=0
            pos_ij = pos_i[j - cum_count]
            result[pos2ind[pos_ij]][j] = 1
        cum_count += len(pos_i)

    st_neuron_path = Path(f"probing_data/BERT/ST_neuron.npy")
    activation = np.load(st_neuron_path)

    pos_corr_path = Path(f"probing_data/BERT/pos_corr.npy")
    pos_corr = calc_corr(result.T, activation)
    np.save(pos_corr_path, pos_corr)


def logistic_reg():
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

    st_neuron_path = Path(f"probing_data/BERT/ST_neuron.npy")
    X = np.load(st_neuron_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # activation.shape,result.shape
    best_acc = 0
    best_C = 0
    for C in np.linspace(0, 2, 30):  # C = 1
        try:
            model = LogisticRegression(penalty="elasticnet",
                                       solver='saga',
                                       n_jobs=20,
                                       l1_ratio=0.3,
                                       max_iter=200,
                                       C=C)
            model.fit(X_train, y_train)

            # Save model
            logger.info(f"Now fitting model for {C}...")
            model_path = Path(f"probing_data/BERT/ST_probe_C_{C}.joblib")
            dump(model, model_path)

            # Show prediction
            acc = np.sum(y_test == model.predict(X_test)) / len(y_test)
            logger.info(f"Model accuracy is {acc}")
            if acc > best_acc:
                best_acc = acc
                best_C = C
        except:
            logger.info(f"Error:C={C}")

    logger.info(f"The best param is {best_C} with acc {best_acc}")


def EDA():

    # Load pos ind
    data_path = Path(f"probing_data/ST/ST-dev.json")
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    pos2ind = {k: v for v, k in enumerate(data["all_pos"])}

    # Load corr
    pos_corr_path = Path(f"probing_data/BERT/pos_corr.npy")
    pos_corr = np.load(pos_corr_path)

    sns.distplot(pos_corr[:, pos2ind["VB"]])
    # sns.distplot(pos_corr[:,pos2ind["NN"]])


def highlight_neuron(neuron_num):  # neuron_num = 1

    # Append to string to color the output
    END = '\033[0m'
    RED = '\033[41m'
    YEL = '\u001b[43m'
    BLUE = '\033[44m'
    CIAN = '\033[46m'

    # # Reindex to the original index
    # layer_num,ind = divmod(neuron_num,768)

    # Get the hid rep
    np_path = Path(f"probing_data/BERT/ST_neuron.npy")
    sst_rep = np.load(np_path)  # sst_rep.shape
    rep = sst_rep[:, neuron_num]
    # Plot the distribution of this neuron
    sns.distplot(rep)

    # Get the top 5% activate neuron
    mean = np.mean(rep)
    std = np.std(rep)

    # Init acc
    result = []
    pos_dict = dict()
    neg_dict = dict()

    data_path = Path(f"probing_data/ST/ST-dev.json")
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cum_count = 0
    pos2ind = {k: v for v, k in enumerate(data["all_pos"])}
    for i, data_i in data["data"].items():  # data_i = data["data"]["1"]
        sentence = " ".join(data_i["word"])
        sentence = "[CLS] " + sentence + " [SEP]"
        tokenized = tokenizer.tokenize(sentence)
        activation = rep[cum_count:cum_count + len(tokenized)]
        cum_count += len(tokenized)

        # Mark if the activation deviates
        is_pos = activation > mean + 1.96 * std
        is_neg = activation < mean - 1.96 * std
        if is_pos.any() or is_neg.any():
            result_i = []
            for p, n, tok in zip(is_pos, is_neg, tokenized):
                if p:
                    result_i.append(RED + tok + END)
                    if tok not in pos_dict:
                        pos_dict[tok] = 0
                    pos_dict[tok] += 1
                elif n:
                    result_i.append(BLUE + tok + END)
                    if tok not in neg_dict:
                        neg_dict[tok] = 0
                    neg_dict[tok] += 1
                else:
                    result_i.append(tok)
            result.append(" ".join(result_i))

        if i == 1000:
            break

    pos_tok = sorted(pos_dict.items(), key=lambda x: x[1], reverse=True)
    neg_tok = sorted(neg_dict.items(), key=lambda x: x[1], reverse=True)

    return result, pos_tok, neg_tok


def calc_corr(A, B):
    """
    Pairwise correlation for matrix with shape of (example_num,feature_num)
    """
    N = B.shape[0]
    sA = A.sum(0)
    sB = B.sum(0)
    p1 = N * np.dot(B.T, A)
    p2 = sA * sB[:, None]
    p3 = N * ((B**2).sum(0)) - (sB**2)
    p4 = N * ((A**2).sum(0)) - (sA**2)
    return ((p1 - p2) / np.sqrt(p4 * p3[:, None]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",
                        action='store_true',
                        help="Debug mode if flagged")  # pos class num = 46
    parser.add_argument(
        "--target",
        default="pos",
        type=str,
        choices=["position", "pos", "head"],
        help="The target variable to be predicted")  # pos class num = 46
    parser.add_argument("--epoch_num",
                        type=int,
                        default=10,
                        help="Number of epoch")  # pos class num = 46
    args = parser.parse_args()
    # args = parser.parse_args(["--debug"])

    # Set logger
    if args.debug:
        level = logging.DEBUG
        args.epoch_num = 3
    else:
        level = logging.INFO

    # Init logger
    file_name = Path().cwd().name
    logger = getLogger(file_name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        '%(process)d-%(asctime)s-%(levelname)s-%(message)s',
        datefmt='%d-%b-%y %H:%M:%S')

    # Logging to stdout
    s_handler = StreamHandler()
    s_handler.setLevel(level)
    s_handler.setFormatter(formatter)

    # Logging to file
    log_file_path = Path(f"log/{file_name}.log")
    f_handler = FileHandler(log_file_path)
    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(formatter)

    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.addHandler(s_handler)
    logger.addHandler(f_handler)

    # train(args)
    calc_hid_rep()
    logistic_reg()

    # args = parser.parse_args(["--debug"])
    # logger.info("Test")
