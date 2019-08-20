"""
For Hydrogen;
%load_ext autoreload
%autoreload 2

Probe BERT hidden representations for SST (Linguistic correlation analysis)
"""
import argparse
import csv
from pathlib import Path
import logging
from logging import getLogger, StreamHandler, FileHandler
from pympler import asizeof
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import torch
from pytorch_transformers import BertForSequenceClassification, BertTokenizer, BertConfig

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained('bert-base-uncased')
config.output_hidden_states = True
config.output_attentions = True

cache = {}
sns.set()


# model_name = "BERT"
# task_name = "CoLA"
def calc_hid_rep(args, model_name: str, task_name: str):

    # Check if the result already exists
    output_data_path = Path(
        f"probing_data/{model_name}/{task_name}_neuron.npy")
    if output_data_path.exists() and not args.override:
        logger.info("Hid rep already exists.")
        return

    # Load model
    model = load_model(model_name, task_name)

    # Activate the neuron (use SST-2 for now)
    data_path = Path(f"glue_data/{args.evaluation_data}/test.tsv")
    with data_path.open("r", encoding="utf-8") as f:
        csv_reader = csv.reader(f, delimiter="\t")
        next(csv_reader)
        for i, row in enumerate(csv_reader):
            sentence = "[CLS] " + row[1] + " [SEP]"
            tokenized = tokenizer.tokenize(sentence)
            encoded = tokenizer.encode(sentence)
            tens = torch.LongTensor(encoded).view(1, -1)
            last_hid, all_hid, all_attention = model(tens)

            # Append the neurons to 1D vec
            array_list = [hid_j[0].detach().numpy() for hid_j in all_hid]
            all_hid_array = np.concatenate(array_list, axis=1)
            if i == 0:
                result_array = all_hid_array
            else:
                result_array = np.concatenate((result_array, all_hid_array),
                                              axis=0)
            # Stop at the specified length
            if args.debug and i == 4:
                break
            if i % 100 == 0:
                logger.info(f"Now at {i}th example")

    # shape = (sum(example_num)=num_words,layer_num*hidden_dim)
    # debug time: (113,9984)
    # Report consumed memory (sample=5 -> 4400MB)
    consumed_bytes = asizeof.asizeof(result_array)
    logger.info(f"Consumed bytes: {consumed_bytes /1024 / 1024 }MB")
    # Save array
    np.save(output_data_path, result_array)
    return


def load_model(model_name: str, task_name: str):
    if model_name not in cache:
        cache[model_name] = dict()
    if task_name not in cache[model_name]:
        model_path = str(Path(f"models/{model_name}/{task_name}/"))
        model = BertForSequenceClassification.from_pretrained(model_path,
                                                              config=config)
        cache[model_name][task_name] = model
    return cache[model_name][task_name]


# A = cola_rep.T
# B = sst_rep.T
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


# neuron_num = 3985
def highlight_neuron(neuron_num):

    # Append to string to color the output
    END = '\033[0m'
    RED = '\033[41m'
    YEL = '\u001b[43m'
    BLUE = '\033[44m'
    CIAN = '\033[46m'

    # # Reindex to the original index
    # layer_num,ind = divmod(neuron_num,768)

    # Get the hid rep
    sst_path = Path(f"probing_data/{model_name}/{task_name2}_neuron.npy")
    sst_rep = np.load(sst_path)
    rep = sst_rep[:, neuron_num]
    # Plot the distribution of this neuron
    sns.distplot(rep)

    # Get the top 5% activate neuron
    mean = np.mean(rep)
    std = np.std(rep)

    # Show the text where this activation occured
    data_path = Path(f"glue_data/{args.evaluation_data}/test.tsv")
    with data_path.open("r") as f:
        csv_reader = csv.reader(f, delimiter="\t")
        next(csv_reader)
        for row in csv_reader:
            sentence = "[CLS] " + row[1] + " [SEP]"
            tokenized = tokenizer.tokenize(sentence)
            encoded = tokenizer.encode(sentence)
            tens = torch.LongTensor(encoded).view(1, -1)
            last_hid, all_hid, all_attention = model(tens)

            # get the activation of
            layer_num, ind = divmod(neuron_num, all_hid[0].shape[2])
            activation = all_hid[0][0][layer_num][ind].item()
            if abs(activation) > std * 2:
                return


def EDA():

    model_name = "BERT"
    task_name1 = "SST-2"
    task_name2 = "CoLA"

    # Load the hid rep
    cola_path = Path(f"probing_data/{model_name}/{task_name1}_neuron.npy")
    cola_rep = np.load(cola_path)
    sst_path = Path(f"probing_data/{model_name}/{task_name2}_neuron.npy")
    sst_rep = np.load(sst_path)

    # Calc the pairwise corr
    corr_path = Path("probing_data/EDA/corr.npy")
    if not corr_path.exists():
        corr_matrix = calc_corr(cola_rep, sst_rep)  # shape==(9984,9984)
        np.save(corr_path, corr_matrix)
        logger.info("Calculated correlation")
    else:
        logger.info("Existing correlation matrix exists: use this")
        corr_matrix = np.load(corr_path)
    sorted_corr = np.sort(corr_matrix, axis=1)

    # There are neurons which moves similarily with each neuron
    # However layer in the lower layer has this tendency strongly
    n = 3
    mean_topn = np.mean(sorted_corr[:, -n:], axis=1)
    fig, ax = plt.subplots()
    ax.plot(range(len(mean_topn)), mean_topn)

    # Maybe neuron with low corr are task specific...
    low_corr_neuron_ind = np.argsort(mean_topn)[:100]


def EDA2():
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",
                        action='store_true',
                        help="Debug mode if flagged")  # pos class num = 46
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--evaluation_data",
                        default="SST-2",
                        help="The data set to activate the neuron")
    args = parser.parse_args()
    # args = parser.parse_args(["--debug"])

    # Init logger
    logger = getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(process)d-%(asctime)s-%(levelname)s-%(message)s',
        datefmt='%d-%b-%y %H:%M:%S')
    # Logging to stdout
    s_handler = StreamHandler()
    s_handler.setLevel(logging.INFO)
    s_handler.setFormatter(formatter)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.addHandler(s_handler)

    # model_list = [
    #     "CoLA", "SST-2", "STS-B", "MNLI", "MRPC", "QNLI", "QQP", "RTE", "WNLI"
    # ]

    # calc_hid_rep(args, "BERT", "CoLA")
    # calc_hid_rep(args, "BERT", "SST-2")

    EDA()
