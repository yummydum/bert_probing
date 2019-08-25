# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: minoconda
#     language: python
#     name: miniconda3-4.3.30
# ---

# + {"hide_input": false}
# %load_ext autoreload
# %autoreload 2
# -

import cross_model_corr
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:150% !important; }</style>"))

# # BERTをSST-2とCoLAでfine tuningしたものの間でニューロンの相関係数を計算

# +
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
else:
    corr_matrix = np.load(corr_path)
sorted_corr = np.sort(corr_matrix, axis=1)
# -

# # 各ニューロンの相関係数の最大値をプロット
# * 下層ほど相関係数が高い=fine tuningで大きく動きが変わっていない=普遍的な情報を表すニューロン?
# * 上層ほど相関係数が低い=fine tuningで動きが変わっている=タスクspecificなニューロン?

max_corr = np.max(sorted_corr, axis=1)
sorted_ind = np.argsort(max_corr)
fig, ax = plt.subplots()
ax.plot(range(len(max_corr)), max_corr)

# # 各ニューロンの分布を可視化&発火している場所を可視化
# * 分布が単峰形でないことが多々ある

# Highlight the neuron
output_dir = Path("results/neuron_highlight/EDA")
np_path = Path(f"probing_data/{model_name}/{task_name2}_neuron.npy")

# ## 適当なニューロン(0番目)

# 1 (sentiment neuron?)
result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(0, sst_path)

for sentence in result:
    print(sentence)

result[0]

pos_tok

neg_tok

# # 相関が高いニューロン

# ## CLSとSEPにほぼ離散的に反応しているニューロン

result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(sorted_ind[-1], sst_path)

for sentence in result:
    print(sentence)

result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(sorted_ind[-2], sst_path)

for sentence in result:
    print(sentence)

result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(sorted_ind[-3], sst_path)

for sentence in result:
    print(sentence)

result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(sorted_ind[-4], sst_path)

for sentence in result:
    print(sentence)

result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(sorted_ind[-5], sst_path)

for sentence in result:
    print(sentence)

result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(sorted_ind[-6], sst_path)

for sentence in result:
    print(sentence)

result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(sorted_ind[-7], sst_path)

for sentence in result:
    print(sentence)

result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(sorted_ind[-8], sst_path)

for sentence in result:
    print(sentence)

result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(sorted_ind[-9], sst_path)

for sentence in result:
    print(sentence)

result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(sorted_ind[-10], sst_path)

for sentence in result:
    print(sentence)

result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(sorted_ind[-11], sst_path)

for sentence in result:
    print(sentence)

result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(sorted_ind[-12], sst_path)

for sentence in result:
    print(sentence)

result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(sorted_ind[-13], sst_path)

for sentence in result:
    print(sentence)

# ## CLSとSEP以外に反応してるやつら

# ### 文頭と文末付近で発火?(だがこれだとSEPとCLSと同じっちゃ同じか...)

result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(sorted_ind[-11], sst_path)

for sentence in result:
    print(sentence)

# Visualize the distribution here 
RED = '\033[41m'
counter = dict()
for sentence in result:
    for i,token in enumerate(sentence.split(" ")):
        if token.startswith(RED):
            if i not in counter:
                counter[i] = 0
            counter[i] += 1
counter = sorted(counter.items(),key=lambda x:x[1])

mean_sentence_len = 0
for sentence in result:
    mean_sentence_len += len(sentence.split(" "))
mean_sentence_len = mean_sentence_len / len(result)
mean_sentence_len

keys = [x[0] for x in counter]
vals = [x[1] for x in counter]
fig,ax = plt.subplots()
ax.scatter(keys,vals)

import seaborn as sns
sentence_lens = []
for sentence in result:
    sentence_lens.append(len(sentence.split(" ")))
sns.distplot(sentence_lens)



result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(sorted_ind[-14], sst_path)

for sentence in result:
    print(sentence)

result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(sorted_ind[-15], sst_path)

for sentence in result:
    print(sentence)

result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(sorted_ind[-20], sst_path)

for sentence in result:
    print(sentence)

result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(sorted_ind[-30], sst_path)

for sentence in result:
    print(sentence)

result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(sorted_ind[-40], sst_path)

for sentence in result:
    print(sentence)

result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(sorted_ind[-50], sst_path)

for sentence in result:
    print(sentence)

result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(sorted_ind[-60], sst_path)

for sentence in result:
    print(sentence)

result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(sorted_ind[-70], sst_path)

for sentence in result:
    print(sentence)

result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(sorted_ind[-80], sst_path)

for sentence in result:
    print(sentence)

result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(sorted_ind[-90], sst_path)

for sentence in result:
    print(sentence)

result, pos_tok, neg_tok = cross_model_corr.highlight_neuron(sorted_ind[-100], sst_path)

for sentence in result:
    print(sentence)

# # 発火している単語の分布が全体の分布と異なっているニューロンを可視化


