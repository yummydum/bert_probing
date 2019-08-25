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

# %load_ext autoreload
# %autoreload 2

import linguistic_corr
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, MinMaxScaler
import seaborn as sns
from joblib import load
import json
#from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:150% !important; }</style>"))
sns.set()

# +
st_neuron_path = Path(f"probing_data/BERT/ST_neuron.npy")
X = np.load(st_neuron_path)
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)
pos_path = Path(f"probing_data/BERT/ST_y.npy")
y = np.load(pos_path)

model_path = Path("probing_data/BERT/temp_probe_model.joblib")
model = load(model_path)
acc = np.sum(y == model.predict(X_norm)) / len(y)
print(acc)

sorted_coef = np.sort(model.coef_, axis=1)
sorted_coef_ind = np.argsort(model.coef_, axis=1)
# -

data_path = Path(f"probing_data/ST/ST-dev.json")
with data_path.open("r", encoding="utf-8") as f:
    data = json.load(f)
pos2ind = {k: v for v, k in enumerate(data["all_pos"])}
ind2pos = {k: v for k, v in enumerate(data["all_pos"])}


# # 文章を取り出す関数
def ith_data(i):  # i = 0
    return data["data"][str(i)]["word"], data["data"][str(i)]["pos"]


# # インデックスから単語へのマッピング
count = 0
ind2word = {}
for i in data["data"].keys():
    words = data["data"][str(i)]["word"]
    for word in words:
        ind2word[count] = word
        count += 1

# # posから正解インデックスへのマッピング
count = 0
pos2correct_word_ind = {}
for i in data["data"].keys():
    pos_list = data["data"][str(i)]["pos"]
    for pos in pos_list:
        if pos not in pos2correct_word_ind:
            pos2correct_word_ind[pos] = []
        pos2correct_word_ind[pos].append(count)
        count += 1

# # クラスの分布

pos_array, count_array = np.unique(y, return_counts=True)
pos_count = zip(pos_array, count_array)
sorted_pos_count = sorted(pos_count, reverse=True, key=lambda x: x[1])
for pos_ind, count in sorted_pos_count:
    print(ind2pos[pos_ind], count)

# # クラス毎の正解率

acc_result = []
for i in range(44):
    acc_i = np.sum(y[y == i] == model.predict(X[y == i])) / np.sum(y == i)
    acc_result.append((acc_i, ind2pos[i]))

for acc_i, pos_name in sorted(acc_result, key=lambda x: x[0], reverse=True):
    print(f"acc for {pos_name} is {acc_i}")

for acc_i, pos_name in sorted(acc_result, key=lambda x: x[0], reverse=True):
    print(pos2ind[pos_name])

# # スコアの分布
score = model.coef_ @ X_norm.T
sorted_score = np.sort(score, axis=1)
sorted_score_ind = np.argsort(score, axis=1)

# # IN

# ## スコアの分布を見る (こいつがでかいー>exp(-score)が小さいー>予測確率が1に近づく)
pos = "IN"
pos_score = sorted_score[pos2ind[pos]]
sns.distplot(pos_score)

# ## 予測確率の分布を見る (ちゃんと分けられている)
pred_prob = 1 / (1 + np.exp(-score[pos2ind[pos]]))
sns.distplot(pred_prob)

# ## スコアが高い単語の確認(ofがほとんど)
sorted_score[pos2ind[pos]][-10:]
inds = sorted_score_ind[pos2ind[pos]][-10:]  # 何番目の単語がこのPOSのスコアが高かったのか
# ちゃんとofとかが出てきている
ind2word[inds[-1]]
ind2word[inds[-2]]
ind2word[inds[-3]]

# ## weight mass (分布が狭いほど少数のfeatureが効いていることを示している(他はノイズ))
weight_mass = np.mean(model.coef_[pos2ind[pos]] * X, axis=0)
sns.distplot(weight_mass)

np.sum(abs(weight_mass - np.mean(weight_mass)) > 5 * np.std(weight_mass))

# # NN

# ## スコアの分布を見る (こいつがでかいー>exp(-score)が小さいー>予測確率が1に近づく)
pos = "NN"
pos_score = sorted_score[pos2ind[pos]]
sns.distplot(pos_score)

# ## 予測確率の分布を見る (ちゃんと分けられている)
pred_prob = 1 / (1 + np.exp(-score[pos2ind[pos]]))
sns.distplot(pred_prob)

# ## スコアが高い単語の確認(ofがほとんど)
sorted_score[pos2ind[pos]][-10:]
inds = sorted_score_ind[pos2ind[pos]][-10:]  # 何番目の単語がこのPOSのスコアが高かったのか
# ちゃんとofとかが出てきている
ind2word[inds[-1]]
ind2word[inds[-2]]
ind2word[inds[-3]]

# ## weight mass (分布が狭いほど少数のfeatureが効いていることを示している(他はノイズ))
weight_mass = np.mean(model.coef_[pos2ind[pos]] * X, axis=0)
sns.distplot(weight_mass)

np.sum(abs(weight_mass - np.mean(weight_mass)) > 5 * np.std(weight_mass))

# # NN

# ## スコアの分布を見る (こいつがでかいー>exp(-score)が小さいー>予測確率が1に近づく)
pos = "CC"
pos_score = sorted_score[pos2ind[pos]]
sns.distplot(pos_score)

# ## 予測確率の分布を見る (ちゃんと分けられている)
pred_prob = 1 / (1 + np.exp(-score[pos2ind[pos]]))
sns.distplot(pred_prob)

# ## スコアが高い単語の確認(ofがほとんど)
sorted_score[pos2ind[pos]][-10:]
inds = sorted_score_ind[pos2ind[pos]][-10:]  # 何番目の単語がこのPOSのスコアが高かったのか
# ちゃんとofとかが出てきている
ind2word[inds[-1]]
ind2word[inds[-2]]
ind2word[inds[-3]]

# ## weight mass (分布が狭いほど少数のfeatureが効いていることを示している(他はノイズ))
weight_mass = np.mean(model.coef_[pos2ind[pos]] * X, axis=0)
sns.distplot(weight_mass)

np.sum(abs(weight_mass - np.mean(weight_mass)) > 5 * np.std(weight_mass))

# # NN

# ## スコアの分布を見る (こいつがでかいー>exp(-score)が小さいー>予測確率が1に近づく)
pos = "DET"
pos_score = sorted_score[pos2ind[pos]]
sns.distplot(pos_score)

# ## 予測確率の分布を見る (ちゃんと分けられている)
pred_prob = 1 / (1 + np.exp(-score[pos2ind[pos]]))
sns.distplot(pred_prob)

# ## スコアが高い単語の確認(ofがほとんど)
sorted_score[pos2ind[pos]][-10:]
inds = sorted_score_ind[pos2ind[pos]][-10:]  # 何番目の単語がこのPOSのスコアが高かったのか
# ちゃんとofとかが出てきている
ind2word[inds[-1]]
ind2word[inds[-2]]
ind2word[inds[-3]]

# ## weight mass (分布が狭いほど少数のfeatureが効いていることを示している(他はノイズ))
weight_mass = np.mean(model.coef_[pos2ind[pos]] * X, axis=0)
sns.distplot(weight_mass)

np.sum(abs(weight_mass - np.mean(weight_mass)) > 5 * np.std(weight_mass))

# # NN

# ## スコアの分布を見る (こいつがでかいー>exp(-score)が小さいー>予測確率が1に近づく)
pos = "JJ"
pos_score = sorted_score[pos2ind[pos]]
sns.distplot(pos_score)

# ## 予測確率の分布を見る (ちゃんと分けられている)
pred_prob = 1 / (1 + np.exp(-score[pos2ind[pos]]))
sns.distplot(pred_prob)

# ## スコアが高い単語の確認(ofがほとんど)
sorted_score[pos2ind[pos]][-10:]
inds = sorted_score_ind[pos2ind[pos]][-10:]  # 何番目の単語がこのPOSのスコアが高かったのか
# ちゃんとofとかが出てきている
ind2word[inds[-1]]
ind2word[inds[-2]]
ind2word[inds[-3]]

# ## weight mass (分布が狭いほど少数のfeatureが効いていることを示している(他はノイズ))
weight_mass = np.mean(model.coef_[pos2ind[pos]] * X, axis=0)
sns.distplot(weight_mass)

np.sum(abs(weight_mass - np.mean(weight_mass)) > 5 * np.std(weight_mass))

# # NN

# ## スコアの分布を見る (こいつがでかいー>exp(-score)が小さいー>予測確率が1に近づく)
pos = "VB"
pos_score = sorted_score[pos2ind[pos]]
sns.distplot(pos_score)

# ## 予測確率の分布を見る (ちゃんと分けられている)
pred_prob = 1 / (1 + np.exp(-score[pos2ind[pos]]))
sns.distplot(pred_prob)

# ## スコアが高い単語の確認(ofがほとんど)
sorted_score[pos2ind[pos]][-10:]
inds = sorted_score_ind[pos2ind[pos]][-10:]  # 何番目の単語がこのPOSのスコアが高かったのか
# ちゃんとofとかが出てきている
ind2word[inds[-1]]
ind2word[inds[-2]]
ind2word[inds[-3]]

# ## weight mass (分布が狭いほど少数のfeatureが効いていることを示している(他はノイズ))
weight_mass = np.mean(model.coef_[pos2ind[pos]] * X, axis=0)
sns.distplot(weight_mass)

np.sum(abs(weight_mass - np.mean(weight_mass)) > 5 * np.std(weight_mass))

# # ## 重要な次元に関して，他の単語とこいつらとの分布の差異を見る (ofの時だけ極端な値を取っていて欲しい)
# correct_word_ind = pos2correct_word_ind["IN"]
# correct_word_ind_set = set(correct_word_ind)
# correct_words = set([ind2word[x] for x in correct_word_ind])  # 正解単語を確認
# correct_bool = []  768*12
# for i in range(40115):
#     if i in correct_word_ind_set:
#         correct_bool.append(True)
#     else:
#         correct_bool.append(False)
# X_correct = X[correct_bool]
# X_not_correct = X[[not x for x in correct_bool]]
#
# important_ind = sorted_weight_mass_ind[0][0]
# sns.distplot(X_correct[:,important_ind])
# sns.distplot(X_not_correct[:,important_ind])
