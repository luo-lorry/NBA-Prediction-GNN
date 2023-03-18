import copy
import os
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from IPython import display
from torch_geometric.nn import GATConv, GATv2Conv
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from torch_geometric_temporal.nn import ASTGCN, GATv2TCN
from sklearn import preprocessing
from tqdm import tqdm


SEQ_LENGTH = 10
OFFSET = 1
player_boxscore_fields = ['PTS', 'AST', 'REB', 'TO', 'STL', 'BLK', 'PLUS_MINUS']
player_boxscore_tracking_fields = ['TCHS', 'PASS', 'DIST']
player_boxscore_advanced_fields = ['PACE', 'USG_PCT', 'TS_PCT']
player_prediction_metrics = ['PTS', 'AST', 'REB', 'TO', 'STL', 'BLK']
player_prediction_metrics_index = [
    (player_boxscore_fields + player_boxscore_tracking_fields + player_boxscore_advanced_fields).index(metric) for
    metric in player_prediction_metrics]


def fill_zeros_with_last(seq):
    seq_ffill = np.zeros_like(seq)
    for i in range(seq.shape[1]):
        arr = seq[:, i]
        prev = np.arange(len(arr))
        prev[arr == 0] = 0
        prev = np.maximum.accumulate(prev)
        seq_ffill[:, i] = arr[prev]

    return seq_ffill

def construct_input_sequences_and_output(z, seq_length=10, offset=1):
    if not isinstance(z, (np.ndarray, np.generic)):
        z = np.array(z)
    if offset == 0:
        x = sliding_window_view(z, seq_length, axis=0)
    else:
        x = sliding_window_view(z[:-offset], seq_length, axis=0)
    y = z[seq_length+offset-1:]
    return x, y

def create_dataset():
    X_seq = pd.read_pickle('data/X_seq.pkl')
    G_seq = pd.read_pickle('data/G_seq.pkl')
    player_id_to_team = pd.read_pickle('player_id2team.pkl')
    player_id_to_position = pd.read_pickle('player_id2position.pkl')

    le = preprocessing.LabelEncoder()
    df_id2team = pd.DataFrame.from_dict(player_id_to_team, orient='index').apply(le.fit_transform)
    enc = preprocessing.OneHotEncoder()
    enc.fit(df_id2team)
    onehotlabels = enc.transform(df_id2team).toarray()
    # team_onehot_seq = np.broadcast_to(onehotlabels, (X_seq.shape[0], X_seq.shape[1], onehotlabels.shape[-1]))
    team_tensor = Variable(torch.FloatTensor(onehotlabels))
    position_tensor = Variable(torch.FloatTensor(np.stack(list(player_id_to_position.values()), axis=0)))

    Xs = np.zeros_like(X_seq)
    for i in range(X_seq.shape[1]):
        Xs[:, i, :] = fill_zeros_with_last(X_seq[:, i, :])

    Gs = []
    for g in G_seq:
        node_dict = {node: i for i, node in enumerate(G_seq[0].nodes())}
        edges = np.array([edge.split(' ') for edge in nx.generate_edgelist(g)])[:, :2].astype(int).T
        edges = np.vectorize(node_dict.__getitem__)(edges)
        Gs.append(torch.LongTensor(edges))

    X_in, X_out = construct_input_sequences_and_output(Xs, seq_length=SEQ_LENGTH, offset=OFFSET)
    # team_in, _ = construct_input_sequences_and_output(team_onehot_seq, seq_length=SEQ_LENGTH, offset=OFFSET)
    G_in, G_out = construct_input_sequences_and_output(Gs, seq_length=SEQ_LENGTH, offset=OFFSET)
    X_in = Variable(torch.FloatTensor(X_in))
    X_out = Variable(torch.FloatTensor(X_out))
    # team_in = Variable(torch.FloatTensor(team_in))

    X_train, X_val, X_test = X_in[:31], X_in[41:41+16], X_in[41+26:]
    y_train, y_val, y_test = X_out[:31], X_out[41:41+16], X_out[41+26:]
    g_train, g_val, g_test = G_in[:31], G_in[41:41+16], G_in[41+26:]
    h_train, h_val, h_test = G_out[:31], G_out[41:41+16], G_out[41+26:]
    # team_train, team_val, team_test = team_in[:31], team_in[41:41+16], team_in[41+26:]
    print(X_train.shape, X_val.shape, X_test.shape)
    print(g_train.shape, g_val.shape, g_test.shape)
    print(h_train.shape, y_train.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val, g_test, h_train, h_val, h_test, team_tensor, position_tensor #, team_train, team_val, team_test


X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val, g_test, h_train, h_val, h_test, team_tensor, position_tensor = create_dataset()
team_embedding_in = team_tensor.shape[-1]
team_embedding_out = 2
team_embedding = nn.Linear(team_embedding_in, team_embedding_out) # nn.Embedding(num_embeddings=team_embedding_in, embedding_dim=team_embedding_out)

position_embedding_in = position_tensor.shape[-1]
position_embedding_out = 2
position_embedding = nn.Linear(position_embedding_in, position_embedding_out) # nn.Embedding(num_embeddings=position_embedding_in, embedding_dim=position_embedding_out)

model_in = y_train.shape[-1] + team_embedding_out + position_embedding_out
model = GATv2TCN(in_channels=model_in,
        out_channels=6,
        len_input=10,
        len_output=1,
        temporal_filter=64,
        # kernel_tcn=2,
        # kernel_conv2d=1,
        out_gatv2conv=32,
        dropout_tcn=0.25,
        dropout_gatv2conv=0.25,
        head_gatv2conv=4)

model_name = 'gatv2tcn-team-position-embedding'
model.load_state_dict(torch.load(f"model/{model_name}/saved_astgcn.pth"))
team_embedding.load_state_dict(torch.load(f"model/{model_name}/team_embedding.pth"))
position_embedding.load_state_dict(torch.load(f"model/{model_name}/position_embedding.pth"))
model.eval()
team_embedding.eval()
position_embedding.eval()
team_embedding_vector = team_embedding(team_tensor)
position_embedding_vector = position_embedding(position_tensor)


X_seq = pd.read_pickle('data/X_seq.pkl')
G_seq = pd.read_pickle('data/G_seq.pkl')
player_id_to_team = pd.read_pickle('player_id2team.pkl')
player_id_to_position = pd.read_pickle('player_id2position.pkl')
player_id_to_name = pd.read_pickle('player_id2name.pkl')

Xs = np.zeros_like(X_seq)
for i in range(X_seq.shape[1]):
    Xs[:, i, :] = fill_zeros_with_last(X_seq[:, i, :])

Gs = []
for g in G_seq:
    node_dict = {node: i for i, node in enumerate(G_seq[0].nodes())}
    edges = np.array([edge.split(' ') for edge in nx.generate_edgelist(g)])[:, :2].astype(int).T
    edges = np.vectorize(node_dict.__getitem__)(edges)
    Gs.append(torch.LongTensor(np.hstack((edges, edges[[1, 0]]))))

X_train = torch.FloatTensor(Xs[-21:-11].transpose(1, 2, 0))
g_train = Gs[-20:-10]

X_list = []
G_list = []
for j in range(SEQ_LENGTH):
    X_list.append(torch.cat([X_train[:, :, j], team_embedding_vector, position_embedding_vector], dim=1))
    G_list.append(g_train[j])
x = torch.stack(X_list, dim=-1)
x = x[None, :, :, :]
x_astgat = model(x, G_list)[0, ...]

## code for creating heatmap
player_id_to_team = pd.read_pickle('player_id2team.pkl')
player_id_to_name = pd.read_pickle('player_id2name.pkl')
a,(b,c)=self._gatv2conv_attention(x=X[0, :, :, t], edge_index=edge_index[t], return_attention_weights=True)
b = b.detach().numpy(); c = c.detach().numpy()
fig, ax = plt.subplots()
att_mat = np.zeros((582, 582))
att_mat[b[0], b[1]] = c.max(axis=1)  # head one
att_mat -= np.diag(np.diag(att_mat))
att_mat /= att_mat.max()
att_mat = np.log(att_mat + 1)
# receiver, sender = np.unravel_index(att_mat.argsort(axis=None)[::-1][:100], att_mat.shape)
# rivals = [(s, r) for s, r in zip(sender, receiver) if (list(player_id_to_team.values())[s] == 'Suns' and list(player_id_to_team.values())[r] == 'Warriors') or (list(player_id_to_team.values())[s] == 'Warriors' and list(player_id_to_team.values())[r] == 'Suns')]
sun_idx = np.array([idx for idx, team in enumerate(player_id_to_team.values()) if team == 'Suns'])
warrior_idx = np.array([idx for idx, team in enumerate(player_id_to_team.values()) if team == 'Warriors'])
att_mat = att_mat[np.hstack([sun_idx, warrior_idx])][:, np.hstack([sun_idx, warrior_idx])]
cax = ax.matshow(att_mat, vmin=0.1, vmax=0.35, zorder=-1)
fig.colorbar(cax)
sender, receiver = np.unravel_index(att_mat.argsort(axis=None)[::-1][:20], att_mat.shape)
sender, receiver = sender[np.bitwise_or(np.bitwise_and(sender<=15, receiver>15), np.bitwise_and(sender>15, receiver<=15))], receiver[np.bitwise_or(np.bitwise_and(sender<=15, receiver>15), np.bitwise_and(sender>15, receiver<=15))]
ax.scatter(receiver, sender, marker='o', s=78, linewidth=1.5, facecolors='none', edgecolors='r')
sw_xticks = np.array(list(player_id_to_name.values()))[np.hstack([sun_idx, warrior_idx])]
x = np.arange(sw_xticks.shape[0])
ax.axvline(x=15.48, ymin=0, ymax=1.3, clip_on=False, color='k', linewidth=0.5)
ax.axhline(y=15.5, xmin=-0.3, xmax=1.0, clip_on=False, color='k', linewidth=0.5)
ax.set_xticks(x, sw_xticks, rotation=90, fontsize=6)
ax.set_yticks(x, sw_xticks, fontsize=6)
# print([(list(player_id_to_name.values())[s], list(player_id_to_name.values())[r]) for s, r in rivals[:5]])
plt.tight_layout()
fig.savefig('Suns_Warriors_attention.png', dpi=200)
plt.show()


#GATv2TCN
astgcn_test = copy.deepcopy(model)
astgcn_test.load_state_dict(torch.load(f"model/{model_name}/saved_astgcn.pth"))
astgcn_test.eval()

test_loss_mse = 0.0
test_loss_l1 = 0.0
test_loss_rmse = 0.0
test_corr = 0.0
test_loss_mape = 0.0

for i in range(X_test.shape[0]):
    y_test_mask = h_test[i].unique()
    X_list = []
    G_list = []
    for j in range(SEQ_LENGTH):
        X_list.append(torch.cat([X_val[i][:, :, j], team_embedding_vector, position_embedding_vector], dim=1))
        G_list.append(g_test[i][j])
    x = torch.stack(X_list, dim=-1)
    x = x[None, :, :, :]
    x_astgcn = astgcn_test(x, G_list)[0, :, :]
    test_loss_mse += F.mse_loss(x_astgcn[y_test_mask], y_test[i][y_test_mask][:, player_prediction_metrics_index])
    test_loss_rmse += mean_squared_error(x_astgcn[y_test_mask].detach().numpy(), y_test[i][y_test_mask][:, player_prediction_metrics_index].detach().numpy(), squared=False) # torch.sqrt(F.mse_loss(x_astgcn[y_test_mask], y_test[i][y_test_mask][:, player_prediction_metrics_index]))
    test_loss_l1 += F.l1_loss(x_astgcn[y_test_mask], y_test[i][y_test_mask][:, player_prediction_metrics_index])
    test_loss_mape += mean_absolute_percentage_error(x_astgcn[y_test_mask].detach().numpy(), y_test[i][y_test_mask][:, player_prediction_metrics_index].detach().numpy()) # torch.sqrt(F.mse_loss(x_astgcn[y_test_mask], y_test[i][y_test_mask][:, player_prediction_metrics_index]))
    test_corr += torch.tanh(torch.mean(torch.stack([torch.arctanh(torch.corrcoef(torch.stack([x_astgcn[y_test_mask][:, metric_idx],
                             y_test[i][y_test_mask][:, player_prediction_metrics_index][:, metric_idx]], dim=0))[0, 1])
                            for metric_idx in range(len(player_prediction_metrics))])))
print(f"MSE: {test_loss_mse.item()/X_test.shape[0]}, RMSE: {test_loss_rmse/X_test.shape[0]}, MAPE: {test_loss_mape/X_test.shape[0]}, CORR: {test_corr.item()/X_test.shape[0]}, MAE: {test_loss_l1.item()/X_test.shape[0]}")



player_id_to_team = pd.read_pickle('player_id2team.pkl')
from nba_api.stats.static import teams, players
nba_teams = teams.get_teams()
team_vec = team_embedding_vector.detach().numpy()
from pandas.plotting._matplotlib.style import get_standard_colors
from matplotlib.lines import Line2D
colors = get_standard_colors(num_colors=len(nba_teams))
markers = list(Line2D.markers.keys())[:len(nba_teams)+1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [2, 1]})
for i, team in enumerate(nba_teams):
    player_in_team = [idx for idx, team_name in enumerate(player_id_to_team.values()) if team_name == team['nickname']]
    ax1.plot(team_vec[player_in_team, 0], team_vec[player_in_team, 1], color=colors[i], marker=markers[i+1], label=team['nickname'])
    ax1.text(team_vec[player_in_team, 0].mean(), team_vec[player_in_team, 1].mean(), team['nickname'])
# fig.savefig('team_embedding.png', dpi=200)
ax1.set_title('Team Embedding')

player_id_to_position = pd.read_pickle('player_id2position.pkl')
position_vec = position_embedding_vector.detach().numpy()

# fig, ax = plt.subplots()
position_dict = {(0, 0, 0): 'No position',
                 (0, 0, 1): 'C',
                 (0, 1, 0): 'G',
                 (1, 0, 0): 'F',
                 (1, 0, 1): 'F/C',
                 (1, 1, 0): 'F/G'}
for i, position in enumerate(np.unique(np.array(list(player_id_to_position.values())), axis=0)):
    if (position==0).all():
        continue
    player_at_position = [idx for idx, player_position in enumerate(player_id_to_position.values()) if (player_position==position).all()]
    label = position_dict[tuple(position)]
    ax2.plot(position_vec[player_at_position, 0], position_vec[player_at_position, 1], color=colors[i], marker=markers[i+1], label=label)
ax2.legend()
ax2.set_title('Position Embedding')
fig.savefig('embedding.png', dpi=200)
pass
