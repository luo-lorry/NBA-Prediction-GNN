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
        Gs.append(torch.LongTensor(np.hstack((edges, edges[[1, 0]]))))

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
# model = ASTGCN(nb_block=2, in_channels=model_in, K=3,
#                nb_chev_filter=1, nb_time_filter=64, time_strides=1, num_for_predict=6,
#                len_input=10, num_of_vertices=582, nb_gatv2conv=16, dropout_gatv2conv=0.25,
#                head_gatv2conv=4)

model = GATv2TCN(in_channels=model_in,
        out_channels=6,
        len_input=10,
        len_output=1,
        temporal_filter=64,
        # kernel_tcn=2,
        # kernel_conv2d=1,
        out_gatv2conv=32,
        dropout_tcn=0.25,
        dropout_gatv2conv=0.5,
        head_gatv2conv=4)

model_name = 'gatv2tcn-team-position-embedding'
# model.load_state_dict(torch.load(f"model/{model_name}/saved_astgcn.pth"))

if not os.path.exists(f"model/{model_name}"):
    os.mkdir(f"model/{model_name}")
# model.load_state_dict(torch.load(f"model/{model_name}/saved_astgcn.pth"))
# team_embedding.load_state_dict(torch.load(f"model/{model_name}/saved_team.pth"))
# model.eval()
# team_embedding.eval()

parameters = list(model.parameters()) + list(team_embedding.parameters()) + list(position_embedding.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.01, weight_decay=0.001)

min_val_loss = np.inf
min_val_iter = -1

# fig, ax = plt.subplots()
PT_INDEX = 0
EPOCHS = 300
BATCH_SIZE = 20
train_loss_history = np.zeros(EPOCHS)
val_loss_history = np.zeros(EPOCHS)
for epoch in tqdm(range(EPOCHS)):
    train_loss = 0.0
    model.train()
    team_embedding_vector = team_embedding(team_tensor)
    position_embedding_vector = position_embedding(position_tensor)
    for i in np.random.choice(np.arange(X_train.shape[0]), size=BATCH_SIZE, replace=False): # range(X_train.shape[0]):
        y_train_mask = h_train[i].unique()
        X_list = []
        G_list = []
        for j in range(SEQ_LENGTH):
            X_list.append(torch.cat([X_train[i][:, :, j], team_embedding_vector, position_embedding_vector], dim=1))
            G_list.append(g_train[i][j])
        x = torch.stack(X_list, dim=-1)
        x = x[None, :, :, :]
        x_astgat = model(x, G_list)[0, ...]
        train_loss += F.mse_loss(x_astgat[y_train_mask], y_train[i][y_train_mask][:, player_prediction_metrics_index])

    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    val_loss = 0.0
    model.eval()
    team_embedding.eval()
    position_embedding.eval()
    team_embedding_vector = team_embedding(team_tensor)
    position_embedding_vector = position_embedding(position_tensor)

    for i in range(X_val.shape[0]):
        y_val_mask = h_val[i].unique()
        X_list = []
        G_list = []
        for j in range(SEQ_LENGTH):
            X_list.append(torch.cat([X_val[i][:, :, j], team_embedding_vector, position_embedding_vector], dim=1))
            G_list.append(g_val[i][j])
        x = torch.stack(X_list, dim=-1)
        x = x[None, :, :, :]
        x_astgat = model(x, G_list)[0, :, :]
        val_loss += F.mse_loss(x_astgat[y_val_mask], y_val[i][y_val_mask][:, player_prediction_metrics_index])

    print(f"validation loss: {val_loss.item()}, training loss: {train_loss.item()}")
    train_loss_history[epoch] = train_loss.item()
    val_loss_history[epoch] = val_loss.item()
    # if epoch % 5 == 0:
    #     if epoch == 0:
    #         ax.plot(y_val[i][y_val_mask, PT_INDEX].detach().numpy(), ls='-.', color='k', lw=1.5, label='real')
    #     if epoch >= 80:
    #         ax.lines.pop(1)
    #     for l in ax.lines[1:]:
    #         l.set_alpha(.3)
    #     ax.plot(x_astgat[y_val_mask, PT_INDEX].detach().numpy(), label=f'{epoch} ({train_loss.item()})')
    #     ax.set_title("Epoch: %d, loss: %1.5f" % (epoch, train_loss.item()))
    #     ax.legend(bbox_to_anchor=(1, 0.5))
    #     # display.clear_output(wait=True)
    #     # display.display(fig)

    if min_val_loss > val_loss:
        print(f"Validation Loss Decreased({min_val_loss:.5f}--->{val_loss:.5f}) \t Saving The Model")
        min_val_loss = val_loss.item()
        min_val_iter = epoch
        # Saving State Dict
        torch.save(model.state_dict(), f"model/{model_name}/saved_astgcn.pth")
        torch.save(team_embedding.state_dict(), f"model/{model_name}/team_embedding.pth")
        torch.save(position_embedding.state_dict(), f"model/{model_name}/position_embedding.pth")

print(min_val_loss, min_val_iter)


#GATv2TCN
astgcn_test = copy.deepcopy(model)
astgcn_test.load_state_dict(torch.load(f"model/{model_name}/saved_astgcn.pth"))
astgcn_test.eval()

team_embedding_test = copy.deepcopy(team_embedding)
team_embedding_test.load_state_dict(torch.load(f"model/{model_name}/team_embedding.pth"))
team_embedding_test.eval()

position_embedding_test = copy.deepcopy(position_embedding)
position_embedding_test.load_state_dict(torch.load(f"model/{model_name}/position_embedding.pth"))
position_embedding_test.eval()

team_embedding_vector = team_embedding_test(team_tensor)
position_embedding_vector = position_embedding_test(position_tensor)

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

fig, ax = plt.subplots()
for i, team in enumerate(nba_teams):
    player_in_team = [idx for idx, team_name in enumerate(player_id_to_team.values()) if team_name == team['nickname']]
    ax.plot(team_vec[player_in_team, 0], team_vec[player_in_team, 1], color=colors[i], marker=markers[i+1], label=team['nickname'])
    plt.text(team_vec[player_in_team, 0].mean(), team_vec[player_in_team, 1].mean(), team['nickname'])

player_id_to_position = pd.read_pickle('player_id2position.pkl')
position_vec = position_embedding_vector.detach().numpy()

fig, ax = plt.subplots()
position_dict = {(0, 0, 0): 'No position',
                 (0, 0, 1): 'C',
                 (0, 1, 0): 'G',
                 (1, 0, 0): 'F',
                 (1, 0, 1): 'F/C',
                 (1, 1, 0): 'F/G'}
for i, position in enumerate(np.unique(np.array(list(player_id_to_position.values())), axis=0)):
    player_at_position = [idx for idx, player_position in enumerate(player_id_to_position.values()) if (player_position==position).all()]
    label = position_dict[tuple(position)]
    ax.plot(position_vec[player_at_position, 0], position_vec[player_at_position, 1], color=colors[i], marker=markers[i+1], label=label)
ax.legend()
pass
