from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric_temporal.nn import GATv2TCN
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


SEQ_LENGTH = 10
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

# nba_teams = teams.get_teams()
# gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable='2022-23', league_id_nullable='00')#, team_id_nullable=1610612738)
# games = gamefinder.get_data_frames()[0]
# games.drop_duplicates('GAME_ID', keep='first', inplace=True)
# games = games.loc[(games['GAME_DATE'] == '2023-01-21')  & (games['TEAM_ID'].isin([team['id'] for team in nba_teams]))][::-1]



X_seq = pd.read_pickle('data/X_seq.pkl')
G_seq = pd.read_pickle('data/G_seq.pkl')
player_id_to_team = pd.read_pickle('player_id2team.pkl')
player_id_to_position = pd.read_pickle('player_id2position.pkl')
player_id_to_name = pd.read_pickle('player_id2name.pkl')
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

team_embedding_in = team_tensor.shape[-1]
team_embedding_out = 2
team_embedding = nn.Linear(team_embedding_in, team_embedding_out) # nn.Embedding(num_embeddings=team_embedding_in, embedding_dim=team_embedding_out)

position_embedding_in = position_tensor.shape[-1]
position_embedding_out = 2
position_embedding = nn.Linear(position_embedding_in, position_embedding_out) # nn.Embedding(num_embeddings=position_embedding_in, embedding_dim=position_embedding_out)

model_in = X_seq.shape[-1] + team_embedding_out + position_embedding_out
model = GATv2TCN(in_channels=model_in,
        out_channels=6,
        len_input=10,
        len_output=1,
        temporal_filter=64,
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

X_list = [torch.cat([torch.FloatTensor(x), team_embedding_vector, position_embedding_vector], dim=1) for x in Xs[-11:-1]]
X = torch.stack(X_list, dim=-1)
X = X[None, :, :, :]
G_list = Gs[-11:-1]
x_astgcn = model(X, G_list)[0, :, :]

today_pickem = {'CJ McCollum': {'PTS': 25.5,
                                'REB': 4.0},
                'Jonas Valanciunas': {'PTS': 19.0,
                                      'REB': 11.5},
                'Trey Murphy III': {'PTS': 15.5},
                'Paolo Banchero': {'PTS': 9.5,
                                   'REB': 6.0},
                'Markelle Fultz': {'PTS': 12.5,
                                   'REB': 10.0,
                                   'AST': 5.5},
                'Wendell Carter Jr.': {'PTS': 16.5,
                                       'REB': 8.5,
                                       'AST': 2.5},
                'Franz Wagner': {'PTS': 19.5,
                                 'REB': 3.5,
                                 'AST': 2.5},
                'Bol Bol': {'PTS': 7.5},
                'Cole Anthony': {'PTS': 8.5,
                                 'REB': 3.5},
                'Moritz Wagner': {'PTS': 9.5,
                                  'REB': 4.0},
                'Gary Harris': {'PTS': 7.5},

                                'Jordan Poole': {'PTS': 27.0,
                                               'AST': 5.0},
                                'Donte DiVincenzo': {'PTS': 12.5,
                                                'REB': 5.5},
                                'Kevon Looney': {'PTS': 6.5,
                                                   'REB': 9.5,
                                                   'AST': 2.5},
                                'Darius Garland': {'PTS': 25.5,
                                                   'AST': 8.5},
                                'Evan Mobley': {'PTS': 16.5,
                                                  'REB': 9.0},

                                'Jimmy Butler': {'PTS': 20.5,
                                                    'REB': 6.0,
                                                    'AST': 4.0,
                                                    'STL': 1.5},
                                "Bam Adebayo": {'PTS': 21.5,
                                                     'REB': 10.5},
                                'Tyler Herro': {'PTS': 18.5,
                                                  'REB': 5.0,
                                                  'AST': 4.0},
                                'Caleb Martin': {'PTS': 8.5,
                                                    'REB': 4.0},
                                'Kyle Lowry': {'PTS': 9.5,
                                               'REB': 3.5},
                                'Max Strus': {'PTS': 10.5},
                                'Victor Oladipo': {'PTS': 10.5},
                                'Luka Doncic': {'PTS': 34.5,
                                                'REB': 10.5,
                                               'AST': 9.0},
                                'Spencer Dinwiddie': {'PTS': 16.5,
                                                     'AST': 5.0},
                                'Tim Hardaway Jr.': {'PTS': 15.5,
                                                'REB': 4.0},
                                'Dorian Finney-Smith': {'PTS': 9.5,
                                                       'REB': 4.5,
                                                       'AST': 1.5,
                                                       'STL': 0.5}
                }

case_study = pd.DataFrame(columns=['Player', 'Statistic', 'Metric', 'Prediction', 'Actual Value'])
row_id = 0
today_pickem_predict = today_pickem.copy()
pickem_correct = 0
pickem_total = 0
predict_values = []
metric_values = []
true_values = []
for player_name, test_metrics in today_pickem.items():
    player_idx = list(player_id_to_name.values()).index(player_name)
    for metric_name, metric_value in test_metrics.items():
        metric_idx = player_prediction_metrics.index(metric_name)
        predict_value = x_astgcn[player_idx, metric_idx].item()
        true_value = X_seq[-1][player_idx, metric_idx]
        if np.isnan(true_value):
            continue
        correct = (true_value > metric_value) == (predict_value > metric_value)
        today_pickem_predict[player_name][metric_name] = (metric_value, predict_value, true_value, correct)
        predict_values.append(round(predict_value))
        metric_values.append(metric_value)
        true_values.append(true_value)
        pickem_total += 1
        case_study.iloc[row_id] = [player_name, metric_name, metric_value, predict_value, true_value]
        row_id += 1
        if correct:
            pickem_correct += 1
print(f"Pick'em accuracy: {pickem_correct} out of {pickem_total}")
print(mean_absolute_percentage_error(predict_values, true_values),
      mean_absolute_percentage_error(metric_values, true_values))
print(mean_squared_error(predict_values, true_values, squared=False),
      mean_squared_error(metric_values, true_values, squared=False))
print(mean_absolute_error(predict_values, true_values),
      mean_absolute_error(metric_values, true_values))

pass