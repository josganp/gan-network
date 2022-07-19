import utils.file_ops as fops
import models.gan
import json
import numpy as np
import tensorflow as tf
import os
import plotly.plotly as py
import plotly.graph_objs as go

hyperparameters = dict(
    num_features=12, num_epochs=1000, normalize=True,
    debug=True, latent_vector_size=72,
    batch_size=1000, ns_param=.5, adpt_l=0,
    res_depth=1, dr_param=1, batch_param=1e-2,
    display_step=10, d_learning_rate=1e-3,
    reg_param=1e-3, g_learning_rate=1e-4
)

loc_name = 'general_results'
loc_str = 'results/{}/'.format(loc_name)
loc_str += '{}.json'


def get_summary(data):
    out = {}

    mats = np.array([entry['confusion_matrix'] for entry in data])

    out['avg_acc'] = np.mean([entry['accuracy'] for entry in data])
    out['std_acc'] = np.std([entry['accuracy'] for entry in data])
    out['avg_mat'] = np.mean(mats, axis=0).tolist()
    out['std_mat'] = np.std(mats, axis=0).tolist()

    return out


loc = 'results/{}'.format(loc_name)
if not os.path.exists(loc):
    os.mkdir(loc)

exploits = ['freak', 'nginx_keyleak', 'nginx_rootdir', 'caleb']

summaries = {'hyperparameters': hyperparameters}
raw_data = []
net_data = {exploit: [] for exploit in exploits}

model = models.gan.GAN(**hyperparameters)

for i in range(5):
    training_data = []

    for tr_exploit in exploits:
        training_data.append(fops.load_data(
            (
                './data/three-step/{}/subset_{}/train_set.csv'
            ).format(tr_exploit, i)
        ))

    trX = np.concatenate([d[0] for d in training_data])
    trY = np.concatenate([d[1] for d in training_data])

    model.train(trX, trY)

    for test_exploit in exploits:
        for j in range(5):
            teX, teY = fops.load_data(
                (
                    './data/three-step/{}/subset_{}/test_set.csv'
                ).format(test_exploit, j)
            )

            d = model.test(teX, teY)
            raw_data.append(d)
            net_data[test_exploit].append(d)

summaries['net'] = get_summary(raw_data)
summaries['raw_data'] = net_data

with open(loc_str.format('raw'), 'w') as f:
    json.dump(summaries, f, indent=2)

# plot data
accs = [[entry['accuracy'] for entry in net_data[ex]] for ex in exploits]

boxes = [go.Box(
    y=accs[i],
    name=exploits[i],
    boxmean='sd'
) for i in range(len(exploits))]

layout = go.Layout(
    title='ADD-GAN: Accuracy per Exploit Trained on All',
    yaxis=dict(title='Accuracy (%)')
)

fig = go.Figure(data=boxes, layout=layout)
py.plot(fig, filename='add-gan-general-results')
