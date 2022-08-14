import pandas as pd

import utils.datasets as ds
import models.gan
import json
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

hyperparameters = dict(
    num_features=72, num_epochs=100, normalize=True,
    debug=True, latent_vector_size=100,
    batch_size=20000, ns_param=0.5, adpt_l=0,
    res_depth=3, dr_param=1, batch_param=1e-2,
    display_step=10, d_learning_rate=1e-2,
    reg_param=1e-3, g_learning_rate=1e-4,
)

model = models.gan.GAN(**hyperparameters)


def get_summary(data):
    out = {}

    mats = np.array([entry['confusion_matrix'] for entry in data])

    out['avg_acc'] = np.mean([entry['accuracy'] for entry in data])
    out['std_acc'] = np.std([entry['accuracy'] for entry in data])
    out['avg_mat'] = np.mean(mats, axis=0).tolist()
    out['std_mat'] = np.std(mats, axis=0).tolist()

    return out


dirs = [el for el in os.listdir('results') if 'trial_' in el]
trials = [int(el.split('_')[1]) for el in dirs]
trials.insert(0, -1)

trial = np.max(trials) + 1
print('Trial number: ' + str(trial))
os.mkdir('results/trial_{}'.format(trial))

# exploits = ['Service_Scan']
exploits = ['UNSW-NB15-modified']
summaries = {'hyperparameters': hyperparameters}
raw_data = []

for exploit in exploits:
    data = []
    top_11_proto, top_10_state = ds.concat_dataset()

    for i in range(1):

        trX, trY = ds.load_data(top_11_proto, top_10_state,
            (
                'UNSW_NB15_training-set.csv'
                # '{}_{}.csv'
                # './data/{}_training-set-39.csv'
            )
            # ).format(exploit, i+1)
        )

        model.train(trX, trY)

        for i in range(1):
            teX, teY = ds.load_data(top_11_proto, top_10_state,
                (
                    'UNSW_NB15_testing-set.csv'
                    # '{}_{}.csv'
                    # './data/{}_testing-set-39.csv'
                )
                # ).format(exploit)
            )

            d = model.test(teX, teY)
            data.append(d)
            raw_data.append(d)

    summaries[exploit] = get_summary(data)

    with open('results/trial_{}/{}.json'.format(trial, exploit), 'w') as f:
        json.dump(data, f, indent=2)

summaries['net'] = get_summary(raw_data)

with open('results/trial_{}/summary.json'.format(trial), 'w') as f:
    json.dump(summaries, f, indent=2)

print('Output in results/trial_{}'.format(trial))

accs = [entry['accuracy'] for entry in data]
cm = [entry['confusion_matrix'] for entry in data]
ax = plt.subplot()
sns.heatmap(np.reshape(cm, (2, -1)), annot=True, fmt='g', ax=ax, cmap='Blues');  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['normal', 'attack'])
ax.yaxis.set_ticklabels(['normal', 'attack'])
# plt.matshow(cm)
# plt.title('Confusion matrix of the classifier')
# plt.colorbar()
plt.savefig(f'graph/confusion_matrix_{trial}_{accs[0]:.4f}.png')
plt.close()
