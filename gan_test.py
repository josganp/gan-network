import utils.datasets as ds
import models.gan
import json

model = models.gan.GAN(
    num_features=44, num_epochs=100, normalize=True,
    debug=True, latent_vector_size=100,
    batch_size=1000, ns_param=0., adpt_l=0,
    res_depth=1, dr_param=1, batch_param=0.,
    display_step=10, learning_rate=0.005,
    reg_param=0.01
)

print('hola')

exploit = 'UNSW_NB15'
# exploit = 'nginx_keyleak'
# exploit = 'nginx_rootdir'

data = []


trX, trY = ds.load_data(
    (
        './data/{}_training-set.csv'
    ).format(exploit)
)

model.train(trX, trY)


teX, teY = ds.load_data(
    (
        './data/{}_testing-set.csv'
    ).format(exploit)
)

data.append(model.test(teX, teY))

with open('results/{}.json'.format(exploit), 'w') as f:
    json.dump(data, f, indent=2)
