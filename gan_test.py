import utils.datasets as ds
import models.gan
import json

model = models.gan.GAN(
    num_features=44, num_epochs=500, normalize=True,
    debug=True, latent_vector_size=9,
    batch_size=1000, ns_param=0.01, adpt_l=0,
    res_depth=1, dr_param=1, batch_param=0.02,
    display_step=10, learning_rate=0.03,
    reg_param=0.01
)

print('hola')

exploit1 = 'UNSW_NB15'
exploit2 = 'Service_Scan'
# exploit = 'nginx_keyleak'
# exploit = 'nginx_rootdir'

data = []

# for i in range(12):
#     trX, trY = ds.load_data(
#         (
#             './data/service_scan_dataset/training/Service_Scan-{}.csv'
#         ).format(i+1)
#     )
#
#     model.train(trX, trY)
#
#     for i in range(3):
#         teX, teY = ds.load_data(
#             (
#                 './data/service_scan_dataset/testing/Service_Scan-{}.csv'
#             ).format(i+12)
#         )
#
#         data.append(model.test(teX, teY))
trX, trY = ds.load_data(
    (
        # './data/{}.csv'
        './data/{}_training-set.csv'
    ).format(exploit1)
)

model.train(trX, trY)


teX, teY = ds.load_data(
    (
        './data/{}_testing-set.csv'
    ).format(exploit1)
)

data.append(model.test(teX, teY))

with open('results/{}.json'.format(exploit1), 'w') as f:
    json.dump(data, f, indent=2)
