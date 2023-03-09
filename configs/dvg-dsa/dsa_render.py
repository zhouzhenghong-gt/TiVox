_base_ = './dsa_default.py'

expname = 'dsa-512-11.23'
basedir = './logs/dsa_render'

data = dict(
    datadir='/data/zhenghongzhou/repo/DirectVoxGO-torch/data/dsa/real',
    dataset_type='dsa_time',
    white_bkgd=True,
    train_num=70,
)