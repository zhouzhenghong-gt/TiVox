_base_ = './dsa_default.py'


expname = 'args_no_use' # no use
basedir = './logs/dsa_real'

data = dict(
    datadir='/data/zhenghongzhou/repo/DirectVoxGO-torch/data/dsa/real',
    dataset_type='dsa_real',
    white_bkgd=True,
) 