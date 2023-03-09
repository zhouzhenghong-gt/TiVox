_base_  = './dsa_default.py'

expname = 'dsa_1grid-220-60-512'
basedir = './logs/dsa_idea_1123'

data = dict(
    datadir='/data/zhenghongzhou/repo/DirectVoxGO-torch/data/dsa/idea_512_1123',
    dataset_type='dsa_idea',
    white_bkgd=True,
    train_num=70,
) #dsa_4grid-20k