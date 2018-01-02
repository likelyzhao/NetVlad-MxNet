from easydict import EasyDict as edict

config = edict()
config.is_contiue = False
config.LOAD_EPOCH = 0
config.NUM_VLAD_CENTERS = 128
config.NUM_LABEL =500
config.LEARNING_RATE = 1
config.FEA_LEN = 512
config.MAX_SHAPE = 800
config.BATCH_SIZE = 32
config.DROP_OUT_RATIO = 0
config.MODEL_PREFIX = 'model/netvlad'
config.TRAIN = edict()
config.TRAIN.TRAIN_LIST_NAME = 'new_train.txt'
config.TRAIN.VAL_LIST_NAME = 'new_val.txt'
config.TRAIN.datapath = '/workspace/data/trainval/{0}_pool5_senet.binary'