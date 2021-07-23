import torch
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

AGGR_MEAN_PARAM = 'mean_param'
AGGR_GEO_MED = 'geom_median'

MAX_UPDATE_NORM = 1000  # reject all updates larger than this amount
geom_median_maxiter = 10

TYPE_LOAN='loan'
TYPE_MNIST='mnist'
TYPE_EMNIST='emnist'