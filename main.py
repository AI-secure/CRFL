import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import json
import datetime
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import time
import numpy as np
import random
import config
import copy

import train
import test
from utils.image_helper import ImageHelper
from utils.loan_helper import LoanHelper
import utils.csv_record as csv_record


logger = logging.getLogger("logger")
# set random seeds
torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)
np.random.seed(1)


if __name__ == '__main__':

    
    # load hyperparameters
    parser = argparse.ArgumentParser(description='CRFL')
    parser.add_argument('--params', description='params file to be loaded')
    args = parser.parse_args()
    with open(f'./{args.params}', 'r') as f:
        params_loaded = yaml.load(f)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')

    # load data
    if params_loaded['type'] == config.TYPE_LOAN:
        helper = LoanHelper(current_time=current_time, params=params_loaded,
                            name=params_loaded.get('name', 'loan'))
        helper.load_data(params_loaded)
  
    elif params_loaded['type'] == config.TYPE_MNIST:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'mnist'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_EMNIST:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'emnist'))
        helper.load_data()
  
    else:
        helper = None

    logger.info(f'load data done')

    # create model 
    helper.create_model()

    # save parameters:
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(helper.params, f)


    if helper.params['is_poison']:
        logger.info(f"Poisoned following participants: {(helper.params['adversary_list'])}")

    for epoch in range(helper.start_epoch, helper.params['epochs'] + 1):
        
        start_time = time.time()
        t = time.time()
        agent_name_keys = helper.participants_list

        # update local models
        submit_params_update_dict, num_samples_dict = train.FLtrain(
            helper=helper,
            start_epoch=epoch,
            local_model=helper.local_model,
            target_model=helper.target_model,
            is_poison=helper.params['is_poison'],
            agent_name_keys=agent_name_keys)
     
        is_updated = True
        # sever aggregation
        if helper.params['aggregation_methods'] == config.AGGR_MEAN_PARAM:
            helper.average_models_params(submit_params_update_dict,agent_name_keys,target_model=helper.target_model)

        elif helper.params['aggregation_methods'] == config.AGGR_GEO_MED:
            maxiter = config.geom_median_maxiter
            num_oracle_calls, is_updated, names, weights, alphas = helper.geometric_median_update(helper.target_model, submit_params_update_dict,agent_name_keys,num_samples_dict, maxiter=maxiter)
        
        # clip the global model
        if params_loaded['type'] == config.TYPE_MNIST:
            dynamic_thres= epoch *0.1+2 
        elif params_loaded['type'] == config.TYPE_LOAN:
            dynamic_thres = epoch*0.025+2
        elif params_loaded['type'] == config.TYPE_EMNIST:
            dynamic_thres= epoch*0.25+4
        param_clip_thres =  helper.params["param_clip_thres"]
        if dynamic_thres < param_clip_thres: 
            param_clip_thres= dynamic_thres

        current_norm = helper.clip_weight_norm(helper.target_model, param_clip_thres )
        csv_record.add_norm_result(current_norm)

        # test acc after clipping
        epoch_loss, epoch_acc, epoch_corret, epoch_total = test.clean_test(helper=helper, epoch=epoch,
                                                                       model=helper.target_model, is_poison=False,
                                                                       visualize=True, agent_name_key="global")
        csv_record.test_result.append(["global", epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])
        csv_record.add_global_acc_result(epoch_acc)

        if helper.params['is_poison'] and epoch >= helper.params['poison_epochs'][0]:
            epoch_loss, epoch_acc_p, epoch_corret, epoch_total = test.adv_test(helper=helper,
                                                                                    epoch=epoch,
                                                                                    model=helper.target_model,
                                                                                    is_poison=True,
                                                                                    visualize=True,
                                                                                    agent_name_key="global")

            csv_record.posiontest_result.append(
                ["global", epoch, epoch_loss, epoch_acc_p, epoch_corret, epoch_total])

        # save model 
        helper.save_model(epoch=epoch, val_loss=epoch_loss)
        # save csv file
        csv_record.save_result_csv(epoch, helper.folder_path)
       
        # add noise
        logger.info(f" epoch: {epoch} add noise on the global model!")
        for name, param in helper.target_model.state_dict().items():
            param.add_(helper.dp_noise(param, helper.params['sigma_param']))

    logger.info(f"This run has a label: {helper.params['current_time']}. ")



