import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import torch
import logging
logger = logging.getLogger("logger")
from utils.image_helper import ImageHelper
import yaml
import time
import datetime



def eval_model(model, helper, is_poison=False):
    model.eval()

    correct = 0
    dataset_size = 0
    if is_poison:
        poison_data_count = 0
        data_iterator = helper.test_data_poison
        for batch_id, batch in enumerate(data_iterator):
            data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=-1, evaluation=True)
            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data)
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
        acc = 100.0 * (float(correct) / float(poison_data_count))  if poison_data_count!=0 else 0
        return acc
    else:
        data_iterator = helper.test_data
        for batch_id, batch in enumerate(data_iterator):
            data, targets = helper.get_batch(data_iterator, batch, evaluation=True)
            dataset_size += len(data)
            output = model(data)
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
        acc = 100.0 * (float(correct) / float(dataset_size))  if dataset_size!=0 else 0
        return acc


def dp_noise(param, sigma):
    noised_layer = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)
    return noised_layer

def smooth_model(target_model, sigma):
    for name, param in target_model.state_dict().items():
        param.add_(dp_noise(param, sigma))


def probability_estimation(labs,model_preds):

    k_as=[]
    p_as=[]
    k_bs=[]
    p_bs=[]
    is_acc=[]
    labs = np.array(labs) # [num_samples]   
    model_preds= np.transpose(np.array(model_preds)) 

    for i in range(len(model_preds)): # over samples 
        line = model_preds[i]
        bincount= np.bincount(line)
        k_a = np.argmax(bincount) 
        is_acc.append(int(k_a == labs[i]))
        
        p_a = bincount[k_a]*1.0/line.shape[0]

        k_as.append(k_a)
        p_as.append(p_a)
        bincount[k_a] = 0# set the max to be the 0
        k_b = np.argmax(bincount)
        p_b = bincount[k_b]*1.0/line.shape[0]
        # print("b", k_b, p_b) # runnerup
        k_bs.append(k_b)
        p_bs.append(p_b)
    
 
    p_as=np.array(p_as) # [num_samples]  
    p_bs=np.array(p_bs) # [num_samples]

    return p_as, p_bs, is_acc



def certificate_over_model(model, model_fname, helper, N_m, sigma):
   

    # evaluate the base model
    model.load_state_dict(torch.load(model_fname)['state_dict']) # reload  the model     
    base_acc_benign = eval_model(model, helper, is_poison=False)
    base_acc_poison = eval_model(model, helper, is_poison=True) #the lable is poisoned 
    logger.info ("BASE model: Benign/Poison ACC %.4f/%.4f"%(base_acc_benign, base_acc_poison))

    acc_benign_list = [] 
    acc_poison_list = []

    model_preds = []
    model_preds_poison = []
    labs = []
    
    for _ in range(0,N_m): # loop over the smoothed model
        if (_ == 0): # the first model :
            start_time = time.time()
        # smooth the model and get the acc/asr
        model.load_state_dict(torch.load(model_fname)['state_dict']) # reload  the model 

        smooth_model(model,sigma)
        acc_benign = eval_model(model, helper, is_poison=False)
        acc_poison = eval_model(model, helper, is_poison=True) #the lable is poisoned 
        logger.info ("SMOOTH model - %d: Benign/Poison ACC %.4f/%.4f"%(_,acc_benign, acc_poison))
        acc_benign_list.append(acc_benign)
        acc_poison_list.append(acc_poison)
        
        data_iterator = helper.test_data 
        all_pred =  np.empty((0), int)
        all_pred_poison =  np.empty((0), int)
      
        
        for batch_id, batch in enumerate(data_iterator):
            if batch_id>50: # use part of the test dataset to save time
                break
            # clean input and clean label 
            data, targets = helper.get_batch(data_iterator, batch, evaluation=True)
            output = model(data)
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            
            if (_ == 0): # the first model 
                labs = labs + list(targets.cpu().numpy())
            all_pred = np.concatenate([all_pred, pred.cpu()], axis=0)

            # backdoor input and clean label  
            for index in range(0, len(data)):
                data[index] = helper.add_pixel_pattern(data[index],-1)

            output = model(data)
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            all_pred_poison = np.concatenate([all_pred_poison, pred.cpu()], axis=0)


        model_preds.append(all_pred) # all_pred: [num_samples, 1] ;  # model_preds: [num_models]
        model_preds_poison.append(all_pred_poison)
        if (_ == 0): # the first model :
            logger.info(f'One Smoothed Model Done in {time.time() - start_time} sec.')
        
    
    logger.info ("SMOOTH model - avg: Benign/Poison ACC %.4f/%.4f"%( sum(acc_benign_list) / len(acc_benign_list) ,   sum(acc_poison_list) / len(acc_poison_list)))
    logger.info ("BASE models: Benign/Poison ACC %.4f/%.4f"%(base_acc_benign, base_acc_poison))
    
    pa_exp, pb_exp, is_acc =  probability_estimation(labs, model_preds)
    pa_exp_poison, pb_exp_poison, is_acc_poison =  probability_estimation(labs, model_preds_poison)
    return  pa_exp, pb_exp, is_acc, pa_exp_poison, pb_exp_poison, is_acc_poison


 
if __name__ == '__main__':
   
    # load parameters for testing
    with open(f'./configs/mnist_smooth_params.yaml', 'r') as f:
        params_loaded = yaml.load(f)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    helper = ImageHelper(current_time=current_time, params=params_loaded,
                                name=params_loaded.get('name', 'mnist'))
    
  
    foldername= helper.params['smoothed_fname'].split('/')
    epoch = int(foldername[-1].split('_')[-1])
    foldername = os.path.join(foldername[0],foldername[1])
    
    # load parameters from training
    training_param_fname= os.path.join(foldername,'params.yaml')
    with open(training_param_fname, 'r') as f:
        training_params = yaml.load(f)

    helper.params['poison_delta']=training_params['poison_delta']
    helper.params['poison_pattern']=training_params['poison_pattern']
    helper.params['poison_label_swap']==training_params['poison_label_swap']
    helper.params['is_poison']==training_params['is_poison']
    helper.params['adversary_list']==training_params['adversary_list']
    


    helper.load_data()
    helper.create_model()
    # save current parameters:
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(helper.params, f)

    # Calculate the expectation and bound of p_A and p_B
    start_time = time.time()
    pa_exp, pb_exp, is_acc, pa_exp_poison, pb_exp_poison, is_acc_poison = certificate_over_model(helper.target_model, helper.params['smoothed_fname'], helper, helper.params['N_m'], helper.params['test_sigma'])
    
    # prepare output file
    
    output_fname = os.path.join(foldername, "pred_clean_Epoch%dM%dSigma%.4f.txt"%(epoch, helper.params['N_m'], helper.params['test_sigma']))
    f = open(output_fname, 'w')
    print("idx\tpa_exp\tpb_exp\tis_acc", file=f, flush=True)

    for i in range(len(pa_exp)): # len of test data set
        print("{}\t{}\t{}\t{}".format(i, pa_exp[i], pb_exp[i], is_acc[i]), file=f, flush=True)
    
    logger.info("is_acc for clean data-clean label %.4f "% (float(sum(is_acc)) / len(is_acc)))
    f.close()
    logger.info("save to %s"% output_fname)


    # prepare output file
    output_fname = os.path.join(foldername, "pred_poison_Epoch%dM%dSigma%.4f.txt"%(epoch, helper.params['N_m'], helper.params['test_sigma']))
    f = open(output_fname, 'w')
    print("idx\tpa_exp\tpb_exp\tis_acc", file=f, flush=True)

    for i in range(len(pa_exp_poison)): # len of test data set
        print("{}\t{}\t{}\t{}".format(i, pa_exp_poison[i], pb_exp_poison[i], is_acc_poison[i]), file=f, flush=True)
    
    logger.info("is_acc for poison data-clean label %.4f "% (float(sum(is_acc_poison)) / len(is_acc_poison)))
    f.close()
    logger.info("save to %s"% output_fname)


    logger.info(f'Done in {time.time() - start_time} sec.')