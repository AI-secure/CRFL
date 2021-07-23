import utils.csv_record as csv_record
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import main
import test
import copy
import config

def FLtrain(helper, start_epoch, local_model, target_model, is_poison,agent_name_keys):
    submit_params_update_dict = dict()
    num_samples_dict = dict()


    for model_id in range(helper.params['num_models']):
    
        agent_name_key = agent_name_keys[model_id]
        ## Synchronize LR and models
        model = local_model
        model.copy_params(target_model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'])
        model.train()

        localmodel_poison_epochs = helper.params['poison_epochs']
        AGENT_POISON_AT_THIS_ROUND = False
        epoch = start_epoch

        if is_poison and agent_name_key in helper.params['adversary_list'] and (epoch in localmodel_poison_epochs):
            AGENT_POISON_AT_THIS_ROUND = True
            main.logger.info(f'poison local model {agent_name_key} ')



        
        target_params = dict()
        for name, param in target_model.named_parameters():
            target_params[name] = target_model.state_dict()[name].clone().detach().requires_grad_(False)
       
        temp_local_epoch = (epoch - 1) * helper.params['internal_epochs']
        for internal_epoch in range(1, helper.params['internal_epochs'] + 1):
            temp_local_epoch += 1

            if helper.params['type'] == config.TYPE_LOAN:
                data_iterator = helper.statehelper_dic[agent_name_key].get_trainloader()
            else:
                _, data_iterator = helper.train_data[agent_name_key]
            total_loss = 0.
            correct = 0
            dataset_size = 0
            poison_data_count = 0 

            model.train()
            for batch_id, batch in enumerate(data_iterator):
          
                optimizer.zero_grad()

                if helper.params['type'] == config.TYPE_LOAN:
                    if AGENT_POISON_AT_THIS_ROUND:
                        data, targets, poison_num = helper.statehelper_dic[agent_name_key].get_poison_batch(batch, feature_dict=helper.feature_dict,evaluation=False)
                        poison_data_count+= poison_num
                    else: 
                        data, targets = helper.statehelper_dic[agent_name_key].get_batch(data_iterator, batch,evaluation=False)
                else:
                    if AGENT_POISON_AT_THIS_ROUND:
                        data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=-1,evaluation=False)
                        poison_data_count+= poison_num
                    else: 
                        data, targets = helper.get_batch(data_iterator, batch,evaluation=False)

                
                dataset_size += len(data)
                output = model(data)
              
                loss = nn.functional.cross_entropy(output, targets)
                loss.backward()

                optimizer.step()
                total_loss += loss.data
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
             
              
            acc = 100.0 * (float(correct) / float(dataset_size))
            total_l = total_loss / dataset_size
       

            if AGENT_POISON_AT_THIS_ROUND:
                main.logger.info(
                '___PoisonTrain {} ,  epoch {:3d}, local model {}, internal_epoch {:3d},  Average loss: {:.4f}, '
                'Accuracy: {}/{} ({:.4f}%), train_poison_data_count: {}'.format(model.name, epoch, agent_name_key,
                                                                                internal_epoch,
                                                                                total_l, correct, dataset_size,
                                                                                acc, poison_data_count))
                
            else:
                main.logger.info(
                    '___Train {},  epoch {:3d}, local model {}, internal_epoch {:3d},  Average loss: {:.4f}, '
                    'Accuracy: {}/{} ({:.4f}%)'.format(model.name, epoch, agent_name_key, internal_epoch,
                                                        total_l, correct, dataset_size,
                                                        acc))
            csv_record.train_result.append([agent_name_key, temp_local_epoch,
                                            epoch, internal_epoch, total_l.item(), acc, correct, dataset_size])

        
            num_samples_dict[agent_name_key] = dataset_size

            
        # scale: no matter poisoning or not
        if  agent_name_key in helper.params['adversary_list'] and (epoch in localmodel_poison_epochs):
            main.logger.info("scaled!!")
            for name, data in model.state_dict().items():
                new_value = target_params[name] + (data - target_params[name]) * helper.params['scale_factor']
                model.state_dict()[name].copy_(new_value)
            
    
        # test local model after internal epochs are finished
        
        epoch_loss, epoch_acc, epoch_corret, epoch_total = test.clean_test(helper=helper, epoch=epoch,
                                                                    model=model, is_poison=AGENT_POISON_AT_THIS_ROUND, visualize=True,
                                                                    agent_name_key=agent_name_key)
        csv_record.test_result.append([agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

     
        
        if AGENT_POISON_AT_THIS_ROUND:
            epoch_loss, epoch_acc, epoch_corret, epoch_total = test.adv_test(helper=helper,
                                                                                    epoch=epoch,
                                                                                    model=model,
                                                                                    is_poison=AGENT_POISON_AT_THIS_ROUND,
                                                                                    visualize=True,
                                                                                    agent_name_key=agent_name_key)
            csv_record.posiontest_result.append(
                [agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

        # update the model params
        client_pramas_update = dict()
        for name, data in model.state_dict().items():
            client_pramas_update[name] = torch.zeros_like(data)
            client_pramas_update[name] = (data - target_params[name])
        
        submit_params_update_dict[agent_name_key] = client_pramas_update
        

    return submit_params_update_dict, num_samples_dict
