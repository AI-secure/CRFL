import torch
import torch.utils.data
from utils.helper import Helper
import random
import logging
from torchvision import datasets, transforms
import numpy as np
logger = logging.getLogger("logger")
import config
from config import device
import copy
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class ImageHelper(Helper):

    def create_model(self):
        local_model=None
        target_model=None
       
        if self.params['type']==config.TYPE_MNIST:
            from models.MnistNet import MnistNet
            local_model = MnistNet(name='Local',
                                   created_time=self.params['current_time'])
            target_model = MnistNet(name='Target',
                                    created_time=self.params['current_time'])
        elif self.params['type']==config.TYPE_EMNIST:
            from models.EmnistNet import EmnistNet
            local_model = EmnistNet(name='Local',
                                    created_time=self.params['current_time'])
            target_model = EmnistNet(name='Target',
                                    created_time=self.params['current_time'])


        local_model=local_model.to(device)
        target_model=target_model.to(device)
        if self.params['resumed_model']:
            if torch.cuda.is_available() :
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}")
            else:
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}",map_location='cpu')
            target_model.load_state_dict(loaded_params['state_dict'])
            self.start_epoch = loaded_params['epoch']+1
            self.params['lr'] = loaded_params.get('lr', self.params['lr'])
            logger.info(f"Loaded parameters from saved model: LR is"
                        f" {self.params['lr']} and current epoch is {self.start_epoch}")
        else:
            self.start_epoch = 1

        self.local_model = local_model
        self.target_model = target_model

    def build_classes_dict(self):
        classes = {}
        for ind, x in enumerate(self.train_dataset):  # for cifar: 50000; for tinyimagenet: 100000
            _, label = x
            if label in classes:
                classes[label].append(ind)
            else:
                classes[label] = [ind]
        return classes

    def poison_test_dataset(self):
        logger.info('get poison test loader')
        # delete the test data with target label
        test_classes = {}
        for ind, x in enumerate(self.test_dataset):
            _, label = x
            if label in test_classes:
                test_classes[label].append(ind)
            else:
                test_classes[label] = [ind]

        range_no_id = list(range(0, len(self.test_dataset)))
        for image_ind in test_classes[self.params['poison_label_swap']]:
            if image_ind in range_no_id:
                range_no_id.remove(image_ind)
        poison_label_inds = test_classes[self.params['poison_label_swap']]

        return torch.utils.data.DataLoader(self.test_dataset,
                           batch_size=self.params['batch_size'],
                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                               range_no_id)), \
               torch.utils.data.DataLoader(self.test_dataset,
                                            batch_size=self.params['batch_size'],
                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                poison_label_inds))
    def load_data(self):
        logger.info('Loading data')
        dataPath = "./data"
    
        if self.params['type'] == config.TYPE_MNIST:

            self.train_dataset = datasets.MNIST(dataPath, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   # transforms.Normalize((0.1307,), (0.3081,))
                               ]))
            self.test_dataset = datasets.MNIST(dataPath, train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.Normalize((0.1307,), (0.3081,))
                ]))
        
        elif self.params['type'] == config.TYPE_EMNIST:
            
            self.train_dataset = datasets.EMNIST(dataPath,split = 'bymerge', train=True, download=True, transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   # transforms.Normalize((0.1307,), (0.3081,))
                               ]))
   
        
            self.test_dataset = datasets.EMNIST(dataPath,split = 'bymerge', train=False, download=True, transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   # transforms.Normalize((0.1307,), (0.3081,))
                               ]))
          
        self.classes_dict = self.build_classes_dict()
        
    
        ## sample indices for participants that are equally
        logger.info('iid')
        all_range = list(range(len(self.train_dataset)))
        random.shuffle(all_range)
        train_loaders = [(pos, self.get_train_iid(all_range, pos))
                            for pos in range(self.params['num_models'])]

        logger.info('train loaders done')
        self.train_data = train_loaders
        self.test_data = self.get_test()
        self.test_data_poison ,self.test_targetlabel_data = self.poison_test_dataset()

        self.advasarial_namelist= []
        if self.params['is_poison']:
            self.advasarial_namelist = self.params['adversary_list']

    
        self.participants_list = list(range(self.params['num_models']))

        self.benign_namelist =list(set(self.participants_list) - set(self.advasarial_namelist))

    def get_train(self, indices):
        """
        This method is used along with Dirichlet distribution
        :param params:
        :param indices:
        :return:
        """
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               indices),pin_memory=True, num_workers=8)
        return train_loader

    def get_train_iid(self, all_range, model_no):
        """
        This method equally splits the dataset.
        :param params:
        :param all_range:
        :param model_no:
        :return:
        """

        data_len = int(len(self.train_dataset) / self.params['num_models'])
        sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               sub_indices))
        return train_loader

    def get_test(self):
        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  shuffle=False)
        return test_loader


    def get_batch(self, train_data, bptt, evaluation=False):
        data, target = bptt
        data = data.to(device)
        target = target.to(device)
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target

    def get_poison_batch(self, bptt,adversarial_index=-1, evaluation=False):

        images, targets = bptt

        poison_count= 0
        new_images=images
        new_targets=targets

        for index in range(0, len(images)):
            if evaluation: # poison all data when testing
                new_targets[index] = self.params['poison_label_swap']
                new_images[index] = self.add_pixel_pattern(images[index],adversarial_index)
                poison_count+=1

            else: # poison part of data when training
                if index < self.params['poisoning_per_batch']:
                    new_targets[index] = self.params['poison_label_swap']
                    new_images[index] = self.add_pixel_pattern(images[index],adversarial_index)
                    poison_count += 1
                else:
                    new_images[index] = images[index]
                    new_targets[index]= targets[index]

        new_images = new_images.to(device)
        new_targets = new_targets.to(device).long()
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)
        return new_images,new_targets,poison_count

    def add_pixel_pattern(self,ori_image,adversarial_index):
        image = copy.deepcopy(ori_image)
        poison_patterns= self.params['poison_pattern']
        delta =  self.params['poison_delta']

        if self.params['type'] == config.TYPE_MNIST or self.params['type'] == config.TYPE_EMNIST:
            for i in range(0, len(poison_patterns)):
                pos = poison_patterns[i]
                image[0][pos[0]][pos[1]] = min( image[0][pos[0]][pos[1]] + delta/np.sqrt(len(poison_patterns)), 1) 

        return image

