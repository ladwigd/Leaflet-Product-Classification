from os import listdir
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from copy import deepcopy
import random
import json
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import json
import datetime


class productRecognitionImageModel:
    
    def __init__(self):
        self.__set_random_seed(1)
        self.__set_random_state()
        
    def __set_random_seed(self, seed_num = 1):
        self.seed_num = 1
        
    def __set_random_state(self):
        SEED = self.seed_num
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        g = torch.Generator()
        g.manual_seed(SEED)
        self.g = g
        
    def __seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            numpy.random.seed(worker_seed)
            random.seed(worker_seed)
            
    def set_configuration(self, configuration = None, hParam_configuration=None):
        self.RUN_COMMENT = "Resnet50: 256, jitter 0.5, optimizer momentum 0.95"
    
        if configuration == None and hParam_configuration == None:
            configuration = {
                "NUMBER_EPOCHS": 12,

                "NUMBER_CLASSES": 832,
                "LEARNING_RATE": 0.001,
                "BATCH_SIZE":16,
                "PARAM_REQUIRES_GRAD": True, # TRUE -> don't freeze layers
                "OPTIMIZER_MOMENTUM": 0.95,
                "IMAGE_RESIZE_SIZE": (256,256),
                "CENTERCROP_SIZE": 224,

                "FINAL_LAYER_INPUT_FEATURES": 2048,
                "FINAL_LAYER_OUTPUT_FEATURES": 1024
            }

            hParam_configuration = deepcopy(configuration)
            hParam_configuration["IMAGE_RESIZE_SIZE_HEIGHT"] = configuration["IMAGE_RESIZE_SIZE"][0]
            hParam_configuration["IMAGE_RESIZE_SIZE_WIDTH"]= configuration["IMAGE_RESIZE_SIZE"][1]
            del hParam_configuration["IMAGE_RESIZE_SIZE"]
        self.configuration = configuration
        self.hParam_configuration = hParam_configuration
      
    def initialize_dataloader(self, train_data_path, test_data_path, val_data_path):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # imagenet values

        self.data_transforms = {
            'train':
            transforms.Compose([
                transforms.Resize(self.configuration["IMAGE_RESIZE_SIZE"]),
                transforms.ColorJitter(saturation=0.5),
                transforms.ToTensor(),
                normalize
            ]),
            'validation':
            transforms.Compose([
                transforms.Resize(self.configuration["IMAGE_RESIZE_SIZE"]),
                transforms.ToTensor(),
                normalize
            ]),
            'test':
            transforms.Compose([
                transforms.Resize(self.configuration["IMAGE_RESIZE_SIZE"]),
                transforms.ToTensor(),
                normalize
            ]),
        }

        self.image_datasets = {
            'train': 
            datasets.ImageFolder(train_data_path, self.data_transforms['train']),
            'test': 
            datasets.ImageFolder(test_data_path, self.data_transforms['test']),
            'validation':
            datasets.ImageFolder(val_data_path, self.data_transforms['validation'])
        }



        self.dataloaders = {
            'train':
            torch.utils.data.DataLoader(self.image_datasets['train'],
                                        batch_size=self.configuration["BATCH_SIZE"],
                                        shuffle=True,
                                        worker_init_fn=self.__seed_worker,
                                        generator=self.g),
            'validation':
            torch.utils.data.DataLoader(self.image_datasets["validation"],
                                        batch_size=self.configuration["BATCH_SIZE"],
                                        shuffle=False,
                                        worker_init_fn=self.__seed_worker,
                                        generator=self.g),
            'test':
            torch.utils.data.DataLoader(self.image_datasets["test"],
                                        batch_size=1,
                                        shuffle=False,
                                        worker_init_fn=self.__seed_worker,
                                        generator=self.g),
        }
        
    def initialize_dataloader_no_val(self, train_data_path, test_data_path):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # imagenet values

        self.data_transforms = {
            'train':
            transforms.Compose([
                transforms.Resize(self.configuration["IMAGE_RESIZE_SIZE"]),
                transforms.ColorJitter(saturation=0.5),
                transforms.ToTensor(),
                normalize
            ]),
            'test':
            transforms.Compose([
                transforms.Resize(self.configuration["IMAGE_RESIZE_SIZE"]),
                transforms.ToTensor(),
                normalize
            ]),
        }

        self.image_datasets = {
            'train': 
            datasets.ImageFolder(train_data_path, self.data_transforms['train']),
            'test': 
            datasets.ImageFolder(test_data_path, self.data_transforms['test'])
        }



        self.dataloaders = {
            'train':
            torch.utils.data.DataLoader(self.image_datasets['train'],
                                        batch_size=self.configuration["BATCH_SIZE"],
                                        shuffle=True,
                                        worker_init_fn=self.__seed_worker,
                                        generator=self.g),
            'test':
            torch.utils.data.DataLoader(self.image_datasets["test"],
                                        batch_size=1,
                                        shuffle=False,
                                        worker_init_fn=self.__seed_worker,
                                        generator=self.g),
        }
        
    def initialize_model(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)
        self.model = models.resnet50(pretrained=True).to(self.device)
        for param in self.model.parameters():
            param.requires_grad = self.configuration["PARAM_REQUIRES_GRAD"]  #freeze layers true/false
        
        #replace fc
        self.model.fc = nn.Sequential(
            nn.Linear(self.configuration["FINAL_LAYER_INPUT_FEATURES"], self.configuration["FINAL_LAYER_OUTPUT_FEATURES"]),
            nn.ReLU(inplace=True),
            nn.Linear(self.configuration["FINAL_LAYER_OUTPUT_FEATURES"], self.configuration["NUMBER_CLASSES"])).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.configuration["LEARNING_RATE"], momentum=self.configuration["OPTIMIZER_MOMENTUM"])
    
    def train_model(self, num_epochs=3):
        writer = SummaryWriter()
        
        try:
            best_valAccuracy = 0
            for epoch in range(num_epochs):
                since = time.time()
                print('Epoch {}/{}'.format(epoch+1, num_epochs))
                print('-' * 10)

                for phase in ['train', 'validation']:
                    if phase == 'train':
                        self.model.train()
                    else:
                        self.model.eval()

                    running_loss = 0.0
                    running_corrects = 0

                    for inputs, labels in self.dataloaders[phase]:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                        _, preds = torch.max(outputs, 1)
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                    epoch_loss = running_loss / len(self.image_datasets[phase])
                    epoch_acc = running_corrects.double() / len(self.image_datasets[phase])

                    if phase == 'validation' and epoch_acc > best_valAccuracy:
                        best_valAccuracy = epoch_acc
                        torch.save(self.model.state_dict(), '/weights/current_run_weights.h5')

                    writer.add_scalar("Loss " + phase, epoch_loss, epoch)
                    writer.add_scalar("Accuracy " + phase, epoch_acc, epoch)

                    print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                                epoch_loss,
                                                                epoch_acc))
                    time_elapsed = time.time() - since
                    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            writer.add_hparams(self.hParam_configuration, {"hparam/best_val_accuracy": best_valAccuracy})
            writer.add_text("comment", self.RUN_COMMENT)
            writer.flush()
            writer.close()
            return self.model
        except KeyboardInterrupt:
            torch.save(self.model.state_dict(), '/weights/weights_interrupt.h5')
            writer.add_hparams(self.hParam_configuration, {"hparam/best_val_accuracy": best_valAccuracy})
            writer.add_text("comment", self.RUN_COMMENT)
            writer.flush()
            writer.close()
            return self.model
        
    def load_best_epoch_weights(self):
        self.model.load_state_dict(torch.load('../src/models/weights/current_run_weights.h5'))
    
    def predict_one_image(self, image_path):
        sample_image_path = image_path
        sample_image = Image.open(sample_image_path)
        sample_image = self.data_transforms["test"](sample_image)
        outputs = self.model(sample_image[None, ...].to(self.device))
        proba, pred = torch.max(outputs, 1)
        percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        return (percentage[pred[0]].item(), pred.item())
    
    def predict_from_dataloader(self, dataloader):
        results = []     
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            proba, pred = torch.max(outputs, 1)
            # separate batches of tensor ouput to use softmax
            for count, i in enumerate(torch.tensor_split(outputs, outputs.shape[0])):
                percentage = torch.nn.functional.softmax(i, dim=1)[0] * 100
                results.append((percentage[pred[count]].item(), pred.tolist()[count]))
        return results

    def predict_from_dataloader_probas(self, dataloader):
        results = []
        probas = []
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            proba, pred = torch.max(outputs, 1)
            # separate batches of tensor ouput to use softmax
            for count, i in enumerate(torch.tensor_split(outputs, outputs.shape[0])):
                percentage = torch.nn.functional.softmax(i, dim=1)[0] * 100
                probas.append(percentage.tolist())
                results.append((percentage[pred[count]].item(), pred.tolist()[count]))
        return results, probas
    
    @staticmethod
    def get_real_class_from_dataloader(dataloader, model_class):
        dataloader_dict = dataloader.dataset.class_to_idx
        return list(dataloader_dict.keys())[list(dataloader_dict.values()).index(model_class)]



