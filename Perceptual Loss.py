import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import save_image

class StyleLoss:
    def __init__(self, target, device, styleW = 100000, contentW = 1):
        self.model=models.vgg19(pretrained=True).features[:29]
        self.req_features= [0,5,10,19,28] 
        self.device = device
        self.target_grams, self.target_features = self.feature_extractor(target)
        self.style_weight = styleW
        self.content_weight = contentW

    def feature_extractor(self, input):
        grams = []
        features = []
        x = input
        for name, layer in enumerate(self.model.children()):
            x = layer(x)
            if name in self.req_features:
                y = self.gram_matrix(x)
                grams.append(y)
                features.append(x)
        return grams, features

    def gram_matrix(self, input):
        B, C, H, W = input.size()
        features = input.view(B, C, H*W)
        gram = torch.bmm(features, features.permute(0,2,1))
        return gram.div(C * H * W)
    
    def find_style(self, input):
        B, C, H, W = input.size()
        grams, features = self.feature_extractor(input)
        loss = torch.zeros((B))
        #for gram, target_gram in zip(features, self.target_gram):
        for i in range(len(self.target_grams)):   
            style_loss = torch.mean((grams[i]-self.target_grams[i]) ** 2, dim=(1,2))
            content_loss = torch.mean((features[i]-self.target_features[i]) ** 2, dim=(1,2,3))
            
            loss += style_loss*self.style_weight + content_loss*self.content_weight
         
        return torch.argmin(loss), loss
            