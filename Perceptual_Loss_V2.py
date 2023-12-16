import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import save_image

class PerceptualLoss:
    def __init__(self, target, device):
        self.model=models.vgg19(pretrained=True).features[:29]
        self.req_features= [0,5,10,19,28] 
        self.device = device
        self.target_grams, self.target_features = self.feature_extractor(target)

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
    
    def find_style(self, input, styleW = 1, contentW = 0):
        B, C, H, W = input.size()
        grams, features = self.feature_extractor(input)
        loss = torch.zeros((B))
        #for gram, target_gram in zip(features, self.target_gram):
        for i in range(len(self.target_grams)):   
            style_loss = torch.mean((grams[i]-self.target_grams[i]) ** 2, dim=(1,2))
            content_loss = torch.mean((features[i]-self.target_features[i]) ** 2, dim=(1,2,3))
            
            loss += style_loss*styleW + content_loss*contentW
        return torch.argmin(loss), loss
    
class ContentLoss:
    def __init__(self, target, device):
        self.model=models.vgg19(pretrained=True).features[:29]
        self.req_features= [0,5,10,19,28] 
        self.device = device
        self.target_features = self.feature_extractor(target)

    def feature_extractor(self, input):
        features = []
        x = input
        for name, layer in enumerate(self.model.children()):
            x = layer(x)
            if name in self.req_features:
                features.append(x)
        return features
    
    def forward(self, input):
        B, C, H, W = input.size()
        features = self.feature_extractor(input)
        content_loss = torch.zeros((B))
        for i in range(len(self.target_features)):  
            # print(features[i].size())
            # print(self.target_features[i].size())
            a = (features[i]-self.target_features[i]) ** 2
            content_loss += torch.mean(a, dim=(1,2,3))
        return content_loss
     
class StyleLoss:
    def __init__(self, target, device):
        self.model=models.vgg19(pretrained=True).features[:29]
        self.req_features= [0,5,10,19,28] 
        self.device = device
        self.target_grams = self.feature_extractor(target)

    def gram_matrix(self, input):
        B, C, H, W = input.size()
        features = input.view(B, C, H*W)
        gram = torch.bmm(features, features.permute(0,2,1))
        return gram.div(C * H * W)
    
    def feature_extractor(self, input):
        grams = []
        x = input
        for name, layer in enumerate(self.model.children()):
            x = layer(x)
            if name in self.req_features:
                y = self.gram_matrix(x)
                grams.append(y)
        return grams

    def forward(self, input):
        B, C, H, W = input.size()
        grams = self.feature_extractor(input)
        style_loss = torch.zeros((B))
        for i in range(len(self.target_grams)):   
            style_loss += torch.mean((grams[i]-self.target_grams[i]) ** 2, dim=(1,2))
        return style_loss
    
class CustomPerceptualLoss:
    def __init__(self, style_target, device):
        self.device = device
        # self.content_loss = ContentLoss(content_target, device)
        self.style_loss = StyleLoss(style_target, device)
    
    def forward(self, input,  styleW = 1, contentW = 0):
        # content_loss = self.content_loss.forward(input).detach()
        style_loss = self.style_loss.forward(input).detach()

        # loss = style_loss*styleW + content_loss*contentW
        loss = style_loss
        print(style_loss*styleW)
        return torch.argmin(loss), loss