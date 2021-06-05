import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl

class df_conv9_fc2_multi(torch.nn.Module):
    def __init__(self):
        super(df_conv9_fc2_multi, self).__init__()
        self.conv1 = torch.nn.Conv1d(2, 16, kernel_size=16, stride=1, padding=0, dilation=1)
        self.conv2 = torch.nn.Conv1d(16, 16, kernel_size=16, stride=1, padding=0, dilation=1)
        self.conv3 = torch.nn.Conv1d(16, 16, kernel_size=16, stride=1, padding=0, dilation=1)
        self.conv4 = torch.nn.Conv1d(16, 32, kernel_size=8, stride=1, padding=0, dilation=1)
        self.conv5 = torch.nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=0, dilation=1)
        self.conv6 = torch.nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=0, dilation=1)
        self.conv7 = torch.nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=0, dilation=1)
        self.conv8 = torch.nn.Conv1d(64, 64, kernel_size=4, stride=1, padding=0, dilation=1)
        self.conv9 = torch.nn.Conv1d(64, 64, kernel_size=4, stride=1, padding=0, dilation=1)
        
        self.fc1 = nn.Linear(64 * 123, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 7)
        
    def forward(self, x):
        x = F.max_pool1d(
            F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))),
            kernel_size=4, stride=4, padding=0, dilation=1
            )
        x = F.max_pool1d(
            F.relu(self.conv6(F.relu(self.conv5(F.relu(self.conv4(x)))))),
            kernel_size=4, stride=4, padding=0, dilation=1
            )
        x = F.max_pool1d(
            F.relu(self.conv9(F.relu(self.conv8(F.relu(self.conv7(x)))))),
            kernel_size=4, stride=4, padding=0, dilation=1
            )
        x = x.view(-1, self.num_flat_features(x))
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
        
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
          num_features *= s
        return num_features

class df_conv9_fc2_dropout(torch.nn.Module):
    def __init__(self):
        super(df_conv9_fc2_dropout, self).__init__()
        self.conv1 = torch.nn.Conv1d(2, 16, kernel_size=16, stride=1, padding=0, dilation=1)
        self.conv2 = torch.nn.Conv1d(16, 16, kernel_size=16, stride=1, padding=0, dilation=1)
        self.conv3 = torch.nn.Conv1d(16, 16, kernel_size=16, stride=1, padding=0, dilation=1)
        self.conv4 = torch.nn.Conv1d(16, 32, kernel_size=8, stride=1, padding=0, dilation=1)
        self.conv5 = torch.nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=0, dilation=1)
        self.conv6 = torch.nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=0, dilation=1)
        self.conv7 = torch.nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=0, dilation=1)
        self.conv8 = torch.nn.Conv1d(64, 64, kernel_size=4, stride=1, padding=0, dilation=1)
        self.conv9 = torch.nn.Conv1d(64, 64, kernel_size=4, stride=1, padding=0, dilation=1)
        
        self.fc1 = nn.Linear(64 * 123, 64)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = F.max_pool1d(
            F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))),
            kernel_size=4, stride=4, padding=0, dilation=1
            )
        x = F.max_pool1d(
            F.relu(self.conv6(F.relu(self.conv5(F.relu(self.conv4(x)))))),
            kernel_size=4, stride=4, padding=0, dilation=1
            )
        x = F.max_pool1d(
            F.relu(self.conv9(F.relu(self.conv8(F.relu(self.conv7(x)))))),
            kernel_size=4, stride=4, padding=0, dilation=1
            )
        x = x.view(-1, self.num_flat_features(x))
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
        
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
          num_features *= s
        return num_features


class df_conv9_fc2(torch.nn.Module):
    def __init__(self):
        super(df_conv9_fc2, self).__init__()
        self.conv1 = torch.nn.Conv1d(2, 16, kernel_size=16, stride=1, padding=0, dilation=1)
        self.conv2 = torch.nn.Conv1d(16, 16, kernel_size=16, stride=1, padding=0, dilation=1)
        self.conv3 = torch.nn.Conv1d(16, 16, kernel_size=16, stride=1, padding=0, dilation=1)
        self.conv4 = torch.nn.Conv1d(16, 32, kernel_size=8, stride=1, padding=0, dilation=1)
        self.conv5 = torch.nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=0, dilation=1)
        self.conv6 = torch.nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=0, dilation=1)
        self.conv7 = torch.nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=0, dilation=1)
        self.conv8 = torch.nn.Conv1d(64, 64, kernel_size=4, stride=1, padding=0, dilation=1)
        self.conv9 = torch.nn.Conv1d(64, 64, kernel_size=4, stride=1, padding=0, dilation=1)
        
        self.fc1 = nn.Linear(64 * 123, 64)
        #self.fc2 = nn.Linear(1024, 32)
        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = F.max_pool1d(
            F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))),
            kernel_size=4, stride=4, padding=0, dilation=1
            )
        x = F.max_pool1d(
            F.relu(self.conv6(F.relu(self.conv5(F.relu(self.conv4(x)))))),
            kernel_size=4, stride=4, padding=0, dilation=1
            )
        x = F.max_pool1d(
            F.relu(self.conv9(F.relu(self.conv8(F.relu(self.conv7(x)))))),
            kernel_size=4, stride=4, padding=0, dilation=1
            )
        x = x.view(-1, self.num_flat_features(x))
        #print(x.shape)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc2(x)
        
        return x
        
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
          num_features *= s
        return num_features


class df_conv6_fc2(torch.nn.Module):
    def __init__(self):
        super(df_conv6_fc2, self).__init__()
        self.conv1 = torch.nn.Conv1d(2, 16, kernel_size=16, stride=1, padding=0, dilation=1)
        self.conv2 = torch.nn.Conv1d(16, 16, kernel_size=16, stride=1, padding=0, dilation=1)
        self.conv3 = torch.nn.Conv1d(16, 32, kernel_size=8, stride=1, padding=0, dilation=1)
        self.conv4 = torch.nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=0, dilation=1)
        self.conv5 = torch.nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=0, dilation=1)
        self.conv6 = torch.nn.Conv1d(64, 64, kernel_size=4, stride=1, padding=0, dilation=1)
        
        self.fc1 = nn.Linear(96 * 123, 64)
        #self.fc2 = nn.Linear(1024, 32)
        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = F.max_pool1d(
            F.relu(self.conv2(F.relu(self.conv1(x)))),
            kernel_size=4, stride=4, padding=0, dilation=1
            )
        x = F.max_pool1d(
            F.relu(self.conv4(F.relu(self.conv3(x)))),
            kernel_size=4, stride=4, padding=0, dilation=1
            )
        x = F.max_pool1d(
            F.relu(self.conv6(F.relu(self.conv5(x)))),
            kernel_size=4, stride=4, padding=0, dilation=1
            )
        x = x.view(-1, self.num_flat_features(x))
        #print(x.shape)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc2(x)
        
        return x
        
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
          num_features *= s
        return num_features

class df_conv9_fc2_dropout_3ch(torch.nn.Module):
    def __init__(self):
        super(df_conv9_fc2_dropout_3ch, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 16, kernel_size=16, stride=1, padding=0, dilation=1)
        self.conv2 = torch.nn.Conv1d(16, 16, kernel_size=16, stride=1, padding=0, dilation=1)
        self.conv3 = torch.nn.Conv1d(16, 16, kernel_size=16, stride=1, padding=0, dilation=1)
        self.conv4 = torch.nn.Conv1d(16, 32, kernel_size=8, stride=1, padding=0, dilation=1)
        self.conv5 = torch.nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=0, dilation=1)
        self.conv6 = torch.nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=0, dilation=1)
        self.conv7 = torch.nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=0, dilation=1)
        self.conv8 = torch.nn.Conv1d(64, 64, kernel_size=4, stride=1, padding=0, dilation=1)
        self.conv9 = torch.nn.Conv1d(64, 64, kernel_size=4, stride=1, padding=0, dilation=1)
        
        self.fc1 = nn.Linear(64 * 123, 64)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = F.max_pool1d(
            F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))),
            kernel_size=4, stride=4, padding=0, dilation=1
            )
        x = F.max_pool1d(
            F.relu(self.conv6(F.relu(self.conv5(F.relu(self.conv4(x)))))),
            kernel_size=4, stride=4, padding=0, dilation=1
            )
        x = F.max_pool1d(
            F.relu(self.conv9(F.relu(self.conv8(F.relu(self.conv7(x)))))),
            kernel_size=4, stride=4, padding=0, dilation=1
            )
        x = x.view(-1, self.num_flat_features(x))
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
        
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
          num_features *= s
        return num_features
        
        
class df_conv12_fc2_dropout_3ch(torch.nn.Module):
    def __init__(self):
        super(df_conv12_fc2_dropout_3ch, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 16, kernel_size=32, stride=1, padding=0, dilation=1)
        self.conv2 = torch.nn.Conv1d(16, 16, kernel_size=32, stride=1, padding=0, dilation=1)
        self.conv3 = torch.nn.Conv1d(16, 16, kernel_size=32, stride=1, padding=0, dilation=1)
        self.conv4 = torch.nn.Conv1d(16, 32, kernel_size=16, stride=1, padding=0, dilation=1)
        self.conv5 = torch.nn.Conv1d(32, 32, kernel_size=16, stride=1, padding=0, dilation=1)
        self.conv6 = torch.nn.Conv1d(32, 32, kernel_size=16, stride=1, padding=0, dilation=1)
        self.conv7 = torch.nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=0, dilation=1)
        self.conv8 = torch.nn.Conv1d(64, 64, kernel_size=8, stride=1, padding=0, dilation=1)
        self.conv9 = torch.nn.Conv1d(64, 64, kernel_size=8, stride=1, padding=0, dilation=1)
        self.conv10 = torch.nn.Conv1d(64, 128, kernel_size=4, stride=1, padding=0, dilation=1)
        self.conv11 = torch.nn.Conv1d(128, 128, kernel_size=4, stride=1, padding=0, dilation=1)
        self.conv12 = torch.nn.Conv1d(128, 128, kernel_size=4, stride=1, padding=0, dilation=1)
        
        self.fc1 = nn.Linear(128 * 92, 96)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(96, 2)
        
    def forward(self, x):
        x = F.max_pool1d(
            F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))),
            kernel_size=3, stride=3, padding=0, dilation=1
            )
        x = F.max_pool1d(
            F.relu(self.conv6(F.relu(self.conv5(F.relu(self.conv4(x)))))),
            kernel_size=3, stride=3, padding=0, dilation=1
            )
        x = F.max_pool1d(
            F.relu(self.conv9(F.relu(self.conv8(F.relu(self.conv7(x)))))),
            kernel_size=3, stride=3, padding=0, dilation=1
            )
        x = F.max_pool1d(
            F.relu(self.conv12(F.relu(self.conv11(F.relu(self.conv10(x)))))),
            kernel_size=3, stride=3, padding=0, dilation=1
            )
        x = x.view(-1, self.num_flat_features(x))
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
        
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
          num_features *= s
        return num_features

class df_conv6_fc2_3ch(torch.nn.Module):
    def __init__(self):
        super(df_conv6_fc2_3ch, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 16, kernel_size=8, stride=1, padding=0, dilation=1)
        self.conv2 = torch.nn.Conv1d(16, 16, kernel_size=8, stride=1, padding=0, dilation=1)
        self.conv3 = torch.nn.Conv1d(16, 24, kernel_size=4, stride=1, padding=0, dilation=4)
        self.conv4 = torch.nn.Conv1d(24, 24, kernel_size=4, stride=1, padding=0, dilation=4)
        self.conv5 = torch.nn.Conv1d(24, 32, kernel_size=2, stride=1, padding=0, dilation=8)
        self.conv6 = torch.nn.Conv1d(32, 32, kernel_size=2, stride=1, padding=0, dilation=8)
        
        x = torch.rand((1,3,8192))
        x = F.max_pool1d(
            F.relu(self.conv2(F.relu(self.conv1(x)))),
            kernel_size=8, stride=8, padding=0, dilation=1
            )
        x = F.max_pool1d(
            F.relu(self.conv4(F.relu(self.conv3(x)))),
            kernel_size=4, stride=4, padding=0, dilation=1
            )
        x = F.max_pool1d(
            F.relu(self.conv6(F.relu(self.conv5(x)))),
            kernel_size=2, stride=2, padding=0, dilation=1
            )
        x = x.view(-1, self.num_flat_features(x))
        
        
        self.fc1 = nn.Linear(int(x.shape[-1]) ,256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 2)
        
    def forward(self, x):
        x = F.max_pool1d(
            F.relu(self.conv2(F.relu(self.conv1(x)))),
            kernel_size=8, stride=8, padding=0, dilation=1
            )
        x = F.max_pool1d(
            F.relu(self.conv4(F.relu(self.conv3(x)))),
            kernel_size=4, stride=4, padding=0, dilation=1
            )
        x = F.max_pool1d(
            F.relu(self.conv6(F.relu(self.conv5(x)))),
            kernel_size=2, stride=2, padding=0, dilation=1
            )
        x = x.view(-1, self.num_flat_features(x))
        #print(x.shape[-1] / 64)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
        
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
          num_features *= s
        return num_features
        
class df_conv6_fc2_3ch_best(torch.nn.Module): # 0.872 - 0.868
    def __init__(self):
        super(df_conv6_fc2_3ch, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 16, kernel_size=8, stride=1, padding=0, dilation=1)
        self.conv2 = torch.nn.Conv1d(16, 16, kernel_size=8, stride=1, padding=0, dilation=1)
        self.conv3 = torch.nn.Conv1d(16, 24, kernel_size=4, stride=1, padding=0, dilation=4)
        self.conv4 = torch.nn.Conv1d(24, 24, kernel_size=4, stride=1, padding=0, dilation=4)
        self.conv5 = torch.nn.Conv1d(24, 32, kernel_size=2, stride=1, padding=0, dilation=8)
        self.conv6 = torch.nn.Conv1d(32, 32, kernel_size=2, stride=1, padding=0, dilation=8)
        
        x = torch.rand((1,3,8192))
        x = F.max_pool1d(
            F.relu(self.conv2(F.relu(self.conv1(x)))),
            kernel_size=8, stride=8, padding=0, dilation=1
            )
        x = F.max_pool1d(
            F.relu(self.conv4(F.relu(self.conv3(x)))),
            kernel_size=4, stride=4, padding=0, dilation=1
            )
        x = F.max_pool1d(
            F.relu(self.conv6(F.relu(self.conv5(x)))),
            kernel_size=2, stride=2, padding=0, dilation=1
            )
        x = x.view(-1, self.num_flat_features(x))
        
        
        self.fc1 = nn.Linear(int(x.shape[-1]) , 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 2)
        
    def forward(self, x):
        x = F.max_pool1d(
            F.relu(self.conv2(F.relu(self.conv1(x)))),
            kernel_size=8, stride=8, padding=0, dilation=1
            )
        x = F.max_pool1d(
            F.relu(self.conv4(F.relu(self.conv3(x)))),
            kernel_size=4, stride=4, padding=0, dilation=1
            )
        x = F.max_pool1d(
            F.relu(self.conv6(F.relu(self.conv5(x)))),
            kernel_size=2, stride=2, padding=0, dilation=1
            )
        x = x.view(-1, self.num_flat_features(x))
        #print(x.shape[-1] / 64)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
        
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
          num_features *= s
        return num_features
        
        
        
class df_conv3_fc2_3ch(torch.nn.Module):
    def __init__(self):
        super(df_conv3_fc2_3ch, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 16, kernel_size=4, stride=1, padding=0, dilation=1)
        self.conv2 = torch.nn.Conv1d(16, 32, kernel_size=4, stride=1, padding=0, dilation=1)
        self.conv3 = torch.nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=0, dilation=1)

        
        x = torch.rand((1,3,8192))
        x = F.max_pool1d(F.relu(self.conv1(x)),
            kernel_size=8, stride=8, padding=0, dilation=1
            )
        x = F.max_pool1d(F.relu(self.conv2(x)),
            kernel_size=4, stride=4, padding=0, dilation=1
            )
        x = F.max_pool1d(F.relu(self.conv3(x)),
            kernel_size=2, stride=2, padding=0, dilation=1
            )
        x = x.view(-1, self.num_flat_features(x))
        
        
        self.fc1 = nn.Linear(int(x.shape[-1]), 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 2)
        
    def forward(self, x):
        x = F.max_pool1d(F.relu(self.conv1(x)),
            kernel_size=8, stride=8, padding=0, dilation=1
            )
        x = F.max_pool1d(F.relu(self.conv2(x)),
            kernel_size=4, stride=4, padding=0, dilation=1
            )
        x = F.max_pool1d(F.relu(self.conv3(x)),
            kernel_size=2, stride=2, padding=0, dilation=1
            )
        x = x.view(-1, self.num_flat_features(x))
        #print(x.shape[-1] / 64)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
        
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
          num_features *= s
        return num_features
        

class df_conv3_fc2_3ch_best(torch.nn.Module): #0.863
    def __init__(self):
        super(df_conv3_fc2_3ch, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 16, kernel_size=4, stride=1, padding=0, dilation=1)
        self.conv2 = torch.nn.Conv1d(16, 32, kernel_size=4, stride=1, padding=0, dilation=1)
        self.conv3 = torch.nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=0, dilation=1)

        
        x = torch.rand((1,3,8192))
        x = F.max_pool1d(F.relu(self.conv1(x)),
            kernel_size=8, stride=8, padding=0, dilation=1
            )
        x = F.max_pool1d(F.relu(self.conv2(x)),
            kernel_size=4, stride=4, padding=0, dilation=1
            )
        x = F.max_pool1d(F.relu(self.conv3(x)),
            kernel_size=2, stride=2, padding=0, dilation=1
            )
        x = x.view(-1, self.num_flat_features(x))
        
        
        self.fc1 = nn.Linear(int(x.shape[-1]), 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 2)
        
    def forward(self, x):
        x = F.max_pool1d(F.relu(self.conv1(x)),
            kernel_size=8, stride=8, padding=0, dilation=1
            )
        x = F.max_pool1d(F.relu(self.conv2(x)),
            kernel_size=4, stride=4, padding=0, dilation=1
            )
        x = F.max_pool1d(F.relu(self.conv3(x)),
            kernel_size=2, stride=2, padding=0, dilation=1
            )
        x = x.view(-1, self.num_flat_features(x))
        #print(x.shape[-1] / 64)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
        
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
          num_features *= s
        return num_features


