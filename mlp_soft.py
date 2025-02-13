import torch.nn as nn
import torch
import settings as st

class MLP_Parent_XenterLoss(nn.Module):
    
    def __init__(self, input_dim_parent, input_dim_child, output_dim):
        super().__init__()
        
        # self.input_dim = (input_dim_parent + input_dim_child)//2
        # self.fc1 = nn.Linear(4*input_dim_parent, output_dim)    
        self.xfeature = nn.Linear(4*input_dim_parent, st.config[st.I]["FC1"])    
        self.fc1 = nn.Linear(st.config[st.I]["FC1"], st.config[st.I]["FC2"])
        # self.prelu_fc = nn.Sigmoid()
        # self.fc2 = nn.Linear(st.FC2, st.FC3)
        self.fc2 = nn.Linear(st.config[st.I]["FC2"], output_dim)
        
    def forward(self, feature1, feature2):
        x1 = torch.pow(feature1, 2) - torch.pow(feature2, 2)
        x2 = torch.pow(feature1 - feature2, 2)
        x3 = feature1 * feature2
        x4 = feature1 + feature2
        #x5 = torch.cat((feature1[:len(feature1)//2], feature2[len(feature2)//2:]), dim=1)
        x = torch.cat((x3, x2, x1, x4), dim=1)
        x = x.view(x.size(0), -1)

        # out = self.fc2(self.dropout1d(self.activation(self.fc1(x))))
        # out = self.fc1(x)
        # x = self.prelu_fc(self.fc1(x))
        
        x = self.xfeature(x)
        out = self.fc2(self.fc1(x))
        # out = self.fc2(x)
        #out = self.fc3(self.fc2(self.fc1(x)))
        
        # x = self.fc1(self.xfeature(x))
        # out = self.fc3(self.fc2(x))
        
        return out, x
    


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)


class MLP_Parent_Child_High_Dim(nn.Module):
    
    def __init__(self, input_dim_parent, input_dim_child, output_dim):
        super().__init__()
        
        # self.input_dim = (input_dim_parent + input_dim_child)//2
        self.fc1 = nn.Linear(4*input_dim_parent, output_dim)        
        # self.fc1 = nn.Linear(640, output_dim)        
        
        # self.fc1 = nn.Linear(2688, 1344)        
        # self.activation = nn.ReLU(True)
        # self.dropout_rate = 0.5
        # self.dropout1d = nn.Dropout(self.dropout_rate)
        # self.fc2 =nn.Linear(1344, 1)

        # activation, batch normalization
        # self.activation = nn.PReLU()
        
        # # dropout
        # self.dropout_rate = 0.9
        # self.dropout1d = nn.Dropout(self.dropout_rate)

        
    def forward(self, feature1, feature2):
        x1 = torch.pow(feature1, 2) - torch.pow(feature2, 2)
        x2 = torch.pow(feature1 - feature2, 2)
        x3 = feature1 * feature2
        x4 = feature1 + feature2
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = x.view(x.size(0), -1)

        # out = self.fc2(self.dropout1d(self.activation(self.fc1(x))))
        out = self.fc1(x)

        return out

class MLP_Parent_Child(nn.Module):
    
    def __init__(self, input_dim_parent, input_dim_child, output_dim):
        super().__init__()
        
        self.input_dim = (input_dim_parent + input_dim_child)//2
        self.fc1 = nn.Linear(self.input_dim, output_dim)        
        # activation, batch normalization
        # self.activation = nn.PReLU()
        # self.activation = nn.ReLU()
        # # dropout
        # self.dropout_rate = 0.9
        # self.dropout1d = nn.Dropout(self.dropout_rate)

        
    def forward(self, x1, x2):
        x = torch.stack((x1,x2)).mean(dim=0)
        # a = torch.stack((x1[:,0:512],x2[:,0:512])).mean(dim=0)
        # b = torch.stack((x1[:,512: 512+128],x2[:,512: 512+128])).mean(dim=0)
        # c = torch.stack((x1[:,512+128:],x2[:,512+128:])).mean(dim=0)
        # x = torch.concat((a,b,c), dim=1)

        out = self.fc1(x)
        return out


class MLP_layer(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, output_dim)        
        # activation, batch normalization
        # self.activation = nn.PReLU()
        # self.activation = nn.ReLU()
        # # dropout
        # self.dropout_rate = 0.9
        # self.dropout1d = nn.Dropout(self.dropout_rate)

        
    def forward(self, x):
        
        out = self.fc1(x)
        return out
