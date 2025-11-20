import torch.nn.functional as F 
import torch.nn as nn 
import torch

    

class Actor(nn.Module):
    def __init__(self,args,hidden_layers=[64,64]):
        super(Actor, self).__init__()
        
        self.ach_goal_dim = args.ach_goal_dim
        self.des_goal_dim = args.des_goal_dim
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.action_max = args.action_max 
        # add in list
        hidden_layers.insert(0,self.ach_goal_dim + self.des_goal_dim + self.obs_dim) # first layer
        hidden_layers.append(self.action_dim) # last layer
        print(hidden_layers)

        # create layers
        layer_list = []
        for i in range(len(hidden_layers)-1):
            input_num = hidden_layers[i]
            output_num = hidden_layers[i+1]
            layer = nn.Linear(input_num,output_num)
            
            layer_list.append(layer)

        # put in ModuleList
        self.layers = nn.ModuleList(layer_list)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    # when actor(s) will activate the function 
    def forward(self,s):

        for layer in range(len(self.layers)-1):
            s = self.relu(self.layers[layer](s))
        s = self.tanh(self.layers[-1](s))
        return s * self.action_max


class Critic(nn.Module):
    def __init__(self, args,hidden_layers=[64,64]):
        super(Critic, self).__init__()
        self.ach_goal_dim = args.ach_goal_dim
        self.des_goal_dim = args.des_goal_dim
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        # add in list
        hidden_layers.insert(0,self.ach_goal_dim + self.des_goal_dim + self.obs_dim + self.action_dim)
        hidden_layers.append(1)
        print(hidden_layers)

        # create layers
        layer_list = []

        for i in range(len(hidden_layers)-1):
            input_num = hidden_layers[i]
            output_num = hidden_layers[i+1]
            layer = nn.Linear(input_num,output_num)
            
            layer_list.append(layer)
        # put in ModuleList
        self.layers = nn.ModuleList(layer_list)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self,s,a):
        input_data = torch.cat((s,a),dim=1)
        for i in range(len(self.layers)-1):
            input_data = self.relu(self.layers[i](input_data))

        # predicet value
        v_s = self.layers[-1](input_data)
        return v_s