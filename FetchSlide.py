

import gymnasium as gym # openai gym
import numpy as np 
import argparse
import warnings
from DDPG_HER.Agent import Agent
from gymnasium.wrappers import RecordVideo
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

# suppress RecordVideo overwrite warning
warnings.filterwarnings('ignore', category=UserWarning, module='gymnasium.wrappers.rendering')


class main():
    def __init__(self,args):
        env_name = 'FetchSlide-v4'
        env = gym.make(env_name)        
        
        # args
        args.env_name = env_name
        args.action_max = env.action_space.high[0]  
        args.ach_goal_dim = env.observation_space["achieved_goal"].shape[0]
        args.des_goal_dim = env.observation_space["desired_goal"].shape[0]
        args.obs_dim = env.observation_space["observation"].shape[0]
        args.action_dim = env.action_space.shape[0]
        

        # print args 
        print("---------------")
        for arg in vars(args):
            print(arg,"=",getattr(args, arg))
        print("---------------")

        # create agent
        hidden_layer_num_list = [256,256,256]
        agent = Agent(args , env , hidden_layer_num_list)

        # trainning
        agent.train() 
        
        # evaluate 
        render_env = gym.make(env_name, render_mode="rgb_array")  
        render_env = RecordVideo(render_env, video_folder = "Video/"+env_name, episode_trigger=lambda x: True)
        agent.evaluate_policy(render_env)
        render_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for DDPG-HER")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate of actor")
    parser.add_argument("--var", type=float, default=0.2, help="Normal noise var")
    parser.add_argument("--tau", type=float, default=0.05, help="Parameter for soft update")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--n_cycles", type=int, default=500, help="Number of cycles per epoch")
    parser.add_argument("--n_batch", type=int, default=40, help="The times of update the network")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="Learning rate of actor")
    parser.add_argument("--normalization", type=bool, default=True, help="Whether to use normalization")
    args = parser.parse_args()

    main(args)