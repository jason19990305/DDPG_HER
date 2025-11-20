from torch.distributions import Normal
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np 
import torch
import time
import copy
import os
# Custom class
from DDPG_HER.ReplayBuffer import ReplayBuffer
from DDPG_HER.ActorCritic import Actor , Critic
from DDPG_HER.Normalization import Normalization 
class Agent():
    def __init__(self,args,env,hidden_layer_num_list=[64,64]):

        # Hyperparameter
        self.ach_goal_dim = args.ach_goal_dim
        self.des_goal_dim = args.des_goal_dim
        self.action_dim = args.action_dim
        self.batch_size = args.batch_size
        self.action_max = args.action_max
        self.n_cycles = args.n_cycles
        self.env_name = args.env_name
        self.n_batch = args.n_batch
        self.obs_dim = args.obs_dim
        self.epochs = args.epochs
        self.gamma = args.gamma
        self.set_var = args.var
        self.var = self.set_var
        self.tau = args.tau
        self.lr = args.lr
        self.use_normalization = args.normalization
        self.max_train_steps = args.epochs * args.n_cycles * env._max_episode_steps

        # Variable
        self.total_steps = 0
        self.training_count = 0
        self.evaluate_count = 0
        warmup_duration = 25000

        # other
        self.env = env
        self.action_max = env.action_space.high[0]
        self.replay_buffer = ReplayBuffer(args)

         
        self.state_normalizer = Normalization(shape=(self.obs_dim), warmup_steps=warmup_duration)
        self.goal_normalizer = Normalization(shape=(self.des_goal_dim), warmup_steps=warmup_duration)
        # Actor-Critic
        self.actor = Actor(args,hidden_layer_num_list.copy())
        self.critic = Critic(args,hidden_layer_num_list.copy())
        self.actor_target =  copy.deepcopy(self.actor)
        self.critic_target =  copy.deepcopy(self.critic)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr, eps=1e-5)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=1e-5)

        print(self.actor)
        print(self.critic)
        print("-----------")

    def choose_action(self,state):

        observation = state["observation"]
        desired_goal = state["desired_goal"]
        achieved_goal = state["achieved_goal"]
        
        # Apply normalization if enabled
        if self.use_normalization:
            observation = self.state_normalizer(observation)
            desired_goal = self.goal_normalizer(desired_goal)
            achieved_goal = self.goal_normalizer(achieved_goal)
        
        state = np.concatenate([observation,desired_goal,achieved_goal],axis=-1)        
        state = torch.tensor(state, dtype=torch.float)
        s = torch.unsqueeze(state,0)
        with torch.no_grad():
            a = self.actor(s)
            noise = torch.randn_like(a) * self.var    
            a = a + noise
            a = torch.clamp(a, -self.action_max, self.action_max)                 # 先 clamp 到 [-1,1]
            
        return a.cpu().numpy().flatten() 

    def evaluate_action(self,state):

        observation = state["observation"]
        desired_goal = state["desired_goal"]
        achieved_goal = state["achieved_goal"]
        
        # Apply normalization if enabled
        if self.use_normalization:
            observation = self.state_normalizer(observation)
            desired_goal = self.goal_normalizer(desired_goal)
            achieved_goal = self.goal_normalizer(achieved_goal)
        
        state = np.concatenate([observation,desired_goal,achieved_goal],axis=-1)        
        state = torch.tensor(state, dtype=torch.float)
        s = torch.unsqueeze(state,0)
        with torch.no_grad():
            a = self.actor(s)     
        return a.cpu().numpy().flatten()

    def evaluate_policy(self, env, render=False):
        times = 20
        success_count = 0
        for i in range(times):
            s, info = env.reset()
            
            done = False
            while True:
                a = self.evaluate_action(s)
                s_, r, done, truncted, _ = env.step(a)
                
                if r == 0:  # Success condition
                    success_count += 1
                    break
                
                s = s_
                
                if done or truncted:
                    break

        success_rate = success_count / times * 100
        return success_rate

    def var_decay(self, total_steps):
        new_var = self.set_var * (1 - total_steps / self.max_train_steps) 
        self.var = max(new_var, 1e-8) # Ensure var is always positive
        
    def update_normalizer(self, state, goal):
        """Update normalizers if normalization is enabled"""
        if self.use_normalization:
            self.state_normalizer.update(state)
            self.goal_normalizer.update(goal)
        
        
    def train(self):
        time_start = time.time()
        episode_count = 0
        epoch_reward_list = []
        epoch_list = []
        
        for epoch in range(self.epochs):
            for cycle in range(self.n_cycles):
                mb_state = []
                mb_ach_goal = []
                mb_des_goal = []
                mb_action = []
                mb_reward = []
                mb_next_state = []
                mb_next_ach_goal = []
                mb_next_des_goal = []
                mb_done = []
                for _ in range(2):
                    ep_state = []
                    ep_ach_goal = []
                    ep_des_goal = []
                    ep_action = []
                    ep_reward = []
                    ep_next_state = []
                    ep_next_ach_goal = []
                    ep_next_des_goal = []
                    ep_done = []
                    s , _ = self.env.reset()           

                    for i in range(self.env._max_episode_steps):
                        a = self.choose_action(s)

                        s_, r, done , truncated , _ = self.env.step(a)
                        done = done or truncated
                        
                        self.replay_buffer.store(s["observation"]
                                                , s["achieved_goal"]
                                                , s["desired_goal"] 
                                                , a
                                                , [r]
                                                , s_["observation"]
                                                , s_["achieved_goal"]
                                                , s_["desired_goal"]
                                                , done)
                        self.total_steps += 1
                        ep_state.append(s["observation"].copy())
                        ep_ach_goal.append(s["achieved_goal"].copy())
                        ep_des_goal.append(s["desired_goal"].copy())
                        ep_action.append(a.copy())
                        ep_reward.append(r)
                        ep_next_state.append(s_["observation"].copy())
                        ep_next_ach_goal.append(s_["achieved_goal"].copy())
                        ep_next_des_goal.append(s_["desired_goal"].copy())
                        ep_done.append(done)
                        
                        # update state
                        s = s_
                    # end of episode
                    mb_state.append(ep_state)
                    mb_ach_goal.append(ep_ach_goal)
                    mb_des_goal.append(ep_des_goal)
                    mb_action.append(ep_action)
                    mb_reward.append(ep_reward)
                    mb_next_state.append(ep_next_state)
                    mb_next_ach_goal.append(ep_next_ach_goal)
                    mb_next_des_goal.append(ep_next_des_goal)
                    mb_done.append(ep_done)
                    episode_count += 1
                    
                # convert into numpy array
                mb_state = np.array(mb_state)
                mb_ach_goal = np.array(mb_ach_goal)
                mb_des_goal = np.array(mb_des_goal)
                mb_action = np.array(mb_action)
                mb_reward = np.array(mb_reward)
                mb_next_state = np.array(mb_next_state)
                mb_next_ach_goal = np.array(mb_next_ach_goal)
                mb_next_des_goal = np.array(mb_next_des_goal)
                mb_done = np.array(mb_done)
                self.her_sample(mb_state, mb_action, mb_next_state, mb_ach_goal, mb_des_goal, mb_next_ach_goal, mb_next_des_goal)
                self.update_normalizer(mb_state, mb_des_goal)
                for _ in range(self.n_batch):
                    self.update()
                self.var_decay(total_steps=self.total_steps)
                # Update target networks
                self.soft_update(self.critic_target,self.critic, self.tau)
                self.soft_update(self.actor_target, self.actor, self.tau)   
            # end of cycle
            evaluate_reward = self.evaluate_policy(self.env)
            epoch_reward_list.append(evaluate_reward)
            epoch_list.append(epoch)
            time_end = time.time()
            h = int((time_end - time_start) // 3600)
            m = int(((time_end - time_start) % 3600) // 60)
            second = int((time_end - time_start) % 60)
            print("---------")
            print("Var : %f"%(self.var))
            print("Total steps : %d"%(self.total_steps))
            print("Time : %02d:%02d:%02d"%(h,m,second))
            print("Epoch : %d / %d\tSuccess rate : %0.2f %%"%(epoch,self.epochs,evaluate_reward))
                        
                        
            
        # Plot the training curve
        plot_dir = "Plot"
        os.makedirs(plot_dir, exist_ok=True)
        plt.plot(epoch_list , epoch_reward_list)
        plt.xlabel("Epoch")
        plt.ylabel("Success Rate (%)")
        plt.title("Training Curve")
        plt.savefig(os.path.join(plot_dir, f"{self.env_name}_training_curve.png"))
        plt.close()
        
    def her_sample(self, mb_state, mb_action, mb_next_state, mb_ach_goal, mb_des_goal, mb_next_ach_goal, mb_next_des_goal):
        T = mb_state.shape[1]
        rollout_batch_size = mb_state.shape[0]
        k = 4  
        for i in range(rollout_batch_size):
            for t in range(T - 1):
                future_indices = np.random.choice(range(t + 1, T), size=min(k, T - t - 1), replace=False)
                for future_t in future_indices:
                    new_goal = mb_next_ach_goal[i, future_t]

                    new_reward = self.env.unwrapped.compute_reward(mb_next_ach_goal[i, t], new_goal, info=None)
                    self.replay_buffer.store(
                        mb_state[i, t],
                        mb_ach_goal[i, t],
                        new_goal, 
                        mb_action[i, t],
                        [new_reward],
                        mb_next_state[i, t],
                        mb_next_ach_goal[i, t],
                        new_goal,
                        False
                    )


    def update(self):
        # Get training data .type is tensor
        s, ach_goal, des_goal, a, r, s_, next_ach_goal, next_des_goal, done = self.replay_buffer.sample_minibatch(self.state_normalizer, self.goal_normalizer)  


        s = torch.cat((s,des_goal,ach_goal),dim=1)        
        s_ = torch.cat((s_,next_des_goal,next_ach_goal),dim=1)
            
        # Get minibatch
        minibatch_s = s
        minibatch_a = a
        minibatch_r = r
        minibatch_s_ = s_
        minibatch_done = done

        # update Actor
        action = self.actor(minibatch_s)
        value = self.critic(minibatch_s,action)
        actor_loss = -torch.mean(value)
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5) # Trick : Clip grad
        self.optimizer_actor.step()


        # Update Critic
        next_action = self.actor_target(minibatch_s_)
        next_value = self.critic_target(minibatch_s_,next_action)
        v_target = minibatch_r + self.gamma * next_value * (1 - minibatch_done)

        value = self.critic(minibatch_s,minibatch_a)
        critic_loss = F.mse_loss(value,v_target)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5) # Trick : Clip grad
        self.optimizer_critic.step()

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
