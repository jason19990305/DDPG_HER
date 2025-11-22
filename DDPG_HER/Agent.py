from torch.distributions import Normal
from gymnasium.vector import AsyncVectorEnv
import gymnasium as gym
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
        self.num_rollouts = 2
        

        # Variable
        self.total_steps = 0
        self.training_count = 0
        self.evaluate_count = 0
        warmup_duration = 25000
        # other
        self.env = env
        self.action_max = env.action_space.high[0]
        self.replay_buffer = ReplayBuffer(args)
        self.num_envs = os.cpu_count() - 1
        env_fns = [lambda : gym.make(self.env_name) for _ in range(self.num_envs)]
        self.venv = AsyncVectorEnv(env_fns , autoreset_mode= gym.vector.AutoresetMode.DISABLED)         
        self.state_normalizer = Normalization(shape=(self.obs_dim), warmup_steps=warmup_duration)
        self.goal_normalizer = Normalization(shape=(self.des_goal_dim), warmup_steps=warmup_duration)
        self.max_train_steps = args.epochs * args.n_cycles * self.num_envs * env._max_episode_steps

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
            # Add noise proportional to current variance schedule
            noise = torch.randn_like(a) * self.var    
            a = a + noise
            a = torch.clamp(a, -self.action_max, self.action_max)
            a = a.squeeze(0)
        return a.cpu().numpy()

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
                s , _ = self.venv.reset()           

                for i in range(self.env._max_episode_steps):
                    a = self.choose_action(s)
                    s_, r, done , truncated , _ = self.venv.step(a)

                    done = np.zeros_like(r)
                    
                    self.total_steps += self.num_envs
                    mb_state.append(s["observation"].copy())
                    mb_ach_goal.append(s["achieved_goal"].copy())
                    mb_des_goal.append(s["desired_goal"].copy())
                    mb_action.append(a.copy())
                    mb_reward.append(r)
                    mb_next_state.append(s_["observation"].copy())
                    mb_next_ach_goal.append(s_["achieved_goal"].copy())
                    mb_next_des_goal.append(s_["desired_goal"].copy())
                    mb_done.append(done)
                        
                    # update state
                    s = s_
                    
                # [time steps , episodes , dim] 
                # swap to [episodes , time steps , dim]
                mb_state = np.array(mb_state).swapaxes(0, 1)
                mb_ach_goal = np.array(mb_ach_goal).swapaxes(0, 1)
                mb_des_goal = np.array(mb_des_goal).swapaxes(0, 1)
                mb_action = np.array(mb_action).swapaxes(0, 1)
                mb_reward = np.array(mb_reward).swapaxes(0, 1).reshape(self.num_envs, -1, 1)
                mb_next_state = np.array(mb_next_state).swapaxes(0, 1)
                mb_next_des_goal = np.array(mb_next_des_goal).swapaxes(0, 1)
                mb_next_ach_goal = np.array(mb_next_ach_goal).swapaxes(0, 1)
                mb_done = np.array(mb_done).swapaxes(0, 1).reshape(self.num_envs, -1, 1)

                    
                self.replay_buffer.store_batch(mb_state.reshape(-1,self.obs_dim)
                                            , mb_ach_goal.reshape(-1,self.ach_goal_dim)
                                            , mb_des_goal.reshape(-1,self.des_goal_dim)
                                            , mb_action.reshape(-1,self.action_dim)
                                            , mb_reward.reshape(-1,1)
                                            , mb_next_state.reshape(-1,self.obs_dim)
                                            , mb_next_ach_goal.reshape(-1,self.ach_goal_dim)
                                            , mb_next_des_goal.reshape(-1,self.des_goal_dim)
                                            , mb_done.reshape(-1,1))       

                self.her_sample(mb_state, mb_action, mb_next_state, mb_ach_goal, mb_des_goal, mb_next_ach_goal, mb_next_des_goal)
                self.update_normalizer(mb_state, mb_des_goal)
                for _ in range(self.n_batch):
                    self.update()
                self.var_decay(total_steps=self.total_steps)  # Enable noise decay
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
        # Vectorized HER sampling
        num_episode = mb_state.shape[0]
        num_steps = mb_state.shape[1]
        k = 4
        batch_size = num_episode * num_steps * k
        
        # Select which rollouts and which timesteps to be used
        episode_index = np.random.randint(0, num_episode, batch_size)
        step_index = np.random.randint(num_steps-1, size=batch_size)
        
        # Gather data
        obs = mb_state[episode_index, step_index]
        actions = mb_action[episode_index, step_index]
        obs_next = mb_next_state[episode_index, step_index]
        ag = mb_ach_goal[episode_index, step_index]
        ag_next = mb_next_ach_goal[episode_index, step_index]
        
        # Select future goals
        # future_offset in [1, num_steps - 1 - t]
        future_offset = (np.random.uniform(size=batch_size) * (num_steps - 1 - step_index)).astype(int) + 1
        future_t = step_index + future_offset
        
        new_goals = mb_next_ach_goal[episode_index, future_t]
        
        # Compute rewards
        new_rewards = self.env.unwrapped.compute_reward(ag_next, new_goals, None)
        new_rewards = np.expand_dims(new_rewards, 1)
        

        # Store in replay buffer
        dones = np.zeros((batch_size, 1), dtype=np.float32)
        
        self.replay_buffer.store_batch(obs, ag, new_goals, actions, new_rewards, obs_next, ag_next, new_goals, dones)
        #for i in range(batch_size):
        #    self.replay_buffer.store(obs[i], ag[i], new_goals[i], actions[i], new_rewards[i], obs_next[i], ag_next[i], new_goals[i], dones[i])

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
        with torch.no_grad():

            next_action = self.actor_target(minibatch_s_)
            next_value = self.critic_target(minibatch_s_,next_action)
            v_target = minibatch_r + self.gamma * next_value 
            # clip the q value
            clip_return = 1 / (1 - self.gamma)
            v_target = torch.clamp(v_target, -clip_return, 0)

        value = self.critic(minibatch_s,minibatch_a)
        critic_loss = F.mse_loss(value,v_target)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5) # Trick : Clip grad
        self.optimizer_critic.step()

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
