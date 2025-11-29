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
        self.use_normalization = args.normalization
        self.ach_goal_dim = args.ach_goal_dim
        self.des_goal_dim = args.des_goal_dim
        self.clip_range = args.clip_range
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
        self.num_rollouts = 2
        

        # Set device for PyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")  # Force using CPU
        if torch.cuda.is_available() and self.device.type == "cuda":
            print("Device name : ",torch.cuda.get_device_name(self.device))
        # Variable
        self.total_steps = 0
        self.training_count = 0
        self.evaluate_count = 0 
        warmup_duration = 0  # 增加預熱步數以獲得更穩定的統計數據
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
        self.actor = Actor(args,hidden_layer_num_list.copy()).to(self.device)
        self.critic = Critic(args,hidden_layer_num_list.copy()).to(self.device)
        self.actor_target =  copy.deepcopy(self.actor).to(self.device)
        self.critic_target =  copy.deepcopy(self.critic).to(self.device)
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
            observation = self.state_normalizer(observation, clip_range=self.clip_range)
            desired_goal = self.goal_normalizer(desired_goal, clip_range=self.clip_range)
            achieved_goal = self.goal_normalizer(achieved_goal, clip_range=self.clip_range)
        
        state = np.concatenate([observation,desired_goal,achieved_goal],axis=-1)        
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        s = torch.unsqueeze(state,0)
        
        # 1. Get deterministic action
        with torch.no_grad():
            action = self.actor(s).cpu().numpy().squeeze()
            
        # 2. Add Gaussian noise
        # action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        # Adapted: self.var corresponds to noise_eps
        action += self.var * self.action_max * np.random.randn(*action.shape)
        
        # 3. Clip action
        action = np.clip(action, -self.action_max, self.action_max)
        
        # 4. Random actions (Epsilon-Greedy)
        random_actions = np.random.uniform(low=-self.action_max, high=self.action_max, size=self.action_dim)
        
        # 5. Choose if use the random actions
        # action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        # Adapted: random_eps = 0.3
        random_eps = 0.3
        action += np.random.binomial(1, random_eps, 1)[0] * (random_actions - action)
        
        return action

    def evaluate_action(self,state):

        observation = state["observation"]
        desired_goal = state["desired_goal"]
        achieved_goal = state["achieved_goal"]
        
        # Apply normalization if enabled
        if self.use_normalization:
            observation = self.state_normalizer(observation, clip_range=self.clip_range)
            desired_goal = self.goal_normalizer(desired_goal, clip_range=self.clip_range)
            achieved_goal = self.goal_normalizer(achieved_goal, clip_range=self.clip_range)
        
        state = np.concatenate([observation,desired_goal,achieved_goal],axis=-1)        
        state = torch.tensor(state, dtype=torch.float, device=self.device)
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
            for _ in range(self.env._max_episode_steps):
                a = self.evaluate_action(s)
                s_, r, done, truncted, _ = env.step(a)
                
                if r == 0:  # Success condition
                    success_count += 1
                    break
                
                s = s_
                

        success_rate = success_count / times * 100
        return success_rate

    def var_decay(self, total_steps):
        new_var = self.set_var * (1 - total_steps / self.max_train_steps) 
        self.var = max(new_var, 1e-8) # Ensure var is always positive
        
    
            
    def update_normalizer(self, state, des_goal, ach_goal):
        if self.use_normalization:
            self.state_normalizer.update(state)
            self.goal_normalizer.update(des_goal)
            self.goal_normalizer.update(ach_goal)
        
        
    def train(self):
        time_start = time.time()
        epoch_reward_list = []
        epoch_list = []
        
        for epoch in range(self.epochs):
            for cycle in range(self.n_cycles):
                
                mb_state = np.zeros((self.env._max_episode_steps,self.num_envs,self.obs_dim), dtype=np.float32)
                mb_ach_goal = np.zeros((self.env._max_episode_steps,self.num_envs,self.ach_goal_dim), dtype=np.float32)
                mb_des_goal = np.zeros((self.env._max_episode_steps,self.num_envs,self.des_goal_dim), dtype=np.float32)
                mb_action = np.zeros((self.env._max_episode_steps,self.num_envs,self.action_dim), dtype=np.float32)
                mb_reward = np.zeros((self.env._max_episode_steps,self.num_envs,1), dtype=np.float32)
                mb_next_state = np.zeros((self.env._max_episode_steps,self.num_envs,self.obs_dim), dtype=np.float32)
                mb_next_ach_goal = np.zeros((self.env._max_episode_steps,self.num_envs,self.ach_goal_dim), dtype=np.float32)
                mb_next_des_goal = np.zeros((self.env._max_episode_steps,self.num_envs,self.des_goal_dim), dtype=np.float32)
                mb_done = np.zeros((self.env._max_episode_steps,self.num_envs,1), dtype=np.float32)
                s , _ = self.venv.reset()           

                for i in range(self.env._max_episode_steps):
                    a = self.choose_action(s)
                    s_, r, done , truncated , _ = self.venv.step(a)

                    done = np.zeros_like(r)
                    
                    self.total_steps += self.num_envs
                    mb_state[i] = s["observation"].copy()
                    mb_ach_goal[i] = s["achieved_goal"].copy()
                    mb_des_goal[i] = s["desired_goal"].copy()
                    mb_action[i] = a.copy()
                    mb_reward[i] = r.reshape(-1,1).copy()
                    mb_next_state[i] = s_["observation"].copy()
                    mb_next_ach_goal[i] = s_["achieved_goal"].copy()
                    mb_next_des_goal[i] = s_["desired_goal"].copy()
                    mb_done[i] = done.reshape(-1,1).copy()
                        
                    # update state
                    s = s_
                    
                # [time steps , episodes , dim] 
                # swap to [episodes , time steps , dim]
                mb_state = mb_state.swapaxes(0, 1)
                mb_ach_goal = mb_ach_goal.swapaxes(0, 1)
                mb_des_goal = mb_des_goal.swapaxes(0, 1)
                mb_action = mb_action.swapaxes(0, 1)
                mb_reward = mb_reward.swapaxes(0, 1).reshape(self.num_envs, -1, 1)
                mb_next_state = mb_next_state.swapaxes(0, 1)
                mb_next_des_goal = mb_next_des_goal.swapaxes(0, 1)
                mb_next_ach_goal = mb_next_ach_goal.swapaxes(0, 1)
                mb_done = mb_done.swapaxes(0, 1).reshape(self.num_envs, -1, 1)

                    
                self.replay_buffer.store_batch(mb_state.reshape(-1,self.obs_dim)
                                            , mb_ach_goal.reshape(-1,self.ach_goal_dim)
                                            , mb_des_goal.reshape(-1,self.des_goal_dim)
                                            , mb_action.reshape(-1,self.action_dim)
                                            , mb_reward.reshape(-1,1)
                                            , mb_next_state.reshape(-1,self.obs_dim)
                                            , mb_next_ach_goal.reshape(-1,self.ach_goal_dim)
                                            , mb_next_des_goal.reshape(-1,self.des_goal_dim)
                                            , mb_done.reshape(-1,1))       
                
                #self.her_sample(mb_state, mb_action, mb_next_state, mb_ach_goal, mb_des_goal, mb_next_ach_goal, mb_next_des_goal)

                self.her_sample_vec(mb_state, mb_action, mb_next_state, mb_ach_goal, mb_des_goal, mb_next_ach_goal, mb_next_des_goal)
                self.update_normalizer(mb_state, mb_des_goal, mb_ach_goal)
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
        # HER sampling with 'future' strategy and k=4
        k = 4
        num_episode = mb_state.shape[0]
        num_steps = mb_state.shape[1]

        for ep in range(num_episode):
            for t in range(num_steps):
                # Get original transition data
                obs = mb_state[ep, t]
                actions = mb_action[ep, t]
                obs_next = mb_next_state[ep, t]
                ag = mb_ach_goal[ep, t]
                ag_next = mb_next_ach_goal[ep, t]

                # Sample k future indices from the same episode
                future_indices = np.random.randint(t, num_steps, size=k)

                # Get future achieved goals as new goals
                future_goals = mb_next_ach_goal[ep, future_indices]

                # Remove debug prints
                # print(ag_next[np.newaxis, :].shape)
                # print(future_goals.shape)

                # Iterate through each future goal to compute reward and store transition
                for i in range(k):
                    new_goal = future_goals[i]
                    # Recompute reward for each new goal individually
                    new_reward = self.env.unwrapped.compute_reward(ag_next, new_goal, None)
                    done = (new_reward == 0).astype(np.float32)
                    self.replay_buffer.store(obs, ag, new_goal, actions, new_reward, obs_next, ag_next, new_goal, done)
                
    def her_sample_vec(self, mb_state, mb_action, mb_next_state, mb_ach_goal, mb_des_goal, mb_next_ach_goal, mb_next_des_goal):
        # Vectorized HER sampling
        # This function is designed to be a vectorized equivalent of her_sample.
        num_episode = mb_state.shape[0]
        num_steps = mb_state.shape[1]
        k = 4
        
        # 1. Create indices for all original transitions
        # episode_idxs will be [0, 0, ..., 1, 1, ..., num_episode-1, ...]
        # t_samples will be [0, 1, ..., num_steps-1, 0, 1, ..., num_steps-1, ...]
        episode_idxs = np.arange(num_episode)
        t_samples = np.arange(num_steps)
        episode_idxs, t_samples = np.meshgrid(episode_idxs, t_samples)
        episode_idxs, t_samples = episode_idxs.flatten(), t_samples.flatten()

        # 2. Repeat these indices k times for k HER samples per transition
        episode_idxs = np.tile(episode_idxs, k)
        t_samples = np.tile(t_samples, k)
        
        # 3. Gather original transition data using the created indices
        obs = mb_state[episode_idxs, t_samples]
        actions = mb_action[episode_idxs, t_samples]
        obs_next = mb_next_state[episode_idxs, t_samples]
        ag = mb_ach_goal[episode_idxs, t_samples]
        ag_next = mb_next_ach_goal[episode_idxs, t_samples]
        
        # 4. Sample future goals, ensuring future_t >= t
        # This matches the logic of np.random.randint(t, num_steps)
        future_t = np.random.randint(t_samples, num_steps)
        new_goals = mb_next_ach_goal[episode_idxs, future_t]
        
        # 5. Recompute rewards and done status
        # Note: This assumes env.compute_reward can handle batched inputs, which is true for gymnasium-robotics
        new_rewards = self.env.unwrapped.compute_reward(ag_next, new_goals, None)
        dones = (new_rewards == 0).astype(np.float32)
        
        # 6. Store the batch of new HER transitions into the replay buffer
        self.replay_buffer.store_batch(obs, ag, new_goals, actions, new_rewards, obs_next, ag_next, new_goals, dones)

    def update(self):
        # Get training data .type is tensor
        s, ach_goal, des_goal, a, r, s_, next_ach_goal, next_des_goal, done = self.replay_buffer.sample_minibatch(self.state_normalizer, self.goal_normalizer)  

        # Move data to the appropriate device
        s = s.to(self.device)
        ach_goal = ach_goal.to(self.device)
        des_goal = des_goal.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        s_ = s_.to(self.device)
        next_ach_goal = next_ach_goal.to(self.device)
        next_des_goal = next_des_goal.to(self.device)
        done = done.to(self.device)

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
