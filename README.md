# DDPG + HER (Hindsight Experience Replay) Implementation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1%2Bcu118-orange)](https://pytorch.org/get-started/previous-versions/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1-green)](https://gymnasium.farama.org/)

My Article : https://hackmd.io/@bGCXESmGSgeAArScMaBxLA/H1thwo4lbx
This repository contains a clean, modular implementation of **Deep Deterministic Policy Gradient (DDPG)** combined with **Hindsight Experience Replay (HER)** using PyTorch.

It is designed to solve sparse-reward robotic control tasks, specifically demonstrating performance on the **Fetch** environments from Gymnasium.

## 📂 Project Structure

```text
.
├── DDPG_HER/           # Core algorithm implementation (Agent, Actor, Critic, Buffer)
├── FetchPush.py        # Training script for FetchPush-v1 (Hard task)
├── FetchReach.py       # Training script for FetchReach-v1 (Easy task)
├── Plot/               # Stores training loss/reward curves
├── Video/              # Stores rendered videos of the agent
└── .vscode/            # VS Code configuration
```

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/jason19990305/DDPG_HER.git
cd DDPG_HER
```

### 2. Install Dependencies
It is recommended to use a virtual environment (Conda or venv). The project is tested with **PyTorch 2.2.1 (CUDA 11.8)** and **Gymnasium 0.29.1**.

```bash
# 1. Install PyTorch with CUDA 11.8 support
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118

# 2. Install Gymnasium and other dependencies
pip install gymnasium==0.29.1 "gymnasium[robotics]" numpy matplotlib
```

*(Note: If you run into MuJoCo errors, please ensure you have `mujoco` installed properly according to your OS.)*

## 🖥️ Usage

### Train on FetchReach (Sanity Check)
The "Reach" task is simple and useful for verifying the algorithm works.
```bash
python FetchReach.py
```

### Train on FetchPush (HER Demonstration)
The "Push" task requires the robot to push a block to a target. This is where HER significantly improves learning efficiency.
```bash
python FetchPush.py
```

## 📊 Results & Visualization

- **Plots**: Training curves (Success Rate & Reward) will be saved in the `Plot/` directory.
- **Videos**: If rendering/recording is enabled in the script, videos will be saved in the `Video/` directory.

**FetchReach** : 

<img width="320" height="240" alt="FetchReach-v4_training_curve" src="https://github.com/user-attachments/assets/61c688c1-1e5e-4328-aeb9-d8d931dda12a" />

**FetchPush** : 

<img width="320" height="240" alt="FetchPush-v4_training_curve" src="https://github.com/user-attachments/assets/8263c9d9-ce29-4d82-bb2d-4e1233b4fb74" />

**FetchPickAndPlace** : 

<img width="320" height="240" alt="FetchPickAndPlace-v4_training_curve" src="https://github.com/user-attachments/assets/c9faa905-9ff7-43d1-aebc-0a793ad00b93" />

**FetchSlide** :

<img width="320" height="240" alt="FetchSlide-v4_training_curve" src="https://github.com/user-attachments/assets/9331d439-2596-4f92-8b2e-5a1e5b08657e" />




