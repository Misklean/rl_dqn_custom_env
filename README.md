# **DQN-Based Reinforcement Learning Project**

## **Overview**
This project implements a **Deep Q-Network (DQN)** to train an agent in a custom environment. The agent learns optimal policies to maximize rewards through trial-and-error interactions with the environment. The project supports **manual gameplay for data collection** and uses **frame stacking** to improve decision-making capabilities.

---

## **Features**
- **Custom Environment**: Built using Pygame, providing dynamic rendering and the ability to train or play manually.
- **Manual Gameplay**: Allows users to control the agent manually to collect gameplay data for supervised pre-training.
- **Frame Stacking**: Captures and processes a sequence of four frames as input, enabling the agent to learn temporal dependencies in the environment.
- **Epsilon Decay**: Gradually decreases the exploration rate epsilon during training to transition from exploration to exploitation.

---

## **Algorithm**
### **Deep Q-Network (DQN)**
- Learns to approximate the **Q-value function**, representing the expected future reward of actions in given states.
- Uses a **replay buffer** to store transitions and enable stable training by sampling randomized batches.
- Incorporates **Double Q-learning** to reduce overestimation of Q-values by using the main network to select actions and a target network to evaluate them.
- Implementation inspired by the [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).

### **Behavior Imitation**
Behavior Imitation allows the agent to mimic human gameplay:
1. **Data Collection**: A human player plays several episodes while the agent records states, actions, and rewards in the replay buffer.
2. **Supervised Pre-training**: The agent trains on the collected data, learning policies based on human behavior.
3. **Reinforcement Learning**: After pre-training, the agent continues training autonomously using reinforcement learning, improving policies through trial and error.

---

## **Libraries Used**
- **Pygame**: Custom environment and rendering for manual and automated gameplay.
- **PyTorch**: Deep learning framework used for implementing and training the DQN.
- **Gymnasium**: Provides a structured environment framework for reinforcement learning.
- **NumPy**: Supports efficient numerical and matrix computations.
- **Matplotlib**: Visualization of training metrics, such as reward progression.

---

## **Usage**
1. **Setup**:
   - Install dependencies:  
     ```bash
     pip install -r requirements.txt
     ```

2. **Run Training**:
   - Start training the agent using DQN and Behavior Imitation:  
     ```bash
     python train_agent.py
     ```
   - Modify hyperparameters in `config.py` if needed.

3. **Play Manually**:
   - Play the game manually to collect gameplay data:  
     ```bash
     python play_game.py
     ```

4. **Evaluate**:
   - Analyze the training results, including saved videos and reward plots, in the `results/` directory.

---

## **Results**
### **Videos**
Here are training highlights from different episodes:
- **Episode 50**: [Insert link or path]  
- **Episode 200**: [Insert link or path]  
- **Episode 500**: [Insert link or path]  
- **Episode 700**: [Insert link or path]  
- **Episode 1000**: [Insert link or path]  

### **Rewards**
Below is the rewards plot showing the agent's progress during training:
- [Insert rewards plot image or path]

---

## **Future Improvements**
- Experiment with advanced algorithms like **Proximal Policy Optimization (PPO)** or **A3C**.
- Add **Prioritized Experience Replay (PER)** to improve training efficiency.
- Incorporate **multi-agent support** for more complex environments.
- Expand the environment with additional game levels or mechanics.