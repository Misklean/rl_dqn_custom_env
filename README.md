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
- **Episode 50**:  
![ep_50](https://github.com/user-attachments/assets/93910b1f-d6f0-42c7-9be9-895efacf15a4)
- **Episode 200**:
![ep_200](https://github.com/user-attachments/assets/d6e3042d-2afb-4f98-90e8-340a7835a6ca)
- **Episode 500**:
![ep_500](https://github.com/user-attachments/assets/dfa9a5fd-e627-49e5-b7de-9614e203ef16)
- **Episode 700**:
![ep_700](https://github.com/user-attachments/assets/7448c61f-57d0-4873-bb16-a17f37c58238)
- **Episode 1000**:
![ep_1000](https://github.com/user-attachments/assets/6426c094-8e6e-4fca-a288-1dd37d40d708)

### **Rewards**
Below is the rewards plot showing the agent's progress during training:
![agent_rewards_plot](https://github.com/user-attachments/assets/78e69f47-a5a2-4fbf-beb8-1bfe3f729c4b)

---

## **Future Improvements**
- Experiment with advanced algorithms like **Proximal Policy Optimization (PPO)** or **A3C**.
- Add **Prioritized Experience Replay (PER)** to improve training efficiency.
- Incorporate **multi-agent support** for more complex environments.
- Expand the environment with additional game levels or mechanics.
