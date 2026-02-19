# MountainCar Reinforcement Learning: Q-Learning vs DQN

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1-brightgreen?style=for-the-badge&logo=openai&logoColor=white)](https://gymnasium.farama.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=for-the-badge)](https://github.com/spouladchang/Mountain-Car-RL-Optimization/graphs/commit-activity)

<div align="center">

**A comprehensive comparative study of Tabular Q-Learning and Deep Q-Network (DQN) for solving the classic MountainCar control problem**

[![Open in Streamlit](https://img.shields.io/badge/🚀_Open_Web_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://mountain-car-rl-optimization.streamlit.app/)

![MountainCar](https://gymnasium.farama.org/_images/mountain_car.gif)

[Features](#-features) • [Web Demo](#-interactive-web-demo) • [Quick Start](#-quick-start) • [Results](#-results) • [Documentation](#-documentation)

</div>

---

## 📋 Table of Contents

- [About](#-about)
- [Features](#-features)
- [Interactive Web Demo](#-interactive-web-demo)
- [Problem Definition](#-problem-definition)
- [Algorithms Implemented](#-algorithms-implemented)
  - [Q-Learning with Discretization](#1-tabular-q-learning-with-discretization)
  - [Deep Q-Network (DQN)](#2-deep-q-network-dqn)
- [Solution to Sparse Rewards](#-solution-to-sparse-rewards)
- [Results](#-results)
- [Showcase](#-showcase)
- [Quick Start](#-quick-start)
- [Repository Structure](#-repository-structure)
- [Key Takeaways](#-key-takeaways)
- [References](#-references)
- [License](#-license)
- [Citation](#-citation)

---

## 🎯 About

This repository provides a **comprehensive implementation and comparison** of two fundamental reinforcement learning algorithms—**Tabular Q-Learning** and **Deep Q-Network (DQN)**—applied to the classic MountainCar-v0 control problem from the Gymnasium library.

**The Challenge**: An underpowered car must reach a flag at the top of a steep hill. The car's engine lacks sufficient power to ascend directly, requiring the agent to learn a counter-intuitive strategy: driving backward to build momentum before accelerating forward. This problem elegantly demonstrates the importance of **exploration**, **delayed gratification**, and **strategic planning** in reinforcement learning.

**What Makes This Project Unique**:
- ✅ **Systematic hyperparameter optimization** through grid search and empirical analysis
- ✅ **Head-to-head algorithm comparison** with detailed performance metrics
- ✅ **Production-ready implementations** with pre-trained weights
- ✅ **Interactive web demo** for hands-on experimentation
- ✅ **Educational focus** with comprehensive documentation and visualizations
- ✅ **Sparse reward challenge** solved without reward shaping

This project serves as both a **learning resource** for RL practitioners and a **benchmark implementation** for solving control problems with sparse rewards.

### 🏷️ Topics

`reinforcement-learning` `deep-learning` `q-learning` `dqn` `deep-q-network` `pytorch` `gymnasium` `openai-gym` `machine-learning` `neural-networks` `control-theory` `hyperparameter-optimization` `sparse-rewards` `temporal-difference` `experience-replay` `streamlit` `interactive-demo` `mountaincar` `jupyter-notebook` `python`

---

## ✨ Features

- 🎓 **Two Complete Implementations**: Tabular Q-Learning and Deep Q-Network (DQN)
- 📊 **Systematic Hyperparameter Tuning**: Grid search results with performance analysis
- 🎯 **High Success Rates**: 75-88% (Q-Learning) and 90-98% (DQN)
- 💾 **Pre-trained Weights**: Ready-to-use models for immediate testing
- 📈 **Interactive Visualizations**: Training progress, convergence metrics, and performance plots
- 🌐 **Web Application**: Live Streamlit demo for experimentation
- 📓 **Jupyter Notebooks**: Well-documented, runnable code with markdown explanations
- 🎬 **Training Showcases**: GIF animations showing learning progression
- 🔬 **No Reward Shaping**: Solves the sparse reward challenge with proper tuning alone

---

## 🌐 Interactive Web Demo

**Try the algorithms yourself without installing anything!**

### 👉 **[Launch Interactive Web Demo](https://mountain-car-rl-optimization.streamlit.app/)**

<div align="center">

[![Streamlit Demo](https://img.shields.io/badge/🎮_Try_It_Now-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=262730)](https://mountain-car-rl-optimization.streamlit.app/)

</div>

**Web Demo Features**:
- 🎛️ **Adjust Hyperparameters**: Modify learning rates, decay schedules, network sizes, and more
- ⚖️ **Compare Algorithms**: Run Q-Learning vs DQN side-by-side
- 📊 **Real-time Visualization**: Watch training progress with interactive Plotly charts
- 🧪 **Experiment Freely**: Test different configurations and observe results
- 📱 **Mobile-Friendly**: Works on desktop, tablet, and mobile devices
- 🚀 **No Installation Required**: Just click and start learning!

**Perfect for**:
- Students learning reinforcement learning concepts
- Researchers exploring hyperparameter effects
- Educators demonstrating RL algorithms
- Practitioners prototyping before local training

---

## 📖 Problem Definition

**Environment**: MountainCar-v0 from Gymnasium

### State Space
- **Position**: Continuous range [-1.2, 0.6]
  - -1.2: Left boundary
  - -0.5: Valley bottom (starting position)
  - 0.5: Goal position (flag)
  
- **Velocity**: Continuous range [-0.07, 0.07]
  - Negative: Moving left
  - Zero: Stationary
  - Positive: Moving right

### Action Space
Three discrete actions:
- **0**: Accelerate Left (push car leftward)
- **1**: No Acceleration (coast)
- **2**: Accelerate Right (push car rightward)

### Reward Structure
- **-1** for every timestep until goal is reached
- **0** when the car reaches position ≥ 0.5
- **Maximum episode length**: 200 steps

### The Challenge
The **sparse reward structure** (-1 per step) provides minimal learning signal. The agent receives no positive reinforcement until randomly discovering the goal through exploration. This makes MountainCar an excellent testbed for:
- Exploration strategies
- Credit assignment over long time horizons
- Handling sparse rewards without reward shaping
- Temporal difference learning

**Key Insight**: The optimal strategy is counter-intuitive—the agent must first move *away* from the goal (backward up the left hill) to build momentum before accelerating toward the goal.

---

## 🤖 Algorithms Implemented

### 1. Tabular Q-Learning with Discretization

📓 **Implementation**: [`1_MountainCar_Q_Learning_Discretization.ipynb`](./1_MountainCar_Q_Learning_Discretization.ipynb)

#### Overview
Tabular Q-Learning discretizes the continuous state space into a finite grid and maintains a lookup table of Q-values for each state-action pair. Updates are performed using the Bellman optimality equation.

#### How It Works
1. **State Discretization**: Converts continuous (position, velocity) into discrete grid cells
2. **Q-Table Storage**: Maintains Q(s,a) values in a 25×25×3 table
3. **Epsilon-Greedy Exploration**: Balances exploration and exploitation
4. **Bellman Updates**: Iteratively improves value estimates

#### Advantages
- ✅ **Simple and Intuitive**: Easy to understand and implement
- ✅ **Interpretable**: Q-table can be visualized and analyzed
- ✅ **Guaranteed Convergence**: With proper exploration and learning rates
- ✅ **Minimal Dependencies**: Only requires NumPy
- ✅ **Fast Training**: ~70 seconds for 7000 episodes

#### Disadvantages
- ❌ **Manual Discretization**: Requires careful choice of grid granularity
- ❌ **Scalability**: Doesn't scale to high-dimensional state spaces
- ❌ **Sample Inefficiency**: Needs 7000 episodes to converge
- ❌ **Lower Success Rate**: 75-88% compared to DQN's 90-98%

#### Hyperparameter Optimization

We systematically tested different discretization granularities:

| Grid Size | States | Success Rate | Training Time | Convergence | Selection |
|-----------|--------|--------------|---------------|-------------|-----------|
| 15×15 | 225 | 65-75% | ~40s | Fast but imprecise | ❌ Too coarse |
| 20×20 | 400 | 70-80% | ~50s | Good baseline | ⚠️ Baseline |
| **25×25** | **625** | **75-88%** | **~70s** | **Optimal** | **✅ Best** |
| 30×30 | 900 | 72-85% | ~90s | Slower, diminishing returns | ❌ Overkill |

#### Optimal Configuration
```python
Buckets: 25×25                    # State space discretization
Learning Rate: 0.2 → 0.01         # Adaptive learning
LR Decay: 0.9998                  # Slow decay for sustained learning
Epsilon: 1.0 → 0.01               # Exploration schedule
Epsilon Decay: 0.9998             # Slow decay maintains exploration
Episodes: 7000                    # Sufficient for convergence
Discount Factor (γ): 0.99         # Long-term planning
```

**Why 25×25?** 
- Provides sufficient precision to distinguish critical states
- Avoids over-discretization that slows learning
- Balances Q-table size with convergence speed
- Empirically validated through grid search

---

### 2. Deep Q-Network (DQN)

📓 **Implementation**: [`2_MountainCar_DQN_PyTorch.ipynb`](./2_MountainCar_DQN_PyTorch.ipynb)

#### Overview
DQN uses a neural network to approximate the Q-function, eliminating the need for manual discretization. It incorporates **experience replay** and a **target network** for stable, sample-efficient learning.

#### Algorithm Components
1. **Neural Network Approximation**: 2 → 256 → 256 → 3 architecture
2. **Experience Replay**: Stores transitions in a buffer for batch learning
3. **Target Network**: Stabilizes Q-value targets during training
4. **Epsilon-Greedy Exploration**: Gradually shifts from exploration to exploitation

#### Advantages
- ✅ **No Discretization**: Handles continuous states naturally
- ✅ **Sample Efficient**: Only needs 2000 episodes (vs 7000 for Q-Learning)
- ✅ **High Performance**: 90-98% success rate
- ✅ **Scalable**: Extends to high-dimensional problems (vision, robotics)
- ✅ **Generalization**: Neural network learns smooth value functions

#### Disadvantages
- ❌ **Complexity**: Requires understanding of deep learning
- ❌ **Hyperparameter Sensitivity**: Performance highly dependent on tuning
- ❌ **Training Time**: ~30 minutes due to neural network computations
- ❌ **Potential Instability**: Can experience catastrophic forgetting
- ❌ **Less Interpretable**: Black-box nature of neural networks

#### Hyperparameter Optimization

DQN's performance critically depends on proper hyperparameter tuning:

##### Network Architecture
```python
Input Layer:        2 neurons (position, velocity)
Hidden Layer 1:   256 neurons + ReLU activation
Hidden Layer 2:   256 neurons + ReLU activation
Output Layer:       3 neurons (Q-values for each action)
```

##### Optimized Hyperparameters
```python
Learning Rate: 0.001              # Adam optimizer
Epsilon Decay: 0.995              # Slow decay - CRITICAL for sparse rewards
Replay Buffer: 50,000             # Large buffer for diverse experiences
Batch Size: 64                    # Mini-batch size
Target Update: Every 10 episodes  # Frequent updates for faster convergence
Episodes: 2000                    # Sufficient with experience replay
Discount Factor (γ): 0.99         # Long-term planning
```

#### Why These Values?

**Slow Epsilon Decay (0.995 vs typical 0.99)**:
- At episode 500: ε ≈ 0.08 (still exploring)
- At episode 1000: ε ≈ 0.006 (mostly exploiting)
- Essential for discovering rare successful trajectories in sparse reward environment

**Large Replay Buffer (50,000 vs typical 10,000)**:
- Stores more diverse experiences
- Reduces correlation between consecutive samples
- Enables learning from infrequent successful episodes

**Frequent Target Updates (10 vs typical 100 episodes)**:
- Faster incorporation of learned knowledge
- Better suited for rapidly changing Q-values early in training
- Balances stability and adaptation

---

## 🎯 Solution to Sparse Rewards

### The Core Challenge

MountainCar's **sparse reward structure** (-1 per timestep) provides almost no learning signal. The agent must:
1. Explore randomly until accidentally reaching the goal
2. Discover that moving *backward* (away from goal) is sometimes necessary
3. Learn the momentum-building strategy through trial and error

This is challenging because:
- Success probability via random actions: **< 0.01%**
- No intermediate rewards to guide learning
- Counter-intuitive optimal strategy (go backward first)

### Our Approach: Hyperparameter Tuning (No Reward Shaping)

We **deliberately avoided reward shaping** (adding intermediate rewards) to demonstrate that proper hyperparameter tuning alone can solve sparse reward problems.

#### Q-Learning Strategy
1. **Slow Decay Rates**: Maintain exploration longer (decay = 0.9998)
2. **Higher Minimum Learning Rate**: Continue adapting late in training (0.01 vs 0.001)
3. **Extended Training**: More episodes (7000) ensure eventual discovery
4. **Optimal Discretization**: 25×25 grid balances precision and efficiency

**Result**: 75-88% success rate without any reward modification

#### DQN Strategy
1. **Experience Replay**: Learn from rare successful episodes multiple times
2. **Large Buffer**: Store 50,000 diverse experiences
3. **Very Slow Epsilon Decay**: Sustain exploration for 1000+ episodes
4. **Neural Network Generalization**: Extrapolate from limited successful experiences

**Result**: 90-98% success rate without any reward modification

### Key Insight

**Proper hyperparameter tuning eliminates the need for reward shaping.** Both algorithms successfully learn optimal policies using only the environment's native sparse rewards, proving that:
- Exploration schedule is critical for sparse rewards
- Sample efficiency can be achieved through experience replay (DQN)
- Patient learning (slow decay) outperforms aggressive optimization

---

## 📊 Results

### Performance Comparison

| Metric | Q-Learning | DQN | Winner |
|--------|-----------|-----|--------|
| **Success Rate** | 75-88% | 90-98% | 🥇 DQN |
| **Training Episodes** | 7000 | 2000 | 🥇 DQN |
| **Training Time** | ~70 seconds | ~30 minutes* | 🥇 Q-Learning |
| **Time per Episode** | ~0.01s | ~0.9s | 🥇 Q-Learning |
| **Steps to Goal** (avg) | 160-180 | 100-110 | 🥇 DQN |
| **Average Reward** | -165 to -175 | -100 to -110 | 🥇 DQN |
| **Sample Efficiency** | ⭐⭐ | ⭐⭐⭐⭐⭐ | 🥇 DQN |
| **Simplicity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 🥇 Q-Learning |
| **Scalability** | ⭐⭐ | ⭐⭐⭐⭐⭐ | 🥇 DQN |
| **Interpretability** | ⭐⭐⭐⭐⭐ | ⭐⭐ | 🥇 Q-Learning |

*DQN takes longer per episode due to neural network computations, but achieves superior results in fewer episodes

### Q-Learning Results

**Final Performance**:
- Success Rate: **75-88%**
- Average Reward: **-165 to -175**
- Training: **7000 episodes** in **~70 seconds**
- State Space: **625 discrete states** (25×25 grid)

**Learning Curve**:

![Q-Learning Training](assets/qlearning_training.png)

*Three-panel visualization showing: (1) episode rewards improving over time, (2) maximum position gradually reaching the goal threshold (0.5), and (3) Q-value changes decreasing as the policy stabilizes, indicating convergence.*

**Key Observations**:
- Learning begins around episode 2000-3000
- Steady improvement through episode 5000
- Converges to stable policy by episode 7000
- Q-value changes drop logarithmically (indicating convergence)

---

### DQN Results

**Final Performance**:
- Success Rate: **90-98%**
- Average Reward: **-100 to -110**
- Training: **2000 episodes** in **~30 minutes**
- Best Average Reward: **-98.93** (100-episode window)
- Network Size: **256-256 neurons**

**Learning Curve**:

![DQN Training](assets/dqn_training.png)

*Four-panel visualization showing: (1) episode rewards, (2) maximum position reached, (3) neural network loss converging, and (4) epsilon decay ensuring sustained exploration throughout training.*

**Key Observations**:
- Rapid learning begins around episode 600-800
- Achieves 80%+ success rate by episode 1000
- Neural network loss stabilizes after episode 1500
- Epsilon maintains exploration until episode 1000+ (ε ≈ 0.08)

---

### Statistical Analysis

**Test Set Performance** (100 episodes, greedy policy):

| Algorithm | Mean Reward | Std Dev | Min Reward | Max Reward | Median Steps |
|-----------|-------------|---------|------------|------------|--------------|
| Q-Learning | -169.4 | ±23.7 | -200 | -107 | 167 |
| DQN | -105.2 | ±8.9 | -200 | -89 | 104 |

**Interpretation**:
- DQN shows **lower variance** (more consistent)
- DQN achieves **better worst-case** performance
- Q-Learning occasionally fails (hits 200-step limit)
- DQN solves problem **~40% faster** on average

---

## 🎬 Showcase

Watch the DQN agent learn to solve MountainCar across different training stages:

### Early Training (Episode 500)
![Episode 500](assets/dqn_episode_500.gif)

**Behavior**: Agent explores randomly, struggles to escape the valley. Maximum position around **-0.60** (far left hill). No understanding of momentum strategy yet.

**Metrics**: 
- Success Rate: ~0%
- Average Reward: -200 (timeout)
- Epsilon: ~0.08 (still heavily exploring)

---

### Mid Training (Episode 1000)
![Episode 1000](assets/dqn_episode_1000.gif)

**Behavior**: Agent discovers oscillation strategy, occasionally reaches the goal. Maximum position frequently reaches **~0.50** (goal threshold). Learning momentum-building technique.

**Metrics**:
- Success Rate: ~60-80%
- Average Reward: -130 to -150
- Epsilon: ~0.006 (mostly exploiting)

---

### Final Training (Episode 2000)
![Episode 2000](assets/dqn_episode_2000.gif)

**Behavior**: Agent consistently solves the problem in **~100 steps** with optimal policy. Efficient momentum building and precise goal approach. No wasted actions.

**Metrics**:
- Success Rate: ~95%+
- Average Reward: -100 to -110
- Epsilon: ~0.01 (full exploitation)

---

## 🚀 Quick Start

### Prerequisites

```bash
Python >= 3.8
pip >= 21.0
```

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/spouladchang/Mountain-Car-RL-Optimization.git
cd Mountain-Car-RL-Optimization
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Running the Notebooks

3. **Launch Jupyter**:
```bash
jupyter notebook
```

4. **Run a notebook**:
   - **Q-Learning**: Open `1_MountainCar_Q_Learning_Discretization.ipynb`
   - **DQN**: Open `2_MountainCar_DQN_PyTorch.ipynb`
   
   Execute cells sequentially (Shift + Enter)

### Using Pre-trained Weights

Both implementations include **pre-trained weights** for immediate testing:

- **Q-Learning**: `trained_weights_discretization.npz` (auto-loads)
- **DQN**: `trained_weights_dqn.pth` (auto-loads)

**The notebooks automatically detect and load these files**, allowing you to:
- ✅ Skip training and see results immediately
- ✅ Test the trained agents
- ✅ Visualize performance
- ✅ Generate animations

### Training from Scratch

To retrain the agents:

**Option 1**: Delete weight files
```bash
rm trained_weights_discretization.npz
rm trained_weights_dqn.pth
```

**Option 2**: Set `RETRAIN = True` in the notebook

Then run the training cells.

**Expected Training Times**:
- Q-Learning: ~70 seconds (7000 episodes)
- DQN: ~30 minutes (2000 episodes)

### Running the Web App Locally

```bash
pip install streamlit
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📁 Repository Structure

```
Mountain-Car-RL-Optimization/
│
├── 📓 1_MountainCar_Q_Learning_Discretization.ipynb    # Q-Learning implementation
├── 📓 2_MountainCar_DQN_PyTorch.ipynb                  # DQN implementation
│
├── 💾 trained_weights_discretization.npz               # Pre-trained Q-table
├── 💾 trained_weights_dqn.pth                          # Pre-trained DQN weights
│
├── 🌐 app.py                                           # Streamlit web application
├── 📋 requirements.txt                                 # Python dependencies
├── 📄 README.md                                        # This file
├── 📜 LICENSE                                          # MIT License
│
└── 📁 assets/                                          # Visualizations & GIFs
    ├── qlearning_training.png                         # Q-Learning results plot
    ├── dqn_training.png                               # DQN results plot
    ├── dqn_episode_500.gif                            # Early training GIF
    ├── dqn_episode_1000.gif                           # Mid training GIF
    └── dqn_episode_2000.gif                           # Final training GIF
```

---

## 💡 Key Takeaways

### When to Use Q-Learning
- ✅ **Learning RL fundamentals**: Best for educational purposes
- ✅ **Low-dimensional problems**: 2-3 state variables
- ✅ **Interpretability required**: Need to understand decision-making
- ✅ **Limited computational resources**: Fast training on CPU
- ✅ **Simple deployment**: No deep learning frameworks needed

**Example Use Cases**: Grid worlds, simple control, pedagogical demonstrations

### When to Use DQN
- ✅ **High performance needed**: Mission-critical applications
- ✅ **Complex state spaces**: Vision, continuous control, robotics
- ✅ **Sample efficiency matters**: Limited interaction budget
- ✅ **Scalability required**: Extends to high-dimensional problems
- ✅ **Transfer learning potential**: Pre-trained networks can be fine-tuned

**Example Use Cases**: Atari games, robotic control, autonomous systems

### Universal Insights

1. **Sparse Rewards Are Solvable**: Both methods succeed without reward shaping when properly tuned

2. **Hyperparameter Tuning Is Critical**: 
   - Poor tuning: 50-60% success
   - Optimal tuning: 75-98% success
   - Impact > algorithm choice

3. **Exploration vs Exploitation Trade-off**:
   - Too fast decay → gets stuck at 50-60%
   - Too slow decay → wastes episodes
   - Optimal: Slow, sustained exploration

4. **Sample Efficiency vs Simplicity**:
   - Q-Learning: Simple but needs 3.5× more episodes
   - DQN: Complex but learns faster

5. **The Credit Assignment Problem**:
   - Moving backward is crucial but receives same reward (-1)
   - Temporal difference learning solves this
   - Experience replay amplifies rare successful episodes (DQN)

---

## 📚 References

### Academic Papers

1. **Watkins, C.J.C.H. and Dayan, P.** (1992). "Q-learning". *Machine Learning*, 8(3-4): 279-292.
   - Original Q-Learning algorithm paper

2. **Mnih, V. et al.** (2015). "Human-level control through deep reinforcement learning". *Nature*, 518(7540): 529-533.
   - DQN breakthrough on Atari games

3. **Sutton, R. S. and Barto, A. G.** (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
   - Comprehensive RL textbook

4. **Van Hasselt, H., Guez, A., and Silver, D.** (2016). "Deep reinforcement learning with double Q-learning". *AAAI*, 30(1).
   - Addresses Q-value overestimation

### Documentation & Resources

5. **Gymnasium Documentation**: https://gymnasium.farama.org/
   - Official environment documentation

6. **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html
   - Deep learning framework reference

7. **Streamlit Documentation**: https://docs.streamlit.io/
   - Web app framework guide

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Summary**: You are free to use, modify, and distribute this code for any purpose, including commercial applications, with attribution.

---

## 👤 Author

**Saeid Pouladchang**

- 🌐 GitHub: [@spouladchang](https://github.com/spouladchang)
- 📧 Email: saeedpooladchang78@gmail.com
- 💼 LinkedIn: [Connect with me](https://www.linkedin.com/in/spouladchang/)

---

## 🌟 Show Your Support

If you find this repository helpful, please consider:

- ⭐ **Starring the repository** on GitHub
- 🍴 **Forking** for your own projects
- 📢 **Sharing** with the RL community
- 🐛 **Reporting issues** or suggesting improvements
- 🤝 **Contributing** via pull requests

**Your support motivates continued development and improvement!**

---

## 📖 Citation

If you use this code in your research or project, please cite:

```bibtex
@misc{pouladchang2026mountaincar,
  author = {Pouladchang, Saeid},
  title = {MountainCar Reinforcement Learning: A Comparative Study of Q-Learning and Deep Q-Networks},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/spouladchang/Mountain-Car-RL-Optimization}},
  note = {Interactive web demo: \url{https://mountain-car-rl-optimization.streamlit.app/}}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a **Pull Request**.

**Areas for Contribution**:
- Additional algorithms (SARSA, Actor-Critic, PPO)
- Hyperparameter search automation (Optuna, Ray Tune)
- Enhanced visualizations
- Performance optimizations
- Documentation improvements
- Bug fixes

---

## 📝 Changelog

### Version 1.0.0 (2026-02-20)
- Initial release
- Q-Learning and DQN implementations
- Pre-trained weights
- Interactive Streamlit web demo
- Comprehensive documentation

---

<div align="center">

**⭐ Star this repository if it helped you learn reinforcement learning! ⭐**

[![GitHub stars](https://img.shields.io/github/stars/spouladchang/Mountain-Car-RL-Optimization?style=social)](https://github.com/spouladchang/Mountain-Car-RL-Optimization/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/spouladchang/Mountain-Car-RL-Optimization?style=social)](https://github.com/spouladchang/Mountain-Car-RL-Optimization/network/members)

Made with ❤️ by [Saeid Pouladchang](https://github.com/spouladchang)

</div>
