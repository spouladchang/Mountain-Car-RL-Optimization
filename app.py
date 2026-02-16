import streamlit as st
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random

st.set_page_config(
    page_title="MountainCar RL Demo",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #ffffff;
        border-radius: 8px 8px 0 0;
        border: 1px solid #dee2e6;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #007bff;
        color: white;
    }

    .card-box {
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
        height: 100%; 
    }
    .card-box:hover {
        transform: translateY(-5px);
    }
    
    .objective-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 6px solid #2196f3;
        color: #0d47a1;
    }
    
    .reward-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 6px solid #4caf50;
        color: #1b5e20;
    }
    
    .action-box {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border-left: 6px solid #9c27b0;
        color: #4a148c;
    }

    .card-box h4 {
        margin: 0 0 10px 0;
        font-weight: 700;
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-box p {
        margin: 0;
        font-size: 0.95rem;
        line-height: 1.5;
        opacity: 0.9;
    }
    
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        color: #856404;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box-generic {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        color: #0c5460;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

device = torch.device('cpu')

class DQNetwork(nn.Module):
    def __init__(self, state_dim=2, action_dim=3, hidden_size=256):
        super(DQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

def discretize_qlearning(state, lower_bounds, upper_bounds, buckets):
    scaling = (state - lower_bounds) / (upper_bounds - lower_bounds)
    indices = (scaling * buckets).astype(int)
    return tuple(np.clip(indices, 0, np.array(buckets) - 1))

def train_qlearning(episodes, buckets, learning_rate, lr_decay, epsilon_decay, gamma=0.99):
    env = gym.make('MountainCar-v0')
    lower_bounds = np.array([-1.2, -0.07])
    upper_bounds = np.array([0.6, 0.07])
    
    q_table = np.zeros(buckets + (3,))
    lr = learning_rate
    epsilon = 1.0
    min_lr = 0.01
    min_epsilon = 0.01
    
    rewards_history = []
    positions_history = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = discretize_qlearning(state, lower_bounds, upper_bounds, buckets)
        episode_reward = 0
        max_position = -1.2
        done = False
        
        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state_discrete = discretize_qlearning(next_obs, lower_bounds, upper_bounds, buckets)
            done = terminated or truncated
            
            old_value = q_table[state][action]
            next_value = 0 if terminated else np.max(q_table[next_state_discrete])
            new_value = old_value + lr * (reward + gamma * next_value - old_value)
            q_table[state][action] = new_value
            
            state = next_state_discrete
            episode_reward += reward
            max_position = max(max_position, next_obs[0])
        
        rewards_history.append(episode_reward)
        positions_history.append(max_position)
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        lr = max(min_lr, lr * lr_decay)
        
        if (episode + 1) % max(1, episodes // 20) == 0:
            progress = (episode + 1) / episodes
            progress_bar.progress(progress)
            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
            status_text.text(f"Episode {episode+1}/{episodes} | Avg Reward: {avg_reward:.1f}")
    
    env.close()
    progress_bar.empty()
    status_text.empty()
    
    policy = np.argmax(q_table, axis=2)
    
    return {
        'q_table': q_table,
        'policy': policy,
        'rewards': rewards_history,
        'positions': positions_history
    }

def test_qlearning(policy, buckets, num_episodes=100):
    env = gym.make('MountainCar-v0')
    lower_bounds = np.array([-1.2, -0.07])
    upper_bounds = np.array([0.6, 0.07])
    
    successes = 0
    rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        state = discretize_qlearning(state, lower_bounds, upper_bounds, buckets)
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 200:
            action = policy[state]
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = discretize_qlearning(next_state, lower_bounds, upper_bounds, buckets)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        rewards.append(episode_reward)
        if next_state[0] >= 0.5:
            successes += 1
    
    env.close()
    
    return {
        'success_rate': (successes / num_episodes) * 100,
        'avg_reward': np.mean(rewards),
        'rewards': rewards
    }

class DQNAgent:
    def __init__(self, hidden_size, learning_rate, gamma, epsilon_decay, batch_size, buffer_size):
        self.policy_net = DQNetwork(hidden_size=hidden_size).to(device)
        self.target_net = DQNetwork(hidden_size=hidden_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(3)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            return self.policy_net(state_tensor).argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([t[0] for t in batch])).to(device)
        actions = torch.LongTensor(np.array([t[1] for t in batch])).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(np.array([t[2] for t in batch])).to(device)
        next_states = torch.FloatTensor(np.array([t[3] for t in batch])).to(device)
        dones = torch.FloatTensor(np.array([t[4] for t in batch])).to(device)
        
        current_q = self.policy_net(states).gather(1, actions).squeeze()
        
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (self.gamma * next_q * (1 - dones))
        
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def train_dqn(episodes, hidden_size, learning_rate, epsilon_decay, batch_size, buffer_size, 
              target_update_freq, gamma=0.99):
    
    env = gym.make('MountainCar-v0')
    agent = DQNAgent(hidden_size, learning_rate, gamma, epsilon_decay, batch_size, buffer_size)
    
    rewards_history = []
    positions_history = []
    losses_history = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        max_position = state[0]
        episode_losses = []
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()
            if loss is not None:
                episode_losses.append(loss)
            
            state = next_state
            episode_reward += reward
            max_position = max(max_position, next_state[0])
        
        agent.update_epsilon()
        
        if (episode + 1) % target_update_freq == 0:
            agent.update_target()
        
        rewards_history.append(episode_reward)
        positions_history.append(max_position)
        losses_history.append(np.mean(episode_losses) if episode_losses else 0)
        
        if (episode + 1) % max(1, episodes // 20) == 0:
            progress = (episode + 1) / episodes
            progress_bar.progress(progress)
            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
            status_text.text(f"Episode {episode+1}/{episodes} | Avg Reward: {avg_reward:.1f} | ε: {agent.epsilon:.3f}")
    
    env.close()
    progress_bar.empty()
    status_text.empty()
    
    return {
        'agent': agent,
        'rewards': rewards_history,
        'positions': positions_history,
        'losses': losses_history
    }

def test_dqn(agent, num_episodes=100):
    env = gym.make('MountainCar-v0')
    agent.policy_net.eval()
    
    successes = 0
    rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 200:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = agent.policy_net(state_tensor).argmax().item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        rewards.append(episode_reward)
        if next_state[0] >= 0.5:
            successes += 1
    
    env.close()
    
    return {
        'success_rate': (successes / num_episodes) * 100,
        'avg_reward': np.mean(rewards),
        'rewards': rewards
    }

def plot_training_results(rewards, positions, algorithm):
    
    window = min(100, len(rewards) // 10)
    if window > 0:
        rewards_ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        positions_ma = np.convolve(positions, np.ones(window)/window, mode='valid')
    else:
        rewards_ma = rewards
        positions_ma = positions
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Episode Rewards', 'Maximum Position Reached'),
        vertical_spacing=0.12
    )
    
    fig.add_trace(
        go.Scatter(y=rewards, name='Raw Reward', opacity=0.3, 
                   line=dict(color='lightcoral', width=1)),
        row=1, col=1
    )
    if window > 0:
        fig.add_trace(
            go.Scatter(x=list(range(window-1, len(rewards))), y=rewards_ma,
                       name=f'{window}-Episode MA', line=dict(color='darkred', width=2)),
            row=1, col=1
        )
    fig.add_hline(y=-110, line_dash="dash", line_color="green",
                  annotation_text="Success Threshold", row=1, col=1)
    
    fig.add_trace(
        go.Scatter(y=positions, name='Max Position', opacity=0.3,
                   line=dict(color='lightblue', width=1)),
        row=2, col=1
    )
    if window > 0:
        fig.add_trace(
            go.Scatter(x=list(range(window-1, len(positions))), y=positions_ma,
                       name=f'{window}-Episode MA', line=dict(color='darkblue', width=2)),
            row=2, col=1
        )
    fig.add_hline(y=0.5, line_dash="dash", line_color="green",
                  annotation_text="Goal", row=2, col=1)
    
    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Position", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text=f"{algorithm} Training Progress",
        title_font_size=20,
        hovermode='x unified'
    )
    
    return fig

def create_performance_comparison(results_dict):
    
    fig = go.Figure()
    
    algorithms = list(results_dict.keys())
    success_rates = [results_dict[alg]['success_rate'] for alg in algorithms]
    avg_rewards = [results_dict[alg]['avg_reward'] for alg in algorithms]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Success Rate (%)', 'Average Reward'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    fig.add_trace(
        go.Bar(x=algorithms, y=success_rates, name='Success Rate',
               marker_color='lightgreen'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=algorithms, y=avg_rewards, name='Avg Reward',
               marker_color='lightblue'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False, title_text="Performance Comparison")
    
    return fig

def main():
    
    st.title("🏔️ MountainCar Reinforcement Learning")
    
    st.image("https://gymnasium.farama.org/_images/mountain_car.gif", caption="MountainCar Environment", use_column_width=False, width=600)

    st.markdown("""
    **Interactive Demo**: Compare Q-Learning and Deep Q-Network (DQN) algorithms on the classic MountainCar problem.
    
    Adjust hyperparameters, train models, and visualize results in real-time!
    """)
    
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")
    
    algorithm = st.sidebar.selectbox(
        "Select Algorithm",
        ["Q-Learning (Tabular)", "Deep Q-Network (DQN)", "Compare Both"],
        help="Choose which algorithm to train and evaluate"
    )
    
    st.sidebar.markdown("---")
    
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    
    if algorithm in ["Q-Learning (Tabular)", "Compare Both"]:
        st.sidebar.subheader("📊 Q-Learning Parameters")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            bucket_x = st.number_input("Buckets (Position)", 10, 50, 25, 5,
                                       help="State space discretization for position")
        with col2:
            bucket_y = st.number_input("Buckets (Velocity)", 10, 50, 25, 5,
                                       help="State space discretization for velocity")
        
        buckets = (bucket_x, bucket_y)
        
        ql_episodes = st.sidebar.slider("Training Episodes", 1000, 10000, 7000, 500,
                                        help="Number of episodes to train")
        
        ql_lr = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.2, 0.01,
                                   help="Initial learning rate (α)")
        
        ql_lr_decay = st.sidebar.slider("LR Decay", 0.995, 0.9999, 0.9998, 0.0001,
                                        help="Learning rate decay factor")
        
        ql_eps_decay = st.sidebar.slider("Epsilon Decay", 0.995, 0.9999, 0.9998, 0.0001,
                                         help="Exploration rate decay")
        
        st.sidebar.markdown("---")
    
    if algorithm in ["Deep Q-Network (DQN)", "Compare Both"]:
        st.sidebar.subheader("🧠 DQN Parameters")
        
        dqn_episodes = st.sidebar.slider("Training Episodes ", 500, 3000, 2000, 100,
                                         help="Number of episodes to train")
        
        hidden_size = st.sidebar.select_slider("Network Size", [128, 256, 512], 256,
                                               help="Number of hidden units")
        
        dqn_lr = st.sidebar.slider("Learning Rate ", 0.0001, 0.01, 0.001, 0.0001,
                                    help="Adam optimizer learning rate")
        
        epsilon_decay = st.sidebar.slider("Epsilon Decay ", 0.99, 0.999, 0.995, 0.001,
                                          help="Exploration rate decay")
        
        batch_size = st.sidebar.select_slider("Batch Size", [32, 64, 128], 64,
                                              help="Mini-batch size for training")
        
        buffer_size = st.sidebar.select_slider("Replay Buffer", [10000, 50000, 100000], 50000,
                                               help="Experience replay buffer size")
        
        target_update = st.sidebar.slider("Target Update Freq", 5, 50, 10, 5,
                                          help="Episodes between target network updates")
    
    st.sidebar.markdown("---")
    
    train_button = st.sidebar.button("🚀 Train Model", type="primary", use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card-box objective-box">
            <h4>🎯 Objective</h4>
            <p>Drive an underpowered car up a steep hill by building momentum (swinging back and forth).</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card-box reward-box">
            <h4>📈 Reward Structure</h4>
            <p><strong>-1</strong> per timestep.<br>Goal is to minimize negative reward (reach flag fast).</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card-box action-box">
            <h4>🎮 Action Space</h4>
            <p>Discrete(3):<br>0: Accelerate Left<br>1: Don't Accelerate<br>2: Accelerate Right</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if train_button:
        
        if algorithm == "Q-Learning (Tabular)":
            
            st.header("📊 Q-Learning Training")
            
            with st.spinner("Training Q-Learning agent..."):
                start_time = time.time()
                results = train_qlearning(ql_episodes, buckets, ql_lr, ql_lr_decay, ql_eps_decay)
                training_time = time.time() - start_time
            
            st.success(f"✅ Training completed in {training_time:.1f} seconds!")
            
            with st.spinner("Testing trained agent..."):
                test_results = test_qlearning(results['policy'], buckets)
            
            st.session_state.trained_models['Q-Learning'] = {
                'train': results,
                'test': test_results,
                'time': training_time,
                'params': {'buckets': buckets, 'episodes': ql_episodes}
            }
            
            tab1, tab2, tab3 = st.tabs(["📈 Training Progress", "🎯 Performance", "ℹ️ Details"])
            
            with tab1:
                fig = plot_training_results(results['rewards'], results['positions'], "Q-Learning")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Success Rate", f"{test_results['success_rate']:.1f}%")
                with col2:
                    st.metric("Avg Reward", f"{test_results['avg_reward']:.1f}")
                with col3:
                    st.metric("Training Time", f"{training_time:.1f}s")
                with col4:
                    st.metric("Episodes", f"{ql_episodes}")
                
                if test_results['success_rate'] >= 80:
                    st.markdown("""
                    <div class="success-box">
                        <strong>Excellent Performance!</strong> The agent successfully learned to solve the task.
                    </div>
                    """, unsafe_allow_html=True)
                elif test_results['success_rate'] >= 60:
                    st.markdown("""
                    <div class="warning-box">
                        <strong>Good Performance.</strong> The agent learned the task but could be improved with more training or better hyperparameters.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="warning-box">
                        <strong>Needs Improvement.</strong> Consider increasing episodes or adjusting learning rate decay.
                    </div>
                    """, unsafe_allow_html=True)
            
            with tab3:
                st.subheader("Configuration")
                st.json({
                    'Buckets': f"{buckets[0]}×{buckets[1]}",
                    'States': buckets[0] * buckets[1],
                    'Episodes': ql_episodes,
                    'Learning Rate': f"{ql_lr} → 0.01",
                    'LR Decay': ql_lr_decay,
                    'Epsilon Decay': ql_eps_decay
                })
        
        elif algorithm == "Deep Q-Network (DQN)":
            
            st.header("🧠 DQN Training")
            
            with st.spinner("Training DQN agent..."):
                start_time = time.time()
                results = train_dqn(dqn_episodes, hidden_size, dqn_lr, epsilon_decay,
                                   batch_size, buffer_size, target_update)
                training_time = time.time() - start_time
            
            st.success(f"✅ Training completed in {training_time:.1f} seconds!")
            
            with st.spinner("Testing trained agent..."):
                test_results = test_dqn(results['agent'])
            
            st.session_state.trained_models['DQN'] = {
                'train': results,
                'test': test_results,
                'time': training_time,
                'params': {'hidden_size': hidden_size, 'episodes': dqn_episodes}
            }
            
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Training Progress", "📉 Loss Curve", "🎯 Performance", "ℹ️ Details"])
            
            with tab1:
                fig = plot_training_results(results['rewards'], results['positions'], "DQN")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                fig = go.Figure()
                window = min(100, len(results['losses']) // 10)
                if window > 0:
                    loss_ma = np.convolve(results['losses'], np.ones(window)/window, mode='valid')
                    fig.add_trace(go.Scatter(y=results['losses'], name='Loss', opacity=0.3,
                                             line=dict(color='lightcoral')))
                    fig.add_trace(go.Scatter(x=list(range(window-1, len(results['losses']))),
                                             y=loss_ma, name=f'{window}-Episode MA',
                                             line=dict(color='darkred', width=2)))
                else:
                    fig.add_trace(go.Scatter(y=results['losses'], name='Loss',
                                             line=dict(color='darkred')))
                
                fig.update_layout(title="Training Loss", xaxis_title="Episode",
                                 yaxis_title="MSE Loss", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Success Rate", f"{test_results['success_rate']:.1f}%")
                with col2:
                    st.metric("Avg Reward", f"{test_results['avg_reward']:.1f}")
                with col3:
                    st.metric("Training Time", f"{training_time:.1f}s")
                with col4:
                    st.metric("Episodes", f"{dqn_episodes}")
                
                if test_results['success_rate'] >= 90:
                    st.markdown("""
                    <div class="success-box">
                        <strong>Excellent Performance!</strong> DQN successfully mastered the task.
                    </div>
                    """, unsafe_allow_html=True)
                elif test_results['success_rate'] >= 70:
                    st.markdown("""
                    <div class="warning-box">
                        <strong>Good Performance.</strong> The network learned well but could benefit from more training.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="warning-box">
                        <strong>Needs Improvement.</strong> Consider increasing network size or training episodes.
                    </div>
                    """, unsafe_allow_html=True)
            
            with tab4:
                st.subheader("Configuration")
                st.json({
                    'Network': f'{hidden_size}-{hidden_size}',
                    'Episodes': dqn_episodes,
                    'Learning Rate': dqn_lr,
                    'Epsilon Decay': epsilon_decay,
                    'Batch Size': batch_size,
                    'Replay Buffer': buffer_size,
                    'Target Update': f'Every {target_update} episodes'
                })
        
        else:
            
            st.header("⚖️ Algorithm Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Q-Learning")
                with st.spinner("Training Q-Learning..."):
                    start_time = time.time()
                    ql_results = train_qlearning(ql_episodes, buckets, ql_lr, ql_lr_decay, ql_eps_decay)
                    ql_time = time.time() - start_time
                    ql_test = test_qlearning(ql_results['policy'], buckets)
                st.success(f"✅ Done in {ql_time:.1f}s")
                st.metric("Success Rate", f"{ql_test['success_rate']:.1f}%")
                st.metric("Avg Reward", f"{ql_test['avg_reward']:.1f}")
            
            with col2:
                st.subheader("🧠 DQN")
                with st.spinner("Training DQN..."):
                    start_time = time.time()
                    dqn_results = train_dqn(dqn_episodes, hidden_size, dqn_lr, epsilon_decay,
                                           batch_size, buffer_size, target_update)
                    dqn_time = time.time() - start_time
                    dqn_test = test_dqn(dqn_results['agent'])
                st.success(f"✅ Done in {dqn_time:.1f}s")
                st.metric("Success Rate", f"{dqn_test['success_rate']:.1f}%")
                st.metric("Avg Reward", f"{dqn_test['avg_reward']:.1f}")
            
            st.markdown("---")
            
            comparison_data = {
                'Q-Learning': ql_test,
                'DQN': dqn_test
            }
            fig = create_performance_comparison(comparison_data)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📊 Analysis")
            
            winner = "DQN" if dqn_test['success_rate'] > ql_test['success_rate'] else "Q-Learning"
            
            st.markdown(f"""
            <div class="info-box-generic">
                <h4>🏆 Winner: {winner}</h4>
                <p><strong>Q-Learning:</strong> {ql_test['success_rate']:.1f}% success rate in {ql_time:.1f}s</p>
                <p><strong>DQN:</strong> {dqn_test['success_rate']:.1f}% success rate in {dqn_time:.1f}s</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.expander("📖 About the Algorithms"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Q-Learning")
            st.markdown("""
            **Type:** Tabular, Model-Free
            
            **How it works:**
            - Discretizes continuous state space
            - Stores Q-values in a lookup table
            - Updates using Bellman equation
            
            **Pros:**
            - Simple and interpretable
            - Guaranteed convergence
            - No neural network required
            
            **Cons:**
            - Requires state discretization
            - Doesn't scale to high dimensions
            - Slower convergence (7000 episodes)
            """)
        
        with col2:
            st.subheader("Deep Q-Network (DQN)")
            st.markdown("""
            **Type:** Deep Learning, Model-Free
            
            **How it works:**
            - Uses neural network to approximate Q-function
            - Experience replay for sample efficiency
            - Target network for stability
            
            **Pros:**
            - No discretization needed
            - Faster convergence (2000 episodes)
            - Scales to complex problems
            
            **Cons:**
            - More complex implementation
            - Requires hyperparameter tuning
            - Longer per-episode time
            """)
    
    with st.expander("🎯 Understanding the Results"):
        st.markdown("""
        ### Success Rate
        - **>90%**: Excellent - agent consistently solves the task
        - **70-90%**: Good - agent learned but could be improved
        - **<70%**: Needs tuning - adjust hyperparameters or increase training
        
        ### Average Reward
        - **>-110**: Very good performance
        - **-110 to -150**: Decent performance
        - **<-150**: Room for improvement
        
        ### Training Progress
        - **Rewards plot**: Should trend upward (less negative)
        - **Position plot**: Should reach 0.5 (the goal)
        - **Loss plot (DQN)**: Should decrease and stabilize
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>Built with Streamlit | <a href='https://github.com/yourusername/mountaincar-rl'>GitHub Repository</a></p>
        <p>MountainCar-v0 Environment from Gymnasium</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
