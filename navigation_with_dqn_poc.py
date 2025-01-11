"""# Environment"""

import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces

class RobotLocomotionEnv(gym.Env):
    def __init__(self):
        # Action space: 8 discrete actions
        self.action_space = spaces.Discrete(8)

        # Observation space: 6D vector
        low = np.array([-1, -1, -np.inf, -np.inf, -np.pi, -np.inf])
        high = np.array([1, 1, np.inf, np.inf, np.pi, np.inf])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Environment constants
        self.dt = 0.1  # Time step
        self.max_steps = 2000
        self.target = np.array([0.0, 0.0])
        self.crash_reward = -100
        self.success_reward = 100
        self.action_cost = -0.1
        self.viewport = 1.0  # x and y between -1 and 1

        # Robot dynamics parameters
        self.max_linear_acceleration = 0.5
        self.max_angular_acceleration = np.pi / 4  # 45 degrees per second
        self.max_linear_speed = 2.0
        self.max_angular_speed = np.pi  # 180 degrees per second

        # Initialize state variables
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.angle = 0.0
        self.angular_velocity = 0.0
        self.step_count = 0

        # Obstacles (if any)
        self.obstacles = []  # List of obstacle positions and sizes

        # Rendering variables
        self.fig = None
        self.ax = None
        self.position_history = []

    def reset(self):
        # Random initial position within viewport
        self.position = np.random.uniform(-self.viewport, self.viewport, size=2)
        # Random initial angle
        self.angle = np.random.uniform(-np.pi, np.pi)
        # Random initial force applied to center of mass
        initial_force = np.random.uniform(-1, 1, size=2)
        # Update velocity based on initial force
        self.velocity += initial_force * self.dt
        # Reset other state variables
        self.velocity = np.clip(self.velocity, -self.max_linear_speed, self.max_linear_speed)
        self.angular_velocity = 0.0
        self.step_count = 0
        self.position_history = [self.position.copy()]

        # Return observation
        observation = np.array([
            self.position[0],
            self.position[1],
            self.velocity[0],
            self.velocity[1],
            self.angle,
            self.angular_velocity
        ])
        return observation, None

    def step(self, action):
        # Apply action to update accelerations
        linear_acc = np.zeros(2)
        angular_acc = 0.0
        if action == 0:
            # Stop: set velocities to zero
            self.velocity = np.zeros(2)
            self.angular_velocity = 0.0
        elif action == 1:
            # Acc left turn
            angular_acc = self.max_angular_acceleration
        elif action == 2:
            # Acc right turn
            angular_acc = -self.max_angular_acceleration
        elif action == 3:
            # Acc forward move
            linear_acc = np.array([self.max_linear_acceleration * np.cos(self.angle),
                                   self.max_linear_acceleration * np.sin(self.angle)])
        elif action == 4:
            # Acc backward move
            linear_acc = np.array([-self.max_linear_acceleration * np.cos(self.angle),
                                   -self.max_linear_acceleration * np.sin(self.angle)])
        elif action == 5:
            # Acc left move
            linear_acc = np.array([-self.max_linear_acceleration * np.sin(self.angle),
                                   self.max_linear_acceleration * np.cos(self.angle)])
        elif action == 6:
            # Acc right move
            linear_acc = np.array([self.max_linear_acceleration * np.sin(self.angle),
                                   -self.max_linear_acceleration * np.cos(self.angle)])
        elif action == 7:
            # Do nothing: no acceleration
            pass
        else:
            raise ValueError("Invalid action")

        # Update angular velocity
        self.angular_velocity += angular_acc * self.dt
        self.angular_velocity = np.clip(self.angular_velocity, -self.max_angular_speed, self.max_angular_speed)

        # Update linear velocity
        self.velocity += linear_acc * self.dt
        self.velocity = np.clip(self.velocity, -self.max_linear_speed, self.max_linear_speed)

        # Update position
        self.position += self.velocity * self.dt
        self.position = np.clip(self.position, -self.viewport, self.viewport)

        # Update angle
        self.angle += self.angular_velocity * self.dt
        self.angle = np.mod(self.angle, 2 * np.pi)
        if self.angle > np.pi:
            self.angle -= 2 * np.pi

        # Increment step count
        self.step_count += 1

        # Calculate reward
        distance = np.linalg.norm(self.position - self.target)
        reward = -distance  # Closer is better
        # Penalize for speed at target
        if distance < 0.1:
            reward -= 0.1 * np.linalg.norm(self.velocity)
        # Action cost if not "do nothing"
        if action != 7:
            reward += self.action_cost
        # Check for episode termination
        terminated = False
        truncated = False
        done = False
        if self._is_out_of_viewport():
            reward += self.crash_reward
            done = True
        elif self._is_at_target():
            reward += self.success_reward
            done = True
        elif self.step_count >= self.max_steps:
            done = True
        terminated = done

        # Return observation, reward, terminated, truncated, info
        observation = np.array([
            self.position[0],
            self.position[1],
            self.velocity[0],
            self.velocity[1],
            self.angle,
            self.angular_velocity
        ])
        return observation, reward, terminated, truncated, None

    def render(self):
      from IPython.display import clear_output, display
      clear_output(wait=True)

      # Create figure if it doesn't exist
      if self.fig is None or self.ax is None:
          self.fig, self.ax = plt.subplots(figsize=(4, 4))
          plt.xlim(-self.viewport - 1, self.viewport + 1)
          plt.ylim(-self.viewport - 1, self.viewport + 1)
          plt.gca().set_aspect('equal', adjustable='box')
          plt.title('Robot Locomotion Environment')

          self.ax.plot(self.target[0], self.target[1], 'go', label='Target')

          self.robot_plot, = self.ax.plot([], [], 'b-', linewidth=2)
          self.trail, = self.ax.plot([], [], 'b:', label='Trail')
          self.ax.legend()

      # Update robot arrow each time render is called
      x, y = self.position
      theta = self.angle
      dx = 0.1 * np.cos(theta)
      dy = 0.1 * np.sin(theta)
      self.robot_plot.set_data([x, x + dx], [y, y + dy])

      # Update the trail with the latest position
      self.position_history.append(self.position.copy())
      self.trail.set_data(
          [p[0] for p in self.position_history],
          [p[1] for p in self.position_history]
      )

      display(self.fig)


    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    def _is_at_target(self):
        distance = np.linalg.norm(self.position - self.target)
        if distance < 0.1 and np.linalg.norm(self.velocity) < 0.1 and abs(self.angular_velocity) < 0.1:
            return True
        else:
            return False

    def _is_out_of_viewport(self):
        if np.any(self.position < -self.viewport) or np.any(self.position > self.viewport):
            return True
        else:
            return False

# Example usage
# if __name__ == "__main__":
#     plt.ion()
#     env = RobotLocomotionEnv()
#     observation, _ = env.reset()
#     env.render()
#     for _ in range(10000):
#         action = env.action_space.sample()
#         observation, reward, terminated, truncated, _ = env.step(action)
#         env.render()
#         if terminated or truncated:
#             break;
#             # print("Episode terminated or truncated.")
#             # observation, _ = env.reset()
#             # env.render()
#     env.close()
#     plt.ioff()
#     plt.show()

"""# Agent"""

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = RobotLocomotionEnv()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_rewards = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    total_reward = 0

    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated
        total_reward += reward

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_rewards.append(total_reward)
            if (i_episode % 10) == 0:
              plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()

"""# Test (RoboLoco Env)"""

def run_and_collect_frames(env, policy_net, device):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    done = False
    while not done:
        with torch.no_grad():
            action_values = policy_net(state)
            action = action_values.max(1).indices.item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()
        if not done:
            state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

env.reset()
run_and_collect_frames(env, policy_net, device)

"""# Test (GYM)"""

!!pip install moviepy

# Import necessary libraries
import gymnasium as gym
import torch
from moviepy.editor import ImageSequenceClip
from IPython.display import Video

# Define a function to run the environment and collect frames
def run_and_collect_frames(env, policy_net, device):
    frames = []
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    done = False
    while not done:
        with torch.no_grad():
            action_values = policy_net(state)
            action = action_values.max(1).indices.item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frame = env.render()
        print(frame)
        frames.append(frame)
        if not done:
            state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
    return frames

# Create the environment with render_mode set to "rgb_array"
env.reset()

# Collect frames during the episode
frames = run_and_collect_frames(env, policy_net, device)
env.close()

# Create the video from the collected frames
clip = ImageSequenceClip(frames, fps=30)
clip.write_videofile("cartpole.mp4", codec='libx264')

# Display the video in Colab
Video("cartpole.mp4")