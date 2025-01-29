import pybullet as p
import pybullet_data
import time
import random
import matplotlib
import matplotlib.pyplot as plt
# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import namedtuple, deque
from itertools import count
import math

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

class RobotLocomotionEnv(gym.Env):
    def __init__(self):
        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)

        # Observation space: 8D vector
        # robot pos, adversary pos, robot vel, adversary vel
        low = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        self.robot_observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.defender_observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Environment constants
        self.max_steps = 400
        self.target = np.array([0.0, 0.0])
        self.success_reward = 100
        self.action_cost = -0.1

        # Simulation parameters
        self.force_magnitude = 500  # Magnitude of the applied force
        self.reset()

    def reset(self):
        self.steps = 0
        # Set up the simulation environment
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        # Create a simple box (brick)
        start_pos = [0, 0, 0.5]
        start_orientation = [0, 0, 0, 1]  # No rotation (quaternion)
        self.adv_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
        self.adv_body = p.createMultiBody(baseMass=1,
                                    baseCollisionShapeIndex=self.adv_id,
                                    basePosition=start_pos,
                                    baseOrientation=start_orientation)

        # Set friction for the box
        p.changeDynamics(self.adv_body, -1, lateralFriction=0.5, spinningFriction=0.5, rollingFriction=0.5)

        # Load a plane
        plane_id = p.loadURDF("plane.urdf")

        # Physics engine parameters
        p.setPhysicsEngineParameter(enableConeFriction=1)

        # Create a simple box (brick)
        agent_x = random.uniform(0.5, 10) * np.random.choice([-1, 1])
        agent_y = random.uniform(0.5, 10) * np.random.choice([-1, 1])
        # Create a simple box (brick)
        start_pos = [agent_x, agent_y, 0.5]
        start_orientation = [0, 0, 0, 1]  # No rotation (quaternion)
        self.ag_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
        self.ag_body = p.createMultiBody(baseMass=1,
                                    baseCollisionShapeIndex=self.ag_id,
                                    basePosition=start_pos,
                                    baseOrientation=start_orientation)

        # Set friction for the box
        p.changeDynamics(self.ag_body, -1, lateralFriction=0.5, spinningFriction=0.5, rollingFriction=0.5)

    def step(self, ag_action, adv_action):
        self.steps += 1
        if self.steps > self.max_steps or self.reached_goal():
            raise ValueError("Max steps reached")
        # Apply action to update accelerations
        def get_action(action):
            linear_acc = np.zeros(2)
            if action == 0:
                # Acc forward move
                linear_acc = np.array([0, self.force_magnitude])
            elif action == 1:
                # Acc backward move
                linear_acc = np.array([0, -self.force_magnitude])
            elif action == 2:
                # Acc left move
                linear_acc = np.array([-self.force_magnitude, 0])
            elif action == 3:
                # Acc right move
                linear_acc = np.array([self.force_magnitude, 0])
            elif action == 4:
                # Do nothing: no acceleration
                pass
            else:
                raise ValueError("Invalid action")
            return [linear_acc[0], linear_acc[1], 0]

        p.applyExternalForce(
            objectUniqueId=self.ag_body,
            linkIndex=-1,
            forceObj=get_action(ag_action),
            posObj=p.getBasePositionAndOrientation(self.ag_body)[0],
            flags=p.WORLD_FRAME,
        )
        p.applyExternalForce(
            objectUniqueId=self.adv_body,
            linkIndex=-1,
            forceObj=get_action(adv_action),
            posObj=p.getBasePositionAndOrientation(self.adv_body)[0],
            flags=p.WORLD_FRAME,
        )
        p.stepSimulation()
        state = self.get_state()
        return state, self.get_agent_reward(), self.get_adversary_reward(), self.reached_goal(), self.steps >= self.max_steps

    def get_state(self):
        ag_pos, ag_orientation = p.getBasePositionAndOrientation(self.ag_body)
        adv_pos, adv_orientation = p.getBasePositionAndOrientation(self.adv_body)
        ag_vel, ag_angular_vel = p.getBaseVelocity(self.ag_body)
        adv_vel, adv_angular_vel = p.getBaseVelocity(self.adv_body)
        return [ag_pos[0], ag_pos[1], ag_vel[0], ag_vel[1], adv_pos[0], adv_pos[1], adv_vel[0], adv_vel[1]]

    def dist_between_entities(self):
        position, orientation = p.getBasePositionAndOrientation(self.ag_body)
        adv_position, adv_orientation = p.getBasePositionAndOrientation(self.adv_body)
        return np.linalg.norm(np.array([position[0], position[1]]) - np.array([adv_position[0], adv_position[1]]))


    def ag_dist_to_target(self):
        position, orientation = p.getBasePositionAndOrientation(self.ag_body)
        return np.linalg.norm(np.array([position[0], position[1]]) - self.target)

    def get_adversary_reward(self):
        if self.reached_goal():
            return -self.success_reward
        return (self.ag_dist_to_target() - (self.dist_between_entities()/2))/16

    def get_agent_reward(self):
        if self.reached_goal():
            return self.success_reward
        else:
            return -self.ag_dist_to_target() / 16

    def reached_goal(self):
        return self.ag_dist_to_target() < 0.5

env = RobotLocomotionEnv()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'ag_action', 'adv_action', 'next_state', 'ag_reward', 'adv_reward'))


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


# Get number of actions from gym action space
n_actions = 5
# Get the number of state observations
env.reset()
state = env.get_state()
n_observations = len(state)

policy_net_ag = DQN(n_observations, n_actions).to(device)

policy_net_adv = DQN(n_observations, n_actions).to(device)

def select_action(state, agent):
    with torch.no_grad():
        # t.max(1) will return the largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        if agent == "ag":
            return policy_net_ag(state).max(1).indices.view(1, 1)
        else:
            return policy_net_adv(state).max(1).indices.view(1, 1)


policy_net_ag.load_state_dict(torch.load("policy_net_agent.pth", map_location=device))
policy_net_adv.load_state_dict(torch.load("policy_net_adversary.pth", map_location=device))

import time
import torch

# Ensure you have already defined/pasted:
# - env = RobotLocomotionEnv()
# - policy_net_ag (the learned agent network)
# - policy_net_adv (optional, if you also needed an adversarial policy network)
# - device
# - select_action function if desired, or inline the policy call as shown below.

# If your environment step is fast, you can slow it down:
time_step = 1./240.  # Adjust for comfortable visualization

# Before the loop, we reset once
env.reset()
state = env.get_state()

while True:
    # Convert the current state to a Torch tensor
    state_t = torch.tensor([state], device=device, dtype=torch.float32)

    # 1) Policy (agent) picks an action
    with torch.no_grad():
        # Take the index of the max Q-value as the chosen action
        ag_action = policy_net_ag(state_t).max(1).indices.item()

    # 2) Keyboard (adversary) picks an action
    # Default to action = 4 ("do nothing")
    adv_action = 4  
    keys = p.getKeyboardEvents()

    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
        adv_action = 0  # forward
    elif p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
        adv_action = 1  # backward
    elif p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
        adv_action = 2  # left
    elif p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
        adv_action = 3  # right

    # 3) Take a step in the environment
    try:
        next_state, ag_reward, adv_reward, reached_goal, done = env.step(ag_action, adv_action)
    except ValueError:
        # If the environment code raises ValueError when max steps is reached:
        print("Episode finished (max steps). Resetting environment.")
        state = env.reset()
        time.sleep(time_step)
        continue

    # 4) Check termination signals
    if reached_goal or done:
        if reached_goal:
            print("Goal reached")
        else:
            print("Episode finished (max steps). Resetting environment.")
        env.reset()
        state = env.get_state()
    else:
        # If not done, update state
        state = next_state

    # 5) Small sleep for real-time rendering
    time.sleep(time_step)