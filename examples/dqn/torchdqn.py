import math
import random
from collections import deque
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cartpole_py import CartPole

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="/tmp/pytorch_logs")
env = CartPole(seed=42)

device = "cpu"

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
MAX_EPISODES = 10_000
MAX_STEPS = 200

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))


class ReplayMemory:
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

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def select_action(policy_net, state, steps_done):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randint(0, 1)]], device=device, dtype=torch.long)


def optimize_model(policy_net, target_net, optimizer, memory, steps_done):
    if len(memory) < BATCH_SIZE:
        return None

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Create tensor batches
    state_batch = torch.cat([s.unsqueeze(0) for s in batch.state])
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat([torch.tensor([r], device=device) for r in batch.reward])

    # Handle next states, accounting for terminal states
    non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s.unsqueeze(0) for s in batch.next_state if s is not None])

    # Compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    # Compute expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute loss and optimize
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Log training metrics
    writer.add_histogram("training/q_values", state_action_values, steps_done)
    writer.add_histogram("training/target_values", expected_state_action_values, steps_done)
    writer.add_scalar("training/loss", loss, steps_done)

    optimizer.zero_grad()
    loss.backward()

    # Clip gradients
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss.item()


def main():
    n_actions = 2
    state = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10_000)

    steps_done = 0
    total_rewards = []

    for episode in range(MAX_EPISODES):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)

        total_reward = 0
        action_sum = 0
        loss_sum = 0
        loss_count = 0

        for t in count():
            action = select_action(policy_net, state.unsqueeze(0), steps_done)
            steps_done += 1
            action_sum += action.item()

            observation, reward, done = env.step(action.item())
            total_reward += reward

            # Store transition
            next_state = None if done else torch.tensor(observation, dtype=torch.float32, device=device)
            memory.push(state, action, next_state, reward, done)
            state = next_state

            # Optimize model
            if steps_done > BATCH_SIZE:
                loss = optimize_model(policy_net, target_net, optimizer, memory, steps_done)
                if loss is not None:
                    loss_sum += loss
                    loss_count += 1

                # Soft update target network
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (
                        1 - TAU
                    )
                target_net.load_state_dict(target_net_state_dict)

            if done or t >= MAX_STEPS:
                total_rewards.append(total_reward)
                avg_action = action_sum / (t + 1)
                avg_loss = loss_sum / loss_count if loss_count > 0 else 0

                # Log episode metrics
                writer.add_scalar("episode/reward", total_reward, episode)
                writer.add_scalar("episode/avg_loss", avg_loss, episode)
                writer.add_scalar("episode/avg_action", avg_action, episode)

                # Calculate and log running average
                if episode >= 100:
                    running_avg = sum(total_rewards[-100:]) / 100
                    writer.add_scalar("episode/running_avg", running_avg, episode)

                    if running_avg >= 195:
                        print(f"Solved in {episode} episodes!")
                        return

                break

    writer.close()


if __name__ == "__main__":
    main()
