import math
import random
from collections import deque
from collections import namedtuple
from itertools import count

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cartpole_py import CartPole

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

env = CartPole(seed=42)

is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


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
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
# n_actions = env.action_space.n
n_actions = 2
# Get the number of state observations
state = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR, amsgrad=False)
memory = ReplayMemory(10000)

learn_step = 0
steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        rand_act = random.randint(0, 1)
        # return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        return torch.tensor([[rand_act]], device=device, dtype=torch.long)


episode_durations = []
losses = []


def plot_durations(show_result=False):
    plt.figure(1)
    plt.clf()  # Clear the current figure but keep subplots intact

    # Set a global title for the figure
    if show_result:
        plt.suptitle("Result")
    else:
        plt.suptitle("Training...")

    # Plot Episode Durations
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy(), label="Duration")
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label="100-episode avg")
    plt.legend()

    # Plot Losses
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot
    plt.xlabel("Optimize Step")
    plt.ylabel("Loss")
    plt.plot(losses, label="Loss", color="orange")
    plt.legend()

    # Adjust the layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    global learn_step
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    writer.add_histogram("values", state_action_values, learn_step)
    # print(
    #     f"values.min={state_action_values.min()} values.std={state_action_values.max()} values.mean={state_action_values.mean()}"
    # )

    writer.add_histogram("next_values", next_state_values[non_final_mask], learn_step)
    writer.add_histogram("exp_tgt_values", expected_state_action_values, learn_step)
    # print(
    #     f"exp_values.min={expected_state_action_values.min()} exp_values.std={expected_state_action_values.max()} exp_values.mean={expected_state_action_values.mean()}"
    # )
    # print(f"==== {loss=}")
    # print()
    losses.append(loss.clone().cpu().item())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for name, param in policy_net.named_parameters():
        # name = "_".join(param.names)
        writer.add_histogram(f"policy/{name}", param.data, learn_step)
        writer.add_histogram(f"policy_grad/{name}", param.grad, learn_step)
    writer.add_scalar("loss", loss, learn_step)
    # writer.add_scalar("loss_grad", loss.grad, learn_step)
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    learn_step += 1
    return loss.item()


num_episodes = 600
max_steps = 200


for i_episode in range(num_episodes):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    total_reward = 0
    action_sum = 0
    loss_sum = 0
    loss_count = 0

    for t in count():
        action = select_action(state)
        action_sum += action.item()
        observation, reward, done = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        total_reward += reward.item()

        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward)
        state = next_state

        loss = optimize_model()
        if loss is not None:
            loss_sum += loss
            loss_count += 1

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done or t >= max_steps:
            episode_durations.append(t + 1)

            avg_action = action_sum / (t + 1)
            avg_loss = loss_sum / loss_count if loss_count > 0 else 0

            writer.add_scalar("Total Reward", total_reward, i_episode)
            writer.add_scalar("Loss", avg_loss, i_episode)
            writer.add_scalar("Average Action", avg_action, i_episode)

            # print(
            #     f"Episode {i_episode + 1}: Total Reward = {total_reward:.2f}, Avg Loss = {avg_loss:.2f}, Avg Action = {avg_action:.2f}, Duration = {t + 1}"
            # )

            plot_durations()
            break
print("Complete")
plot_durations(show_result=True)
plt.ioff()
plt.show()

writer.close()
