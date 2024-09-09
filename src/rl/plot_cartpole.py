import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from cartpole_py import CartPole


def pd_controller(state):
    theta = state[2]  # pole angle
    theta_dot = state[3]  # pole angular velocity
    Kp = 1.0
    Kd = 1.0
    control = Kp * theta + Kd * theta_dot
    return 1 if control >= 0 else 0


env = CartPole(seed=42)
state = env.reset()

history = {
    "cart_positions": [],
    "pole_angles": [],
    "pole_velocities": [],
    "actions": [],
}


def animate(frame):
    global env, state, history

    if frame == 0:
        state = env.reset()
        history = {key: [] for key in history}

    action = pd_controller(state)
    state, reward, done = env.step(action)

    history["cart_positions"].append(state[0])
    history["pole_angles"].append(state[2])
    history["pole_velocities"].append(state[3])
    history["actions"].append(action)

    cart_position, pole_angle = state[0], state[2]

    ax1.clear()
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-1, 1.5)
    ax1.set_aspect("equal", adjustable="box")

    cart_width, cart_height = 0.5, 0.25
    cart = plt.Rectangle(
        (cart_position - cart_width / 2, -cart_height / 2),
        cart_width,
        cart_height,
        fill=True,
    )
    ax1.add_patch(cart)

    pole_length = 1
    pole_end_x = cart_position + pole_length * np.sin(pole_angle)
    pole_end_y = pole_length * np.cos(pole_angle)
    ax1.plot([cart_position, pole_end_x], [0, pole_end_y], "r-", linewidth=3)

    ax1.plot([-2.5, 2.5], [0, 0], "k-", linewidth=2)

    msg = (
        f"Step: {frame}, Action: {action} Reward {reward}\n"
        f"Cart Pos: {cart_position}, Pole Angle: {pole_angle:.4f}rad\n"
        f"Pole Velocity: {state[3]:.4f}rad/s"
    )
    print(str(frame).center(20, "-"))
    print(msg)
    ax1.set_title(msg)

    if frame > 0:
        ax2.clear()
        ax2.plot(history["cart_positions"], label="Cart Position")
        ax2.plot(history["pole_angles"], label="Pole Angle")
        ax2.plot(history["pole_velocities"], label="Pole Velocity")
        ax2.plot(history["actions"], label="Action", drawstyle="steps-post")
        ax2.legend(loc="upper left")
        ax2.set_xlabel("Step")
        ax2.set_title("State History")
    time.sleep(0.2)

    if done:
        input()
        plt.close()


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
plt.tight_layout(pad=3.0)

ani = animation.FuncAnimation(fig, animate, frames=500, interval=100, repeat=False)
plt.show(block=True)

print("Final state:")
print(f"Cart Position: {state[0]}")
print(f"Cart Velocity: {state[1]}")
print(f"Pole Angle: {state[2]} radians")
print(f"Pole Angular Velocity: {state[3]} radians/s")

input("Press Enter to close...")
