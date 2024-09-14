import ctypes
from ctypes import c_double
from ctypes import c_uint32
from ctypes import c_uint64
from ctypes import POINTER
import pathlib

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

libpath = pathlib.Path("./libcartpole_wrapper.so")
if libpath.exists():
    lib = ctypes.CDLL(str(libpath))
elif (libpath := libpath.with_suffix(".dylib")).exists():
    lib = ctypes.CDLL(str(libpath))
else:
    raise FileNotFoundError()

lib.cartpole_init.argtypes = [c_uint64]
lib.cartpole_init.restype = ctypes.c_void_p

lib.cartpole_reset.argtypes = [ctypes.c_void_p]
lib.cartpole_reset.restype = POINTER(c_double * 4)

lib.cartpole_step.argtypes = [
    ctypes.c_void_p,
    c_uint32,
    POINTER(c_double * 4),
    POINTER(c_double),
    POINTER(ctypes.c_ubyte),
]
lib.cartpole_step.restype = None

lib.cartpole_delete.argtypes = [ctypes.c_void_p]
lib.cartpole_delete.restype = None


class CartPole:
    def __init__(self, seed=42):
        self.obj = lib.cartpole_init(seed)
        if not self.obj:
            raise MemoryError("Failed to create CartPole object")

    def reset(self):
        result = lib.cartpole_reset(self.obj)
        return list(result.contents)

    def step(self, action):
        state = (c_double * 4)()
        reward = c_double()
        done = ctypes.c_ubyte()
        lib.cartpole_step(self.obj, action, state, ctypes.byref(reward), ctypes.byref(done))
        return list(state), reward.value, bool(done.value)

    def __del__(self):
        if hasattr(self, "obj"):
            lib.cartpole_delete(self.obj)


def test():
    env = CartPole(seed=42)
    _ = env.reset()
    for _ in range(1000):
        action = 0
        _, reward, done = env.step(action)
        print(reward)
        if done:
            break


def animate(frame):
    global env, state
    if frame == 0:
        state = env.reset()

    # theta = state[2]
    action = 0
    state, _, done = env.step(action)

    cart_position, pole_angle = state[0], state[2]

    ax.clear()
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1, 1.5)

    cart = plt.Rectangle((cart_position - 0.5, -0.1), 1, 0.2, fill=True)
    ax.add_patch(cart)

    pole_length = 1
    pole_end_x = cart_position + pole_length * np.sin(pole_angle)
    pole_end_y = pole_length * np.cos(pole_angle)
    ax.plot([cart_position, pole_end_x], [0, pole_end_y], "r-")

    if done:
        plt.close()


if __name__ == "__main__":
    test()
    env = CartPole(seed=42)
    fig, ax = plt.subplots()
    ani = animation.FuncAnimation(fig, animate, frames=200, interval=50, repeat=False)
    plt.show()
