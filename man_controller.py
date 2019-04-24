import numpy as np

def reach_xy_contr(state):
    cube_pos = state["achieved_goal"]
    robot_pos = state["observation"][:3]
    diff = cube_pos - robot_pos
    action = diff*10
    action[2] = 0.005
    action = np.concatenate((action, [0.]))
    return action, 1

def reach_x_open(state):
    cube_pos = state["achieved_goal"]
    robot_pos = state["observation"][:3]
    diff = cube_pos - robot_pos
    action = diff*5
    action = np.concatenate((action, [1.]))
    return action, 1

def pick(state):
    return [0.,0.,0.,-1.], 10

def go_to_goal(state):
    cube_pos = state["achieved_goal"]
    diff = state["desired_goal"] - cube_pos
    action = diff * 5
    action = np.concatenate((action, [-1.]))
    return action, 1

def controller(state, picked):
    cube_pos = state["achieved_goal"]
    robot_pos = state["observation"][:3]
    diff = cube_pos - robot_pos
    if picked[0] == False:
        if np.linalg.norm(diff[:2]) > 0.01:
            action, steps = reach_xy_contr(state)
        elif np.linalg.norm(diff[2])> 0.005:
            action, steps = reach_x_open(state)
        else:
            action, steps = pick(state)
            picked[0] = True
    else:
        action, steps = go_to_goal(state)
    return action, steps


def get_demo(env, state, render = False):
    states, actions = [], []
    picked = [False]
    for i in range(200):
        action, steps = controller(state, picked)
        for s in range(steps):
            states.append(np.concatenate((state["observation"],
                                        state["achieved_goal"],
                                        state["desired_goal"])))
            new_state, *_ = env.step(action)
            if render: env.render()
            actions.append(action)
            state = new_state
        if not np.linalg.norm((state["achieved_goal"]- state["desired_goal"])) > 0.01:
            break
    return states, actions