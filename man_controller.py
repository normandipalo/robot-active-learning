import numpy as np

def reach_xy_contr(state, norm = False):
    cube_pos = state["achieved_goal"]
    robot_pos = state["observation"][:3]
    diff = cube_pos - robot_pos
    if not norm:
        action = diff*10
    else:
        action = diff/np.linalg.norm(diff)*5
    action[2] = 0.005
    action = np.concatenate((action, [0.]))
    return action, 1

def reach_x_open(state, norm = False):
    cube_pos = state["achieved_goal"]
    robot_pos = state["observation"][:3]
    diff = cube_pos - robot_pos
    if not norm:
        action = diff*5
    else:
        action = 5*diff/(np.linalg.norm(diff) + 0.1)
    action = np.concatenate((action, [1.]))
    return action, 1

def pick(state):
    return [0.,0.,0.,-1.], 10

def go_to_goal(state, norm = False):
    cube_pos = state["achieved_goal"]
    diff = state["desired_goal"] - cube_pos
    if not norm:
        action = diff * 5
    else:
        action = diff/np.linalg.norm(diff)
    action = np.concatenate((action, [-1.]))
    return action, 1

def controller(state, picked, norm = False):
    cube_pos = state["achieved_goal"]
    robot_pos = state["observation"][:3]
    diff = cube_pos - robot_pos
    if picked[0] == False:
        if np.linalg.norm(diff[:2]) > 0.01:
            action, steps = reach_xy_contr(state, norm)
        elif np.linalg.norm(diff[2])> 0.005:
            action, steps = reach_x_open(state, norm)
        else:
            action, steps = pick(state)
            picked[0] = True
    else:
        action, steps = go_to_goal(state, norm)
    return action, steps


def get_demo(env, state, norm = False, render = False):
    states, actions = [], []
    picked = [False]
    for i in range(200):
        action, steps = controller(state, picked, norm)
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
