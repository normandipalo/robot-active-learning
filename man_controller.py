import numpy as np
import copy

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
    if cube_pos[2] > 0.43: picked[0] = True
    print("cube pos", cube_pos)
    diff = cube_pos - robot_pos
    if picked[0] == False:
        if np.linalg.norm(diff[:2]) > 0.01:
            action, steps = reach_xy_contr(state, norm)
            if not np.linalg.norm(diff[2]) > 0.1: #If the robot is at the cube level
                action[2] += 1                      # go up or it will push it away
        elif np.linalg.norm(diff[2])> 0.005:
            action, steps = reach_x_open(state, norm)
        else:
            action, steps = pick(state)
            picked[0] = True

    else:
        action, steps = go_to_goal(state, norm)
    return action, steps

def get_demo_cam(env, c_state, norm = False, render = False, depth = False):
    state = copy.copy(c_state)
    states, actions = [], []
    picked = [False]
    _ = env.render(mode = "rgb_array")
    for i in range(200):
        action, steps = controller(state, picked, norm)
        for s in range(steps):

            new_state, *_ = env.step(action)
            if render: env.render("rgb_array")
            #Have to call two times to make it work.
            s = env.env.viewer.sim.render(50, 50, camera_name = 'external_camera_0')[::-1,:,:].astype(np.float64)/255.
            a, d = env.env.sim.render(50, 50, depth = True, camera_name = 'external_camera_0')
            s = env.env.viewer.sim.render(50, 50, camera_name = 'external_camera_0')[::-1,:,:].astype(np.float64)/255.
            a, d = env.env.sim.render(50, 50, depth = True, camera_name = 'external_camera_0')
            d = d[::-1,:].astype(np.float64)
            if not depth:
                states.append(s)
            else:
                states.append(np.concatenate((s,d[:,:,None]), -1))
            actions.append(action)
            state = new_state
           # print(state["achieved_goal"], state["desired_goal"])
           # print(action)
        if not np.linalg.norm((state["achieved_goal"]- state["desired_goal"])) > 0.05:
            break
    return states, actions

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
        if not np.linalg.norm((state["achieved_goal"]- state["desired_goal"])) > 0.05:
            break
    return states, actions


def get_demo_cam2(env, c_state, norm = False, render = False):
    state = copy.copy(c_state)
    states, actions = [], []
    picked = [False]
    for i in range(100):
        action, steps = controller(state[0], picked, norm)
        for s in range(steps):
            #Concat im rgb and depth.
            states.append(np.concatenate((state[1], state[2][:,:,None]), -1))
            new_state, _, _, _ = env.step(action)
            if render: env.render()
            actions.append(action)
            state = new_state
        if not np.linalg.norm((state[0]["achieved_goal"]- state[0]["desired_goal"])) > 0.05:
            break
    return states, actions

def get_demo_cam_full(env, c_state, norm = False, render = False):
    state = copy.copy(c_state)
    states, actions, rob_states = [], [], []
    picked = [False]
    for i in range(100):
        action, steps = controller(state[0], picked, norm)
        for s in range(steps):
            #Concat im rgb and depth.
            states.append(np.concatenate((state[1], state[2][:,:,None]), -1))
            rob_states.append(state[0]["observation"][:3])
            new_state, _, _, _ = env.step(action)
            if render: env.render()
            actions.append(action)
            state = new_state
        if not np.linalg.norm((state[0]["achieved_goal"]- state[0]["desired_goal"])) > 0.05:
            break
    return states, rob_states, actions

def get_demo_cam_random_pick(env, c_state, norm = False, render = False):
    #Used to pick the cube and go to a random position,
    # to then obtain a demo from that state.
    state = copy.copy(c_state)
    picked = [False]
    for i in range(100):
        if picked[0] == True: break
        action, steps = controller(state[0], picked, norm)
        for s in range(steps):
            #Concat im rgb and depth.
            new_state, _, _, _ = env.step(action)
            if render: env.render()
            state = new_state

    random_dir = np.random.randn(4)*0.4
    random_dir[3] = -1

    for s in range(20):
        new_state, _, _, _ = env.step(random_dir)
        if render: env.render()
        state = new_state

    return state
