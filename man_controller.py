import numpy as np
import copy

def reach_xy_contr(state,cube = 0, norm = False):
    cube_pos0 = state["achieved_goal"][:3]
    cube_pos1 = state["achieved_goal"][3:6]
    robot_pos = state["observation"][:3]
    diff = cube_pos0 - robot_pos if not cube else cube_pos1 - robot_pos
    if not norm:
        action = diff*10
    else:
        action = diff/np.linalg.norm(diff)*5
    action[2] = 0.005
    action = np.concatenate((action, [0.]))
    return action, 1

def reach_x_open(state, cube = 0, norm = False):
    cube_pos0 = state["achieved_goal"][:3]
    cube_pos1 = state["achieved_goal"][3:6]
    robot_pos = state["observation"][:3]
    diff = cube_pos0 - robot_pos if not cube else cube_pos1 - robot_pos
    if not norm:
        action = diff*5
    else:
        action = 5*diff/(np.linalg.norm(diff) + 0.1)
    action = np.concatenate((action, [1.]))
    return action, 1

def pick(state):
    return [0.,0.,0.,-1.], 10

def go_to_goal(state, cube = 0, norm = False, go_up = 0, curr_goal = None):
    cube_pos0 = state["achieved_goal"][:3]
    cube_pos1 = state["achieved_goal"][3:6]
    if curr_goal is None:
        curr_goal = state["desired_goal"]
    diff = curr_goal - cube_pos0 if not cube else (curr_goal - cube_pos1)
    if not norm:
        action = diff * 5
    else:
        action = diff/np.linalg.norm(diff)
    action = np.concatenate((action, [-1.]))
    if go_up:
        action[2] += 1
    return action, 1

def controller(state, picked, in_position, norm = True):
    action, steps = np.array([0.,0.,0.,0.]), 1 #placeholder
    cube_pos0 = state["achieved_goal"][:3]
    cube_pos1 = state["achieved_goal"][3:6]
    robot_pos = state["observation"][:3]
    if cube_pos0[2] > 0.43: picked[0] = True
    if cube_pos1[2] > 0.43: picked[1] = True
    diff0 = cube_pos0 - robot_pos
    diff1 = cube_pos1 - robot_pos
    if not in_position[0]:
        cube = 0
        alligned_z = 0
        if picked[0] == False:
            if np.linalg.norm(diff0[:2]) > 0.01:
                action, steps = reach_xy_contr(state, cube, norm)
                #If the robot is at the cube level go up or it will push it away.
                # Go particularly up to avoid touching the other cube.
                if not np.linalg.norm(diff0[2]) > 0.5:
                    action[2] += 0.2
            elif np.linalg.norm(diff0[2])> 0.005:
                action, steps = reach_x_open(state, cube, norm)
            else:
                action, steps = pick(state)
                picked[0] = True

        else:
            if np.linalg.norm(cube_pos0[:2] - state["desired_goal"][:2]).__lt__(0.03): alligned_z = True
            if not alligned_z:
                #use this to reach the goal from above and avoid touching
                # the other cube, then go down to release the cube in place.
                action, steps = go_to_goal(state, cube, norm, go_up = False, curr_goal = state["desired_goal"] + np.array([0.,0,0.1]))
            #    if np.linalg.norm(diff0[:2]).__lt__(0.01): alligned_z = True
            else:
                action, steps = go_to_goal(state, cube, norm)
                if np.linalg.norm(cube_pos0[:] - state["desired_goal"][:]).__lt__(0.01):
                    in_position[0] = True
                    action, steps = np.array([0.,0.,0.5,1.]), 5
    else:
        cube = 1
        if not in_position[1]:

            alligned_z = 0
            if picked[1] == False:
                if np.linalg.norm(diff1[:2]) > 0.01:
                    action, steps = reach_xy_contr(state, cube, norm)
                    #If the robot is at the cube level go up or it will push it away.
                    # Go particularly up to avoid touching the other cube.
                    if not np.linalg.norm(diff1[2]) > 0.5:
                        action[2] += 0.2
                elif np.linalg.norm(diff1[2])> 0.005:
                    action, steps = reach_x_open(state, cube, norm)
                else:
                    action, steps = pick(state)
                    picked[1] = True

            else:
                if np.linalg.norm(cube_pos1[:2] - state["desired_goal"][:2]).__lt__(0.03): alligned_z = True
                if not alligned_z:
                    #use this to reach the goal from above and avoid touching
                    # the other cube, then go down to release the cube in place.
                    action, steps = go_to_goal(state, cube, norm, go_up = False, curr_goal = state["desired_goal"] + np.array([0.,0.,0.2]))
                #    if np.linalg.norm(diff0[:2]).__lt__(0.01): alligned_z = True
                else:
                    action, steps = go_to_goal(state, cube, norm, curr_goal = state["desired_goal"])
                    if np.linalg.norm(cube_pos1[:] - state["desired_goal"][:]).__lt__(0.1):
                        in_position[1] = True
                        action, steps = np.array([0.,0.,0.5,1.]), 5
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
    picked = [False, False]
    in_position = [False, False]
    for i in range(200):
        action, steps = controller(state, picked, in_position, norm)
        for s in range(steps):
            states.append(np.concatenate((state["observation"],
                                        state["achieved_goal"],
                                        state["desired_goal"])))
            new_state, *_ = env.step(action)
            if render: env.render()
            actions.append(action)
            state = new_state
        if not np.linalg.norm((state["achieved_goal"][:3]- state["desired_goal"])) > 0.03 and not np.linalg.norm((state["achieved_goal"][3:6]- state["desired_goal"])) > 0.07:
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
        action, steps = controller(state, picked, norm)
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
