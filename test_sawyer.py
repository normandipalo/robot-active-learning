import argparse
import numpy as np

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper

env = robosuite.make(
        "SawyerNutAssemblyRound",
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        gripper_visualization=True,
        reward_shaping=True,
        control_freq=20,
    )


env = IKWrapper(env)

while True:
        obs = env.reset()
        env.viewer.set_camera(camera_id=2)
        env.render()

        # rotate the gripper so we can see it easily
        env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
        dpos = np.array([0.0,0.0,0.])
        while True:

            nut_eul = T.mat2euler(T.quat2mat(obs["RoundNut0_quat"]))

            pos_handle = np.array([+0.065*np.cos(nut_eul[2]),0.065*np.sin(nut_eul[2]), 0.])
            print(pos_handle, "\n\n", obs["RoundNut0_pos"])
            r_z = T.rotation_matrix(nut_eul[2], [0.,0.,1.])
            rotation = np.eye(3)#np.array([0.,0.0,0.0,1.])
            r_y = T.rotation_matrix(np.pi, [0.,1.,0.])
            rotation = r_y[:3,:3] #rotation.dot(r_y[:3,:3])
            grasp = 1
            rotation = r_z[:3,:3].dot(r_y[:3,:3])

            error = -obs["eef_pos"] + obs["RoundNut0_pos"] + pos_handle
            error[2]*=0
            #error[2]*=-1
            dpos = np.clip(error, -0.001, 0.001)

            # convert into a suitable end effector action for the environment
            current = env._right_hand_orn
            # relative rotation of desired from current
            drotation = current.T.dot(rotation)
            dquat =  T.mat2quat(drotation)
            #dquat = obs["RoundNut0_to_eef_quat"]
            # map 0 to -1 (open) and 1 to 0 (closed halfway)
            grasp = grasp - 1.



            action = np.concatenate([dpos, dquat, [grasp]])
            obs, reward, done, info = env.step(action)
            env.render()
