from typing import OrderedDict
from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture
from franka_env.utils.rotations import euler_2_quat
import numpy as np
import requests
import copy
import gymnasium as gym
import time
import jax
from franka_env.envs.franka_env import FrankaEnv

class USBEnv(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def init_cameras(self, name_serial_dict=None):
        """Init both wrist cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            if cam_name == "side_classifier" :
                self.cap["side_classifier"] = self.cap["side_policy"]
            elif cam_name == "side_stage_classifier":
                self.cap["side_stage_classifier"] = self.cap["side_policy"]
            else:
                cap = VideoCapture(
                    RSCapture(name=cam_name, **kwargs)
                )
                self.cap[cam_name] = cap

    def reset(self, **kwargs):
        self._recover()
        self._update_currpos()
        self._send_movel_command(self.currpos)
        time.sleep(0.1)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        self._send_gripper_command(1.0)
        
        # Move above the target pose, that go down and grab the USB
        target = copy.deepcopy(self.currpos)
        target[2] = self.config.TARGET_POSE[2] + 0.05
        self.interpolate_move(target, timeout=0.5)
        target = copy.deepcopy(self.config.TARGET_POSE)
        target[2] = self.config.TARGET_POSE[2] + 0.05
        self.interpolate_move(target, timeout=0.5)
        self.interpolate_move(self.config.TARGET_POSE, timeout=0.5)
        self._send_gripper_command(-1.0)
        time.sleep(0.4)

        # grab out the USB
        self._update_currpos()
        reset_pose = copy.deepcopy(self.config.TARGET_POSE)
        reset_pose[0] += 0.02
        self.interpolate_move(reset_pose, timeout=0.5)

        # put the USB into a random position
        usb_reset_pose = self._RESET_POSE.copy()
        usb_reset_pose[2] = self.config.USB_RESET_HEIGHT
        usb_reset_pose[:2] += np.random.uniform(-self.random_xy_range, self.random_xy_range, (2,))
        # usb_reset_pose[3:] = euler_2_quat(self._RESET_POSE[3:].copy())
        self.interpolate_move(usb_reset_pose, timeout=0.25)
        self._send_gripper_command(1.0)
        time.sleep(0.2)
        usb_reset_pose[2] = self._RESET_POSE[2]
        self.interpolate_move(usb_reset_pose, timeout=0.5)

        obs, info = super().reset(**kwargs)
        time.sleep(0.3)
        self.success = False
        self._update_currpos()
        obs = self._get_obs()
        return obs, info
    
    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position with linear interpolation."""
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        self._send_movel_command(goal)
        while np.max(self.currpos - goal) > 0.005:
            time.sleep(0.01)
            self._update_currpos()
        time.sleep(timeout)
        self._update_currpos()
    
    def go_to_reset(self, joint_reset=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """

        # Perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)

        # Perform Carteasian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self.interpolate_move(reset_pose, timeout=1)
        else:
            reset_pose = self.resetpos.copy()
            self.interpolate_move(reset_pose, timeout=1)

        # Change to compliance mode
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)


class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.9
        ):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info


class StageFeatureWrapper(gym.Wrapper):
    """
    使用冻结的 stage classifier 提取特征向量，并添加到观测空间中。
    
    注意：此 Wrapper 应在 ChunkingWrapper 之前应用，
    此时图像形状为 (H, W, C)，没有时间维度。
    """
    def __init__(self, env: gym.Env, classifier, feature_dim=256):
        super().__init__(env)
        self.classifier = classifier
        self.feature_dim = feature_dim
        
        # 扩展观测空间，注册 'stage_features'
        self.observation_space = copy.deepcopy(env.observation_space)
        self.observation_space.spaces['stage_features'] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(feature_dim,), dtype=np.float32
        )
        
        @jax.jit
        def _extract_features(params, classifier_input):
            return classifier.apply_fn(
                {"params": params},
                classifier_input,
                train=False,
                return_encoded=True
            )
        
        self._extract_features = _extract_features
        self._params = classifier.params
        
        # 预热 JIT（首次调用会编译，避免运行时卡顿）
        try:
            dummy_img = env.observation_space.spaces["side_stage_classifier"].sample()
            dummy_input = {"side_stage_classifier": dummy_img[None, None, ...]}
            _ = self._extract_features(self._params, dummy_input)
            print("[+] StageFeatureWrapper: JIT warmup complete")
        except Exception as e:
            print(f"[!] StageFeatureWrapper: JIT warmup failed: {e}")
    
    def step(self, action):
        """执行一步环境交互，并添加 stage_features 到观测中。"""
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = self._add_stage_features(observation)
        return observation, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """重置环境，并添加 stage_features 到初始观测中。"""
        obs, info = self.env.reset(**kwargs)
        obs = self._add_stage_features(obs)
        return obs, info
    
    def _add_stage_features(self, obs):
        """从 side_stage_classifier 图像提取特征并添加到观测中。"""
        if "side_stage_classifier" in obs:
            img = obs["side_stage_classifier"]
            
            # 此 Wrapper 在 ChunkingWrapper 之前，图像形状为 (H, W, C)
            # 分类器期望 (B, T, H, W, C)，其中 B=1, T=1
            if len(img.shape) == 3:  # (H, W, C) - 正常情况
                classifier_input = {"side_stage_classifier": img[None, None, ...]}  # -> (1, 1, H, W, C)
            elif len(img.shape) == 4:  # (T, H, W, C) - 如果意外有时间维度
                classifier_input = {"side_stage_classifier": img[None, ...]}  # -> (1, T, H, W, C)
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}, expected (H, W, C) or (T, H, W, C)")
            
            # 使用预编译的函数提取特征
            features = self._extract_features(self._params, classifier_input)
            
            # 移除 batch 维度并转为 numpy: (1, feature_dim) -> (feature_dim,)
            features = np.asarray(features).squeeze(axis=0)
            obs['stage_features'] = features.astype(np.float32)
        else:
            # 容错处理：如果没有 side_stage_classifier 图像，给个零向量
            print("[!] Warning: side_stage_classifier not found in observation, using zero vector")
            obs['stage_features'] = np.zeros(self.feature_dim, dtype=np.float32)
        
        return obs
