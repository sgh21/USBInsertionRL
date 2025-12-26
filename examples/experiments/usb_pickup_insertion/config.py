import os
import cv2
import jax
import numpy as np
import jax.numpy as jnp

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

# !:新加入的创建stage_classifier并加载权重
from serl_launcher.networks.reward_classifier import create_classifier
from flax.training import checkpoints

from experiments.config import DefaultTrainingConfig
from experiments.usb_pickup_insertion.wrapper import USBEnv, GripperPenaltyWrapper


class EnvConfig(DefaultEnvConfig):
    SERVER_URL: str = "http://localhost:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "230422272349",
            "dim": (640, 480),
            "exposure": 20000,
        },
        "wrist_2": {
            "serial_number": "419122270589",
            "dim": (640, 480),
            "exposure": 20000,
        },
        "side_policy": {
            "serial_number": "130322274099",
            "dim": (1280, 720),
            "exposure": 40000,
        },
        "side_classifier": {
            "serial_number": "130322274099",
            "dim": (1280, 720),
            "exposure": 40000,
        },
        "side_stage_classifier": {
            "serial_number": "130322274099",
            "dim": (1280, 720), 
            "exposure": 40000,
            "im_shape": (224, 224, 3),
        },
    }
    IMAGE_CROP = {"wrist_1":        lambda img: cv2.resize(img[0:480, :], None, fx=128 / 480, fy=128 / 480),
                  "wrist_2":        lambda img: cv2.resize(img[0:480, :], None, fx=128 / 480, fy=128 / 480),
                  "side_policy":    lambda img: cv2.resize(img[224:(224 + 256), 365:(365 + 256)], None, fx=0.5, fy=0.5),
                  "side_classifier":lambda img: img[224:(224+128), 365:(365+128)],
                  "side_stage_classifier": lambda img: cv2.resize(img[:540,:960], (224, 224)),  # 保持原始分辨率
                 }
    TARGET_POSE =   np.array([-0.060, -0.660, 0.008, np.pi, 0.0, 0.5 * np.pi])
    RESET_POSE =    np.array([-0.030, -0.660, 0.030, np.pi, 0.0, 0.5 * np.pi])
    USB_RESET_HEIGHT = 0.010
    ACTION_SCALE = np.array([0.05, 0.1, 1])
    RANDOM_RESET = True
    DISPLAY_IMAGE = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.060, 0.040, 0.050, 0.1, 0.1, 0.5])
    ABS_POSE_LIMIT_LOW  = TARGET_POSE - np.array([0.030, 0.040, 0.030, 0.1, 0.1, 0.5])
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.006,
        "translational_clip_y": 0.0059,
        "translational_clip_z": 0.0035,
        "translational_clip_neg_x": 0.005,
        "translational_clip_neg_y": 0.005,
        "translational_clip_neg_z": 0.0035,
        "rotational_clip_x": 0.02,
        "rotational_clip_y": 0.02,
        "rotational_clip_z": 0.015,
        "rotational_clip_neg_x": 0.02,
        "rotational_clip_neg_y": 0.02,
        "rotational_clip_neg_z": 0.015,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.03,
        "rotational_clip_y": 0.03,
        "rotational_clip_z": 0.03,
        "rotational_clip_neg_x": 0.03,
        "rotational_clip_neg_y": 0.03,
        "rotational_clip_neg_z": 0.03,
        "rotational_Ki": 0.0,
    }
    MAX_EPISODE_LENGTH = 120


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["side_policy", "wrist_1", "wrist_2"]
    classifier_keys = ["side_classifier"]
    # [新增] 定义阶段分类器使用的图像键
    stage_classifier_keys = ["side_stage_classifier"]
    stage_num_classes = 5          # 阶段数量
    stage_feature_dim = 256        # 特征维度
    stage_encoder_type = "resnet18"

    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    checkpoint_period = 200
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False, stage_classifier=False):
        env = USBEnv(
            fake_env=fake_env, save_video=save_video, config=EnvConfig()
        )
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)

        # !: [新增] 加载并应用阶段特征提取器（在 ChunkingWrapper 之前）
        if stage_classifier:
            from experiments.usb_pickup_insertion.wrapper import StageFeatureWrapper
            
            rng = jax.random.PRNGKey(0)
            # 创建 dummy 数据来初始化网络结构（需要匹配 ChunkingWrapper 之前的形状）
            dummy_obs = env.observation_space.sample()
            # 手动添加 batch 和 time 维度
            dummy_img = dummy_obs["side_stage_classifier"]
            dummy_classifier_input = {"side_stage_classifier": dummy_img[None, None, ...]}
            
            # 初始化分类器结构
            
            classifier_state = create_classifier(
                rng, 
                dummy_classifier_input, 
                self.stage_classifier_keys, 
                n_way=self.stage_num_classes,  # 阶段数
                encoder_type=self.stage_encoder_type
            )
            
            # 加载训练好的权重
            ckpt_path = os.path.abspath("stage_classifier_ckpt/")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"stage_classifier checkpoint dir not found: {ckpt_path}")
            
            classifier_state = checkpoints.restore_checkpoint(ckpt_path, classifier_state)
            print(f"[+] Loaded stage classifier from {ckpt_path}")
            
            # 应用 Wrapper（ResNet18 编码输出 256 维特征）
            env = StageFeatureWrapper(env, classifier_state, feature_dim=self.stage_feature_dim)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        
        if classifier:
            # !: remove stage_features from sample before loading classifier
            classifier_sample = env.observation_space.sample()
            if "stage_features" in classifier_sample:
                del classifier_sample["stage_features"]
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=classifier_sample,
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                
                # ![修复] 计算奖励时，从观测中移除 stage_features
                classifier_obs = {k: v for k, v in obs.items() if k != "stage_features"}
                logits = classifier(classifier_obs)
                logits = jnp.squeeze(logits)          # 变成标量，比如 -4.066795

                score = sigmoid(logits)               # 标量，比如 0.017...
                cond = (score > 0.7) & (obs["state"][0, 0] > 0.4)
                return int(jnp.where(cond, 1, 0).item())      # 返回 JAX 标量 1.0 或 0.0

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env

