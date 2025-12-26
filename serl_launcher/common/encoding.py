from typing import Dict, Iterable, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange, repeat


class EncodingWrapper(nn.Module):
    """
    Encodes observations into a single flat encoding, adding additional
    functionality for adding proprioception and stopping the gradient.

    Args:
        encoder: The encoder network.
        use_proprio: Whether to concatenate proprioception (after encoding).
    """

    encoder: nn.Module
    use_proprio: bool
    proprio_latent_dim: int = 64
    enable_stacking: bool = False
    image_keys: Iterable[str] = ("image",)

    @nn.compact
    def __call__(
        self,
        observations: Dict[str, jnp.ndarray],
        train=False,
        stop_gradient=False,
        is_encoded=False,
    ) -> jnp.ndarray:
        # encode images with encoder
        encoded = []
        for image_key in self.image_keys:
            image = observations[image_key]
            if not is_encoded:
                if self.enable_stacking:
                    # Combine stacking and channels into a single dimension
                    if len(image.shape) == 4:
                        image = rearrange(image, "T H W C -> H W (T C)")
                    if len(image.shape) == 5:
                        image = rearrange(image, "B T H W C -> B H W (T C)")

            image = self.encoder[image_key](image, train=train, encode=not is_encoded)

            if stop_gradient:
                image = jax.lax.stop_gradient(image)

            encoded.append(image)

        encoded = jnp.concatenate(encoded, axis=-1)

        if self.use_proprio:
            # project state to embeddings as well
            state = observations["state"]
            if self.enable_stacking:
                # Combine stacking and channels into a single dimension
                if len(state.shape) == 2:
                    state = rearrange(state, "T C -> (T C)")
                    encoded = encoded.reshape(-1)
                if len(state.shape) == 3:
                    state = rearrange(state, "B T C -> B (T C)")
            state = nn.Dense(
                self.proprio_latent_dim, kernel_init=nn.initializers.xavier_uniform()
            )(state)
            state = nn.LayerNorm()(state)
            state = nn.tanh(state)
            encoded = jnp.concatenate([encoded, state], axis=-1)
        
        #! : 3. [新增] 拼接阶段分类器特征 (Stage Features)
        # 这些特征来自 Frozen Classifier，直接使用
        if "stage_features" in observations:
            # debug：
            # jax.debug.print("add stage_features to agent state")
            stage_features = observations["stage_features"]
            
            # 处理 Chunking 带来的时间维度 (T)
            if self.enable_stacking:
                # (Batch, Time, Feature) -> (Batch, Time*Feature)
                if len(stage_features.shape) == 3:
                    stage_features = rearrange(stage_features, "B T F -> B (T F)")
                # (Time, Feature) -> (Time*Feature)
                elif len(stage_features.shape) == 2:
                    stage_features = rearrange(stage_features, "T F -> (T F)")
            
            # 直接拼接到最终的特征向量中
            encoded = jnp.concatenate([encoded, stage_features], axis=-1)
            # print("encoded shape with stage_features:", encoded.shape)
        return encoded
