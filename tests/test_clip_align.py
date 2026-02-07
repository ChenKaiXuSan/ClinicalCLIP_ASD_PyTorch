from types import SimpleNamespace

import torch

from project.models.clip_align import VideoAttentionCLIP


def test_clip_align_map_guided_forward():
    hparams = SimpleNamespace(
        model=SimpleNamespace(
            clip_feature_dim=32,
            clip_embed_dim=16,
            model_class_num=3,
            attn_in_channels=1,
            clip_backbone="3dcnn",
            map_guided=True,
            map_guided_type="channel",
            map_guided_hidden_dim=16,
        )
    )

    model = VideoAttentionCLIP(hparams)
    video = torch.randn(2, 3, 4, 8, 8)
    attn_map = torch.randn(2, 1, 4, 8, 8)

    outputs = model(video, attn_map)

    assert outputs["logits"].shape == (2, 3)
    assert outputs["video_embed"].shape == (2, 16)
    assert outputs["attn_embed"].shape == (2, 16)
    assert outputs["video_gate"] is not None
    assert outputs["video_gate"].shape == (2, 16)
    assert outputs["video_tokens"] is not None
    assert outputs["video_tokens"].shape == (2, 16, 2, 4, 4)
