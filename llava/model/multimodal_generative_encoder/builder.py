def build_generative_vision_tower(config):
    tower_type = getattr(config, "generative_vision_tower_type", getattr(config, "generative_encoder_type", "wan_vace_online"))

    if tower_type in {"wan_t2v_online", "wan_t2v"}:
        from .wan_t2v_encoder import WanT2VOnlineEncoder

        return WanT2VOnlineEncoder(config)

    if tower_type == "wan_vace_online":
        from .wan_vace_encoder import WanVaceOnlineEncoder

        return WanVaceOnlineEncoder(config)

    if tower_type in {"sd21_online", "sd2.1_online", "diffusion_online"}:
        from .sd21_online_encoder import SD21OnlineEncoder

        return SD21OnlineEncoder(config)

    if tower_type == "svd_online":
        from .svd_online_encoder import SVDOnlineEncoder

        return SVDOnlineEncoder(config)

    if tower_type in {"vjepa_online", "jepa_online"}:
        from .vjepa_online_encoder import VJEPAOnlineEncoder

        return VJEPAOnlineEncoder(config)

    if tower_type in {"dinov3_online", "dino_online"}:
        from .dinov3_online_encoder import DINOv3OnlineEncoder

        return DINOv3OnlineEncoder(config)

    if tower_type in {"vggt_online", "vggt"}:
        from .vggt_online_encoder import VGGTOnlineEncoder

        return VGGTOnlineEncoder(config)

    if tower_type == "vae_online":
        from .vae_online_encoder import VAEOnlineEncoder

        return VAEOnlineEncoder(config)

    raise ValueError(f"Unknown generative vision tower: {tower_type}")


# Backward-compat alias
def build_generative_encoder(config):
    return build_generative_vision_tower(config)
