from omegaconf import OmegaConf


def load_config(config_path: str):
    """加载 YAML 配置文件并返回一个 OmegaConf 对象。"""
    try:
        cfg = OmegaConf.load(config_path)
        print(f"[*] 成功加载配置文件：{config_path}")
        return cfg
    except Exception as e:
        print(f"[!] 加载配置文件失败：{e}")
        raise
