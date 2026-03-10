import os




if __name__ == '__main__':
    # 切换到项目根目录
    proj_root = os.path.abspath(os.path.join(__file__, "../.."))
    os.chdir(proj_root)
    print(f"[*] 项目根目录：{proj_root}")

    # 加载配置
    from src.utils.config_loader import load_config
    cfg = load_config("configs/basic_cfg.yaml")
    dataset_name = cfg.dataset.name
    score_file = cfg.dataset.score_file
    acoustic_feats = cfg.acoustic_feats

    outputs_root = os.path.join(proj_root, "outputs")
    raw_feats_dir = os.path.join(outputs_root, "raw_feats", dataset_name)
    print("[*] 开始提取特征统计信息...")
    from src.feat_extractor import extract_feats_stats_from_csv
    stats = extract_feats_stats_from_csv(raw_feats_dir)