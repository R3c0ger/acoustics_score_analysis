import os
import re
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)


# ==============================================================================
# 配置与常量
# ==============================================================================

# 定义数据集子集组合，分别为 A+1、B+1、A+B+1（All）
SUBSET_GROUPS = [
    ["A", "1"],
    ["B", "1"],
    ["A", "B", "1"]  # All
]
# 绘图相关常量
PLOT_JITTER_RANGE = 0.12
FIG_SIZE = (16, 6)
DPI = 100
ALPHA = 0.8
MARKER_SIZE = 80
# 样式映射常量
PITCH_MARKERS = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
SCORE_COLOR_MAP = {1: 'red', 3: 'green', 5: 'blue'}


# ==============================================================================
# 数据解析与处理工具函数
# ==============================================================================

def parse_label(filename):
    """获取文件名中最后一个数字序列，作为打分标签"""
    base = os.path.splitext(filename)[0]
    nums = re.findall(r"\d+", base)
    if nums:
        return int(nums[-1])
    return None


def parse_suffix_type(filename):
    """根据文件名后缀判断数据集子集类型，返回"A"、"B"、"1"或None"""
    base = os.path.splitext(filename)[0]
    if re.search(r"-A$", base, flags=re.IGNORECASE):
        return "A"
    if re.search(r"-B$", base, flags=re.IGNORECASE):
        return "B"
    if re.search(r"-1$", base):
        return "1"
    return None


def parse_pitch_digit(filename):
    """从文件名中提取音高信息，返回数字部分（如4），如果没有匹配则返回None"""
    base = os.path.splitext(filename)[0]
    match = re.search(r"([A-Ga-g])(\d)", base)
    if match:
        return int(match.group(2))
    return None


def get_label_subset_pitch_feats(feats_stats, record_name=True):
    """
    从 feats_stats 中提取：打分标签、数据集子集标签、音高、声学特征统计信息（与文件名）。
    其中 feats 是一个列表，每个元素是一个字典，包含一个文件的所有声学特征统计信息。
    """
    labels, subsets, pitches, feats, filenames = [], [], [], [], []
    for filename, stats in feats_stats.items():
        labels.append(parse_label(filename))
        subsets.append(parse_suffix_type(filename))
        pitches.append(parse_pitch_digit(filename))
        feats.append(stats)
        if record_name:
            filenames.append(filename)
    if record_name:
        return labels, subsets, pitches, feats, filenames
    else:
        return labels, subsets, pitches, feats


def filter_data_by_subset(structured_data, allowed_subsets, feature_names=None):
    """
    根据允许的子集标签过滤结构化数据。

    Args:
        structured_data: get_label_subset_pitch_feats 返回的字典格式数据。
        allowed_subsets: 允许保留的子集标签列表 (e.g., ['A', '1'])。
        feature_names: 可选，需要提取的特征名列表。如果为 None，则保留原始字典结构。

    Returns:
        包含 numpy 数组的字典，键为 'labels', 'subsets', 'pitches', 'features', 'filenames'。
    """
    indices = [
        i for i, sub in enumerate(structured_data['subsets'])
        if sub in allowed_subsets
    ]
    if not indices:
        return {k: np.array([]) for k in structured_data.keys()}

    result = {}
    for key in ['labels', 'subsets', 'pitches', 'filenames']:
        result[key] = np.array([structured_data[key][i] for i in indices])

    # 专门处理特征数据
    if feature_names:
        # 将特定特征提取为二维数组
        feat_data = []
        for i in indices:
            row = [structured_data['feats'][i].get(f, np.nan) for f in feature_names]
            feat_data.append(row)
        result['features'] = np.array(feat_data)
    else:
        # 保留原始字典列表结构（如果需要）
        result['features'] = [structured_data['feats'][i] for i in indices]

    return result


def remove_outlier_jitter(feats_stats):
    """
    从 feats_stats 中找到 jitter 特征值最大的文件，将其从 feats_stats 中移除。
    用于避免个别 jitter 特征值过大导致其他点挤在一起。
    """
    outlier_file = None
    max_jitter = -float('inf')
    for filename, stats in feats_stats.items():
        jitter_value = stats.get("Jitter", None)
        if jitter_value is not None and jitter_value > max_jitter:
            max_jitter = jitter_value
            outlier_file = filename
    if outlier_file:
        print(f"[!] 移除异常值：{outlier_file}，jitter={max_jitter}")
        # 返回新字典以避免副作用，或者直接在原字典删除
        return {k: v for k, v in feats_stats.items() if k != outlier_file}
    else:
        print("[*] 未找到异常值，无需移除。")
    return feats_stats


# ==============================================================================
# 绘图工具函数
# ==============================================================================

def get_pitch_style_map(unique_pitches):
    """生成一致的音高颜色和形状映射"""
    color_map = {p: plt.cm.tab10(i) for i, p in enumerate(unique_pitches)}
    marker_map = {
        p: marker
        for p, marker in zip(unique_pitches, PITCH_MARKERS[:len(unique_pitches)])
    }
    return color_map, marker_map


def create_legend_handles_pitch(color_map, marker_map, ax):
    """创建音高图例句柄"""
    handles = [
        plt.Line2D(
            [0], [0],
            marker=marker_map[p],
            color='w',
            label=f'Pitch {p}',
            markerfacecolor=color_map[p],
            markersize=8,
        )
        for p in sorted(color_map.keys())
    ]
    return handles


def create_legend_handles_score(ax):
    """创建评分图例句柄"""
    handles = [
        plt.Line2D(
            [0], [0],
            marker='o', color='w',
            label=f'Score {score}',
            markerfacecolor=SCORE_COLOR_MAP[score],
            markersize=8,
        )
        for score in sorted(SCORE_COLOR_MAP.keys())
    ]
    return handles


def plot_scatter_ndim(feats_stats, outputs_root, ndim):
    """
    绘制散点图的统一入口函数，根据 ndim 参数决定绘制 1D/2D/3D 散点图。

    在 outputs_root 下的 "plot_1d" / "plot_2d" / "plot_3d" 目录中保存所有生成的图像。
    根据 ndim 参数，排列组合出所有 ndim 个特征统计值的组合，
    为每个特征统计值组合创建一个图像。
    每个图像分三个子图，分别展示不同数据子集（A+1、B+1、A+B+1（All））的特征值与打分标签的关系。

    - 1D: x 轴为评分标签，y 轴为一个特征的统计量，点的颜色和形状根据音高区分。
    - 2D/3D: x, y(, z) 轴为声学特征统计值，点的颜色代表评分标签。

    Args:
        feats_stats: 从 CSV 文件提取的特征统计信息。
            格式为 {filename: {feat_name: value, ...}, ...}。
            可从 filename 中解析出打分标签、数据集子集标签、音高等信息。
        outputs_root: 输出目录路径，用于保存生成的图像。
        ndim: 维度 (1, 2, 或 3)。
    """
    if ndim < 1 or ndim > 3:
        raise ValueError("ndim 必须为 1, 2 或 3")

    # 1. 准备数据
    # 从 feats_stats 中提取标签、子集、音高、特征统计信息和文件名，构建结构化数据字典
    labels, subsets, pitches, feats, filenames \
        = get_label_subset_pitch_feats(feats_stats)
    structured_data = {
        'labels': labels,
        'subsets': subsets,
        'pitches': pitches,
        'feats': feats,
        'filenames': filenames
    }

    # 提取所有声学特征统计量名称
    all_feat_names = set()
    for stats in feats_stats.values():
        all_feat_names.update(stats.keys())
        break
    all_feat_names = sorted(all_feat_names)

    # 根据 ndim 确定特征组合
    if ndim == 1:  # 1D 情况下，遍历每个单独的特征
        feat_combinations = [(f,) for f in all_feat_names]
    else:  # 2D/3D 情况下，遍历特征组合
        feat_combinations = list(combinations(all_feat_names, ndim))

    # 创建输出目录
    plot_dir = os.path.join(outputs_root, f"plot_{ndim}d")
    os.makedirs(plot_dir, exist_ok=True)

    # 2. 遍历特征组合进行绘图
    for feat_tuple in feat_combinations:
        fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
        # 设置子图
        if ndim == 3:
            axes = [fig.add_subplot(1, 3, i + 1, projection='3d') for i in range(3)]
        else:
            # 1D 和 2D 使用普通子图，1D 共享 Y 轴
            sharey = (ndim == 1)
            # plt.subplots 返回 (fig, axes) 或 (fig, axes_array)
            _, axes_temp = plt.subplots(1, 3, figsize=FIG_SIZE, sharey=sharey)
            if not isinstance(axes_temp, (list, np.ndarray)):
                axes = [axes_temp]
            else:
                axes = axes_temp

        # 遍历每个子集组 (A+1, B+1, All)
        for i, subset_group in enumerate(SUBSET_GROUPS):
            ax = axes[i]
            # 筛选当前子图对应的数据集子集
            filtered = filter_data_by_subset(structured_data, subset_group, list(feat_tuple))
            if filtered['labels'].size == 0:
                ax.set_axis_off()
                ax.set_title(f"Subset: {'+'.join(subset_group)} (No Data)")
                continue

            # 提取坐标数据
            scores = filtered['labels'].astype(float)
            pitches_arr = filtered['pitches']

            if ndim == 1:
                # --- 1D 绘图逻辑 ---
                # 为分数标签添加 jitter，避免点重叠
                x_coords = scores + np.random.uniform(
                    -PLOT_JITTER_RANGE, PLOT_JITTER_RANGE, size=scores.shape
                )
                y_coords = filtered['features'][:, 0]  # Shape (N, 1) -> (N,)

                # 根据音高区分样式
                unique_p = sorted(list(set(pitches_arr)))
                c_map, m_map = get_pitch_style_map(unique_p)
                for p_val in unique_p:
                    mask = pitches_arr == p_val
                    if np.any(mask):
                        ax.scatter(
                            x_coords[mask], y_coords[mask],
                            c=[c_map[p_val]], marker=m_map[p_val],
                            s=MARKER_SIZE, alpha=ALPHA,
                            )
                ax.set_xlabel("Score Label")
                ax.set_ylabel(f"{feat_tuple[0]} Value")
                ax.set_xticks([1, 3, 5])
                # 只在最后一个子图添加图例
                if i == 2:
                    handles = create_legend_handles_pitch(c_map, m_map, ax)
                    ax.legend(handles=handles, title="Pitch", loc='upper right', fontsize=8)

            elif ndim == 2:
                # --- 2D 绘图逻辑 ---
                x_coords = filtered['features'][:, 0]
                y_coords = filtered['features'][:, 1]
                colors = [SCORE_COLOR_MAP.get(s, 'gray') for s in scores]
                ax.scatter(
                    x_coords, y_coords,
                    c=colors, s=MARKER_SIZE,
                    alpha=ALPHA,
                )
                ax.set_xlabel(feat_tuple[0])
                ax.set_ylabel(feat_tuple[1])
                # 添加图例
                if i == 2:
                    handles = create_legend_handles_score(ax)
                    ax.legend(handles=handles, title="Score Label", loc='upper right', fontsize=8)

            elif ndim == 3:
                # --- 3D 绘图逻辑 ---
                x_coords = filtered['features'][:, 0]
                y_coords = filtered['features'][:, 1]
                z_coords = filtered['features'][:, 2]
                colors = [SCORE_COLOR_MAP.get(s, 'gray') for s in scores]
                ax.scatter(
                    x_coords, y_coords, z_coords,
                    c=colors, s=MARKER_SIZE, alpha=ALPHA,
                )
                ax.set_xlabel(feat_tuple[0])
                ax.set_ylabel(feat_tuple[1])
                ax.set_zlabel(feat_tuple[2])
                # 3D 图例位置调整
                handles = create_legend_handles_score(ax)
                ax.legend(handles=handles, title="Score", loc='upper left', fontsize=8)

            ax.set_title(f"Subset: {'+'.join(subset_group)}", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.3)

        # 设置总标题
        title_suffix = f"{feat_tuple[0]}" if ndim == 1 else f"{', '.join(feat_tuple)}"
        plt.suptitle(f"{ndim}D Scatter Plot for Acoustic Features: {title_suffix}", fontsize=16)
        # plt.tight_layout(rect=[0, 0, 1, 0.9])

        # 保存图像
        filename_safe = "_".join(feat_tuple)
        plot_path = os.path.join(plot_dir, f"scatter_{ndim}d_{filename_safe}.png")
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"[+] 已保存图像：{plot_path}")


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
    print("[*] 开始绘制散点图...")
    stats = remove_outlier_jitter(stats)
    for ndim in [1, 2, 3]:
        print(f"[*] 绘制 {ndim}D 散点图...")
        plot_scatter_ndim(stats, outputs_root, ndim)
    print("[+] 散点图绘制完成！")
