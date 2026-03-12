import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel

from src.combined_data import CombinedData


def _prepare_xy(
        dataset: CombinedData,
        tech_name: str,
        subsets: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """从 CombinedData 对象中提取特定技巧和子集的 X, y。"""
    df_subset = dataset.get_one_score_feats_subset(tech_name, subsets)
    y = df_subset[tech_name].values.astype(np.float32)
    X = df_subset[dataset.feat_cols].values.astype(np.float32)
    return df_subset, X, y


def run_pca_analysis(
        dataset: CombinedData,
        tech_name: str,
        output_dir: str,
        subset_types: Optional[List[str]] = None,
):
    """执行 PCA 分析。"""
    subset_types = subset_types if subset_types else ["A", "B", "1"]
    subset_label = "+".join(subset_types)
    print(f"[*] 运行 PCA | 技巧：{tech_name} | 子集：{subset_label}")

    # 提取子集数据
    df_subset, X, y = _prepare_xy(dataset, tech_name, subset_types)
    if X.size == 0 or len(y) == 0:
        print("[!] 跳过：有效样本不足。")
        return

    # 标准化 & PCA
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    n_comp = min(len(dataset.feat_cols), Xz.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(Xz)

    # 结果保存
    out_path = os.path.join(output_dir, tech_name, subset_label)
    os.makedirs(out_path, exist_ok=True)
    # 保存原始数据
    df_subset.to_csv(os.path.join(out_path, f"feats_with_{tech_name}_score.csv"), index=False)

    # 1. Loadings CSV
    comps = pd.DataFrame(
        pca.components_,
        columns=dataset.feat_cols,
        index=[f"PC{i + 1}" for i in range(n_comp)]
    )
    comps.to_csv(os.path.join(out_path, "pca_loadings.csv"))

    # 2. Variance CSV
    var_df = pd.DataFrame(
        {
            "PC": [f"PC{i + 1}" for i in range(n_comp)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        },
    )
    var_df.to_csv(os.path.join(out_path, "pca_explained_variance.csv"), index=False)

    # 3. PC1 Formula
    pc1 = comps.loc["PC1"].sort_values(key=abs, ascending=False)
    formula = " + ".join([f"{v:.4f}*{k}" for k, v in pc1.items()])
    with open(os.path.join(out_path, "pc1_formula.txt"), "w", encoding="utf-8") as f:
        f.write(formula)

    # 4. Heatmap
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    im = ax.imshow(comps.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(dataset.feat_cols)))
    ax.set_xticklabels(dataset.feat_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_comp))
    ax.set_yticklabels([f"PC{i + 1}" for i in range(n_comp)], fontsize=9)
    for i in range(n_comp):
        for j in range(len(dataset.feat_cols)):
            val = comps.iloc[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                color="white" if abs(val) > 0.6 else "black", fontsize=7,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"PCA Loadings ({tech_name} - {subset_label})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_path, "pca_heatmap.png"), dpi=300)
    plt.close(fig)

    print(f"[+] PCA 完成 (解释方差 Top2: {pca.explained_variance_ratio_[:2].sum():.2%})")


def run_lasso_analysis(
        dataset: CombinedData,
        tech_name: str,
        output_dir: str,
        subset_types: Optional[List[str]] = None,
):
    """执行 LASSO 分析。"""
    subset_types = subset_types if subset_types else ["A", "B", "1"]
    subset_label = "+".join(subset_types)
    print(f"[*] 运行 LASSO | 技巧：{tech_name} | 子集：{subset_label}")

    # 提取子集数据
    df_subset, X, y = _prepare_xy(dataset, tech_name, subset_types)
    if X.size == 0 or len(y) == 0:
        print("[!] 跳过：有效样本不足。")
        return

    # 标准化 & LASSO
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    model = LassoCV(cv=5, random_state=42, n_alphas=100, max_iter=10000)
    model.fit(Xz, y)

    # 结果整理
    coefs = pd.Series(model.coef_, index=dataset.feat_cols)
    coef_df = pd.DataFrame(
        {
            "feature": dataset.feat_cols,
            "coef": coefs.values,
            "abs_coef": np.abs(coefs.values),
        },
    ).sort_values(by="abs_coef", ascending=False)
    coef_df["selected"] = np.abs(coef_df["coef"]) >= 0.01

    # 保存
    out_path = os.path.join(output_dir, tech_name, subset_label)
    os.makedirs(out_path, exist_ok=True)
    # 0. 保存原始数据
    df_subset.to_csv(os.path.join(out_path, f"feats_with_{tech_name}_score.csv"), index=False)
    # 1. 保存系数 CSV
    coef_df.to_csv(os.path.join(out_path, "lasso_coefficients.csv"), index=False)
    # 2. 绘图
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    # 准备数据
    features = coef_df["feature"].tolist()
    coefficients = coef_df["coef"].values
    n_features = len(features)
    # 生成颜色列表
    colors = ['#1f77b4' if c > 0 else '#d62728' for c in coefficients]
    ax.bar(features, coefficients, color=colors, edgecolor='gray', linewidth=0.5)
    ax.axhline(0, color="#333", linewidth=1)
    ax.set_ylabel("Coefficient (Standardized)")
    ax.set_title(f"LASSO Coefficients ({tech_name} - {subset_label})")
    ax.set_ylim(-1, 1)
    ax.set_xticks(range(n_features))
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_path, "lasso_barplot.png"), dpi=300)
    plt.close(fig)

    selected = coef_df[coef_df["selected"]]["feature"].tolist()
    print(f"[+] LASSO 完成 (选中特征: {len(selected)})")


def run_ordinal_regression(
        dataset: CombinedData,
        tech_name: str,
        output_dir: str,
        subset_types: Optional[List[str]] = None,
):
    """
    执行序数回归分析 (Ordinal Regression)。
    使用 statsmodels OrderedModel (Probit/Logit link) 分析特征对有序评分的影响。
    """
    subset_types = subset_types if subset_types else ["A", "B", "1"]
    subset_label = "+".join(subset_types)
    print(f"[*] 运行序数回归 | 技巧：{tech_name} | 子集：{subset_label}")

    # 提取子集数据
    df_subset, X, y = _prepare_xy(dataset, tech_name, subset_types)
    if X.size == 0 or len(y) == 0:
        print("[!] 跳过：有效样本不足。")
        return
    unique_scores = sorted(np.unique(y))

    # 标准化特征 (X)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    feature_cols = dataset.feat_cols
    # 构建 DataFrame 以便 statsmodels 使用
    df_model = pd.DataFrame(Xz, columns=feature_cols)
    df_model['target'] = y.astype(int)

    # 构建模型
    y_cat = pd.Categorical(df_model['target'], categories=unique_scores, ordered=True)
    try:
        model = OrderedModel(endog=y_cat, exog=df_model[feature_cols], distr='probit')
        result = model.fit(method='bfgs', maxiter=10000, disp=False)
    except Exception as e:
        print(f"[!] 模型拟合失败: {e}")
        return

    # 保存结果
    out_path = os.path.join(output_dir, tech_name, subset_label)
    os.makedirs(out_path, exist_ok=True)
    # 0. 保存原始数据集 (含评分和原始特征)
    df_output = df_subset.copy()
    df_output.to_csv(os.path.join(out_path, f"feats_with_{tech_name}_score.csv"), index=False)
    # 1. 保存标准化后的数据集 (用于建模的数据)
    df_scaled = df_model.copy()
    df_scaled.to_csv(os.path.join(out_path, f"feats_with_{tech_name}_score_scaled.csv"), index=False)
    # 2. 保存回归系数表
    # 提取统计量，整理结果表格
    mask = [idx in feature_cols for idx in result.params.index]
    coef_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "coef": result.params[mask].values,
            "std_err": result.bse[mask].values,
            "z_value": result.tvalues[mask].values,
            "p_value": result.pvalues[mask].values,
            "odds_ratio": np.exp(result.params[mask].values),
            "or_ci_lower": np.exp(result.conf_int()[mask][0]),
            "or_ci_upper": np.exp(result.conf_int()[mask][1]),
        }
    )
    # 按系数绝对值排序 (影响度)
    coef_df["impact"] = np.abs(coef_df["coef"])
    coef_df = coef_df.sort_values(by="impact", ascending=False).reset_index(drop=True)
    coef_df.to_csv(os.path.join(out_path, "ordinal_regression_coefficients.csv"), index=False)
    # 3. 保存模型摘要文本
    summary_text = result.summary().as_text()
    with open(os.path.join(out_path, "ordinal_regression_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Ordinal Regression Summary\n")
        f.write(f"Technique: {tech_name}\n")
        f.write(f"Subset: {subset_label}\n")
        f.write(f"Link Function: Probit\n")
        f.write("=" * 60 + "\n\n")
        f.write(summary_text)

    significant_feats = coef_df[coef_df["p_value"] < 0.05]["feature"].tolist()
    print(f"[+] 序数回归完成 | 显著特征 (p<0.05): {len(significant_feats)}")
    if significant_feats:
        print(f"    -> {', '.join(significant_feats[:5])}{'...' if len(significant_feats) > 5 else ''}")


if __name__ == '__main__':
    # 切换到项目根目录
    proj_root = os.path.abspath(os.path.join(__file__, "../.."))
    os.chdir(proj_root)
    print(f"[*] 项目根目录：{proj_root}")

    # 加载配置
    from src.utils.config_loader import load_config
    cfg = load_config("configs/basic_cfg.yaml")
    data_root = cfg.dataset.root_dir
    dataset_name = cfg.dataset.name
    score_file = cfg.dataset.score_file
    subset_groups = cfg.dataset.subset_groups
    print(f"[*] 数据集：{dataset_name} | 评分文件：{score_file} | 子集分组：{subset_groups}")
    outputs_root = os.path.join(proj_root, "outputs")
    raw_feats_dir = os.path.join(outputs_root, "raw_feats", dataset_name)
    analysis_dir = os.path.join(outputs_root, "analysis")

    print("[*] 开始提取特征统计信息...")
    from src.feat_extractor import extract_feats_stats_from_csv
    df_stats = extract_feats_stats_from_csv(raw_feats_dir)
    print(df_stats.head())
    print(f"[*] 加载评分矩阵：{score_file}")
    from src.data_loader import load_score_matrix
    score_path = os.path.join(data_root, dataset_name, score_file)
    df_score = load_score_matrix(score_path)
    print(df_score.head())
    print("[*] 开始合并评分矩阵和特征统计信息...")
    from src.combined_data import CombinedData
    combined_data = CombinedData(df_score, df_stats)
    print("[+] 合并完成！")

    print("[*] 开始分析...")
    for tech in combined_data.tech_cols:
        for subset_group in subset_groups:
            run_pca_analysis(combined_data, tech, analysis_dir, subset_group)
            run_lasso_analysis(combined_data, tech, analysis_dir, subset_group)
            run_ordinal_regression(combined_data, tech, analysis_dir, subset_group)
