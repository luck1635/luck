#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学习规划助手（熵权法）
v7.0 - 性能优化：降低时间与空间复杂度
- allocate_time: 子任务按学科预分组，避免重复过滤 O(P*C)->O(P+C)
- reorder_by_subject: 单次排序替代 iterrows 逐行拼接 O(n log n)
- show_allocation_preview: 向量化缩进替代 apply O(n)->向量化
- save_history: 向量化构建历史记录替代 iterrows
- entropy_weight: np.isnan 替代 pd.isnull，减少不必要拷贝
"""

import numpy as np
import pandas as pd
import os
import json
import re
from datetime import datetime

# ---------- 设置 Pandas 显示选项 ----------
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_colwidth', 20)
pd.set_option('display.float_format', '{:.2f}'.format)

# ---------- 学科满分映射 ----------
FULL_MARKS = {
    '语文': 150,
    '数学': 150,
    '英语': 150,
    '物理': 100,
    '化学': 100,
    '生物': 100
}
def get_full_mark(subject):
    return FULL_MARKS.get(subject, 150)

# -------------------------- 熵权法核心 --------------------------
def entropy_weight(df, positive_cols=None, method='range'):
    # 直接取 numpy 数组，减少 DataFrame 拷贝开销
    if isinstance(df, pd.DataFrame):
        arr = df.to_numpy(dtype=float)
        cols = df.columns
    else:
        arr = np.asarray(df, dtype=float)
        cols = None
    n_samples, n_cols = arr.shape
    if n_samples <= 1:
        idx = cols if cols is not None else range(n_cols)
        return pd.Series([1.0 / n_cols] * n_cols, index=idx)
    # np.isnan 比 pd.isnull 更快（纯 numpy 数组）
    null_mask = np.any(np.isnan(arr), axis=0)
    if np.any(null_mask):
        first_null = int(np.where(null_mask)[0][0])
        col_name = cols[first_null] if cols is not None else first_null
        raise ValueError(f"列 '{col_name}' 存在缺失值")
    pos_set = set(positive_cols) if positive_cols is not None else set()
    is_pos = np.array([c in pos_set for c in cols]) if cols is not None else np.zeros(n_cols, dtype=bool)
    if method == 'range':
        min_vals = arr.min(axis=0)
        max_vals = arr.max(axis=0)
        ranges = max_vals - min_vals
        nonzero = ranges != 0
        coef = np.where(is_pos, 1.0, -1.0)
        offset = np.where(is_pos, min_vals, max_vals)
        std_arr = np.zeros_like(arr)
        if np.any(nonzero):
            std_arr[:, nonzero] = (coef[nonzero] * (arr[:, nonzero] - offset[nonzero]) / ranges[nonzero])
    elif method == 'zscore':
        mean_vals = arr.mean(axis=0)
        std_vals = arr.std(axis=0, ddof=1)
        nonzero = std_vals != 0
        coef = np.where(is_pos, 1.0, -1.0)
        std_arr = np.zeros_like(arr)
        if np.any(nonzero):
            std_arr[:, nonzero] = (coef[nonzero] * (arr[:, nonzero] - mean_vals[nonzero]) / std_vals[nonzero])
        col_mins = std_arr.min(axis=0)
        shift = np.where(col_mins < 0, -col_mins + 1e-6, 0.0)
        std_arr += shift.reshape(1, -1)  # 原地加，减少一次拷贝
    else:
        std_arr = arr.copy()
    std_arr += 1e-8
    col_sums = std_arr.sum(axis=0)
    p_matrix = std_arr / col_sums
    k = 1.0 / np.log(n_samples)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = np.log(p_matrix)
    log_p[~np.isfinite(log_p)] = 0.0
    entropy = -k * (p_matrix * log_p).sum(axis=0)
    diff = 1 - entropy
    diff_sum = diff.sum()
    idx = cols if cols is not None else range(n_cols)
    if diff_sum == 0:
        weights = pd.Series(1.0 / n_cols, index=idx)
    else:
        weights = pd.Series(diff / diff_sum, index=idx)
    return weights

# -------------------------- 任务分配核心（两步分配）-------------------------
def allocate_time(total_minutes, subject_data, custom_tasks_list):
    """
    两步分配：学科间分配 + 学科内部分配
    优化：子任务按学科预分组，避免每个学科重复过滤 custom_df
    """
    # 清理 subject_data 索引
    subject_data_clean = subject_data.copy()
    subject_data_clean.index = subject_data_clean.index.str.strip()
    df_sub = pd.DataFrame({
        '重要度': subject_data_clean['量化的重要度'],
        '紧急度': subject_data_clean['紧急度']
    }, index=subject_data_clean.index)

    if df_sub.empty:
        print("无学科任务，无法分配")
        return pd.DataFrame()

    # 子任务按学科预分组（O(C) 一次扫描，避免每个学科重复过滤）
    children_by_subject = {}
    if custom_tasks_list:
        for t in custom_tasks_list:
            sub = str(t.get('subject', '其他')).strip()
            children_by_subject.setdefault(sub, []).append(t)

    # 处理孤立子任务：学科不在列表中的归入"其他"
    known_subjects = set(df_sub.index)
    orphan_subs = [s for s in children_by_subject if s not in known_subjects]
    if orphan_subs:
        other_children = []
        for s in orphan_subs:
            other_children.extend(children_by_subject.pop(s))
        if other_children:
            children_by_subject.setdefault('其他', []).extend(other_children)
            if '其他' not in df_sub.index:
                df_sub.loc['其他'] = {'重要度': 5.0, '紧急度': 5.0}

    # ---- 第一步：学科间分配 ----
    try:
        sub_weights = entropy_weight(df_sub[['重要度', '紧急度']], positive_cols=['重要度', '紧急度'], method='range')
    except Exception:
        sub_weights = pd.Series([0.5, 0.5], index=['重要度', '紧急度'])
    w_imp, w_urg = sub_weights['重要度'], sub_weights['紧急度']
    sub_scores = df_sub['重要度'] * w_imp + df_sub['紧急度'] * w_urg
    sub_total = sub_scores.sum()
    if sub_total == 0:
        sub_allocated = pd.Series(total_minutes / len(sub_scores), index=sub_scores.index)
    else:
        sub_allocated = (sub_scores / sub_total) * total_minutes

    # ---- 第二步：每个学科内部再分配 ----
    all_results = []
    for sub in df_sub.index:
        parent_time = sub_allocated[sub]
        sub_children = children_by_subject.get(sub, [])
        # 构建内部任务数据（母任务 + 子任务）
        inner_names = [sub]
        inner_imp = [df_sub.loc[sub, '重要度']]
        inner_urg = [df_sub.loc[sub, '紧急度']]
        for t in sub_children:
            inner_names.append(t['name'])
            inner_imp.append(t['importance'])
            inner_urg.append(t['urgency'])
        n_inner = len(inner_names)
        if n_inner == 1:
            # 只有母任务
            all_results.append({
                '任务名称': sub,
                '重要度': inner_imp[0],
                '紧急度': inner_urg[0],
                '综合得分': sub_scores.get(sub, 0),
                '建议时间(分钟)': parent_time,
                '所属学科': None,
                '层级': 0
            })
        else:
            # 有子任务，内部计算权重和得分
            inner_arr = np.column_stack([inner_imp, inner_urg])
            inner_df = pd.DataFrame(inner_arr, columns=['重要度', '紧急度'])
            try:
                inner_weights = entropy_weight(inner_df, positive_cols=['重要度', '紧急度'], method='range')
                iw_imp, iw_urg = inner_weights['重要度'], inner_weights['紧急度']
            except Exception:
                iw_imp, iw_urg = 0.5, 0.5
            inner_scores = inner_arr[:, 0] * iw_imp + inner_arr[:, 1] * iw_urg
            inner_total = inner_scores.sum()
            if inner_total == 0:
                inner_alloc = np.full(n_inner, parent_time / n_inner)
            else:
                inner_alloc = (inner_scores / inner_total) * parent_time
            # 批量追加结果，减少 append 次数
            for i in range(n_inner):
                is_parent = (i == 0)
                all_results.append({
                    '任务名称': inner_names[i],
                    '重要度': inner_imp[i],
                    '紧急度': inner_urg[i],
                    '综合得分': inner_scores[i],
                    '建议时间(分钟)': inner_alloc[i],
                    '所属学科': sub if not is_parent else None,
                    '层级': 0 if is_parent else 1
                })

    return pd.DataFrame(all_results)

# -------------------------- 分级重排函数（单次排序，O(n log n)）-------------------------
def reorder_by_subject(df):
    """
    优化：用单次排序替代 iterrows 逐行拼接
    - 添加 _group_score 列：母任务用自身综合得分，子任务用其母任务的综合得分
    - 按 _group_score 降序 → 层级升序(母在前) → 综合得分降序 排序
    """
    if df.empty:
        return df
    df = df.copy()
    if '层级' not in df.columns:
        # 兼容旧数据：用所属学科判断层级
        _null_set = {'None', 'nan', 'NaN', 'NaT', ''}
        subj_clean = df['所属学科'].astype(str).str.strip()
        df['层级'] = np.where(subj_clean.isin(_null_set) | subj_clean.isna(), 0, 1)
    # 建立母任务名称 -> 综合得分 的映射（O(P)）
    parents = df[df['层级'] == 0]
    parent_score_map = dict(zip(parents['任务名称'].astype(str).str.strip(), parents['综合得分']))
    # 子任务的 _group_score = 其母任务的综合得分；母任务的 _group_score = 自身综合得分
    task_name_clean = df['任务名称'].astype(str).str.strip()
    subject_clean = df['所属学科'].astype(str).str.strip()
    # 子任务通过 所属学科 查找母任务得分
    df['_group_score'] = np.where(
        df['层级'] == 0,
        df['综合得分'],
        subject_clean.map(parent_score_map).fillna(-1).values
    )
    # 单次排序：组得分降序 → 层级升序(母0在前子1在后) → 综合得分降序
    df = df.sort_values(
        by=['_group_score', '层级', '综合得分'],
        ascending=[False, True, False],
        kind='mergesort'  # 稳定排序
    )
    df = df.drop(columns=['_group_score'], errors='ignore')
    return df

# -------------------------- 任务管理模块（实时预览）-------------------------
def show_allocation_preview(total_minutes, subject_data, custom_tasks_list, auto_weights=None):
    result = allocate_time(total_minutes, subject_data, custom_tasks_list)
    if result.empty:
        print("无任务，无法分配")
        return None
    result_ordered = reorder_by_subject(result)
    print("\n===== 当前时间分配预览 =====")
    preview = result_ordered.copy()
    # 向量化缩进：层级==1 加4空格，否则不加（替代 apply 逐行调用）
    indent = np.where(preview['层级'].values == 1, '    ', '')
    preview['任务名称'] = indent + preview['任务名称'].astype(str).values
    display_cols = ['任务名称', '重要度', '紧急度', '综合得分', '建议时间(分钟)']
    print(preview[display_cols].to_string(index=False))
    return None

def _print_custom_task_list(tasks):
    """打印自定义任务编号列表，编号与 tasks 列表索引严格对应"""
    if not tasks:
        print("（暂无自主任务）")
        return
    print("\n当前自主任务列表（仅可操作自主任务，学科母任务不可操作）：")
    # 用 join 一次性构建输出，减少 print 调用
    lines = [
        f"  {i}. [{t['subject']}] {t['name']} | 重要度:{t['importance']:.1f} | 紧急度:{t['urgency']:.1f}"
        for i, t in enumerate(tasks, 1)
    ]
    print('\n'.join(lines))

def manage_custom_tasks(subject_importance_dict, total_minutes, subject_data, existing_tasks=None, last_time_range=""):
    if existing_tasks is None:
        tasks = []
    else:
        tasks = existing_tasks.copy()
        for t in tasks:
            if 'subject' not in t:
                t['subject'] = '其他'
    subjects = sorted(subject_importance_dict.keys())
    print("\n===== 自主安排任务管理 =====")
    print("以下任务将参与时间分配（学科任务已自动加入）")
    print(f"当前总时间: {total_minutes} 分钟 (时间段: {last_time_range if last_time_range else '未记录'})")
    show_allocation_preview(total_minutes, subject_data, tasks, None)
    while True:
        print("\n选项： [1]添加任务  [2]修改任务  [3]删除任务  [4]完成并保存")
        choice = input("请输入选项(1/2/3/4): ").strip()
        if choice == '1':
            print("\n可选学科：")
            for idx, sub in enumerate(subjects, 1):
                print(f"  {idx}. {sub} (重要度:{subject_importance_dict.get(sub,5.0):.1f})")
            print(f"  {len(subjects)+1}. 其他 (重要度:5.0)")
            print("  0. 手动输入学科名")
            sub_choice = input("请选择学科编号 (或输入学科名称): ").strip()
            if sub_choice == '0':
                subject = input("请输入学科名称: ").strip()
                if not subject:
                    print("学科名不能为空")
                    continue
                default_imp = 5.0
            else:
                try:
                    idx = int(sub_choice)
                    if 1 <= idx <= len(subjects):
                        subject = subjects[idx-1]
                        default_imp = subject_importance_dict.get(subject, 5.0)
                    elif idx == len(subjects)+1:
                        subject = '其他'
                        default_imp = 5.0
                    else:
                        print("无效编号")
                        continue
                except ValueError:
                    subject = sub_choice
                    default_imp = subject_importance_dict.get(subject, 5.0)
            subject = subject.strip()
            name = input("任务名称: ").strip()
            if not name:
                print("任务名称不能为空")
                continue
            imp_str = input(f"重要度(1-10, 默认{default_imp:.1f}): ").strip()
            imp = default_imp if imp_str == '' else float(imp_str) if imp_str.replace('.','').isdigit() else default_imp
            urg_str = input("紧急度(1-10): ").strip()
            try:
                urg = float(urg_str)
                if urg < 1 or urg > 10:
                    raise ValueError
            except Exception:
                print("紧急度输入无效，设为5.0")
                urg = 5.0
            tasks.append({'name': name, 'subject': subject, 'importance': imp, 'urgency': urg})
            print(f"任务 '{name}' ({subject}) 已添加")
            show_allocation_preview(total_minutes, subject_data, tasks, None)
        elif choice == '2':
            if not tasks:
                print("暂无自主任务")
                continue
            _print_custom_task_list(tasks)
            try:
                idx = int(input("\n请输入要修改的编号: ")) - 1
                if idx < 0 or idx >= len(tasks):
                    print("无效编号")
                    continue
                t = tasks[idx]
                print(f"当前任务: [{t['subject']}] {t['name']} | 重要度:{t['importance']} | 紧急度:{t['urgency']}")
                new_name = input(f"新任务名称(原:{t['name']}): ").strip()
                if new_name:
                    t['name'] = new_name
                change_sub = input("是否修改学科? (y/n): ").strip().lower()
                if change_sub == 'y':
                    print("\n可选学科：")
                    for i2, sub in enumerate(subjects, 1):
                        print(f"  {i2}. {sub} (重要度:{subject_importance_dict.get(sub,5.0):.1f})")
                    print(f"  {len(subjects)+1}. 其他 (重要度:5.0)")
                    sub_choice = input("请选择学科编号 (或输入学科名称): ").strip()
                    if sub_choice == '0':
                        new_sub = input("请输入学科名称: ").strip()
                        if new_sub:
                            t['subject'] = new_sub
                            default_imp = subject_importance_dict.get(new_sub, 5.0)
                    else:
                        try:
                            idx2 = int(sub_choice)
                            if 1 <= idx2 <= len(subjects):
                                t['subject'] = subjects[idx2-1]
                                default_imp = subject_importance_dict.get(t['subject'], 5.0)
                            elif idx2 == len(subjects)+1:
                                t['subject'] = '其他'
                                default_imp = 5.0
                            else:
                                print("无效编号，学科未修改")
                        except ValueError:
                            t['subject'] = sub_choice
                            default_imp = subject_importance_dict.get(t['subject'], 5.0)
                    t['subject'] = t['subject'].strip()
                    update_imp = input(f"是否更新重要度为此学科默认值 {default_imp:.1f}? (y/n): ").strip().lower()
                    if update_imp == 'y':
                        t['importance'] = default_imp
                new_imp = input(f"新重要度(原:{t['importance']}): ").strip()
                if new_imp and new_imp.replace('.','').isdigit():
                    t['importance'] = float(new_imp)
                new_urg = input(f"新紧急度(原:{t['urgency']}): ").strip()
                if new_urg and new_urg.replace('.','').isdigit():
                    t['urgency'] = float(new_urg)
                print("任务已修改")
                show_allocation_preview(total_minutes, subject_data, tasks, None)
            except Exception as e:
                print(f"输入无效: {e}")
        elif choice == '3':
            if not tasks:
                print("暂无自主任务")
                continue
            _print_custom_task_list(tasks)
            try:
                idx = int(input("\n请输入要删除的编号: ")) - 1
                if 0 <= idx < len(tasks):
                    removed = tasks.pop(idx)
                    print(f"已删除 [{removed['subject']}] {removed['name']}")
                    show_allocation_preview(total_minutes, subject_data, tasks, None)
                else:
                    print("无效编号")
            except ValueError:
                print("输入无效")
        elif choice == '4':
            break
        else:
            print("无效选项")
    return tasks, None

# -------------------------- 解析时间段 -------------------------
def parse_time_range_to_minutes(user_input):
    segments = re.split(r'[;；]', user_input)
    total_minutes = 0
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        seg = seg.replace('：', ':')
        pattern = r'(\d{1,2}):(\d{2})\s*[-~]\s*(\d{1,2}):(\d{2})|(\d{1,2}):(\d{2})\s+(\d{1,2}):(\d{2})'
        m = re.match(pattern, seg)
        if not m:
            raise ValueError(f"时间段 '{seg}' 格式错误，请使用如 '08:00-12:00' 或 '08:00 12:00'")
        if m.group(1):
            h1, m1, h2, m2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        else:
            h1, m1, h2, m2 = int(m.group(5)), int(m.group(6)), int(m.group(7)), int(m.group(8))
        start = h1 * 60 + m1
        end = h2 * 60 + m2
        if end <= start:
            end += 24 * 60
        total_minutes += (end - start)
    if total_minutes <= 0:
        raise ValueError("总时间必须大于0")
    return total_minutes

# -------------------------- 读取与预处理 --------------------------
def read_dataframe(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext == '.xlsx':
        return pd.read_excel(file_path, engine='openpyxl')
    elif ext == '.xls':
        return pd.read_excel(file_path, engine='xlrd')
    else:
        try:
            return pd.read_excel(file_path, engine='openpyxl')
        except Exception:
            return pd.read_csv(file_path)

def extract_subject(name):
    if pd.isna(name):
        return '其他'
    s = str(name)
    if '化：' in s:
        return '化学'
    if '物：' in s:
        return '物理'
    if '数：' in s:
        return '数学'
    if '生：' in s:
        return '生物'
    if '语：' in s:
        return '语文'
    if '英：' in s or '外：' in s:
        return '英语'
    return '其他'

def map_difficulty(diff):
    if pd.isna(diff):
        return 0.3
    s = str(diff)
    if '难' in s:
        return 0.8
    if '较难' in s:
        return 0.6
    return 0.3

def clean_path(raw):
    raw = raw.strip()
    if raw.startswith('&'):
        raw = raw[1:].strip()
    if raw.startswith("'") and raw.endswith("'"):
        raw = raw[1:-1]
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1]
    raw = raw.replace('\\ ', ' ')
    raw = re.sub(r'\s+', ' ', raw)
    return raw

def get_file_path(prompt, allow_skip=False):
    print(prompt)
    print("提示：可直接拖拽文件到窗口")
    while True:
        raw = input("> ").strip()
        if raw == '' and allow_skip:
            return None
        path = clean_path(raw)
        if path and os.path.exists(path):
            return path
        else:
            print(f"文件不存在: {path}")

# -------------------------- 加载成绩表格 --------------------------
def load_score_table(file_path):
    df = pd.read_excel(file_path, header=None, engine='openpyxl')
    if df.empty:
        raise ValueError("成绩表格为空")
    header_row = df.iloc[0]
    subject_columns = []
    subject_names = []
    for idx, val in header_row.items():
        val_str = str(val).strip()
        if val_str and val_str not in ['实分', '年排', 'nan', ''] and not val_str.lower().startswith('实分'):
            subject_columns.append(idx)
            subject_names.append(val_str)
    if subject_columns:
        real_score_row_index = None
        for i in range(len(df)):
            first_cell = str(df.iloc[i, 0]).strip() if len(df.columns) > 0 else ''
            if first_cell == '实分':
                row_values = df.iloc[i][subject_columns]
                if not row_values.isnull().all():
                    real_score_row_index = i
                    break  # 找到即停止，避免不必要的遍历
        if real_score_row_index is not None:
            score_row = df.iloc[real_score_row_index]
            scores = {}
            for col, sub in zip(subject_columns, subject_names):
                val = score_row[col]
                if pd.notna(val):
                    try:
                        scores[sub] = float(val)
                    except (TypeError, ValueError):
                        pass
            return scores, subject_names
        else:
            return {}, subject_names
    df_header = pd.read_excel(file_path, header=0, engine='openpyxl')
    subject_col = None
    score_col = None
    for col in df_header.columns:
        col_lower = str(col).lower()
        if any(k in col_lower for k in ['学科', '科目', '课程', '名称', 'subject']):
            subject_col = col
        if any(k in col_lower for k in ['成绩', '分数', '得分', 'score']):
            score_col = col
    if subject_col is not None and score_col is not None:
        sub_series = df_header[subject_col].dropna()
        score_series = df_header[score_col].dropna()
        common_idx = sub_series.index.intersection(score_series.index)
        if len(common_idx) > 0:
            sub_vals = sub_series.loc[common_idx].astype(str).str.strip()
            score_vals = pd.to_numeric(score_series.loc[common_idx], errors='coerce')
            temp_df = pd.DataFrame({'subject': sub_vals, 'score': score_vals}).dropna()
            if not temp_df.empty:
                scores = temp_df.groupby('subject')['score'].mean().to_dict()
                subjects = list(scores.keys())
                return scores, subjects
    raise ValueError("未检测到有效成绩格式")

# -------------------------- 保存与加载 --------------------------
SAVE_FILE = "study_planner_data.json"

def save_data(weights, tasks, last_time_range="", target_scores=None, subject_data=None):
    if target_scores is None:
        target_scores = {}
    subject_info = {}
    if subject_data is not None and not subject_data.empty:
        for sub in subject_data.index:
            subject_info[sub] = {
                '量化的重要度': subject_data.loc[sub, '量化的重要度'],
                '基础紧急度': subject_data.loc[sub, '基础紧急度'] if '基础紧急度' in subject_data.columns else subject_data.loc[sub, '紧急度']
            }
    data = {
        'weights': {k: v.to_dict() for k, v in weights.items() if v is not None and not v.empty},
        'custom_tasks': tasks,
        'last_time_range': last_time_range,
        'target_scores': target_scores,
        'subject_info': subject_info
    }
    with open(SAVE_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_data():
    if not os.path.exists(SAVE_FILE):
        return None, None, "", {}, None
    with open(SAVE_FILE, 'r', encoding='utf-8') as f:
        d = json.load(f)
    weights = {}
    for k, vdict in d.get('weights', {}).items():
        weights[k] = pd.Series(vdict)
    tasks = d.get('custom_tasks', [])
    last_time_range = d.get('last_time_range', "")
    target_scores = d.get('target_scores', {})
    subject_info = d.get('subject_info', {})
    return weights, tasks, last_time_range, target_scores, subject_info

def save_data_excel(subject_stats, allocation_result, weights_dict, full_data=True):
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    file_path = os.path.join(desktop, "数据.xlsx")
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        if allocation_result is not None and not allocation_result.empty:
            ordered_result = reorder_by_subject(allocation_result)
        else:
            ordered_result = allocation_result
        if '层级' in ordered_result.columns:
            ordered_result = ordered_result.drop(columns=['层级'])
        ordered_result.to_excel(writer, sheet_name='时间分配建议', index=False)
        weight_df = pd.DataFrame({
            '模型': list(weights_dict.keys()),
            '权重详情': [str(w.to_dict()) for w in weights_dict.values()]
        })
        weight_df.to_excel(writer, sheet_name='模型权重', index=False)
        if full_data and subject_stats is not None and not subject_stats.empty:
            subject_stats.to_excel(writer, sheet_name='学科统计', index=False)
        else:
            pd.DataFrame({'说明': ['未加载新表格，无学科统计数据']}).to_excel(writer, sheet_name='说明', index=False)
    print(f"✅ 核心数据已保存至桌面: {file_path}")

# -------------------------- 历史记录（向量化构建）--------------------------
def save_history(result_df, total_minutes, time_range, target_scores):
    if result_df.empty:
        return
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    history_file = os.path.join(desktop, "历史记录.xlsx")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    target_str = '; '.join([f"{k}:{v:.1f}" for k, v in target_scores.items()])
    # 向量化构建：直接从 DataFrame 列取值，避免 iterrows 逐行 Python 循环
    new_history = pd.DataFrame({
        '时间戳': timestamp,
        '时间段': time_range,
        '总时间(分钟)': total_minutes,
        '任务名称': result_df['任务名称'].to_numpy(),
        '重要度': result_df['重要度'].to_numpy(),
        '紧急度': result_df['紧急度'].to_numpy(),
        '综合得分': result_df['综合得分'].to_numpy(),
        '建议时间(分钟)': result_df['建议时间(分钟)'].to_numpy(),
        '目标分': target_str
    })
    if os.path.exists(history_file):
        try:
            existing = pd.read_excel(history_file, engine='openpyxl')
            combined = pd.concat([existing, new_history], ignore_index=True)
            combined.to_excel(history_file, index=False, engine='openpyxl')
        except Exception as e:
            print(f"读取历史文件失败，将重新创建: {e}")
            new_history.to_excel(history_file, index=False, engine='openpyxl')
    else:
        new_history.to_excel(history_file, index=False, engine='openpyxl')
    print(f"✅ 历史记录已追加至桌面：{history_file}")

# -------------------------- 主程序 --------------------------
def main():
    print("=" * 60)
    print("学习规划助手（熵权法）")
    print("顺序：成绩表格 → 目标分 → 作业记录 → 时间段 → 任务管理")
    print("=" * 60)

    saved_weights, saved_tasks, last_time_range, saved_target_scores, saved_subject_info = load_data()

    # ---------- 1. 成绩表格 ----------
    print("\n--- 第一步：提供成绩表格（提取科目） ---")
    avg_scores = {}
    score_subjects = []
    score_choice = 'n'
    if saved_weights:
        print("发现上次保存数据，可选择跳过或重新输入。")
        score_choice = input("直接回车使用上次成绩表格，输入 n 重新输入: ").strip().lower()
        if score_choice == 'n':
            score_path = get_file_path("拖入成绩表格路径（不可跳过）:", allow_skip=False)
            if not score_path:
                print("未提供成绩表格，无法继续。")
                return
            try:
                avg_scores, score_subjects = load_score_table(score_path)
                print(f"\n从成绩表格中提取到以下科目：{score_subjects}")
            except Exception as e:
                print(f"读取成绩表格失败: {e}，程序终止")
                return
        else:
            if saved_subject_info and saved_target_scores:
                score_subjects = list(saved_target_scores.keys())
                avg_scores = {}
                print(f"\n使用上次保存的科目列表：{score_subjects}")
            else:
                print("上次未保存有效成绩数据，请重新输入。")
                score_path = get_file_path("拖入成绩表格路径（不可跳过）:", allow_skip=False)
                if not score_path:
                    return
                try:
                    avg_scores, score_subjects = load_score_table(score_path)
                    print(f"\n从成绩表格中提取到以下科目：{score_subjects}")
                except Exception as e:
                    print(f"读取成绩表格失败: {e}，程序终止")
                    return
    else:
        score_path = get_file_path("拖入成绩表格路径（不可跳过）:", allow_skip=False)
        if not score_path:
            print("未提供成绩表格，无法继续。")
            return
        try:
            avg_scores, score_subjects = load_score_table(score_path)
            print(f"\n从成绩表格中提取到以下科目：{score_subjects}")
        except Exception as e:
            print(f"读取成绩表格失败: {e}，程序终止")
            return

    # ---------- 2. 目标分 ----------
    print("\n--- 第二步：输入各学科目标分 ---")
    if saved_target_scores and score_choice != 'n':
        print("上次已保存的各学科目标分：")
        for sub, val in saved_target_scores.items():
            full = get_full_mark(sub)
            print(f"  {sub}: {val:.1f} (满分{full})")
        target_choice = input("直接回车沿用，输入 n 重新输入: ").strip().lower()
        if target_choice == 'n':
            target_scores = {}
            for sub in score_subjects:
                full = get_full_mark(sub)
                actual = avg_scores.get(sub, None)
                prompt = f"  {sub} 的目标分 (直接回车默认100"
                if actual is not None:
                    prompt += f"，实际成绩: {actual:.1f}"
                prompt += f"，满分{full})"
                while True:
                    score_str = input(prompt + ": ").strip()
                    if score_str == "":
                        target_scores[sub] = 100.0
                        break
                    try:
                        score = float(score_str)
                        if score <= 0:
                            print("目标分必须大于0")
                            continue
                        target_scores[sub] = score
                        break
                    except (TypeError, ValueError):
                        print("输入无效，请输入数字")
        else:
            target_scores = saved_target_scores
    else:
        target_scores = {}
        for sub in score_subjects:
            full = get_full_mark(sub)
            actual = avg_scores.get(sub, None)
            prompt = f"  {sub} 的目标分 (直接回车默认100"
            if actual is not None:
                prompt += f"，实际成绩: {actual:.1f}"
            prompt += f"，满分{full})"
            while True:
                score_str = input(prompt + ": ").strip()
                if score_str == "":
                    target_scores[sub] = 100.0
                    break
                try:
                    score = float(score_str)
                    if score <= 0:
                        print("目标分必须大于0")
                        continue
                    target_scores[sub] = score
                    break
                except (TypeError, ValueError):
                    print("输入无效，请输入数字")

    # ---------- 3. 作业记录 ----------
    print("\n--- 第三步：输入作业记录（用于计算学科重要度） ---")
    job_choice = 'y'
    file_path = None
    if saved_weights and score_choice != 'n' and 'target_choice' in locals() and target_choice != 'n':
        job_choice = input("是否重新输入作业记录？(y/n，直接回车跳过): ").strip().lower()
        if job_choice != 'y':
            if saved_subject_info:
                subject_data = pd.DataFrame.from_dict(saved_subject_info, orient='index')
                subject_data.index.name = '学科'
                subject_data.rename(columns={'量化的重要度': '量化的重要度', '基础紧急度': '基础紧急度'}, inplace=True)
                subject_data.index = subject_data.index.str.strip()
                for sub in score_subjects:
                    sub_clean = sub.strip()
                    if sub_clean not in subject_data.index:
                        subject_data.loc[sub_clean, '量化的重要度'] = 5.0
                        subject_data.loc[sub_clean, '基础紧急度'] = max(1.0, min(10.0, target_scores.get(sub_clean, 100.0) / 100.0))
                        subject_data.loc[sub_clean, '紧急度'] = subject_data.loc[sub_clean, '基础紧急度']
                print("\n已恢复上次保存的学科重要度和基础紧急度：")
                print(subject_data[['量化的重要度', '基础紧急度']])
                time_weights = pd.Series([1 / 3, 1 / 3, 1 / 3], index=['平均耗时', '平均错题率', '平均难度'])
            else:
                print("未找到上次学科数据，请重新输入。")
                job_choice = 'y'

    if job_choice == 'y':
        file_path = get_file_path("请拖拽或输入Excel文件路径（作业记录）:", allow_skip=False)
        if not file_path:
            print("未提供作业记录，无法继续。")
            return
        try:
            df_raw = read_dataframe(file_path)
            print(f"读取成功，共{df_raw.shape[0]}条记录")
        except Exception as e:
            print(f"读取失败: {e}")
            return
        df_raw['学科'] = df_raw['作业名称'].apply(extract_subject)
        df_raw['难度数值'] = df_raw['难度'].apply(map_difficulty)
        df_raw['质量'] = pd.to_numeric(df_raw['质量（错/全）'], errors='coerce')
        df_raw['时间'] = pd.to_numeric(df_raw['时间/min'], errors='coerce')
        df_clean = df_raw.dropna(subset=['时间', '质量', '难度数值'])
        if df_clean.empty:
            print("有效数据不足，无法计算")
            return
        subject_stats = df_clean.groupby('学科').agg({
            '时间': 'mean',
            '质量': 'mean',
            '难度数值': 'mean'
        }).reset_index()
        subject_stats.columns = ['学科', '平均耗时(min)', '平均错题率', '平均难度']
        print("\n各学科统计数据：")
        print(subject_stats)
        if subject_stats.shape[0] < 3:
            print("学科数量不足，使用等权重")
            time_weights = pd.Series([1 / 3, 1 / 3, 1 / 3], index=['平均耗时', '平均错题率', '平均难度'])
        else:
            try:
                time_weights = entropy_weight(subject_stats[['平均耗时(min)', '平均错题率', '平均难度']],
                                              positive_cols=['平均耗时'], method='range')
                print("\n时间预计模型因子权重：")
                print(time_weights)
            except Exception as e:
                print(f"熵权计算失败: {e}，使用等权重")
                time_weights = pd.Series([1 / 3, 1 / 3, 1 / 3], index=['平均耗时', '平均错题率', '平均难度'])
        weights_results = {"时间预计模型": time_weights}
        print("\n⚠️ 任务排序模型缺少期望分数与实际分数差等数据，已跳过")
        w_t = time_weights.get('平均耗时', 0)
        w_e = time_weights.get('平均错题率', 0)
        w_d = time_weights.get('平均难度', 0)
        # 向量化计算综合得分，避免逐行 Series 操作
        subject_stats['综合得分'] = (
            subject_stats['平均耗时(min)'].to_numpy() * w_t +
            subject_stats['平均错题率'].to_numpy() * w_e +
            subject_stats['平均难度'].to_numpy() * w_d
        )
        col_max = subject_stats['综合得分'].max()
        col_min = subject_stats['综合得分'].min()
        if col_max > col_min:
            subject_stats['量化的重要度'] = 1 + 9 * (subject_stats['综合得分'] - col_min) / (col_max - col_min)
        else:
            subject_stats['量化的重要度'] = 5.0
        subject_data = subject_stats.set_index('学科')[['量化的重要度']].copy()
        subject_data.index = subject_data.index.str.strip()
        for sub in score_subjects:
            sub_clean = sub.strip()
            if sub_clean not in subject_data.index:
                subject_data.loc[sub_clean, '量化的重要度'] = 5.0

    # 紧急度调整（向量化）
    target_series = pd.Series(target_scores)
    subject_data['基础紧急度'] = subject_data.index.map(
        lambda sub: max(1.0, min(10.0, target_scores.get(sub, 100.0) / 100.0))
    )
    if avg_scores:
        base_arr = subject_data['基础紧急度'].to_numpy()
        sub_names = subject_data.index.to_numpy()
        actual_arr = np.array([avg_scores.get(s, None) for s in sub_names])
        target_arr = np.array([target_scores.get(s, 100.0) for s in sub_names])
        # 有实际成绩的学科：factor = 1 + min(2, gap/100)
        has_actual = ~pd.isna(actual_arr)
        gap = np.maximum(0.0, target_arr - np.where(has_actual, actual_arr, 0.0))
        factor = np.where(has_actual, 1 + np.minimum(2.0, gap / 100.0), 1.0)
        new_urg = np.clip(base_arr * factor, 1.0, 10.0)
        subject_data['紧急度'] = new_urg
        print("\n各学科基础紧急度和调整后紧急度：")
        print(subject_data[['量化的重要度', '基础紧急度', '紧急度']])
    else:
        subject_data['紧急度'] = subject_data['基础紧急度']
        print("\n各学科基础紧急度（无成绩调整）：")
        print(subject_data[['量化的重要度', '基础紧急度']])

    # ---------- 4. 时间段 ----------
    print("\n--- 第四步：输入可用时间段 ---")
    if last_time_range:
        print(f"上次使用的时间段: {last_time_range}")
        time_choice = input("直接回车使用上次时间段，输入 n 重新输入: ").strip().lower()
        if time_choice == 'n':
            time_str = input("请输入可用时间段(支持中英文分号，如 07:00-07:20;12:00-12:20): ").strip()
            while True:
                try:
                    total_min = parse_time_range_to_minutes(time_str)
                    last_time_range = time_str
                    break
                except Exception as e:
                    print(f"错误: {e}")
                    time_str = input("请重新输入: ").strip()
        else:
            total_min = parse_time_range_to_minutes(last_time_range)
    else:
        time_str = input("请输入可用时间段(支持中英文分号，如 07:00-07:20;12:00-12:20): ").strip()
        while True:
            try:
                total_min = parse_time_range_to_minutes(time_str)
                last_time_range = time_str
                break
            except Exception as e:
                print(f"错误: {e}")
                time_str = input("请重新输入: ").strip()

    # ---------- 5. 任务管理 ----------
    print("\n--- 第五步：自主任务管理 ---")
    subject_imp_dict = subject_data['量化的重要度'].to_dict()
    final_tasks, _ = manage_custom_tasks(subject_imp_dict, total_min, subject_data, saved_tasks if saved_weights else None, last_time_range)

    result = allocate_time(total_min, subject_data, final_tasks)
    if result.empty:
        print("无任务可分配，退出。")
        return

    result_ordered = reorder_by_subject(result)

    if 'time_weights' not in locals():
        time_weights = pd.Series([1 / 3, 1 / 3, 1 / 3], index=['平均耗时', '平均错题率', '平均难度'])
    weights_results = {"时间预计模型": time_weights}

    if file_path:
        output_path = os.path.splitext(file_path)[0] + "_计划结果.xlsx"
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df_raw.to_excel(writer, sheet_name='原始数据', index=False)
                subject_stats.to_excel(writer, sheet_name='学科统计', index=False)
                export_result = result_ordered.copy()
                if '层级' in export_result.columns:
                    export_result = export_result.drop(columns=['层级'])
                export_result.to_excel(writer, sheet_name='时间分配建议', index=False)
                pd.DataFrame({'模型': list(weights_results.keys()),
                              '权重详情': [str(w.to_dict()) for w in weights_results.values()]}).to_excel(writer, sheet_name='模型权重', index=False)
            print(f"\n✅ 详细结果已保存至: {output_path}")
        except Exception as e:
            print(f"保存详细结果失败: {e}")

    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    result_folder = os.path.join(desktop, "结果")
    os.makedirs(result_folder, exist_ok=True)
    result_file = os.path.join(result_folder, "运行结果.xlsx")
    export_desktop = result_ordered.copy()
    if '层级' in export_desktop.columns:
        export_desktop = export_desktop.drop(columns=['层级'])
    export_desktop.to_excel(result_file, index=False)
    print(f"✅ 时间分配结果已保存至：{result_file}")

    subject_stats_for_excel = subject_stats if 'subject_stats' in locals() else None
    save_data_excel(subject_stats_for_excel, result, weights_results, full_data=True)

    save_history(result, total_min, last_time_range, target_scores)
    save_data(weights_results, final_tasks, last_time_range, target_scores, subject_data)

if __name__ == '__main__':
    main()
