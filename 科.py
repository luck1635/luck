#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学习规划助手（熵权法）
v6.13 - 优化分级显示（母任务▼/子任务└─），修复孤立子任务丢失
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
    data = df.astype(float)
    n_samples, n_cols = data.shape
    if n_samples <= 1:
        return pd.Series([1.0 / n_cols] * n_cols, index=data.columns)
    arr = data.values
    null_mask = np.any(pd.isnull(arr), axis=0)
    if np.any(null_mask):
        first_null = data.columns[null_mask][0]
        raise ValueError(f"列 '{first_null}' 存在缺失值")
    pos_set = positive_cols if positive_cols is not None else []
    is_pos = np.array([c in pos_set for c in data.columns])
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
        std_arr = std_arr + shift.reshape(1, -1)
    else:
        std_arr = arr.copy()
    std_arr = std_arr + 1e-8
    col_sums = std_arr.sum(axis=0)
    p_matrix = std_arr / col_sums
    k = 1.0 / np.log(n_samples)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = np.log(p_matrix)
    log_p[~np.isfinite(log_p)] = 0.0
    entropy = -k * (p_matrix * log_p).sum(axis=0)
    diff = 1 - entropy
    diff_sum = diff.sum()
    if diff_sum == 0:
        weights = pd.Series(1.0 / n_cols, index=data.columns)
    else:
        weights = pd.Series(diff / diff_sum, index=data.columns)
    return weights

# -------------------------- 任务分配核心（两步分配）-------------------------
def allocate_time(total_minutes, subject_data, custom_tasks_list):
    """
    两步分配：
    1. 学科任务先分配总时间（仅学科）
    2. 每个学科内部，母任务与子任务再分配该学科总时间
    返回分配结果的DataFrame（包含所有任务）
    """
    # 清理 subject_data 索引
    subject_data_clean = subject_data.copy()
    subject_data_clean.index = subject_data_clean.index.str.strip()
    df_sub = pd.DataFrame({
        '重要度': subject_data_clean['量化的重要度'],
        '紧急度': subject_data_clean['紧急度']
    }, index=subject_data_clean.index)
    df_sub['type'] = 'subject'

    if custom_tasks_list:
        custom_df = pd.DataFrame(custom_tasks_list).set_index('name')
        custom_df.rename(columns={'importance':'重要度', 'urgency':'紧急度'}, inplace=True)
        custom_df = custom_df[['重要度','紧急度']]
        custom_df['type'] = 'custom'
        # 清理学科名称前后的空格
        custom_df['subject'] = [str(t['subject']).strip() for t in custom_tasks_list]
    else:
        custom_df = pd.DataFrame(columns=['重要度','紧急度','type','subject'])

    if df_sub.empty:
        print("无学科任务，无法分配")
        return pd.DataFrame()

    # ---- 修复：处理学科不在学科列表中的孤立子任务 ----
    # 将这些子任务统一归入"其他"学科，并创建"其他"母任务
    if not custom_df.empty:
        known_subjects = set(df_sub.index)
        orphan_mask = ~custom_df['subject'].isin(known_subjects)
        if orphan_mask.any():
            custom_df.loc[orphan_mask, 'subject'] = '其他'
            if '其他' not in df_sub.index:
                df_sub.loc['其他'] = {'重要度': 5.0, '紧急度': 5.0, 'type': 'subject'}

    # ---- 第一步：学科间分配 ----
    try:
        sub_weights = entropy_weight(df_sub[['重要度','紧急度']], positive_cols=['重要度','紧急度'], method='range')
    except:
        sub_weights = pd.Series([0.5,0.5], index=['重要度','紧急度'])
    sub_scores = df_sub['重要度'] * sub_weights['重要度'] + df_sub['紧急度'] * sub_weights['紧急度']
    if sub_scores.sum() == 0:
        sub_allocated = pd.Series(total_minutes / len(sub_scores), index=sub_scores.index)
    else:
        sub_allocated = (sub_scores / sub_scores.sum()) * total_minutes

    # ---- 第二步：每个学科内部再分配 ----
    all_results = []
    for sub in df_sub.index:
        parent_time = sub_allocated[sub]
        children = custom_df[custom_df['subject'] == sub] if not custom_df.empty else pd.DataFrame()
        inner_tasks = []
        inner_tasks.append({
            'name': sub,
            '重要度': df_sub.loc[sub, '重要度'],
            '紧急度': df_sub.loc[sub, '紧急度'],
            'type': 'subject'
        })
        if not children.empty:
            for idx, row in children.iterrows():
                inner_tasks.append({
                    'name': idx,
                    '重要度': row['重要度'],
                    '紧急度': row['紧急度'],
                    'type': 'custom'
                })
        inner_df = pd.DataFrame(inner_tasks).set_index('name')
        if len(inner_df) == 1:
            # 只有母任务，综合得分使用学科间得分
            parent_score = sub_scores[sub] if sub in sub_scores else 0
            all_results.append({
                '任务名称': sub,
                '重要度': inner_df.loc[sub, '重要度'],
                '紧急度': inner_df.loc[sub, '紧急度'],
                '综合得分': parent_score,
                '建议时间(分钟)': parent_time,
                '所属学科': None
            })
        else:
            # 有子任务，内部重新分配时间并计算得分
            try:
                inner_weights = entropy_weight(inner_df[['重要度','紧急度']], positive_cols=['重要度','紧急度'], method='range')
            except:
                inner_weights = pd.Series([0.5,0.5], index=['重要度','紧急度'])
            inner_scores = inner_df['重要度'] * inner_weights['重要度'] + inner_df['紧急度'] * inner_weights['紧急度']
            if inner_scores.sum() == 0:
                inner_alloc = pd.Series(parent_time / len(inner_scores), index=inner_scores.index)
            else:
                inner_alloc = (inner_scores / inner_scores.sum()) * parent_time
            for task_name, time_val in inner_alloc.items():
                all_results.append({
                    '任务名称': task_name,
                    '重要度': inner_df.loc[task_name, '重要度'],
                    '紧急度': inner_df.loc[task_name, '紧急度'],
                    '综合得分': inner_scores[task_name] if task_name in inner_scores else 0,
                    '建议时间(分钟)': time_val,
                    '所属学科': sub if task_name != sub else None
                })

    result_df = pd.DataFrame(all_results)
    return result_df

# -------------------------- 分级重排函数（清理空格）-------------------------
def reorder_by_subject(df):
    """
    将分配结果按学科分组，母任务在前，子任务缩进（子任务按综合得分降序）
    返回重排后的 DataFrame
    """
    if df.empty:
        return df
    df = df.copy()
    # 清理可能的空格
    df['任务名称_clean'] = df['任务名称'].astype(str).str.strip()
    if '所属学科' in df.columns:
        df['所属学科_clean'] = df['所属学科'].astype(str).str.strip()
    else:
        df['所属学科_clean'] = None
    # 分离母任务（所属学科为 None 或 'None' 或空）和子任务
    parents = df[df['所属学科_clean'].isna() | (df['所属学科_clean'] == 'None') | (df['所属学科_clean'] == '')].copy()
    children = df[df['所属学科_clean'].notna() & (df['所属学科_clean'] != 'None') & (df['所属学科_clean'] != '')].copy()
    ordered_rows = []
    # 母任务按综合得分降序排列
    parents_sorted = parents.sort_values('综合得分', ascending=False)
    for _, parent in parents_sorted.iterrows():
        ordered_rows.append(parent)
        # 获取该母任务的所有子任务
        child_rows = children[children['所属学科_clean'] == parent['任务名称_clean']]
        child_rows = child_rows.sort_values('综合得分', ascending=False)
        for _, child in child_rows.iterrows():
            ordered_rows.append(child)
    # 如果还有子任务找不到母任务（理论上不会），放在最后
    remaining_children = children[~children['所属学科_clean'].isin(parents['任务名称_clean'].tolist())]
    if not remaining_children.empty:
        for _, child in remaining_children.iterrows():
            ordered_rows.append(child)
    result = pd.DataFrame(ordered_rows)
    # 删除辅助列
    result = result.drop(columns=['任务名称_clean', '所属学科_clean'], errors='ignore')
    return result

# -------------------------- 任务管理模块（实时预览）-------------------------
def show_allocation_preview(total_minutes, subject_data, custom_tasks_list, auto_weights=None):
    result = allocate_time(total_minutes, subject_data, custom_tasks_list)
    if result.empty:
        print("无任务，无法分配")
        return None
    # 分级重排
    result_ordered = reorder_by_subject(result)
    print("\n===== 当前时间分配预览 =====")
    preview = result_ordered.copy()
    # 用 所属学科 字段判断是否子任务，确保分级标记准确
    def format_name(row):
        subject = str(row.get('所属学科', '')).strip()
        if subject and subject != 'None':
            # 子任务：缩进 + 连接符
            return '  └─ ' + str(row['任务名称'])
        else:
            # 母任务：展开标记
            return '▼ ' + str(row['任务名称'])
    preview['任务名称'] = preview.apply(format_name, axis=1)
    display_cols = ['任务名称', '重要度', '紧急度', '综合得分', '建议时间(分钟)']
    print(preview[display_cols].to_string(index=False))
    return None

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
            # 清理学科名称前后的空格
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
            except:
                print("紧急度输入无效，设为5.0")
                urg = 5.0
            tasks.append({'name': name, 'subject': subject, 'importance': imp, 'urgency': urg})
            print(f"任务 '{name}' ({subject}) 已添加")
            show_allocation_preview(total_minutes, subject_data, tasks, None)
        elif choice == '2':
            if not tasks:
                print("暂无任务")
                continue
            try:
                idx = int(input("编号: ")) - 1
                if idx < 0 or idx >= len(tasks):
                    print("无效")
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
                print("暂无任务")
                continue
            try:
                idx = int(input("编号: ")) - 1
                if 0 <= idx < len(tasks):
                    removed = tasks.pop(idx)
                    print(f"已删除 '{removed['name']}'")
                    show_allocation_preview(total_minutes, subject_data, tasks, None)
                else:
                    print("无效")
            except ValueError:
                print("输入无效")
        elif choice == '4':
            break
        else:
            print("无效选项")
    return tasks, None

# -------------------------- 解析时间段（兼容中英文符号）-------------------------
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
        start = h1*60 + m1
        end = h2*60 + m2
        if end <= start:
            end += 24*60
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
        except:
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
    header_row = df.iloc[0].copy()
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
        if real_score_row_index is not None:
            score_row = df.iloc[real_score_row_index]
            scores = {}
            for col, sub in zip(subject_columns, subject_names):
                val = score_row[col]
                if pd.notna(val):
                    try:
                        scores[sub] = float(val)
                    except:
                        pass
            return scores, subject_names
        else:
            return {}, subject_names
    df_header = pd.read_excel(file_path, header=0, engine='openpyxl')
    subject_col = None
    score_col = None
    for col in df_header.columns:
        col_lower = str(col).lower()
        if '学科' in col_lower or '科目' in col_lower or '课程' in col_lower or '名称' in col_lower or 'subject' in col_lower:
            subject_col = col
        if '成绩' in col_lower or '分数' in col_lower or '得分' in col_lower or 'score' in col_lower:
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

# -------------------------- 历史记录 --------------------------
def save_history(result_df, total_minutes, time_range, target_scores):
    if result_df.empty:
        return
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    history_file = os.path.join(desktop, "历史记录.xlsx")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    target_str = '; '.join([f"{k}:{v:.1f}" for k, v in target_scores.items()])
    meta_rows = []
    for idx, row in result_df.iterrows():
        meta_rows.append({
            '时间戳': timestamp,
            '时间段': time_range,
            '总时间(分钟)': total_minutes,
            '任务名称': row['任务名称'],
            '重要度': row['重要度'],
            '紧急度': row['紧急度'],
            '综合得分': row['综合得分'],
            '建议时间(分钟)': row['建议时间(分钟)'],
            '目标分': target_str
        })
    new_history = pd.DataFrame(meta_rows)
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
    print("="*60)
    print("学习规划助手（熵权法）")
    print("顺序：成绩表格 → 目标分 → 作业记录 → 时间段 → 任务管理")
    print("="*60)

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
                    except:
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
                except:
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
                # 清理索引
                subject_data.index = subject_data.index.str.strip()
                for sub in score_subjects:
                    sub_clean = sub.strip()
                    if sub_clean not in subject_data.index:
                        subject_data.loc[sub_clean, '量化的重要度'] = 5.0
                        subject_data.loc[sub_clean, '基础紧急度'] = max(1.0, min(10.0, target_scores.get(sub_clean, 100.0) / 100.0))
                        subject_data.loc[sub_clean, '紧急度'] = subject_data.loc[sub_clean, '基础紧急度']
                print("\n已恢复上次保存的学科重要度和基础紧急度：")
                print(subject_data[['量化的重要度', '基础紧急度']])
                time_weights = pd.Series([1/3,1/3,1/3], index=['平均耗时','平均错题率','平均难度'])
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
        df_clean = df_raw.dropna(subset=['时间','质量','难度数值'])
        if df_clean.empty:
            print("有效数据不足，无法计算")
            return
        subject_stats = df_clean.groupby('学科').agg({
            '时间': 'mean',
            '质量': 'mean',
            '难度数值': 'mean'
        }).reset_index()
        subject_stats.columns = ['学科','平均耗时(min)','平均错题率','平均难度']
        print("\n各学科统计数据：")
        print(subject_stats)
        if subject_stats.shape[0] < 3:
            print("学科数量不足，使用等权重")
            time_weights = pd.Series([1/3,1/3,1/3], index=['平均耗时','平均错题率','平均难度'])
        else:
            try:
                time_weights = entropy_weight(subject_stats[['平均耗时(min)','平均错题率','平均难度']],
                                              positive_cols=['平均耗时'], method='range')
                print("\n时间预计模型因子权重：")
                print(time_weights)
            except Exception as e:
                print(f"熵权计算失败: {e}，使用等权重")
                time_weights = pd.Series([1/3,1/3,1/3], index=['平均耗时','平均错题率','平均难度'])
        weights_results = {"时间预计模型": time_weights}
        print("\n⚠️ 任务排序模型缺少期望分数与实际分数差等数据，已跳过")
        subject_stats['综合得分'] = (subject_stats['平均耗时(min)'] * time_weights.get('平均耗时',0) +
                                     subject_stats['平均错题率'] * time_weights.get('平均错题率',0) +
                                     subject_stats['平均难度'] * time_weights.get('平均难度',0))
        if subject_stats['综合得分'].max() > subject_stats['综合得分'].min():
            subject_stats['量化的重要度'] = 1 + 9 * (subject_stats['综合得分'] - subject_stats['综合得分'].min()) / (subject_stats['综合得分'].max() - subject_stats['综合得分'].min())
        else:
            subject_stats['量化的重要度'] = 5.0
        subject_data = subject_stats.set_index('学科')[['量化的重要度']].copy()
        # 清理索引
        subject_data.index = subject_data.index.str.strip()
        for sub in score_subjects:
            sub_clean = sub.strip()
            if sub_clean not in subject_data.index:
                subject_data.loc[sub_clean, '量化的重要度'] = 5.0

    # 紧急度调整
    subject_data['基础紧急度'] = subject_data.index.map(lambda sub: max(1.0, min(10.0, target_scores.get(sub, 100.0) / 100.0)))
    if avg_scores:
        adjusted_urgency = {}
        for sub in subject_data.index:
            base = subject_data.loc[sub, '基础紧急度']
            actual = avg_scores.get(sub, None)
            if actual is not None:
                target = target_scores.get(sub, 100.0)
                gap = max(0.0, target - actual)
                factor = 1 + min(2.0, gap / 100.0)
                new_urg = base * factor
                new_urg = max(1.0, min(10.0, new_urg))
                adjusted_urgency[sub] = new_urg
            else:
                adjusted_urgency[sub] = base
        subject_data['紧急度'] = subject_data.index.map(lambda sub: adjusted_urgency.get(sub, subject_data.loc[sub, '基础紧急度']))
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

    # 对最终结果进行分级重排（用于显示和保存）
    result_ordered = reorder_by_subject(result)

    # 组装权重结果（用于保存）
    if 'time_weights' not in locals():
        time_weights = pd.Series([1/3,1/3,1/3], index=['平均耗时','平均错题率','平均难度'])
    weights_results = {"时间预计模型": time_weights}

    # 保存详细结果（如果有文件）
    if file_path:
        output_path = os.path.splitext(file_path)[0] + "_计划结果.xlsx"
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df_raw.to_excel(writer, sheet_name='原始数据', index=False)
                subject_stats.to_excel(writer, sheet_name='学科统计', index=False)
                result_ordered.to_excel(writer, sheet_name='时间分配建议', index=False)
                pd.DataFrame({'模型': list(weights_results.keys()),
                              '权重详情': [str(w.to_dict()) for w in weights_results.values()]}).to_excel(writer, sheet_name='模型权重', index=False)
            print(f"\n✅ 详细结果已保存至: {output_path}")
        except Exception as e:
            print(f"保存详细结果失败: {e}")

    # 保存到桌面（使用重排后的结果）
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    desktop_file = os.path.join(desktop, "学习规划结果.xlsx")
    result_ordered.to_excel(desktop_file, index=False)
    print(f"✅ 时间分配结果已保存到桌面：{desktop_file}")

    # 保存数据.xlsx（内部会重排）
    subject_stats_for_excel = subject_stats if 'subject_stats' in locals() else None
    save_data_excel(subject_stats_for_excel, result, weights_results, full_data=True)

    # 历史记录保存原始未重排的结果（保留原始排序）
    save_history(result, total_min, last_time_range, target_scores)

    # 保存当前状态
    save_data(weights_results, final_tasks, last_time_range, target_scores, subject_data)

if __name__ == '__main__':
    main()
