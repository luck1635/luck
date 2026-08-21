#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学习规划助手（熵权法）
完整版 v5.0 - 支持中英文分号/冒号分隔时间段，兼容中文输入
"""

import numpy as np
import pandas as pd
import os
import json
import re

# -------------------------- 熵权法核心 --------------------------
def entropy_weight(df, positive_cols=None, method='range'):
    data = df.copy().astype(float)
    n_samples = data.shape[0]
    data_std = pd.DataFrame(index=data.index, columns=data.columns)
    for col in data.columns:
        col_data = data[col]
        if col_data.isnull().any():
            raise ValueError(f"列 '{col}' 存在缺失值")
        is_pos = (positive_cols is not None and col in positive_cols)
        if method == 'range':
            min_v, max_v = col_data.min(), col_data.max()
            if max_v - min_v == 0:
                data_std[col] = 0.0
                continue
            if is_pos:
                data_std[col] = (col_data - min_v) / (max_v - min_v)
            else:
                data_std[col] = (max_v - col_data) / (max_v - min_v)
        elif method == 'zscore':
            mean_v, std_v = col_data.mean(), col_data.std()
            if std_v == 0:
                data_std[col] = 0.0
                continue
            if is_pos:
                data_std[col] = (col_data - mean_v) / std_v
            else:
                data_std[col] = -(col_data - mean_v) / std_v
    if method == 'zscore':
        for col in data_std.columns:
            min_val = data_std[col].min()
            if min_val < 0:
                data_std[col] = data_std[col] - min_val + 1e-6
    data_std = data_std + 1e-8
    p_matrix = data_std.div(data_std.sum(axis=0), axis=1)
    k = 1.0 / np.log(n_samples)
    entropy = -k * (p_matrix * np.log(p_matrix)).sum(axis=0)
    diff = 1 - entropy
    weights = diff / diff.sum()
    return weights

# -------------------------- 任务管理模块（实时预览）--------------------------
def show_allocation_preview(total_minutes, subject_data, custom_tasks, auto_weights):
    df_sub = pd.DataFrame({
        '重要度': subject_data['量化的重要度'],
        '紧急度': subject_data['紧急度']
    }, index=subject_data.index)
    if custom_tasks:
        custom_df = pd.DataFrame(custom_tasks).set_index('name')
        custom_df.rename(columns={'importance':'重要度', 'urgency':'紧急度'}, inplace=True)
        custom_df = custom_df[['重要度','紧急度']]
    else:
        custom_df = pd.DataFrame(columns=['重要度','紧急度'])
    all_tasks = pd.concat([df_sub, custom_df], axis=0)
    if all_tasks.empty:
        print("无任务，无法分配")
        return auto_weights
    if auto_weights is not None and len(auto_weights)==2:
        w = auto_weights
    else:
        try:
            w = entropy_weight(all_tasks[['重要度','紧急度']], positive_cols=['重要度','紧急度'], method='range')
        except:
            w = pd.Series([0.5,0.5], index=['重要度','紧急度'])
    scores = all_tasks['重要度'] * w['重要度'] + all_tasks['紧急度'] * w['紧急度']
    total_score = scores.sum()
    if total_score == 0:
        allocated = pd.Series(total_minutes / len(scores), index=scores.index)
    else:
        allocated = (scores / total_score) * total_minutes
    print("\n===== 当前时间分配预览 =====")
    preview = pd.DataFrame({
        '任务': all_tasks.index,
        '重要度': all_tasks['重要度'],
        '紧急度': all_tasks['紧急度'],
        '综合得分': scores,
        '建议时间(分钟)': allocated
    })
    print(preview.to_string(index=False))
    return w

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
    auto_weights = show_allocation_preview(total_minutes, subject_data, tasks, None)
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
            auto_weights = show_allocation_preview(total_minutes, subject_data, tasks, auto_weights)
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
                auto_weights = show_allocation_preview(total_minutes, subject_data, tasks, auto_weights)
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
                    auto_weights = show_allocation_preview(total_minutes, subject_data, tasks, auto_weights)
                else:
                    print("无效")
            except ValueError:
                print("输入无效")
        elif choice == '4':
            break
        else:
            print("无效选项")
    return tasks, auto_weights

# -------------------------- 解析时间段（兼容中英文符号）-------------------------
def parse_time_range_to_minutes(user_input):
    """
    解析用户输入的时间段字符串，支持中英文分号（；;）分割多个时段，
    每个时段支持中英文冒号（：:）和多种连接符（-~空格）。
    返回总分钟数（int）。
    """
    # 用中英文分号分割
    segments = re.split(r'[;；]', user_input)
    total_minutes = 0
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        # 将中文冒号替换为英文冒号
        seg = seg.replace('：', ':')
        # 匹配时间段，支持 - ~ 和空格
        pattern = r'(\d{1,2}):(\d{2})\s*[-~]\s*(\d{1,2}):(\d{2})|(\d{1,2}):(\d{2})\s+(\d{1,2}):(\d{2})'
        m = re.match(pattern, seg)
        if not m:
            raise ValueError(f"时间段 '{seg}' 格式错误，请使用如 '08:00-12:00' 或 '08:00 12:00'")
        if m.group(1):  # 包含 - 或 ~ 的格式
            h1, m1, h2, m2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        else:  # 空格分隔的格式
            h1, m1, h2, m2 = int(m.group(5)), int(m.group(6)), int(m.group(7)), int(m.group(8))
        start = h1*60 + m1
        end = h2*60 + m2
        if end <= start:   # 跨夜处理
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

# -------------------------- 保存与加载（含 target_scores）-------------------------
SAVE_FILE = "study_planner_data.json"

def save_data(weights, tasks, last_time_range="", target_scores=None):
    if target_scores is None:
        target_scores = {}
    data = {
        'weights': {k: v.to_dict() for k, v in weights.items() if v is not None and not v.empty},
        'custom_tasks': tasks,
        'last_time_range': last_time_range,
        'target_scores': target_scores
    }
    with open(SAVE_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_data():
    if not os.path.exists(SAVE_FILE):
        return None, None, "", {}
    with open(SAVE_FILE, 'r', encoding='utf-8') as f:
        d = json.load(f)
    weights = {}
    for k, vdict in d.get('weights', {}).items():
        weights[k] = pd.Series(vdict)
    tasks = d.get('custom_tasks', [])
    last_time_range = d.get('last_time_range', "")
    target_scores = d.get('target_scores', {})
    return weights, tasks, last_time_range, target_scores

def save_data_excel(subject_stats, allocation_result, weights_dict, full_data=True):
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    file_path = os.path.join(desktop, "数据.xlsx")
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        allocation_result.to_excel(writer, sheet_name='时间分配建议', index=False)
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

# -------------------------- 主程序 --------------------------
def main():
    print("="*60)
    print("学习规划助手（熵权法）")
    print("时间预计模型 → 任务排序模型(跳过) → 自主安排模型")
    print("="*60)

    saved_weights, saved_tasks, last_time_range, saved_target_scores = load_data()
    if saved_weights:
        print("\n发现上次保存的权重和任务列表。")
        if last_time_range:
            print(f"上次使用的时间段: {last_time_range}")
        choice = input("直接回车(或1)使用上次数据；输入新表格路径则重新计算: ").strip()
        if choice == '' or choice == '1':
            print("使用上次数据。")
            if last_time_range:
                use_last = input(f"使用上次的时间段 {last_time_range}？(直接回车使用，输入 n 重新输入): ").strip().lower()
                if use_last == 'n':
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
            dummy_subject_data = pd.DataFrame({'量化的重要度': [], '紧急度': []}, index=[])
            dummy_imp_dict = {}
            final_tasks, auto_weights = manage_custom_tasks(dummy_imp_dict, total_min, dummy_subject_data, saved_tasks, last_time_range)
            if final_tasks:
                custom_df = pd.DataFrame(final_tasks).set_index('name')
                custom_df.rename(columns={'importance':'重要度','urgency':'紧急度'}, inplace=True)
                custom_df = custom_df[['重要度','紧急度']]
            else:
                custom_df = pd.DataFrame(columns=['重要度','紧急度'])
            all_tasks_final = custom_df
            if not all_tasks_final.empty:
                scores = all_tasks_final['重要度']*auto_weights['重要度'] + all_tasks_final['紧急度']*auto_weights['紧急度']
                if scores.sum()==0:
                    allocated = pd.Series(total_min/len(scores), index=scores.index)
                else:
                    allocated = (scores/scores.sum())*total_min
                result = pd.DataFrame({
                    '任务名称': all_tasks_final.index,
                    '重要度': all_tasks_final['重要度'],
                    '紧急度': all_tasks_final['紧急度'],
                    '综合得分': scores,
                    '建议时间(分钟)': allocated
                })
            else:
                result = pd.DataFrame()
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            desktop_file = os.path.join(desktop, "学习规划结果.xlsx")
            result.to_excel(desktop_file, index=False)
            print(f"\n✅ 时间分配结果已保存到桌面：{desktop_file}")
            save_data_excel(None, result, saved_weights, full_data=False)
            save_data(saved_weights, final_tasks, last_time_range, saved_target_scores)
            return
        else:
            file_path = clean_path(choice)
            if not os.path.exists(file_path):
                print("文件不存在")
                return
    else:
        file_path = get_file_path("请拖拽或输入Excel文件路径:", allow_skip=False)
        if not file_path:
            return

    # 读取原始数据
    try:
        df_raw = read_dataframe(file_path)
        print(f"读取成功，共{df_raw.shape[0]}条记录")
    except Exception as e:
        print(f"读取失败: {e}")
        return

    # 预处理
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

    # 时间预计模型
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

    # 计算学科综合得分和量化重要度
    subject_stats['综合得分'] = (subject_stats['平均耗时(min)'] * time_weights.get('平均耗时',0) +
                                 subject_stats['平均错题率'] * time_weights.get('平均错题率',0) +
                                 subject_stats['平均难度'] * time_weights.get('平均难度',0))
    if subject_stats['综合得分'].max() > subject_stats['综合得分'].min():
        subject_stats['量化的重要度'] = 1 + 9 * (subject_stats['综合得分'] - subject_stats['综合得分'].min()) / (subject_stats['综合得分'].max() - subject_stats['综合得分'].min())
    else:
        subject_stats['量化的重要度'] = 5.0
    subject_data = subject_stats.set_index('学科')[['量化的重要度']].copy()

    # ------------------- 固定顺序询问目标分 -------------------
    subject_list = list(subject_data.index)
    priority_order = ['语文', '数学', '英语', '物理', '化学', '生物']
    def sort_key(sub):
        if sub in priority_order:
            return (priority_order.index(sub), sub)
        else:
            return (len(priority_order), sub)
    subject_list_sorted = sorted(subject_list, key=sort_key)

    if saved_target_scores:
        print("\n发现已保存的各学科目标分：")
        for sub in subject_list_sorted:
            if sub in saved_target_scores:
                print(f"  {sub}: {saved_target_scores[sub]:.1f}")
        use_last = input("是否沿用以上目标分？(直接回车沿用，输入 n 重新输入): ").strip().lower()
        if use_last != 'n':
            target_scores = {sub: saved_target_scores.get(sub, 100.0) for sub in subject_list_sorted}
        else:
            target_scores = {}
            for sub in subject_list_sorted:
                while True:
                    score_str = input(f"  {sub} 的目标分 (直接回车默认100): ").strip()
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
        print("\n请按顺序输入各学科的目标分数（用于计算学科紧急度 = 目标分/100，范围1~10）：")
        target_scores = {}
        for sub in subject_list_sorted:
            while True:
                score_str = input(f"  {sub} 的目标分 (直接回车默认100): ").strip()
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

    # 计算紧急度
    subject_data['紧急度'] = subject_data.index.map(lambda sub: max(1.0, min(10.0, target_scores.get(sub, 100.0) / 100.0)))
    print("\n各学科紧急度（基于目标分）：")
    print(subject_data[['量化的重要度', '紧急度']])

    # ------------------- 时间段输入（兼容中英文符号） -------------------
    if last_time_range:
        print(f"上一次使用的时间段: {last_time_range}")
        use_last = input("是否使用上一次的时间段？(直接回车使用，输入 n 重新输入): ").strip().lower()
        if use_last != 'n':
            time_str = last_time_range
            total_min = parse_time_range_to_minutes(time_str)
        else:
            time_str = input("请输入可用时间段(支持中英文分号，如 07:00-07:20;12:00-12:20): ").strip()
            while True:
                try:
                    total_min = parse_time_range_to_minutes(time_str)
                    break
                except Exception as e:
                    print(f"错误: {e}")
                    time_str = input("请重新输入: ").strip()
    else:
        time_str = input("请输入可用时间段(支持中英文分号，如 07:00-07:20;12:00-12:20): ").strip()
        while True:
            try:
                total_min = parse_time_range_to_minutes(time_str)
                break
            except Exception as e:
                print(f"错误: {e}")
                time_str = input("请重新输入: ").strip()
    last_time_range = time_str

    # 进入任务管理
    subject_imp_dict = subject_data['量化的重要度'].to_dict()
    final_tasks, auto_weights = manage_custom_tasks(subject_imp_dict, total_min, subject_data, None, last_time_range)

    # 最终结果
    df_sub_final = pd.DataFrame({
        '重要度': subject_data['量化的重要度'],
        '紧急度': subject_data['紧急度']
    }, index=subject_data.index)
    if final_tasks:
        custom_df_final = pd.DataFrame(final_tasks).set_index('name')
        custom_df_final.rename(columns={'importance':'重要度','urgency':'紧急度'}, inplace=True)
        custom_df_final = custom_df_final[['重要度','紧急度']]
    else:
        custom_df_final = pd.DataFrame(columns=['重要度','紧急度'])
    all_tasks_final = pd.concat([df_sub_final, custom_df_final], axis=0)
    scores_final = all_tasks_final['重要度']*auto_weights['重要度'] + all_tasks_final['紧急度']*auto_weights['紧急度']
    if scores_final.sum() == 0:
        allocated_final = pd.Series(total_min / len(scores_final), index=scores_final.index)
    else:
        allocated_final = (scores_final / scores_final.sum()) * total_min
    result = pd.DataFrame({
        '任务名称': all_tasks_final.index,
        '重要度': all_tasks_final['重要度'],
        '紧急度': all_tasks_final['紧急度'],
        '综合得分': scores_final,
        '建议时间(分钟)': allocated_final
    })
    weights_results["自主安排模型"] = auto_weights

    # 保存详细结果
    base_name = os.path.splitext(file_path)[0]
    output_path = base_name + "_计划结果.xlsx"
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_raw.to_excel(writer, sheet_name='原始数据', index=False)
            subject_stats.to_excel(writer, sheet_name='学科统计', index=False)
            result.to_excel(writer, sheet_name='时间分配建议', index=False)
            pd.DataFrame({'模型': list(weights_results.keys()),
                          '权重详情': [str(w.to_dict()) for w in weights_results.values()]}).to_excel(writer, sheet_name='模型权重', index=False)
        print(f"\n✅ 详细结果已保存至: {output_path}")
    except Exception as e:
        print(f"保存详细结果失败: {e}")

    # 保存到桌面
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    desktop_file = os.path.join(desktop, "学习规划结果.xlsx")
    result.to_excel(desktop_file, index=False)
    print(f"✅ 时间分配结果已保存到桌面：{desktop_file}")

    save_data_excel(subject_stats, result, weights_results, full_data=True)
    save_data(weights_results, final_tasks, last_time_range, target_scores)

if __name__ == '__main__':
    main()