import argparse
import csv
import json
import os
import random
import re
from pathlib import Path

import requests

API_URL = os.environ.get("PROMPT_NEXT_API_URL", "https://api2.aigcbest.top/v1/chat/completions")
API_MODEL = os.environ.get("PROMPT_NEXT_API_MODEL", "gemini-2.5-flash-thinking")
API_KEY = os.environ.get("PROMPT_NEXT_API_KEY", "")
ALL_CATEGORIES = ["combat", "martial_arts", "dance", "sport", "gymnastics"]


def call_generation_api(messages, max_tokens):
    if not API_KEY:
        raise EnvironmentError("Missing PROMPT_NEXT_API_KEY")
    payload = json.dumps({
        "model": API_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens
    })
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(API_URL, headers=headers, data=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        return [data['choices'][0]['message']['content'].strip()]
    except requests.exceptions.HTTPError as http_err:
        status_code = response.status_code if 'response' in locals() and response is not None else "unknown"
        error_message = ""
        try:
            error_json = response.json()
            error_info = error_json.get("error", {})
            error_message = f" code={error_info.get('code')} message={error_info.get('message')}"
        except Exception:
            pass
        print(f"API HTTP错误: status={status_code} err={http_err}{error_message}，响应内容: {response.text if 'response' in locals() and response is not None else '无响应体'}")
        return []
    except Exception as e:
        print(f"API请求失败: {str(e)}")
        return []


def normalize_prompt_text(text):
    normalized = text.lower().replace("_", " ").replace("→", " ")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def load_prompt_manifest(manifest_path):
    if not manifest_path or not os.path.exists(manifest_path):
        return []

    records = []
    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt_name = (row.get("prompt_name") or "").strip()
            category = (row.get("category") or "").strip()
            if not prompt_name or not category:
                continue
            records.append({
                "prompt_name": prompt_name,
                "category": category,
                "normalized_prompt": normalize_prompt_text(
                    row.get("normalized_prompt") or prompt_name
                ),
            })
    return records


def infer_category_from_manifest(text, manifest_records):
    if not text or not manifest_records:
        return ""

    normalized_text = normalize_prompt_text(text)
    if not normalized_text:
        return ""

    for record in manifest_records:
        if normalized_text == record["normalized_prompt"]:
            return record["category"]

    best_match = ("", 0)
    for record in manifest_records:
        candidate = record["normalized_prompt"]
        if not candidate:
            continue
        if normalized_text in candidate or candidate in normalized_text:
            score = min(len(normalized_text), len(candidate))
            if score > best_match[1]:
                best_match = (record["category"], score)
    return best_match[0]


def load_csv_by_column(csv_path, strict_count=None, prompt_manifest_path=None):
    """
    根据明确列名解析CSV数据，避免索引错位
    参数: csv_path - CSV文件路径
    返回: parsed_data - 解析后的字典列表（每条含所有列字段）
    """
    # 定义CSV必须包含的列名（与用户提供的列名完全一致）
    required_columns = [
        "prompt_name",
        "gpt4o_difficulty_score", "gpt4o_action_sequence", 
        "gpt4o_technical_complexity", "gpt4o_movement_intensity", 
        "gpt4o_balance_requirement", "gpt4o_continuity", "gpt4o_scoring_reason",
        "qwen_difficulty_score", "qwen_action_sequence", 
        "qwen_technical_complexity", "qwen_movement_intensity", 
        "qwen_balance_requirement", "qwen_continuity", "qwen_scoring_reason",
        "mpjpe_g", "mpjpe_l", "mpjpe_pa", "accel_dist", "vel_dist", "success"
    ]
    
    manifest_records = load_prompt_manifest(prompt_manifest_path)
    parsed_data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        # 按列名读取，自动忽略表头（若CSV首行是列名）
        reader = csv.DictReader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        
        # 校验CSV列名是否完整
        csv_columns = reader.fieldnames
        # import pdb; pdb.set_trace()
        missing_cols = [col for col in required_columns if col not in csv_columns]
        if missing_cols:
            raise ValueError(f"CSV缺少必要列: {', '.join(missing_cols)}，请检查文件格式")
        
        # 解析每一行数据
        for row_num, row in enumerate(reader, 2):  # row_num从2开始（跳过表头行）
            # 过滤空行
            if not row["prompt_name"].strip():
                print(f"跳过空行（第{row_num}行）")
                continue
            
            # 转换数值型字段（避免字符串格式导致函数参数错误）
            try:
                mpjpe_g = float(row["mpjpe_g"].strip())
                mpjpe_l = float(row["mpjpe_l"].strip())
                mpjpe_pa = float(row["mpjpe_pa"].strip())
                accel_dist = float(row["accel_dist"].strip())
                vel_dist = float(row["vel_dist"].strip())
                success = row["success"].strip().upper()  # 统一为YES/NO
            except ValueError as e:
                print(f"第{row_num}行数值字段错误（{str(e)}），跳过该行")
                continue
            
            # 组装单条数据（与优化函数参数一一对应）
            prompt_name = row["prompt_name"].strip()
            video_name = row.get("video_name", "").strip()
            category = (
                row.get("category", "").strip()
                or infer_category_from_manifest(prompt_name, manifest_records)
                or infer_category_from_manifest(video_name, manifest_records)
                or infer_category_from_prompt(prompt_name)
            )
            single_data = {
                "category": category,
                "video_name": video_name,
                # 1. 原始prompt（函数参数：original_prompts）
                "original_prompt": prompt_name,
                
                # 2. 跟踪指标（函数参数：tracking_metrics_list）
                "tracking_metrics": {
                    "mpjpe_g": mpjpe_g,
                    "mpjpe_l": mpjpe_l,
                    "mpjpe_pa": mpjpe_pa,
                    "accel_dist": accel_dist,
                    "vel_dist": vel_dist,
                    "success": success if success in ["YES", "NO"] else "YES"  # 容错处理
                },
                
                # 3. GPT-4o VLM数据（函数参数：gpt4o_vlm_*）
                "gpt4o_vlm": {
                    "score": row["gpt4o_difficulty_score"].strip(),
                    "reason": row["gpt4o_scoring_reason"].strip(),
                    "analysis": {
                        "action_sequence": row["gpt4o_action_sequence"].strip(),
                        "technical_complexity": row["gpt4o_technical_complexity"].strip(),
                        "movement_intensity": row["gpt4o_movement_intensity"].strip(),
                        "balance_requirement": row["gpt4o_balance_requirement"].strip(),
                        "continuity": row["gpt4o_continuity"].strip()
                    }
                },
                
                # 4. Qwen VLM数据（函数参数：qwen_vlm_*）
                "qwen_vlm": {
                    "score": row["qwen_difficulty_score"].strip(),
                    "reason": row["qwen_scoring_reason"].strip(),
                    "analysis": {
                        "action_sequence": row["qwen_action_sequence"].strip(),
                        "technical_complexity": row["qwen_technical_complexity"].strip(),
                        "movement_intensity": row["qwen_movement_intensity"].strip(),
                        "balance_requirement": row["qwen_balance_requirement"].strip(),
                        "continuity": row["qwen_continuity"].strip()
                    }
                }
            }
            parsed_data.append(single_data)
    
    if strict_count is not None and len(parsed_data) != strict_count:
        raise ValueError(f"CSV解析后共{len(parsed_data)}条数据，需{strict_count}条，请检查文件完整性")

    print(f"✅ CSV解析完成，共获取{len(parsed_data)}条有效数据")
    return parsed_data

def split_into_groups(parsed_data, group_size=5, shuffle=True):
    """
    将40条数据随机打乱后平均分成8组（每组5条）
    参数: parsed_data - 解析后的全量数据；group_size - 每组条数（固定5）
    返回: grouped_data - 8组数据列表（每组5条）
    """
    data = list(parsed_data)
    if shuffle:
        random.shuffle(data)

    grouped_data = [
        data[i:i + group_size]
        for i in range(0, len(data), group_size)
    ]
    grouped_data = [group for group in grouped_data if group]

    print(f"✅ 数据已分成{len(grouped_data)}组，目标每组{group_size}条")
    return grouped_data


def infer_category_from_prompt(prompt_text):
    text = prompt_text.lower()
    if any(token in text for token in ["pirouette", "grand", "pas de", "fouett", "dancer", "allegro"]):
        return "dance"
    if any(token in text for token in ["vault", "tsukahara", "gymnast", "cartwheel", "aerial"]):
        return "gymnastics"
    if any(token in text for token in ["palm strike", "bow stance", "practitioner", "dragon", "tiger"]):
        return "martial_arts"
    if any(token in text for token in ["fighter", "combatant", "hook punch", "roundhouse", "flying knee"]):
        return "combat"
    return "sport"


def severity_key(item):
    metrics = item["tracking_metrics"]
    success = str(metrics.get("success", "YES")).upper()
    success_rank = 0 if success == "NO" else 1
    return (
        success_rank,
        -float(metrics.get("mpjpe_g", 0.0)),
        -float(metrics.get("mpjpe_l", 0.0)),
        -float(metrics.get("mpjpe_pa", 0.0)),
        -float(metrics.get("accel_dist", 0.0)),
        -float(metrics.get("vel_dist", 0.0)),
    )


def pad_category_rows(rows, target_count=40):
    if not rows:
        raise ValueError("Cannot pad an empty category.")

    sorted_rows = sorted(rows, key=severity_key)
    padded = list(sorted_rows)
    idx = 0
    while len(padded) < target_count:
        source = dict(sorted_rows[idx % len(sorted_rows)])
        source["duplicate_of"] = source.get("video_name") or source.get("original_prompt")
        padded.append(source)
        idx += 1
    return padded[:target_count]


def split_and_pad_by_category(parsed_data, target_count=40):
    categorized = {category: [] for category in ALL_CATEGORIES}
    for item in parsed_data:
        categorized[item["category"]].append(item)

    missing = [category for category, rows in categorized.items() if not rows]
    if missing:
        raise ValueError(
            "Missing categories in input CSV: "
            + ", ".join(missing)
            + ". Current rule cannot synthesize an empty category because there is no same-class sample to duplicate."
        )

    return {
        category: pad_category_rows(rows, target_count=target_count)
        for category, rows in categorized.items()
    }


def write_category_csv(rows, output_path):
    if not rows:
        return
    fieldnames = [
        "category",
        "video_name",
        "prompt_name",
        "gpt4o_difficulty_score",
        "gpt4o_action_sequence",
        "gpt4o_technical_complexity",
        "gpt4o_movement_intensity",
        "gpt4o_balance_requirement",
        "gpt4o_continuity",
        "gpt4o_scoring_reason",
        "qwen_difficulty_score",
        "qwen_action_sequence",
        "qwen_technical_complexity",
        "qwen_movement_intensity",
        "qwen_balance_requirement",
        "qwen_continuity",
        "qwen_scoring_reason",
        "mpjpe_g",
        "mpjpe_l",
        "mpjpe_pa",
        "accel_dist",
        "vel_dist",
        "success",
        "duplicate_of",
    ]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for item in rows:
            metrics = item["tracking_metrics"]
            gpt = item["gpt4o_vlm"]
            qwen = item["qwen_vlm"]
            writer.writerow({
                "category": item["category"],
                "video_name": item.get("video_name", ""),
                "prompt_name": item["original_prompt"],
                "gpt4o_difficulty_score": gpt["score"],
                "gpt4o_action_sequence": gpt["analysis"]["action_sequence"],
                "gpt4o_technical_complexity": gpt["analysis"]["technical_complexity"],
                "gpt4o_movement_intensity": gpt["analysis"]["movement_intensity"],
                "gpt4o_balance_requirement": gpt["analysis"]["balance_requirement"],
                "gpt4o_continuity": gpt["analysis"]["continuity"],
                "gpt4o_scoring_reason": gpt["reason"],
                "qwen_difficulty_score": qwen["score"],
                "qwen_action_sequence": qwen["analysis"]["action_sequence"],
                "qwen_technical_complexity": qwen["analysis"]["technical_complexity"],
                "qwen_movement_intensity": qwen["analysis"]["movement_intensity"],
                "qwen_balance_requirement": qwen["analysis"]["balance_requirement"],
                "qwen_continuity": qwen["analysis"]["continuity"],
                "qwen_scoring_reason": qwen["reason"],
                "mpjpe_g": metrics["mpjpe_g"],
                "mpjpe_l": metrics["mpjpe_l"],
                "mpjpe_pa": metrics["mpjpe_pa"],
                "accel_dist": metrics["accel_dist"],
                "vel_dist": metrics["vel_dist"],
                "success": metrics["success"],
                "duplicate_of": item.get("duplicate_of", ""),
            })


def write_prompt_manifest(records, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["prompt_name", "category", "normalized_prompt", "source_group_file"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for record in records:
            writer.writerow({
                "prompt_name": record["prompt_name"],
                "category": record["category"],
                "normalized_prompt": normalize_prompt_text(record["prompt_name"]),
                "source_group_file": record.get("source_group_file", ""),
            })


def generate_optimized_mixed_prompt(
    original_prompts,
    tracking_metrics_list,
    gpt4o_vlm_scores,
    gpt4o_vlm_reasons,
    gpt4o_vlm_analyses,
    qwen_vlm_scores,
    qwen_vlm_reasons,
    qwen_vlm_analyses
):
    messages = []
    messages.append({
        "role": "user",
        "content": [{
            "type": "text",
            "text": """
You are improving prompts for the next CLAIMS loop.
Generate one harder prompt for each input sample.

Requirements:
- Keep the motion category consistent with the original prompt.
- Make the optimized prompt more challenging but still realistic and trackable.
- Use PHC tracking metrics and both VLM analyses to decide how much to increase difficulty.
- If tracking failed badly or metrics are very high, increase difficulty only slightly.
- If tracking was relatively good, increase difficulty more aggressively.

Return plain text only in this exact format:
Group Analysis:
[short analysis]

Optimized Prompt for Set 1:
Prompt: ...

Optimized Prompt for Set 2:
Prompt: ...
""".strip()
        }]
    })

    data_summary = []
    for idx in range(len(original_prompts)):
        metrics = tracking_metrics_list[idx]
        gpt4o_analysis = gpt4o_vlm_analyses[idx]
        qwen_analysis = qwen_vlm_analyses[idx]
        category = infer_category_from_prompt(original_prompts[idx])
        data_summary.append(f"""
[Set {idx + 1}]
Category: {category}
Original Prompt: {original_prompts[idx]}
Tracking: success={metrics['success']}, mpjpe_g={metrics['mpjpe_g']}, mpjpe_l={metrics['mpjpe_l']}, mpjpe_pa={metrics['mpjpe_pa']}, accel_dist={metrics['accel_dist']}, vel_dist={metrics['vel_dist']}
GPT4o: score={gpt4o_vlm_scores[idx]}, reason={gpt4o_vlm_reasons[idx]}, action_sequence={gpt4o_analysis['action_sequence']}, technical_complexity={gpt4o_analysis['technical_complexity']}, movement_intensity={gpt4o_analysis['movement_intensity']}, balance_requirement={gpt4o_analysis['balance_requirement']}, continuity={gpt4o_analysis['continuity']}
Qwen: score={qwen_vlm_scores[idx]}, reason={qwen_vlm_reasons[idx]}, action_sequence={qwen_analysis['action_sequence']}, technical_complexity={qwen_analysis['technical_complexity']}, movement_intensity={qwen_analysis['movement_intensity']}, balance_requirement={qwen_analysis['balance_requirement']}, continuity={qwen_analysis['continuity']}
""".strip())

    messages.append({
        "role": "user",
        "content": [{
            "type": "text",
            "text": "\n\n".join(data_summary)
        }]
    })

    return call_generation_api(messages, max_tokens=2500)


def extract_prompts_from_result(result_text):
    prompts = []
    for line in result_text.splitlines():
        match = re.match(r"\s*(?:D\.\s*)?Prompt:\s*(.+)", line)
        if match:
            prompts.append(match.group(1).strip())
    return prompts


def process_release_loop(input_csv, output_dir, group_size=5, seed=7, target_count=40, prompt_manifest_path=None):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    parsed_data = load_csv_by_column(
        input_csv,
        strict_count=None,
        prompt_manifest_path=prompt_manifest_path,
    )
    categorized = split_and_pad_by_category(parsed_data, target_count=target_count)

    function_map = {
        "combat": generate_optimized_combat_prompts,
        "martial_arts": generate_optimized_martial_arts_prompt,
        "dance": generate_optimized_dance_prompt,
        "sport": generate_optimized_sport_prompt,
        "gymnastics": generate_optimized_gymnastics_prompt,
    }

    categorized_dir = os.path.join(output_dir, "category_csv")
    categorized_prompt_dir = os.path.join(output_dir, "category_prompts")
    all_prompts = []
    all_prompt_records = []
    category_prompt_records = {category: [] for category in ALL_CATEGORIES}
    generated_files = []
    total_groups = 0

    for category in ALL_CATEGORIES:
        rows = categorized[category]
        write_category_csv(rows, os.path.join(categorized_dir, f"{category}.csv"))
        grouped_data = split_into_groups(rows, group_size=group_size, shuffle=False)
        total_groups += len(grouped_data)

        for group_idx, group in enumerate(grouped_data, 1):
            func_params = build_function_params(group)
            result = function_map[category](*func_params)
            if not result:
                continue

            raw_text = result[0]
            group_path = os.path.join(output_dir, f"{category}_group{group_idx}.txt")
            with open(group_path, "w", encoding="utf-8") as f:
                f.write(raw_text + "\n")
            generated_files.append(group_path)

            group_prompts = extract_prompts_from_result(raw_text)
            all_prompts.extend(group_prompts)
            category_prompt_records[category].extend(group_prompts)
            all_prompt_records.extend(
                {
                    "prompt_name": prompt,
                    "category": category,
                    "source_group_file": group_path,
                }
                for prompt in group_prompts
            )

    prompts_path = os.path.join(output_dir, "loop1_prompts.txt")
    with open(prompts_path, "w", encoding="utf-8") as f:
        for prompt in all_prompts:
            f.write(prompt + "\n")

    os.makedirs(categorized_prompt_dir, exist_ok=True)
    for category, prompts in category_prompt_records.items():
        prompt_path = os.path.join(categorized_prompt_dir, f"{category}_prompt.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            for prompt in prompts:
                f.write(prompt + "\n")

    manifest_path = os.path.join(output_dir, "loop1_prompt_manifest.csv")
    write_prompt_manifest(all_prompt_records, manifest_path)

    return {
        "groups": total_groups,
        "prompt_count": len(all_prompts),
        "output_dir": output_dir,
        "prompts_path": prompts_path,
        "manifest_path": manifest_path,
        "categorized_dir": categorized_dir,
        "categorized_prompt_dir": categorized_prompt_dir,
        "generated_files": generated_files,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate next-loop prompts from merged PHC+VLM CSV.")
    parser.add_argument("--input-csv", type=str, help="Merged CSV input.")
    parser.add_argument("--output-dir", type=str, help="Output directory.")
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-count", type=int, default=40)
    parser.add_argument("--prompt-manifest", type=str, default=None)
    return parser.parse_args()

def build_function_params(single_group):
    """
    将单组数据（5条）构造成优化函数的输入参数
    参数: single_group - 单组5条数据
    返回: 优化函数所需的8个参数（按顺序对应）
    """
    original_prompts = []
    tracking_metrics_list = []
    gpt4o_vlm_scores = []
    gpt4o_vlm_reasons = []
    gpt4o_vlm_analyses = []
    qwen_vlm_scores = []
    qwen_vlm_reasons = []
    qwen_vlm_analyses = []
    
    for data in single_group:
        # 1. original_prompts
        original_prompts.append(data["original_prompt"])
        
        # 2. tracking_metrics_list
        tracking_metrics_list.append(data["tracking_metrics"])
        
        # 3. GPT-4o VLM相关
        gpt4o_vlm_scores.append(data["gpt4o_vlm"]["score"])
        gpt4o_vlm_reasons.append(data["gpt4o_vlm"]["reason"])
        gpt4o_vlm_analyses.append(data["gpt4o_vlm"]["analysis"])
        
        # 4. Qwen VLM相关
        qwen_vlm_scores.append(data["qwen_vlm"]["score"])
        qwen_vlm_reasons.append(data["qwen_vlm"]["reason"])
        qwen_vlm_analyses.append(data["qwen_vlm"]["analysis"])
    
    return (
        original_prompts,
        tracking_metrics_list,
        gpt4o_vlm_scores,
        gpt4o_vlm_reasons,
        gpt4o_vlm_analyses,
        qwen_vlm_scores,
        qwen_vlm_reasons,
        qwen_vlm_analyses
    )
def generate_optimized_sport_prompt(    
    original_prompts, 
    tracking_metrics_list, 
    # GPT-4o VLM 输入（原有）
    gpt4o_vlm_scores, 
    gpt4o_vlm_reasons, 
    gpt4o_vlm_analyses,
    # Qwen VLM 输入（新增，格式与GPT-4o一致）
    qwen_vlm_scores, 
    qwen_vlm_reasons, 
    qwen_vlm_analyses
):
    """
    输入5个格斗动作prompt、跟踪数据及双VLM（GPT-4o + Qwen）反馈，输出5个拔高难度的优化prompt
    参数:
        original_prompts: 5个原始prompt列表（长度必须为5）
        tracking_metrics_list: 5个跟踪指标字典列表（含success/mpjpe_g等，长度必须为5）
        # GPT-4o VLM 参数（原有）
        gpt4o_vlm_scores: GPT-4o VLM评分列表（如["8/10"...]，长度5）
        gpt4o_vlm_reasons: GPT-4o VLM评分理由列表（长度5）
        gpt4o_vlm_analyses: GPT-4o VLM详细分析字典列表（含action_sequence等，长度5）
        # Qwen VLM 参数（新增，格式与GPT-4o一致）
        qwen_vlm_scores: Qwen VLM评分列表（如["7/10"...]，长度5）
        qwen_vlm_reasons: Qwen VLM评分理由列表（长度5）
        qwen_vlm_analyses: Qwen VLM详细分析字典列表（含action_sequence等，长度5）
    返回:
        包含组内分析和5个优化后prompt的列表
    """
    # -------------------------- 1. 参数校验（新增Qwen VLM校验，与GPT-4o格式对齐） --------------------------
    # 整合所有输入列表（含双VLM）
    input_lists = [
        original_prompts, tracking_metrics_list,
        # GPT-4o VLM
        gpt4o_vlm_scores, gpt4o_vlm_reasons, gpt4o_vlm_analyses,
        # Qwen VLM（新增）
        qwen_vlm_scores, qwen_vlm_reasons, qwen_vlm_analyses
    ]
    input_names = [
        "original_prompts", "tracking_metrics_list",
        "gpt4o_vlm_scores", "gpt4o_vlm_reasons", "gpt4o_vlm_analyses",
        "qwen_vlm_scores", "qwen_vlm_reasons", "qwen_vlm_analyses"
    ]
    
    # 1.1 检查所有列表长度为5且一致
    for idx, (lst, name) in enumerate(zip(input_lists, input_names)):
        if len(lst) != 5:
            raise ValueError(f"{name}长度必须为5（当前为{len(lst)}），请输入5个动作的数据")
    if len(set(len(lst) for lst in input_lists)) != 1:
        raise ValueError("所有输入列表长度不一致，请确保每个列表都包含5个元素")

    # 1.2 检查跟踪指标字典键完整（原有逻辑不变）
    required_metric_keys = ["success", "mpjpe_g", "mpjpe_l", "mpjpe_pa", "accel_dist", "vel_dist"]
    for idx, metrics in enumerate(tracking_metrics_list):
        missing_keys = [k for k in required_metric_keys if k not in metrics]
        if missing_keys:
            raise ValueError(f"第{idx+1}个tracking_metrics缺少键：{', '.join(missing_keys)}")

    # 1.3 检查双VLM详细分析字典键完整（与原有GPT-4o格式一致）
    required_analysis_keys = ["action_sequence", "technical_complexity", "movement_intensity", "balance_requirement", "continuity"]
    # 校验GPT-4o VLM分析
    for idx, analysis in enumerate(gpt4o_vlm_analyses):
        missing_keys = [k for k in required_analysis_keys if k not in analysis]
        if missing_keys:
            raise ValueError(f"第{idx+1}个gpt4o_vlm_analyses缺少键：{', '.join(missing_keys)}")
    # 校验Qwen VLM分析（新增）
    for idx, analysis in enumerate(qwen_vlm_analyses):
        missing_keys = [k for k in required_analysis_keys if k not in analysis]
        if missing_keys:
            raise ValueError(f"第{idx+1}个qwen_vlm_analyses缺少键：{', '.join(missing_keys)}")


    # 1. 运动无级变量库（全英文，混合所有难度动作）
    # 1. 优化后的运动无级变量库（移除数值和括号，删除scene和direction）
    sports_vars = {
        "base_action": [
            "jogging with steady pace and relaxed arm swing",
            "basic jump with two-foot takeoff and landing",
            "push-up with knee support and slow movement",
            "soccer forward pass with slow arm swing",
            "walking lunges with steady step and bent knees",
            "static plank with forearm support and straight body",
            "basketball chest pass with close range and slow release",
            "sprint start with three-point stance and moderate acceleration",
            "vertical jump with countermovement and controlled landing",
            "lunge with dynamic step and proper knee alignment",
            "basketball dribble with steady bounce and low height",
            "tennis forehand with moderate swing and controlled range",
            "swimming freestyle with steady pace and consistent stroke",
            "cycling with moderate speed and regular pedal rotation",
            "100m sprint with maximum acceleration and rapid steps",
            "long jump with fast approach run and airborne flight",
            "high jump with Fosbury flop technique and back arch",
            "soccer kick with full instep strike and long range",
            "burpee with rapid transitions between movements",
            "400m hurdles with consistent rhythm between obstacles",
            "basketball slam dunk with high vertical jump and one-hand grip",
            "parkour vault with fast runup and smooth obstacle clearance",
            "snowboard half-pipe with high airborne and spinning movement",
            "rugby tackle with explosive drive and low center of gravity",
            "tennis serve with high speed and topspin rotation",
            "triathlon transition with seamless switch between disciplines",
            "volleyball spike with high jump and powerful downward strike",
            "skateboard ollie with board flip and balanced landing",
            "hurdle race with rapid steps between obstacles",
            "martial arts kickboxing combo with quick successive strikes"
        ],
        "combo_action": [
            "jogging → basic jump → walking recovery",
            "push-up → walking lunges → static plank",
            "basketball chest pass → slow dribble → chest pass",
            "sprint start → short acceleration → slow stop",
            "tennis forehand → backhand → slow run to net",
            "swimming freestyle → turn → freestyle continuation",
            "sprint start → rapid acceleration → slide stop",
            "dribble → crossover → jump shot → rebound",
            "long jump approach → takeoff → landing → recovery",
            "400m hurdles → sprint finish → cool-down jog",
            "parkour vault → run → ollie → landing roll",
            "volleyball set → jump spike → defensive block",
            "rugby tackle → standup → pass → sprint"
        ],
        "detail": [
            "jogging with consistent stride length and parallel arm swing",
            "push-up with elbows at proper angle and tight core",
            "forward pass with hands positioned on ball sides and follow-through",
            "plank with elbows under shoulders and engaged glutes",
            "sprint start with front knee over toe and bent back leg",
            "dribble with spread fingers and ball contact with pads",
            "tennis forehand with proper racket angle and fluid swing",
            "long jump takeoff with extended front leg and tucked back leg",
            "Fosbury flop with arched back and hips positioned over bar",
            "soccer kick with properly placed plant foot and hip rotation",
            "slam dunk with powerful approach and high hand placement",
            "parkour speed vault with balanced hand support and parallel leg swing",
            "snowboard half-pipe with weight shifted forward and shoulder-initiated spin",
            "tennis serve with high toss and optimal impact position",
            "volleyball spike with two-leg jump and cocked arm position"
        ],
        "speed_rhythm": [
            "jogging with consistent pace and regular step frequency",
            "push-up with slow descent and controlled ascent",
            "walking lunges with steady tempo and no speed variation",
            "sprint with gradual acceleration and maintained speed",
            "dribble with steady bounce rate followed by quick pass",
            "swimming with consistent stroke timing and efficient turns",
            "100m sprint with rapid acceleration and sustained maximum speed",
            "burpee with quick transitions between push-up and jump",
            "parkour with fast runup and rapid sequence execution",
            "tennis serve with quick toss and explosive swing",
            "hurdle race with rapid sprint between obstacle jumps"
        ]
    }

    # 2. 优化后的运动无级模板库（移除scene和direction相关内容）
    sports_templates = [
        "The athlete performed {combo_action} with {detail}, maintaining {speed_rhythm}.",
        "During training, the sportsperson executed {base_action} focusing on {detail}, following {speed_rhythm}.",
        "The athlete completed {combo_action} with {detail} (power control), adhering to {speed_rhythm}.",
        "In the key moment, the athlete transitioned from {combo_action} to {base_action} with {detail}, using {speed_rhythm}.",
        "The sportsperson combined {base_action} and {combo_action} with {detail}, synchronizing with {speed_rhythm}.",
        "The sequence began with {combo_action}, then shifted into {base_action} with {detail}, regulated by {speed_rhythm}.",
        "During warm-up, the athlete practiced {base_action} with {detail}, following {speed_rhythm}.",
        "In the final phase, the athlete executed {base_action} followed by {combo_action} with {detail}, optimized for {speed_rhythm}."
    ]


    # -------------------------- 3. 构建多轮消息（补充VLM难度评分参考） --------------------------
    messages = []

    # -------------------------- 第一轮：核心目标（新增VLM评分作为参考说明） --------------------------
    background_msg = {
        "role": "user",
        "content": [{"type": "text", "text": """
        Core Goal: Generate optimized prompts that push the motion tracker’s performance limit. These prompts must be more challenging than the original ones.

        References for optimization:
        1. Intra-group metric comparison (5 sets vs each other): Identify which actions the tracker handles relatively easily (lower metrics) or struggles with (higher metrics).
        2. Dual VLM feedback (GPT-4o + Qwen): Use their analysis and difficulty scores (0-10, 10 being hardest) of the original actions to guide variable selection from {sports_vars}.
        """.strip()}]  
    }
    messages.append(background_msg)

    # -------------------------- 第二轮：5组数据（保留原有VLM评分展示，无额外修改） --------------------------
    data_summary = ""
    for idx in range(5):
        metrics = tracking_metrics_list[idx]
        gpt4o_analysis = gpt4o_vlm_analyses[idx]
        qwen_analysis = qwen_vlm_analyses[idx]
        
        data_summary += f"""
        [Set {idx+1} Data]
        1. Original Prompt: "{original_prompts[idx]}"
        2. Tracking Metrics:
        - Success: {metrics['success']}
        - mpjpe_g: {metrics['mpjpe_g']} mm | mpjpe_l: {metrics['mpjpe_l']} mm | mpjpe_pa: {metrics['mpjpe_pa']} mm
        - accel_dist: {metrics['accel_dist']} mm/frame² | vel_dist: {metrics['vel_dist']} mm/frame
        3. Dual VLM Score and Analysis:
        - GPT-4o: Action Difficulty Score = {gpt4o_vlm_scores[idx]} (0-10, 10 hardest); Action Sequence={gpt4o_analysis['action_sequence']}; Technical Analysis={gpt4o_analysis['technical_complexity']}; Movement Intensity={gpt4o_analysis['movement_intensity']}; Balance Requirement={gpt4o_analysis['balance_requirement']}; Reason="{gpt4o_vlm_reasons[idx]}"
        - Qwen: Action Difficulty Score = {qwen_vlm_scores[idx]} (0-10, 10 hardest); Action Sequence={qwen_analysis['action_sequence']}; Technical Analysis={qwen_analysis['technical_complexity']};  Movement Intensity={qwen_analysis['movement_intensity']}; Balance Requirement={qwen_analysis['balance_requirement']}; Reason="{qwen_vlm_reasons[idx]}"
        """  # 仅在评分后补充“(0-10, 10 hardest)”说明

    # -------------------------- 第二轮：5组数据（细化难度增加幅度策略） --------------------------
    data_msg = {
        "role": "user",
        "content": [{"type": "text", "text": f"""
        Below are 5 sets of data. Optimize each prompt to be more challenging than the original, using variables from {sports_vars}, with adjustment幅度 based on tracker performance and VLM scores:
        - For actions the tracker handles relatively easily (lower metrics in the group) AND with lower VLM scores: Select variables from {sports_vars} to SIGNIFICANTLY increase difficulty.
        - For actions the tracker handles relatively easily (lower metrics) BUT with higher VLM scores: Select variables to MODERATELY increase difficulty.
        - For actions the tracker struggles with (higher metrics) BUT with lower VLM scores: Select variables to MODERATELY increase difficulty.
        - For actions the tracker struggles with (higher metrics) AND with higher VLM scores: Select variables to SLIGHTLY increase difficulty (closer to the original but still more challenging).

        5 Sets of Data:
        {data_summary.strip()}
        """.strip()}]
    }
    messages.append(data_msg)


    # -------------------------- 第三轮：{sports_vars}库（完全不变） --------------------------
    resource_msg = {
        "role": "user",
        "content": [{"type": "text", "text": f"""
        Use ONLY variables from the following {sports_vars} library and templates to generate optimized prompts:

        1. {sports_vars} Library:
        - base_action: {sports_vars['base_action']}
        - combo_action: {sports_vars['combo_action']}
        - detail: {sports_vars['detail']}
        - speed_rhythm: {sports_vars['speed_rhythm']}

        2. Sports Templates:
        {sports_templates}
        """.strip()}]
    }
    messages.append(resource_msg)

    # -------------------------- 第四轮：任务要求（新增VLM评分在分析中的引用） --------------------------
    task_msg = {
        "role": "user",
        "content": [{"type": "text", "text": """
        Your Tasks:
        1. GROUP ANALYSIS:
        A. Tracker Performance Comparison: Compare the 5 sets’ metrics and VLM difficulty scores to identify which are handled relatively easily or with difficulty.
        B. Dual VLM Consensus: Identify common points in GPT-4o and Qwen’s analysis and score trends for each set.

        2. PROMPT OPTIMIZATION (1-to-1):
        For each set (labeled "Optimized Prompt for Set X"):
        A. Original vs Optimized: Explain how the optimized prompt is more challenging than the original, referencing VLM scores.
        B. Variables Selection: Which variables were chosen and why (link to tracker performance, VLM feedback, and scores).
        C. Template Selection: Which template was used and why.
        D. Final Optimized Prompt: English sentence using selected variables and template.

        Output Format:
        1. Group Analysis
        A. Tracker Performance Comparison:
            [Your comparison here, including VLM scores]
        B. Dual VLM Consensus:
            [Your consensus here, including score trends]

        2. Optimized Prompts
        - Optimized Prompt for Set 1:
            A. Original vs Optimized: [Explanation with VLM score reference]
            B. Variables Selection: [Variables + Reason linked to scores]
            C. Template Selection: [Template + Reason]
            D. Prompt: [Final prompt]
        
        - Optimized Prompt for Set 2: [Same structure]
        - ... (all 5 sets)
        """.strip()}]  # 仅在分析和优化逻辑中新增“引用VLM评分”的要求
    }
    messages.append(task_msg)


    # -------------------------- 4. 调用Gemini API（统一模型与鉴权） --------------------------
    return call_generation_api(messages, max_tokens=8000)

def generate_optimized_gymnastics_prompt(    original_prompts, 
    tracking_metrics_list, 
    # GPT-4o VLM 输入（原有）
    gpt4o_vlm_scores, 
    gpt4o_vlm_reasons, 
    gpt4o_vlm_analyses,
    # Qwen VLM 输入（新增，格式与GPT-4o一致）
    qwen_vlm_scores, 
    qwen_vlm_reasons, 
    qwen_vlm_analyses
):
    """
    输入5个格斗动作prompt、跟踪数据及双VLM（GPT-4o + Qwen）反馈，输出5个拔高难度的优化prompt
    参数:
        original_prompts: 5个原始prompt列表（长度必须为5）
        tracking_metrics_list: 5个跟踪指标字典列表（含success/mpjpe_g等，长度必须为5）
        # GPT-4o VLM 参数（原有）
        gpt4o_vlm_scores: GPT-4o VLM评分列表（如["8/10"...]，长度5）
        gpt4o_vlm_reasons: GPT-4o VLM评分理由列表（长度5）
        gpt4o_vlm_analyses: GPT-4o VLM详细分析字典列表（含action_sequence等，长度5）
        # Qwen VLM 参数（新增，格式与GPT-4o一致）
        qwen_vlm_scores: Qwen VLM评分列表（如["7/10"...]，长度5）
        qwen_vlm_reasons: Qwen VLM评分理由列表（长度5）
        qwen_vlm_analyses: Qwen VLM详细分析字典列表（含action_sequence等，长度5）
    返回:
        包含组内分析和5个优化后prompt的列表
    """
    # -------------------------- 1. 参数校验（新增Qwen VLM校验，与GPT-4o格式对齐） --------------------------
    # 整合所有输入列表（含双VLM）
    input_lists = [
        original_prompts, tracking_metrics_list,
        # GPT-4o VLM
        gpt4o_vlm_scores, gpt4o_vlm_reasons, gpt4o_vlm_analyses,
        # Qwen VLM（新增）
        qwen_vlm_scores, qwen_vlm_reasons, qwen_vlm_analyses
    ]
    input_names = [
        "original_prompts", "tracking_metrics_list",
        "gpt4o_vlm_scores", "gpt4o_vlm_reasons", "gpt4o_vlm_analyses",
        "qwen_vlm_scores", "qwen_vlm_reasons", "qwen_vlm_analyses"
    ]
    
    # 1.1 检查所有列表长度为5且一致
    for idx, (lst, name) in enumerate(zip(input_lists, input_names)):
        if len(lst) != 5:
            raise ValueError(f"{name}长度必须为5（当前为{len(lst)}），请输入5个动作的数据")
    if len(set(len(lst) for lst in input_lists)) != 1:
        raise ValueError("所有输入列表长度不一致，请确保每个列表都包含5个元素")

    # 1.2 检查跟踪指标字典键完整（原有逻辑不变）
    required_metric_keys = ["success", "mpjpe_g", "mpjpe_l", "mpjpe_pa", "accel_dist", "vel_dist"]
    for idx, metrics in enumerate(tracking_metrics_list):
        missing_keys = [k for k in required_metric_keys if k not in metrics]
        if missing_keys:
            raise ValueError(f"第{idx+1}个tracking_metrics缺少键：{', '.join(missing_keys)}")

    # 1.3 检查双VLM详细分析字典键完整（与原有GPT-4o格式一致）
    required_analysis_keys = ["action_sequence", "technical_complexity", "movement_intensity", "balance_requirement", "continuity"]
    # 校验GPT-4o VLM分析
    for idx, analysis in enumerate(gpt4o_vlm_analyses):
        missing_keys = [k for k in required_analysis_keys if k not in analysis]
        if missing_keys:
            raise ValueError(f"第{idx+1}个gpt4o_vlm_analyses缺少键：{', '.join(missing_keys)}")
    # 校验Qwen VLM分析（新增）
    for idx, analysis in enumerate(qwen_vlm_analyses):
        missing_keys = [k for k in required_analysis_keys if k not in analysis]
        if missing_keys:
            raise ValueError(f"第{idx+1}个qwen_vlm_analyses缺少键：{', '.join(missing_keys)}")


    # 1. 优化后的体操无级变量库（移除数值/括号，删除scene和direction）
    gymnastics_vars = {
        "base_action": [
            "basic squat with bent knees and flat feet",
            "simple arm stretch with slow overhead reach",
            "balance beam walk with steady pace and outstretched arms",
            "vault approach with straight walk",
            "static plank with forearm support and straight body",
            "slow leg lift with sideways movement and controlled lowering",
            "parallel bars hang with straight arms",
            "cartwheel with sideways movement and two-hand support",
            "single leg lift with static balance on beam",
            "floor jump with vertical leap and soft landing",
            "rings hang with slow shoulder shrugs",
            "saddle horse mount with slow leg swing and steady grip",
            "uneven bars pull-up with controlled ascent",
            "floor roll with forward movement",
            "double backflip with airborne rotation and tuck position",
            "balance beam split leap with leg extension and airborne movement",
            "vault tsukahara with springboard takeoff and twisting movement",
            "parallel bars swing with giant circle and momentum-based movement",
            "triple backflip with consecutive airborne spins and pike position",
            "balance beam back handspring with no-hand flip and rotation",
            "vault yurchenko with round-off onto springboard and twisting movement",
            "rings iron cross with horizontal arm hold and core tension",
            "parallel bars dismount with double backflip off bars",
            "floor arabesque leap with extended leg and airborne movement",
            "uneven bars kip with dynamic swing to support",
            "saddle horse double leg circle with rapid leg swing",
            "floor full-twisting double backflip with spins and twisting movement",
            "balance beam front handspring into split leap with seamless transition",
            "rings swing to cross with dynamic swing into iron cross"
        ],
        "combo_action": [
            "basic squat → simple arm stretch → floor jump",
            "balance beam walk → single leg lift → step down",
            "plank → slow leg lift → arm stretch",
            "cartwheel → floor roll → standing jump",
            "uneven bars pull-up → hang → slow swing",
            "saddle horse mount → single leg circle → dismount",
            "vault approach → tsukahara → landing roll",
            "floor cartwheel → double backflip → split leap",
            "triple backflip → floor arabesque leap → full-twisting backflip",
            "uneven bars kip → swing → parallel bars dismount",
            "rings swing → iron cross → dynamic dismount",
            "balance beam back handspring → split leap → front handspring"
        ],
        "detail": [
            "squat with knees over toes, straight back and forward-facing arms",
            "beam walk with steady steps, forward gaze and relaxed shoulders",
            "plank with elbows under shoulders, engaged core and no hip sag",
            "hang with straight arms, shoulders away from ears and firm grip",
            "cartwheel with shoulder-width hand placement, straight legs and together landing feet",
            "pull-up with chin over bar and fully extended elbows at bottom",
            "floor roll with tucked chin, shoulder-initiated rotation and no head impact",
            "double backflip with knees tucked to chest at peak height and timely untucking before landing",
            "split leap with squared hips, pointed toes and straight back",
            "tsukahara with arched back at takeoff and twisting movement initiated mid-air",
            "triple backflip with strong leg drive at takeoff and maintained pike position",
            "iron cross with arms parallel to ground, proper shoulder alignment and locked core",
            "yurchenko vault with hands on springboard during round-off and post-takeoff twist",
            "full-twisting backflip with twist initiation after first spin and fixed gaze on landing",
            "beam back handspring with heel-driven takeoff, arched body mid-air and ball-of-foot landing"
        ],
        "speed_rhythm": [
            "balance beam walk with slow steps and steady arm movement",
            "squat with controlled descent, held position and steady ascent",
            "plank with held position, rest period and repetition",
            "cartwheel with timely completion, rest period and floor roll",
            "pull-up with controlled ascent, held position and steady descent",
            "balance beam walk with steady steps and held leg lift",
            "vault approach with progressive speed increase and explosive takeoff",
            "swing with gradual buildup, sudden release for dismount and increased speed",
            "backflip sequence with quick takeoff, airborne spins and fast landing",
            "rings swing with steady forward and backward movement and explosive transition to cross",
            "balance beam combo with quick back handspring, split leap and front handspring"
        ]
    }

    # 2. 优化后的体操无级模板库（移除scene和direction相关内容）
    gymnastics_templates = [
        "The gymnast performed {combo_action} with {detail}, following {speed_rhythm}.",
        "During practice, the athlete executed {base_action} focusing on {detail}, maintaining {speed_rhythm}.",
        "The gymnast completed {combo_action} with {detail} (airborne control), adhering to {speed_rhythm}.",
        "In the dismount sequence, the athlete executed {base_action} followed by {combo_action} with {detail}, using {speed_rhythm}.",
        "The routine began with {combo_action}, then shifted into {base_action} with {detail}, regulated by {speed_rhythm}.",
        "During warm-up, the gymnast practiced {base_action} with {detail}, following {speed_rhythm}.",
        "In competition, the gymnast combined {base_action} and {combo_action} with {detail}, optimized for {speed_rhythm}.",
        "The athlete transitioned from {combo_action} to {base_action} with {detail}, synchronizing with {speed_rhythm}."
    ]

    # -------------------------- 3. 构建多轮消息（补充VLM难度评分参考） --------------------------
    messages = []

    # -------------------------- 第一轮：核心目标（新增VLM评分作为参考说明） --------------------------
    background_msg = {
        "role": "user",
        "content": [{"type": "text", "text": """
        Core Goal: Generate optimized prompts that push the motion tracker’s performance limit. These prompts must be more challenging than the original ones.

        References for optimization:
        1. Intra-group metric comparison (5 sets vs each other): Identify which actions the tracker handles relatively easily (lower metrics) or struggles with (higher metrics).
        2. Dual VLM feedback (GPT-4o + Qwen): Use their analysis and difficulty scores (0-10, 10 being hardest) of the original actions to guide variable selection from {gymnastics_vars}.
        """.strip()}]  
    }
    messages.append(background_msg)

    # -------------------------- 第二轮：5组数据（保留原有VLM评分展示，无额外修改） --------------------------
    data_summary = ""
    for idx in range(5):
        metrics = tracking_metrics_list[idx]
        gpt4o_analysis = gpt4o_vlm_analyses[idx]
        qwen_analysis = qwen_vlm_analyses[idx]
        
        data_summary += f"""
        [Set {idx+1} Data]
        1. Original Prompt: "{original_prompts[idx]}"
        2. Tracking Metrics:
        - Success: {metrics['success']}
        - mpjpe_g: {metrics['mpjpe_g']} mm | mpjpe_l: {metrics['mpjpe_l']} mm | mpjpe_pa: {metrics['mpjpe_pa']} mm
        - accel_dist: {metrics['accel_dist']} mm/frame² | vel_dist: {metrics['vel_dist']} mm/frame
        3. Dual VLM Score and Analysis:
        - GPT-4o: Action Difficulty Score = {gpt4o_vlm_scores[idx]} (0-10, 10 hardest); Action Sequence={gpt4o_analysis['action_sequence']}; Technical Analysis={gpt4o_analysis['technical_complexity']}; Movement Intensity={gpt4o_analysis['movement_intensity']}; Balance Requirement={gpt4o_analysis['balance_requirement']}; Reason="{gpt4o_vlm_reasons[idx]}"
        - Qwen: Action Difficulty Score = {qwen_vlm_scores[idx]} (0-10, 10 hardest); Action Sequence={qwen_analysis['action_sequence']}; Technical Analysis={qwen_analysis['technical_complexity']};  Movement Intensity={qwen_analysis['movement_intensity']}; Balance Requirement={qwen_analysis['balance_requirement']}; Reason="{qwen_vlm_reasons[idx]}"
        """  # 仅在评分后补充“(0-10, 10 hardest)”说明

    # -------------------------- 第二轮：5组数据（细化难度增加幅度策略） --------------------------
    data_msg = {
        "role": "user",
        "content": [{"type": "text", "text": f"""
        Below are 5 sets of data. Optimize each prompt to be more challenging than the original, using variables from {gymnastics_vars}, with adjustment幅度 based on tracker performance and VLM scores:
        - For actions the tracker handles relatively easily (lower metrics in the group) AND with lower VLM scores: Select variables from {gymnastics_vars} to SIGNIFICANTLY increase difficulty.
        - For actions the tracker handles relatively easily (lower metrics) BUT with higher VLM scores: Select variables to MODERATELY increase difficulty.
        - For actions the tracker struggles with (higher metrics) BUT with lower VLM scores: Select variables to MODERATELY increase difficulty.
        - For actions the tracker struggles with (higher metrics) AND with higher VLM scores: Select variables to SLIGHTLY increase difficulty (closer to the original but still more challenging).

        5 Sets of Data:
        {data_summary.strip()}
        """.strip()}]
    }
    messages.append(data_msg)


    # -------------------------- 第三轮：{gymnastics_vars}库（完全不变） --------------------------
    resource_msg = {
        "role": "user",
        "content": [{"type": "text", "text": f"""
        Use ONLY variables from the following {gymnastics_vars} library and templates to generate optimized prompts:

        1. {gymnastics_vars} Library:
        - base_action: {gymnastics_vars['base_action']}
        - combo_action: {gymnastics_vars['combo_action']}
        - detail: {gymnastics_vars['detail']}
        - speed_rhythm: {gymnastics_vars['speed_rhythm']}

        2. Gymnastics Templates:
        {gymnastics_templates}
        """.strip()}]
    }
    messages.append(resource_msg)

    # -------------------------- 第四轮：任务要求（新增VLM评分在分析中的引用） --------------------------
    task_msg = {
        "role": "user",
        "content": [{"type": "text", "text": """
        Your Tasks:
        1. GROUP ANALYSIS:
        A. Tracker Performance Comparison: Compare the 5 sets’ metrics and VLM difficulty scores to identify which are handled relatively easily or with difficulty.
        B. Dual VLM Consensus: Identify common points in GPT-4o and Qwen’s analysis and score trends for each set.

        2. PROMPT OPTIMIZATION (1-to-1):
        For each set (labeled "Optimized Prompt for Set X"):
        A. Original vs Optimized: Explain how the optimized prompt is more challenging than the original, referencing VLM scores.
        B. Variables Selection: Which variables were chosen and why (link to tracker performance, VLM feedback, and scores).
        C. Template Selection: Which template was used and why.
        D. Final Optimized Prompt: English sentence using selected variables and template.

        Output Format:
        1. Group Analysis
        A. Tracker Performance Comparison:
            [Your comparison here, including VLM scores]
        B. Dual VLM Consensus:
            [Your consensus here, including score trends]

        2. Optimized Prompts
        - Optimized Prompt for Set 1:
            A. Original vs Optimized: [Explanation with VLM score reference]
            B. Variables Selection: [Variables + Reason linked to scores]
            C. Template Selection: [Template + Reason]
            D. Prompt: [Final prompt]
        
        - Optimized Prompt for Set 2: [Same structure]
        - ... (all 5 sets)
        """.strip()}]  # 仅在分析和优化逻辑中新增“引用VLM评分”的要求
    }
    messages.append(task_msg)


    # -------------------------- 4. 调用Gemini API（统一模型与鉴权） --------------------------
    return call_generation_api(messages, max_tokens=8000)
def generate_optimized_martial_arts_prompt(    original_prompts, 
    tracking_metrics_list, 
    # GPT-4o VLM 输入（原有）
    gpt4o_vlm_scores, 
    gpt4o_vlm_reasons, 
    gpt4o_vlm_analyses,
    # Qwen VLM 输入（新增，格式与GPT-4o一致）
    qwen_vlm_scores, 
    qwen_vlm_reasons, 
    qwen_vlm_analyses
):
    """
    输入5个格斗动作prompt、跟踪数据及双VLM（GPT-4o + Qwen）反馈，输出5个拔高难度的优化prompt
    参数:
        original_prompts: 5个原始prompt列表（长度必须为5）
        tracking_metrics_list: 5个跟踪指标字典列表（含success/mpjpe_g等，长度必须为5）
        # GPT-4o VLM 参数（原有）
        gpt4o_vlm_scores: GPT-4o VLM评分列表（如["8/10"...]，长度5）
        gpt4o_vlm_reasons: GPT-4o VLM评分理由列表（长度5）
        gpt4o_vlm_analyses: GPT-4o VLM详细分析字典列表（含action_sequence等，长度5）
        # Qwen VLM 参数（新增，格式与GPT-4o一致）
        qwen_vlm_scores: Qwen VLM评分列表（如["7/10"...]，长度5）
        qwen_vlm_reasons: Qwen VLM评分理由列表（长度5）
        qwen_vlm_analyses: Qwen VLM详细分析字典列表（含action_sequence等，长度5）
    返回:
        包含组内分析和5个优化后prompt的列表
    """
    # -------------------------- 1. 参数校验（新增Qwen VLM校验，与GPT-4o格式对齐） --------------------------
    # 整合所有输入列表（含双VLM）
    input_lists = [
        original_prompts, tracking_metrics_list,
        # GPT-4o VLM
        gpt4o_vlm_scores, gpt4o_vlm_reasons, gpt4o_vlm_analyses,
        # Qwen VLM（新增）
        qwen_vlm_scores, qwen_vlm_reasons, qwen_vlm_analyses
    ]
    input_names = [
        "original_prompts", "tracking_metrics_list",
        "gpt4o_vlm_scores", "gpt4o_vlm_reasons", "gpt4o_vlm_analyses",
        "qwen_vlm_scores", "qwen_vlm_reasons", "qwen_vlm_analyses"
    ]
    
    # 1.1 检查所有列表长度为5且一致
    for idx, (lst, name) in enumerate(zip(input_lists, input_names)):
        if len(lst) != 5:
            raise ValueError(f"{name}长度必须为5（当前为{len(lst)}），请输入5个动作的数据")
    if len(set(len(lst) for lst in input_lists)) != 1:
        raise ValueError("所有输入列表长度不一致，请确保每个列表都包含5个元素")

    # 1.2 检查跟踪指标字典键完整（原有逻辑不变）
    required_metric_keys = ["success", "mpjpe_g", "mpjpe_l", "mpjpe_pa", "accel_dist", "vel_dist"]
    for idx, metrics in enumerate(tracking_metrics_list):
        missing_keys = [k for k in required_metric_keys if k not in metrics]
        if missing_keys:
            raise ValueError(f"第{idx+1}个tracking_metrics缺少键：{', '.join(missing_keys)}")

    # 1.3 检查双VLM详细分析字典键完整（与原有GPT-4o格式一致）
    required_analysis_keys = ["action_sequence", "technical_complexity", "movement_intensity", "balance_requirement", "continuity"]
    # 校验GPT-4o VLM分析
    for idx, analysis in enumerate(gpt4o_vlm_analyses):
        missing_keys = [k for k in required_analysis_keys if k not in analysis]
        if missing_keys:
            raise ValueError(f"第{idx+1}个gpt4o_vlm_analyses缺少键：{', '.join(missing_keys)}")
    # 校验Qwen VLM分析（新增）
    for idx, analysis in enumerate(qwen_vlm_analyses):
        missing_keys = [k for k in required_analysis_keys if k not in analysis]
        if missing_keys:
            raise ValueError(f"第{idx+1}个qwen_vlm_analyses缺少键：{', '.join(missing_keys)}")


    # 1. 优化后的武术无级变量库（移除数值/括号，删除scene和direction）
    martial_arts_vars = {
        "base_action": [
            "basic bow stance with static posture and shoulder-width feet",
            "simple cross fist with slow arm swing and minimal hip movement",
            "basic horse stance with steady posture and bent knees",
            "simple palm strike with straight arm and slow retraction",
            "static mountain stance with parallel feet and even weight distribution",
            "slow downward chop with arm swing from shoulder and low speed",
            "basic front kick with knee-height movement and slow extension",
            "standard hook punch with moderate hip rotation and steady strike",
            "front kick with waist-height movement and controlled landing",
            "side stance shift with smooth weight transfer",
            "basic push hands with gentle force redirection and slow reaction",
            "low sweep kick with ankle-height movement and slow leg swing",
            "single whip stance with dynamic weight shift and steady transition",
            "basic elbow strike with short range and moderate force",
            "whirlwind kick with spinning movement, airborne leg extension and rapid rotation",
            "double kick with rapid consecutive leg strikes and no pause",
            "iron bridge with backbend posture, arm support and core tension",
            "snake-like fist with rapid wrist flick and zigzag arm movement",
            "triple spinning kick with consecutive spins and airborne leg extension",
            "flying side kick with airborne lateral movement and focused strike",
            "reverse tornado kick with backward spinning movement and heel strike",
            "leaping tiger claw with forward jump, extended fingers and grappling focus",
            "dragon tail whip with rapid leg swing from low to high and arcing movement",
            "jumping double palm strike with airborne posture and extended body",
            "side flip kick with lateral flip movement, leg strike and no hand support",
            "shadowless kick with rapid low movement and barely visible leg",
            "eight-directional step punch with dynamic stepping and multi-angle strikes",
            "ground sweep to standup with seamless transition from prone to jumping kick"
        ],
        "combo_action": [
            "basic bow stance → simple cross fist → horse stance",
            "front kick (knee-height) → basic palm strike → side stance shift",
            "mountain stance → downward chop → static hold",
            "hook punch → low sweep kick → push hands",
            "side stance shift → elbow strike → single whip stance",
            "front kick (waist-height) → cross fist → backward step",
            "whirlwind kick → landing hook punch → double kick",
            "push hands → snake-like fist → iron bridge transition",
            "triple spinning kick → flying side kick → reverse tornado kick",
            "leaping tiger claw → dragon tail whip → jumping double palm strike",
            "ground sweep → standup → shadowless kick → eight-directional step punch"
        ],
        "detail": [
            "bow stance with bent knees, straight back and forward-facing toes",
            "cross fist with fists crossed at chest and outward-facing knuckles",
            "horse stance with wide foot placement, engaged core and no sway",
            "palm strike with together fingers, flat palm and impact at palm heel",
            "hook punch with bent elbow, force from torso twist and target-focused landing",
            "push hands with relaxed arms, opponent force following and no rigid resistance",
            "sweep kick with straight leg, pointed foot and contact at opponent's lower leg",
            "whirlwind kick with pivot foot rotation, tight core for balance and target-focused gaze",
            "double kick with first strike to thigh, second to waist and no pause between",
            "iron bridge with locked elbows, squeezed shoulder blades and steady breath",
            "triple spinning kick with back foot push for spin initiation and extended legs mid-air",
            "flying side kick with back leg takeoff, straight front leg and landing on takeoff leg",
            "reverse tornado kick with arm swing for spin initiation and heel strike to opponent's torso",
            "shadowless kick with low leg position, instep strike and rapid retraction"
        ],
        "speed_rhythm": [
            "stance transitions with slow movement and no sudden speed change",
            "punching with steady speed and consistent force",
            "stance hold → slow strike → reset with steady pace",
            "alternating speed: slow punch → moderate kick → slow step",
            "reaction with moderate speed in push hands practice",
            "preparation → steady strike → recovery with balanced timing",
            "stance buildup with slow movement → explosive kick → rapid recovery",
            "push hands with slow movement → sudden snake-like fist with burst speed",
            "spinning with accelerating speed: initial slow spin → gradual speed increase",
            "jump-strike with slow approach → explosive takeoff → airborne strike → quick landing",
            "ground-to-air transition with slow sweep → rapid standup → explosive kick"
        ]
    }

    # 2. 优化后的武术无级模板库（移除scene和direction相关内容）
    martial_arts_templates = [
        "The martial artist executed {combo_action} with {detail}, following {speed_rhythm}.",
        "During training, the practitioner performed {base_action} focusing on {detail}, maintaining {speed_rhythm}.",
        "The martial artist completed {combo_action} with {detail} (power control), adhering to {speed_rhythm}.",
        "In the combat sequence, the martial artist transitioned from {combo_action} to {base_action} with {detail}, using {speed_rhythm}.",
        "The practitioner combined {base_action} and {combo_action} with {detail}, synchronizing with {speed_rhythm}.",
        "The routine began with {combo_action}, then shifted into {base_action} with {detail}, regulated by {speed_rhythm}.",
        "During form practice, the practitioner performed {base_action} with {detail}, following {speed_rhythm}.",
        "In dynamic practice, the martial artist executed {base_action} followed by {combo_action} with {detail}, optimized for {speed_rhythm}."
    ]

    # -------------------------- 3. 构建多轮消息（补充VLM难度评分参考） --------------------------
    messages = []

    # -------------------------- 第一轮：核心目标（新增VLM评分作为参考说明） --------------------------
    background_msg = {
        "role": "user",
        "content": [{"type": "text", "text": """
        Core Goal: Generate optimized prompts that push the motion tracker’s performance limit. These prompts must be more challenging than the original ones.

        References for optimization:
        1. Intra-group metric comparison (5 sets vs each other): Identify which actions the tracker handles relatively easily (lower metrics) or struggles with (higher metrics).
        2. Dual VLM feedback (GPT-4o + Qwen): Use their analysis and difficulty scores (0-10, 10 being hardest) of the original actions to guide variable selection from {martial_arts_vars}.
        """.strip()}]  
    }
    messages.append(background_msg)

    # -------------------------- 第二轮：5组数据（保留原有VLM评分展示，无额外修改） --------------------------
    data_summary = ""
    for idx in range(5):
        metrics = tracking_metrics_list[idx]
        gpt4o_analysis = gpt4o_vlm_analyses[idx]
        qwen_analysis = qwen_vlm_analyses[idx]
        
        data_summary += f"""
        [Set {idx+1} Data]
        1. Original Prompt: "{original_prompts[idx]}"
        2. Tracking Metrics:
        - Success: {metrics['success']}
        - mpjpe_g: {metrics['mpjpe_g']} mm | mpjpe_l: {metrics['mpjpe_l']} mm | mpjpe_pa: {metrics['mpjpe_pa']} mm
        - accel_dist: {metrics['accel_dist']} mm/frame² | vel_dist: {metrics['vel_dist']} mm/frame
        3. Dual VLM Score and Analysis:
        - GPT-4o: Action Difficulty Score = {gpt4o_vlm_scores[idx]} (0-10, 10 hardest); Action Sequence={gpt4o_analysis['action_sequence']}; Technical Analysis={gpt4o_analysis['technical_complexity']}; Movement Intensity={gpt4o_analysis['movement_intensity']}; Balance Requirement={gpt4o_analysis['balance_requirement']}; Reason="{gpt4o_vlm_reasons[idx]}"
        - Qwen: Action Difficulty Score = {qwen_vlm_scores[idx]} (0-10, 10 hardest); Action Sequence={qwen_analysis['action_sequence']}; Technical Analysis={qwen_analysis['technical_complexity']};  Movement Intensity={qwen_analysis['movement_intensity']}; Balance Requirement={qwen_analysis['balance_requirement']}; Reason="{qwen_vlm_reasons[idx]}"
        """  # 仅在评分后补充“(0-10, 10 hardest)”说明

    # -------------------------- 第二轮：5组数据（细化难度增加幅度策略） --------------------------
    data_msg = {
        "role": "user",
        "content": [{"type": "text", "text": f"""
        Below are 5 sets of data. Optimize each prompt to be more challenging than the original, using variables from {martial_arts_vars}, with adjustment幅度 based on tracker performance and VLM scores:
        - For actions the tracker handles relatively easily (lower metrics in the group) AND with lower VLM scores: Select variables from {martial_arts_vars} to SIGNIFICANTLY increase difficulty.
        - For actions the tracker handles relatively easily (lower metrics) BUT with higher VLM scores: Select variables to MODERATELY increase difficulty.
        - For actions the tracker struggles with (higher metrics) BUT with lower VLM scores: Select variables to MODERATELY increase difficulty.
        - For actions the tracker struggles with (higher metrics) AND with higher VLM scores: Select variables to SLIGHTLY increase difficulty (closer to the original but still more challenging).

        5 Sets of Data:
        {data_summary.strip()}
        """.strip()}]
    }
    messages.append(data_msg)


    # -------------------------- 第三轮：{martial_arts_vars}库（完全不变） --------------------------
    resource_msg = {
        "role": "user",
        "content": [{"type": "text", "text": f"""
        Use ONLY variables from the following {martial_arts_vars} library and templates to generate optimized prompts:

        1. {martial_arts_vars} Library:
        - base_action: {martial_arts_vars['base_action']}
        - combo_action: {martial_arts_vars['combo_action']}
        - detail: {martial_arts_vars['detail']}
        - speed_rhythm: {martial_arts_vars['speed_rhythm']}

        2. Martial arts Templates:
        {martial_arts_templates}
        """.strip()}]
    }
    messages.append(resource_msg)

    # -------------------------- 第四轮：任务要求（新增VLM评分在分析中的引用） --------------------------
    task_msg = {
        "role": "user",
        "content": [{"type": "text", "text": """
        Your Tasks:
        1. GROUP ANALYSIS:
        A. Tracker Performance Comparison: Compare the 5 sets’ metrics and VLM difficulty scores to identify which are handled relatively easily or with difficulty.
        B. Dual VLM Consensus: Identify common points in GPT-4o and Qwen’s analysis and score trends for each set.

        2. PROMPT OPTIMIZATION (1-to-1):
        For each set (labeled "Optimized Prompt for Set X"):
        A. Original vs Optimized: Explain how the optimized prompt is more challenging than the original, referencing VLM scores.
        B. Variables Selection: Which variables were chosen and why (link to tracker performance, VLM feedback, and scores).
        C. Template Selection: Which template was used and why.
        D. Final Optimized Prompt: English sentence using selected variables and template.

        Output Format:
        1. Group Analysis
        A. Tracker Performance Comparison:
            [Your comparison here, including VLM scores]
        B. Dual VLM Consensus:
            [Your consensus here, including score trends]

        2. Optimized Prompts
        - Optimized Prompt for Set 1:
            A. Original vs Optimized: [Explanation with VLM score reference]
            B. Variables Selection: [Variables + Reason linked to scores]
            C. Template Selection: [Template + Reason]
            D. Prompt: [Final prompt]
        
        - Optimized Prompt for Set 2: [Same structure]
        - ... (all 5 sets)
        """.strip()}]  # 仅在分析和优化逻辑中新增“引用VLM评分”的要求
    }
    messages.append(task_msg)


    # -------------------------- 4. 调用Gemini API（统一模型与鉴权） --------------------------
    return call_generation_api(messages, max_tokens=8000)
def generate_optimized_dance_prompt(    original_prompts, 
    tracking_metrics_list, 
    # GPT-4o VLM 输入（原有）
    gpt4o_vlm_scores, 
    gpt4o_vlm_reasons, 
    gpt4o_vlm_analyses,
    # Qwen VLM 输入（新增，格式与GPT-4o一致）
    qwen_vlm_scores, 
    qwen_vlm_reasons, 
    qwen_vlm_analyses
):
    """
    输入5个格斗动作prompt、跟踪数据及双VLM（GPT-4o + Qwen）反馈，输出5个拔高难度的优化prompt
    参数:
        original_prompts: 5个原始prompt列表（长度必须为5）
        tracking_metrics_list: 5个跟踪指标字典列表（含success/mpjpe_g等，长度必须为5）
        # GPT-4o VLM 参数（原有）
        gpt4o_vlm_scores: GPT-4o VLM评分列表（如["8/10"...]，长度5）
        gpt4o_vlm_reasons: GPT-4o VLM评分理由列表（长度5）
        gpt4o_vlm_analyses: GPT-4o VLM详细分析字典列表（含action_sequence等，长度5）
        # Qwen VLM 参数（新增，格式与GPT-4o一致）
        qwen_vlm_scores: Qwen VLM评分列表（如["7/10"...]，长度5）
        qwen_vlm_reasons: Qwen VLM评分理由列表（长度5）
        qwen_vlm_analyses: Qwen VLM详细分析字典列表（含action_sequence等，长度5）
    返回:
        包含组内分析和5个优化后prompt的列表
    """
    # -------------------------- 1. 参数校验（新增Qwen VLM校验，与GPT-4o格式对齐） --------------------------
    # 整合所有输入列表（含双VLM）
    input_lists = [
        original_prompts, tracking_metrics_list,
        # GPT-4o VLM
        gpt4o_vlm_scores, gpt4o_vlm_reasons, gpt4o_vlm_analyses,
        # Qwen VLM（新增）
        qwen_vlm_scores, qwen_vlm_reasons, qwen_vlm_analyses
    ]
    input_names = [
        "original_prompts", "tracking_metrics_list",
        "gpt4o_vlm_scores", "gpt4o_vlm_reasons", "gpt4o_vlm_analyses",
        "qwen_vlm_scores", "qwen_vlm_reasons", "qwen_vlm_analyses"
    ]
    
    # 1.1 检查所有列表长度为5且一致
    for idx, (lst, name) in enumerate(zip(input_lists, input_names)):
        if len(lst) != 5:
            raise ValueError(f"{name}长度必须为5（当前为{len(lst)}），请输入5个动作的数据")
    if len(set(len(lst) for lst in input_lists)) != 1:
        raise ValueError("所有输入列表长度不一致，请确保每个列表都包含5个元素")

    # 1.2 检查跟踪指标字典键完整（原有逻辑不变）
    required_metric_keys = ["success", "mpjpe_g", "mpjpe_l", "mpjpe_pa", "accel_dist", "vel_dist"]
    for idx, metrics in enumerate(tracking_metrics_list):
        missing_keys = [k for k in required_metric_keys if k not in metrics]
        if missing_keys:
            raise ValueError(f"第{idx+1}个tracking_metrics缺少键：{', '.join(missing_keys)}")

    # 1.3 检查双VLM详细分析字典键完整（与原有GPT-4o格式一致）
    required_analysis_keys = ["action_sequence", "technical_complexity", "movement_intensity", "balance_requirement", "continuity"]
    # 校验GPT-4o VLM分析
    for idx, analysis in enumerate(gpt4o_vlm_analyses):
        missing_keys = [k for k in required_analysis_keys if k not in analysis]
        if missing_keys:
            raise ValueError(f"第{idx+1}个gpt4o_vlm_analyses缺少键：{', '.join(missing_keys)}")
    # 校验Qwen VLM分析（新增）
    for idx, analysis in enumerate(qwen_vlm_analyses):
        missing_keys = [k for k in required_analysis_keys if k not in analysis]
        if missing_keys:
            raise ValueError(f"第{idx+1}个qwen_vlm_analyses缺少键：{', '.join(missing_keys)}")


    # 1. 优化后的舞蹈无级变量库（移除数值/括号，删除scene和direction）
    dance_vars = {
        "base_action": [
            "basic tendu with slow leg slide and small movement",
            "simple plié with gentle knee bend and steady rhythm",
            "basic port de bras with slow arm sweep and no torso twist",
            "simple step with flat foot and minimal weight shift",
            "static arabesque with held posture and minimal sway",
            "slow arm circle with forward rotation and steady movement",
            "heel-toe tap with slow alternating motion and no weight transfer",
            "relevé with slow rise onto toes and controlled balance",
            "chassé with gliding step and moderate speed",
            "pirouette with single turn and steady rotation",
            "grand battement with medium leg lift and controlled descent",
            "sauté with small jump and two-foot takeoff landing",
            "pas de chat with cat-like step and moderate height",
            "chainé turns with continuous slow spins and steady rotation",
            "lunge with arm extension and balanced posture",
            "fouetté turn with rapid spinning, leg flick and high balance demand",
            "grand jeté with leap, split posture and airborne extension",
            "pas de bourrée with quick footwork sequence and fast weight shifts",
            "contraction-release with abrupt torso twist and dynamic core control",
            "triple pirouette with consecutive spins and spotting technique",
            "aerial cartwheel with no-hand airborne flip and rotation",
            "grand allegro with high leap, leg extension and airborne movement",
            "bourrée en pointe with rapid tiptoe steps and light footwork",
            "saut de basque with turning leap, leg swing and full rotation",
            "port de bras with backbend, dynamic spinal arch and synchronized arm sweep",
            "jeté battu with rapid alternating leg beats mid-air",
            "tour en l'air with airborne full rotation and tucked position",
            "pique turn chain with continuous sharp turns on pointe",
            "floorwork roll into jump with seamless transition from prone to aerial"
        ],
        "combo_action": [
            "basic tendu → simple plié → basic port de bras",
            "simple step → relevé → slow arm drop",
            "static arabesque → heel-toe tap → arm circle",
            "sauté → slow lunge → port de bras",
            "chassé → single pirouette → grand battement",
            "chainé turns → pas de chat → relevé hold",
            "lunge with arm extension → sauté → side step",
            "pas de bourrée → fouetté turn → grand jeté",
            "contraction-release → grand battement → chassé → pirouette",
            "triple pirouette → aerial cartwheel → grand allegro",
            "saut de basque → jeté battu → bourrée en pointe",
            "floorwork roll → tour en l'air → pique turn chain → landing plié"
        ],
        "detail": [
            "tendu with pointed toes and lifted heel",
            "arm movement with relaxed shoulders and slightly bent elbows",
            "plié with appropriate depth and evenly distributed weight",
            "arabesque with straight back leg, aligned hips and squared shoulders",
            "step with heel strike first and smooth toe follow-through",
            "pirouette with fixed point spotting, engaged core and no sway",
            "chassé with feet barely leaving ground and smooth gliding motion",
            "sauté with bent knees on takeoff and soft absorption on landing",
            "chainé turns with arms in first position and rotation from torso",
            "fouetté turn with head spotting for direction, engaged core and controlled leg flick",
            "grand jeté with airborne split, fully extended legs and straight back",
            "pas de bourrée with rapid foot taps and minimal upper body movement",
            "triple pirouette with back foot push, maintained turnout and steady rotation",
            "aerial cartwheel with shoulder rotation for takeoff and elevated hips mid-air",
            "grand allegro with extended airborne time, fully stretched legs and balanced posture",
            "tour en l'air with timely tuck initiation and smooth untuck before landing",
            "pique turn chain with pointe foot aligned to hip and arm swing for turn initiation"
        ],
        "speed_rhythm": [
            "steady tempo with consistent movement speed",
            "slow leg movement matched to arm speed",
            "held posture → slow movement → held posture with balanced timing",
            "half-time steps with deliberate transitions",
            "alternating count movements with varied timing",
            "moderate tempo with brief pauses",
            "slow step → medium turn → slow step with smooth transitions",
            "slow contraction → sudden release with speed burst",
            "slow steps → fast pirouette → suspended landing with dynamic contrast",
            "accelerating turns with gradual speed increase",
            "leap sequence with slow preparation → explosive takeoff → suspended mid-air → quick landing",
            "floorwork with slow movement → fast jump → fast turn → held pose"
        ]
    }

    # 2. 优化后的舞蹈无级模板库（移除scene和direction相关内容）
    dance_templates = [
        "The dancer performed {combo_action} with {detail}, following {speed_rhythm}.",
        "During practice, the performer executed {base_action} focusing on {detail}, maintaining {speed_rhythm}.",
        "The dancer completed {combo_action} with {detail} (dynamic control), adhering to {speed_rhythm}.",
        "In the sequence climax, the dancer executed {base_action} followed by {combo_action} with {detail}, using {speed_rhythm}.",
        "The performer combined {base_action} and {combo_action} with {detail}, synchronizing with {speed_rhythm}.",
        "The sequence began with {combo_action}, then shifted into {base_action} with {detail}, regulated by {speed_rhythm}.",
        "During warm-up, the dancer practiced {base_action} with {detail}, following {speed_rhythm}.",
        "In dynamic performance, the dancer executed {base_action} with {detail}, transitioning to {combo_action} with {speed_rhythm}."
    ]

    # -------------------------- 3. 构建多轮消息（补充VLM难度评分参考） --------------------------
    messages = []

    # -------------------------- 第一轮：核心目标（新增VLM评分作为参考说明） --------------------------
    background_msg = {
        "role": "user",
        "content": [{"type": "text", "text": """
        Core Goal: Generate optimized prompts that push the motion tracker’s performance limit. These prompts must be more challenging than the original ones.

        References for optimization:
        1. Intra-group metric comparison (5 sets vs each other): Identify which actions the tracker handles relatively easily (lower metrics) or struggles with (higher metrics).
        2. Dual VLM feedback (GPT-4o + Qwen): Use their analysis and difficulty scores (0-10, 10 being hardest) of the original actions to guide variable selection from {dance_vars}.
        """.strip()}]  
    }
    messages.append(background_msg)

    # -------------------------- 第二轮：5组数据（保留原有VLM评分展示，无额外修改） --------------------------
    data_summary = ""
    for idx in range(5):
        metrics = tracking_metrics_list[idx]
        gpt4o_analysis = gpt4o_vlm_analyses[idx]
        qwen_analysis = qwen_vlm_analyses[idx]
        
        data_summary += f"""
        [Set {idx+1} Data]
        1. Original Prompt: "{original_prompts[idx]}"
        2. Tracking Metrics:
        - Success: {metrics['success']}
        - mpjpe_g: {metrics['mpjpe_g']} mm | mpjpe_l: {metrics['mpjpe_l']} mm | mpjpe_pa: {metrics['mpjpe_pa']} mm
        - accel_dist: {metrics['accel_dist']} mm/frame² | vel_dist: {metrics['vel_dist']} mm/frame
        3. Dual VLM Score and Analysis:
        - GPT-4o: Action Difficulty Score = {gpt4o_vlm_scores[idx]} (0-10, 10 hardest); Action Sequence={gpt4o_analysis['action_sequence']}; Technical Analysis={gpt4o_analysis['technical_complexity']}; Movement Intensity={gpt4o_analysis['movement_intensity']}; Balance Requirement={gpt4o_analysis['balance_requirement']}; Reason="{gpt4o_vlm_reasons[idx]}"
        - Qwen: Action Difficulty Score = {qwen_vlm_scores[idx]} (0-10, 10 hardest); Action Sequence={qwen_analysis['action_sequence']}; Technical Analysis={qwen_analysis['technical_complexity']};  Movement Intensity={qwen_analysis['movement_intensity']}; Balance Requirement={qwen_analysis['balance_requirement']}; Reason="{qwen_vlm_reasons[idx]}"
        """  # 仅在评分后补充“(0-10, 10 hardest)”说明

    # -------------------------- 第二轮：5组数据（细化难度增加幅度策略） --------------------------
    data_msg = {
        "role": "user",
        "content": [{"type": "text", "text": f"""
        Below are 5 sets of data. Optimize each prompt to be more challenging than the original, using variables from {dance_vars}, with adjustment幅度 based on tracker performance and VLM scores:
        - For actions the tracker handles relatively easily (lower metrics in the group) AND with lower VLM scores: Select variables from {dance_vars} to SIGNIFICANTLY increase difficulty.
        - For actions the tracker handles relatively easily (lower metrics) BUT with higher VLM scores: Select variables to MODERATELY increase difficulty.
        - For actions the tracker struggles with (higher metrics) BUT with lower VLM scores: Select variables to MODERATELY increase difficulty.
        - For actions the tracker struggles with (higher metrics) AND with higher VLM scores: Select variables to SLIGHTLY increase difficulty (closer to the original but still more challenging).

        5 Sets of Data:
        {data_summary.strip()}
        """.strip()}]
    }
    messages.append(data_msg)


    # -------------------------- 第三轮：{dance_vars}库（完全不变） --------------------------
    resource_msg = {
        "role": "user",
        "content": [{"type": "text", "text": f"""
        Use ONLY variables from the following {dance_vars} library and templates to generate optimized prompts:

        1. {dance_vars} Library:
        - base_action: {dance_vars['base_action']}
        - combo_action: {dance_vars['combo_action']}
        - detail: {dance_vars['detail']}
        - speed_rhythm: {dance_vars['speed_rhythm']}

        2. Dance Templates:
        {dance_templates}
        """.strip()}]
    }
    messages.append(resource_msg)

    # -------------------------- 第四轮：任务要求（新增VLM评分在分析中的引用） --------------------------
    task_msg = {
        "role": "user",
        "content": [{"type": "text", "text": """
        Your Tasks:
        1. GROUP ANALYSIS:
        A. Tracker Performance Comparison: Compare the 5 sets’ metrics and VLM difficulty scores to identify which are handled relatively easily or with difficulty.
        B. Dual VLM Consensus: Identify common points in GPT-4o and Qwen’s analysis and score trends for each set.

        2. PROMPT OPTIMIZATION (1-to-1):
        For each set (labeled "Optimized Prompt for Set X"):
        A. Original vs Optimized: Explain how the optimized prompt is more challenging than the original, referencing VLM scores.
        B. Variables Selection: Which variables were chosen and why (link to tracker performance, VLM feedback, and scores).
        C. Template Selection: Which template was used and why.
        D. Final Optimized Prompt: English sentence using selected variables and template.

        Output Format:
        1. Group Analysis
        A. Tracker Performance Comparison:
            [Your comparison here, including VLM scores]
        B. Dual VLM Consensus:
            [Your consensus here, including score trends]

        2. Optimized Prompts
        - Optimized Prompt for Set 1:
            A. Original vs Optimized: [Explanation with VLM score reference]
            B. Variables Selection: [Variables + Reason linked to scores]
            C. Template Selection: [Template + Reason]
            D. Prompt: [Final prompt]
        
        - Optimized Prompt for Set 2: [Same structure]
        - ... (all 5 sets)
        """.strip()}]  # 仅在分析和优化逻辑中新增“引用VLM评分”的要求
    }
    messages.append(task_msg)


    # -------------------------- 4. 调用Gemini API（统一模型与鉴权） --------------------------
    return call_generation_api(messages, max_tokens=8000)
def generate_optimized_combat_prompts(
    original_prompts, 
    tracking_metrics_list, 
    # GPT-4o VLM 输入（原有）
    gpt4o_vlm_scores, 
    gpt4o_vlm_reasons, 
    gpt4o_vlm_analyses,
    # Qwen VLM 输入（新增，格式与GPT-4o一致）
    qwen_vlm_scores, 
    qwen_vlm_reasons, 
    qwen_vlm_analyses
):
    """
    输入5个格斗动作prompt、跟踪数据及双VLM（GPT-4o + Qwen）反馈，输出5个拔高难度的优化prompt
    参数:
        original_prompts: 5个原始prompt列表（长度必须为5）
        tracking_metrics_list: 5个跟踪指标字典列表（含success/mpjpe_g等，长度必须为5）
        # GPT-4o VLM 参数（原有）
        gpt4o_vlm_scores: GPT-4o VLM评分列表（如["8/10"...]，长度5）
        gpt4o_vlm_reasons: GPT-4o VLM评分理由列表（长度5）
        gpt4o_vlm_analyses: GPT-4o VLM详细分析字典列表（含action_sequence等，长度5）
        # Qwen VLM 参数（新增，格式与GPT-4o一致）
        qwen_vlm_scores: Qwen VLM评分列表（如["7/10"...]，长度5）
        qwen_vlm_reasons: Qwen VLM评分理由列表（长度5）
        qwen_vlm_analyses: Qwen VLM详细分析字典列表（含action_sequence等，长度5）
    返回:
        包含组内分析和5个优化后prompt的列表
    """
    # -------------------------- 1. 参数校验（新增Qwen VLM校验，与GPT-4o格式对齐） --------------------------
    # 整合所有输入列表（含双VLM）
    input_lists = [
        original_prompts, tracking_metrics_list,
        # GPT-4o VLM
        gpt4o_vlm_scores, gpt4o_vlm_reasons, gpt4o_vlm_analyses,
        # Qwen VLM（新增）
        qwen_vlm_scores, qwen_vlm_reasons, qwen_vlm_analyses
    ]
    input_names = [
        "original_prompts", "tracking_metrics_list",
        "gpt4o_vlm_scores", "gpt4o_vlm_reasons", "gpt4o_vlm_analyses",
        "qwen_vlm_scores", "qwen_vlm_reasons", "qwen_vlm_analyses"
    ]
    
    # 1.1 检查所有列表长度为5且一致
    for idx, (lst, name) in enumerate(zip(input_lists, input_names)):
        if len(lst) != 5:
            raise ValueError(f"{name}长度必须为5（当前为{len(lst)}），请输入5个动作的数据")
    if len(set(len(lst) for lst in input_lists)) != 1:
        raise ValueError("所有输入列表长度不一致，请确保每个列表都包含5个元素")

    # 1.2 检查跟踪指标字典键完整（原有逻辑不变）
    required_metric_keys = ["success", "mpjpe_g", "mpjpe_l", "mpjpe_pa", "accel_dist", "vel_dist"]
    for idx, metrics in enumerate(tracking_metrics_list):
        missing_keys = [k for k in required_metric_keys if k not in metrics]
        if missing_keys:
            raise ValueError(f"第{idx+1}个tracking_metrics缺少键：{', '.join(missing_keys)}")

    # 1.3 检查双VLM详细分析字典键完整（与原有GPT-4o格式一致）
    required_analysis_keys = ["action_sequence", "technical_complexity", "movement_intensity", "balance_requirement", "continuity"]
    # 校验GPT-4o VLM分析
    for idx, analysis in enumerate(gpt4o_vlm_analyses):
        missing_keys = [k for k in required_analysis_keys if k not in analysis]
        if missing_keys:
            raise ValueError(f"第{idx+1}个gpt4o_vlm_analyses缺少键：{', '.join(missing_keys)}")
    # 校验Qwen VLM分析（新增）
    for idx, analysis in enumerate(qwen_vlm_analyses):
        missing_keys = [k for k in required_analysis_keys if k not in analysis]
        if missing_keys:
            raise ValueError(f"第{idx+1}个qwen_vlm_analyses缺少键：{', '.join(missing_keys)}")


    # -------------------------- 2. 格斗变量库和模板库（保持不变） --------------------------
    combat_vars = {
        "base_action": [
            "basic jab with slow rhythm and small arm movement",
            "simple slip with small lateral shift and no sudden speed change",
            "basic footwork slide with steady pace and shoulder-width stance",
            "basic parry with minor arm adjustment and low force",
            "straight punch with no hip rotation and slow extension",
            "simple knee tap with low height and minimal core engagement",
            "static guard with arms up and no movement",
            "slow front kick with thigh-height movement and no follow-through",
            "standard cross with moderate hip rotation and steady speed",
            "basic roundhouse kick with small hip swing and flat foot landing",
            "basic full mount with static control and no rapid transition",
            "hook punch with bent arm and moderate torque",
            "side kick with mid-thigh height and controlled retraction",
            "half guard sweep with slow weight shift",
            "elbow strike with short range and moderate force",
            "rear naked choke with slow arm wrapping and no immediate pressure",
            "front headlock with static hold and no takedown attempt",
            "spinning back fist with rotational burst and fast arm swing",
            "flying knee with airborne explosion and core tension for balance",
            "jump switch kick with mid-air leg switch and hip alignment control",
            "armbar from closed guard with rapid joint lock and elbow pressure focus",
            "superman punch with body extension burst and airborne reach",
            "wheel kick with full leg rotation and wide swing",
            "triangle choke with rapid leg entanglement and neck pressure",
            "double-leg takedown with explosive drive and low center of gravity",
            "switch kick feint to spinning heel strike with feint and full rotation",
            "jumping switch knee with double-leg airborne movement and core locked for balance",
            "flying armbar with airborne diving movement, body folded mid-air and rapid joint control",
            "reverse spinning elbow with turning movement, rotation driven by shoulders and quick reset after impact",
            "leaping guillotine choke with forward diving movement, arms cinched instantly and body weight applied",
            "540° tornado kick with single-leg takeoff, leg arcing during rotation",
            "ankle lock from open guard with rapid lock and heel pressure plus body torsion",
            "diving double punch with forward diving movement, body leaned forward and quick transition to defensive stance",
            "spinning back kick to the body with turning movement, hip hyperextension and body rotation for reset",
            "cartwheel guard pass with cartwheel-style movement, instant arm support for force and crossed legs to pass guard"
        ],
        "combo_action": [
            "basic jab → standard cross → standard retreat step",
            "simple parry → basic hook → simple slip",
            "straight punch → knee tap → footwork slide",
            "static guard → elbow strike → side step",
            "hook punch → roundhouse kick → half guard transition",
            "side kick → cross punch → backward shuffle",
            "parry → elbow strike → forward lunge",
            "front headlock → hip toss → mount",
            "simple slip → superman punch → flying knee → pivot escape",
            "cartwheel kick → landing slide → spinning back fist → weave",
            "double-leg takedown → mount → armbar → sweep",
            "wheel kick → landing spin → jump switch kick → defensive guard",
            "switch kick feint → spinning heel kick → diving double punch → retreat roll",
            "leaping guillotine → ankle lock → standup → 540° tornado kick"
        ],
        "detail": [
            "fist form with aligned knuckles and locked wrist during punch",
            "footwork with parallel feet and shoulder-width stance during slide",
            "guard position with tucked elbows and vertical forearms",
            "knee tap with flexed hip and flat foot before impact",
            "punch retraction with speed matching extension and no lag",
            "roundhouse kick with rotated supporting foot and raised knee first",
            "hook punch with elbow kept at right angle and force from torso twist",
            "half guard with legs wrapped above knee and applied hip pressure",
            "rear naked choke with forearm across windpipe and bicep pressed against jaw",
            "flying knee with tight core and body lean mid-air to maintain balance",
            "spinning back fist with eyes locked on target and heel-first landing for buffering",
            "superman punch with rear leg driving forward and extended torso",
            "triangle choke with crossed legs at ankles and elevated hip",
            "double-leg takedown with head pressed into opponent's chest and arms wrapped behind knees",
            "540° tornado kick with pivoted takeoff foot, arms swung for momentum and eyes fixed on target until landing",
            "flying armbar with body forming a 'C' shape mid-air, hips thrust immediately after arm lock and elbow joint kept perpendicular to ground",
            "jumping switch knee with knees tucked inward during leg switch, forefoot used for buffering on landing and center of gravity shifted to striking leg",
            "reverse spinning elbow with shoulders leading hip rotation during turn, elbow kept level with ears and arm tensed at impact"
        ],
        "speed_rhythm": [
            "steady punching speed with consistent movement pace",
            "slow footwork with no sudden direction changes",
            "guard hold → slow punch → reset with balanced timing",
            "alternating speed: slow jab → medium cross → medium hook",
            "slow step → moderate kick → slow retreat",
            "stance hold → strike → recovery with steady timing",
            "slow slide → explosive punch → immediate stop post-impact",
            "slow pre-spin windup → speed burst during spinning back fist → fast defensive reset",
            "takedown drive with acceleration → sudden weight shift → submission lock",
            "slow feint → fast real strike → rapid direction change",
            "airborne movement with takeoff → mid-air posture control → landing buffer → quick transition to next move"
        ]
    }

    combat_templates = [
        "The fighter executed {combo_action} with {detail}, following {speed_rhythm}.",
        "During training, the combatant performed {base_action} focusing on {detail}, maintaining {speed_rhythm}.",
        "The fighter completed {combo_action} at {speed_rhythm}, with {detail} (precision control).",
        "The combatant launched {base_action} with {detail}, maintaining {speed_rhythm}.",
        "The fighter combined {base_action} and {combo_action}, using {detail} to optimize {speed_rhythm}.",
        "The combatant transitioned from {combo_action} to {base_action} with {detail}, guiding {speed_rhythm}."
    ]


    # -------------------------- 3. 构建多轮消息（补充VLM难度评分参考） --------------------------
    messages = []

    # -------------------------- 第一轮：核心目标（新增VLM评分作为参考说明） --------------------------
    background_msg = {
        "role": "user",
        "content": [{"type": "text", "text": """
        Core Goal: Generate optimized prompts that push the motion tracker’s performance limit. These prompts must be more challenging than the original ones.

        References for optimization:
        1. Intra-group metric comparison (5 sets vs each other): Identify which actions the tracker handles relatively easily (lower metrics) or struggles with (higher metrics).
        2. Dual VLM feedback (GPT-4o + Qwen): Use their analysis and difficulty scores (0-10, 10 being hardest) of the original actions to guide variable selection from {combat_vars}.
        """.strip()}]  
    }
    messages.append(background_msg)

    # -------------------------- 第二轮：5组数据（保留原有VLM评分展示，无额外修改） --------------------------
    data_summary = ""
    for idx in range(5):
        metrics = tracking_metrics_list[idx]
        gpt4o_analysis = gpt4o_vlm_analyses[idx]
        qwen_analysis = qwen_vlm_analyses[idx]
        
        data_summary += f"""
        [Set {idx+1} Data]
        1. Original Prompt: "{original_prompts[idx]}"
        2. Tracking Metrics:
        - Success: {metrics['success']}
        - mpjpe_g: {metrics['mpjpe_g']} mm | mpjpe_l: {metrics['mpjpe_l']} mm | mpjpe_pa: {metrics['mpjpe_pa']} mm
        - accel_dist: {metrics['accel_dist']} mm/frame² | vel_dist: {metrics['vel_dist']} mm/frame
        3. Dual VLM Score and Analysis:
        - GPT-4o: Action Difficulty Score = {gpt4o_vlm_scores[idx]} (0-10, 10 hardest); Action Sequence={gpt4o_analysis['action_sequence']}; Technical Analysis={gpt4o_analysis['technical_complexity']}; Movement Intensity={gpt4o_analysis['movement_intensity']}; Balance Requirement={gpt4o_analysis['balance_requirement']}; Reason="{gpt4o_vlm_reasons[idx]}"
        - Qwen: Action Difficulty Score = {qwen_vlm_scores[idx]} (0-10, 10 hardest); Action Sequence={qwen_analysis['action_sequence']}; Technical Analysis={qwen_analysis['technical_complexity']};  Movement Intensity={qwen_analysis['movement_intensity']}; Balance Requirement={qwen_analysis['balance_requirement']}; Reason="{qwen_vlm_reasons[idx]}"
        """  # 仅在评分后补充“(0-10, 10 hardest)”说明

    # -------------------------- 第二轮：5组数据（细化难度增加幅度策略） --------------------------
    data_msg = {
        "role": "user",
        "content": [{"type": "text", "text": f"""
        Below are 5 sets of data. Optimize each prompt to be more challenging than the original, using variables from {combat_vars}, with adjustment幅度 based on tracker performance and VLM scores:
        - For actions the tracker handles relatively easily (lower metrics in the group) AND with lower VLM scores: Select variables from {combat_vars} to SIGNIFICANTLY increase difficulty.
        - For actions the tracker handles relatively easily (lower metrics) BUT with higher VLM scores: Select variables to MODERATELY increase difficulty.
        - For actions the tracker struggles with (higher metrics) BUT with lower VLM scores: Select variables to MODERATELY increase difficulty.
        - For actions the tracker struggles with (higher metrics) AND with higher VLM scores: Select variables to SLIGHTLY increase difficulty (closer to the original but still more challenging).

        5 Sets of Data:
        {data_summary.strip()}
        """.strip()}]
    }
    messages.append(data_msg)


    # -------------------------- 第三轮：{combat_vars}库（完全不变） --------------------------
    resource_msg = {
        "role": "user",
        "content": [{"type": "text", "text": f"""
        Use ONLY variables from the following {combat_vars} library and templates to generate optimized prompts:

        1. {combat_vars} Library:
        - base_action: {combat_vars['base_action']}
        - combo_action: {combat_vars['combo_action']}
        - detail: {combat_vars['detail']}
        - speed_rhythm: {combat_vars['speed_rhythm']}

        2. Combat Templates:
        {combat_templates}
        """.strip()}]
    }
    messages.append(resource_msg)

    # -------------------------- 第四轮：任务要求（新增VLM评分在分析中的引用） --------------------------
    task_msg = {
        "role": "user",
        "content": [{"type": "text", "text": """
        Your Tasks:
        1. GROUP ANALYSIS:
        A. Tracker Performance Comparison: Compare the 5 sets’ metrics and VLM difficulty scores to identify which are handled relatively easily or with difficulty.
        B. Dual VLM Consensus: Identify common points in GPT-4o and Qwen’s analysis and score trends for each set.

        2. PROMPT OPTIMIZATION (1-to-1):
        For each set (labeled "Optimized Prompt for Set X"):
        A. Original vs Optimized: Explain how the optimized prompt is more challenging than the original, referencing VLM scores.
        B. Variables Selection: Which variables were chosen and why (link to tracker performance, VLM feedback, and scores).
        C. Template Selection: Which template was used and why.
        D. Final Optimized Prompt: English sentence using selected variables and template.

        Output Format:
        1. Group Analysis
        A. Tracker Performance Comparison:
            [Your comparison here, including VLM scores]
        B. Dual VLM Consensus:
            [Your consensus here, including score trends]

        2. Optimized Prompts
        - Optimized Prompt for Set 1:
            A. Original vs Optimized: [Explanation with VLM score reference]
            B. Variables Selection: [Variables + Reason linked to scores]
            C. Template Selection: [Template + Reason]
            D. Prompt: [Final prompt]
        
        - Optimized Prompt for Set 2: [Same structure]
        - ... (all 5 sets)
        """.strip()}]  # 仅在分析和优化逻辑中新增“引用VLM评分”的要求
    }
    messages.append(task_msg)




    # -------------------------- 4. 调用Gemini API（统一模型与鉴权） --------------------------
    return call_generation_api(messages, max_tokens=2000)



# -------------------------- 主执行流程（全自动化处理） --------------------------
if __name__ == "__main__":
    args = parse_args()

    if args.input_csv and args.output_dir:
        result = process_release_loop(
            input_csv=args.input_csv,
            output_dir=args.output_dir,
            group_size=args.group_size,
            seed=args.seed,
            target_count=args.target_count,
            prompt_manifest_path=args.prompt_manifest,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        raise SystemExit("Please provide --input-csv and --output-dir for release usage.")
