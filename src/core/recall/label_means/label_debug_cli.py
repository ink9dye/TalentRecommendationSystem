import traceback

import faiss

from src.core.recall.label_path import LabelRecallPath


def run_label_debug_cli() -> None:
    # 开启 verbose 以输出完整 Step2 / bridge 诊断日志
    l_path = LabelRecallPath(recall_limit=200, verbose=True)
    encoder = l_path._query_encoder

    try:
        domain_choice = input("\n请选择领域编号 (0跳过): ").strip() or "0"

        while True:
            user_input = input("\n请输入岗位需求 (q退出): ").strip()
            if not user_input or user_input.lower() == "q":
                break

            query_vec, _ = encoder.encode(user_input)
            faiss.normalize_L2(query_vec)

            top_ids, search_time = l_path.recall(query_vec, domain_id=domain_choice, query_text=user_input)

            db = l_path.last_debug_info
            print("\n" + "🔍 [深度诊断流水线]" + "-" * 98)

            domains = db.get("active_domains", [])
            domain_str = " | ".join(domains) if domains else "未限制"
            print(f"【Step 1: 领域探测】目标领域: [{domain_str}] 置信度: {db.get('dominance')}")
            job_previews = db.get("job_previews", [])
            if job_previews:
                print("      Top5 岗位名称:")
                for i, jp in enumerate(job_previews[:5], 1):
                    print(f"        #{i} {(jp.get('name') or '')[:55]}")

            i_kws = db.get("industrial_kws", [])
            melt_stats = db.get("anchor_melt_stats") or {}
            print(
                f"【Step 2: 工业锚点】技能数: {len(i_kws)}  熔断前={melt_stats.get('before_melt', 0)} 熔断后={melt_stats.get('after_melt', 0)} "
                f"Top30后={melt_stats.get('after_top30', 0)} 最终锚点数: {len(db.get('anchor_detail', []))}"
            )

            CORE_KEYWORDS = ["动力学", "运动学", "轨迹规划", "状态估计", "最优控制"]
            print("【锚点主干词存活检查】")
            for kw in CORE_KEYWORDS:
                in_cleaned = any(kw in t for t in melt_stats.get("cleaned_terms_sample", []))
                in_before = any(kw in t for t in melt_stats.get("terms_before_melt", []))
                in_after_melt = any(kw in t for t in melt_stats.get("terms_after_melt", []))
                in_after_top30 = any(kw in t for t in melt_stats.get("terms_after_top30", []))
                in_after_sim = any(kw in t for t in melt_stats.get("terms_after_sim", []))
                in_final = any(kw in t for t in i_kws)
                reason = (
                    "in_final"
                    if in_final
                    else (
                        "sim_drop"
                        if in_after_top30 and not in_after_sim
                        else (
                            "topk_drop"
                            if in_after_melt and not in_after_top30
                            else ("melt_fail" if in_before and not in_after_melt else ("clean_fail" if not in_before else "?"))
                        )
                    )
                )
                print(
                    f"  {kw}: cleaned={in_cleaned} before_melt={in_before} after_melt={in_after_melt} after_top30={in_after_top30} "
                    f"after_sim={in_after_sim} final_anchor={in_final} 原因={reason}"
                )

            fcl = db.get("filter_closed_loop") or {}
            raw_tids = fcl.get("similar_to_raw_tids", [])
            pass_tids = fcl.get("similar_to_pass_tids", [])
            final_tids = fcl.get("final_term_ids_for_paper", [])
            n_final = fcl.get("final_term_count", 0)
            print("【词过滤闭环】")
            print(f"  similar_to_raw_tids 数量: {len(raw_tids)}  前30: {raw_tids[:30]}")
            print(f"  similar_to_pass_tids 数量: {len(pass_tids)}  前30: {pass_tids[:30]}")
            print(f"  final_term_ids_for_paper 数量: {n_final}  前30: {final_tids[:30]}")
            for label, in_final in (fcl.get("contains_check") or {}).items():
                print(f"  contains {label}: {in_final}")

            top_contrib = db.get("top_terms_final_contrib") or []
            if top_contrib:
                print(
                    "【Top term 最终贡献表】term | tid | final_weight | main_role | role_penalty | paper_count_hit | top_paper_contrib | task_advantage"
                )
                for r in top_contrib[:20]:
                    ta = r.get("task_advantage")
                    ta_s = f"{ta:.3f}" if ta is not None else "-"
                    print(
                        f"  {(r.get('term') or '')[:28]:28s} | {r.get('tid')} | {r.get('final_weight', 0):.4f} | "
                        f"{(r.get('main_role') or '')[:12]:12s} | {r.get('role_penalty') or 0:.3f} | {r.get('paper_count_hit', 0)} | "
                        f"{r.get('top_paper_contrib', 0):.6f} | {ta_s}"
                    )

            vocab_count = db.get("recall_vocab_count", 0)
            w_count = db.get("work_count", 0)
            a_count = db.get("author_count", 0)
            print(f"【Step 4: 召回规模】参与检索学术词数: {vocab_count}  检索论文: {w_count}  锁定作者: {a_count}")

            print("【Top20 作者来源】author_id | final_score | top_terms_by_contribution(前3) | best_paper | best_paper_terms")
            for i, item in enumerate(db.get("top_samples", [])[:20], 1):
                aid = item.get("aid", "")
                score = item.get("score", 0)
                top_terms = item.get("top_terms_by_contribution", [])[:3]
                terms_s = ", ".join(f"{t}({c})" for t, c in top_terms) if top_terms else "-"
                tp = item.get("top_paper", {}) or {}
                best_title = (tp.get("title") or "")[:40]
                best_hits = ", ".join((tp.get("hits") or [])[:3])
                print(f"  #{i:2d} {aid} | {score:.4f} | {terms_s} | 《{best_title}》 | {best_hits}")

            print("-" * 98)
            print(f"{'排名':<6} | {'作者 ID':<14} | {'得分':<10} | {'代表作 (命中标签)'}")
            print("-" * 98)
            for i, item in enumerate(db.get("top_samples", [])[:30], 1):
                aid = item.get("aid", "")
                score = item.get("score", 0)
                tp = item.get("top_paper", {}) or {}
                title = (tp.get("title") or "")[:45]
                hit_tags = ", ".join((tp.get("hits") or [])[:4])
                top_terms = item.get("top_terms_by_contribution", [])[:2]
                src = ", ".join(f"{t}({c})" for t, c in top_terms) if top_terms else ""
                print(f"#{i:<5} | {aid:<14} | {score:.4f}    | 《{title}》 命中: {hit_tags}  来源: {src}")
            print("-" * 98)
            print(f"[*] 诊断完成。全链路耗时: {search_time:.2f}ms")

    except Exception as e:
        print(f"运行出错: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    run_label_debug_cli()

