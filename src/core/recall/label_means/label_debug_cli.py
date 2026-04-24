import os
import traceback
from typing import Optional

import faiss

from src.core.recall.label_path import LabelRecallPath
from src.utils.tools import extract_skills, SKILL_SPLIT_PATTERN, normalize_skill, split_space_terms, is_bad_skill

LABEL_DEBUG_CLI_EXTRA_TABLES = False  # 学术词命运表/Stage3 来源回溯表：默认关闭


def run_single_label_debug(
    l_path: LabelRecallPath,
    domain_choice: str,
    verbose: bool,
    raw_text: str,
    semantic_text: Optional[str] = None,
) -> None:
    if semantic_text is None:
        semantic_text = raw_text
    encoder = l_path._query_encoder

    if verbose:
        parts = [p for p in SKILL_SPLIT_PATTERN.split(raw_text) if p and p.strip()]
        normalized_terms = []
        final_skills = []
        for p in parts:
            norm = normalize_skill(p)
            if not norm:
                continue
            normalized_terms.append(norm)
            for sub in split_space_terms(norm):
                if is_bad_skill(sub):
                    continue
                final_skills.append(sub)
        final_skills = list(set(final_skills))
        print("【JD 技能清洗链路】")
        print(f"  raw_phrases 样本({min(30, len(parts))}/{len(parts)}): {parts[:30]}")
        print(f"  normalized_phrases 样本({min(30, len(normalized_terms))}/{len(normalized_terms)}): {normalized_terms[:30]}")
        print(f"  cleaned_skills(最终技能短语) 样本({min(50, len(final_skills))}/{len(final_skills)}): {final_skills[:50]}")

    query_vec, _ = encoder.encode(semantic_text)
    faiss.normalize_L2(query_vec)

    top_ids, search_time = l_path.recall(
        query_vec,
        domain_id=domain_choice,
        query_text=raw_text,
        semantic_query_text=semantic_text,
    )

    db = getattr(l_path, "last_debug_info", None) or {}

    if verbose:
        # Windows 默认控制台编码可能为 GBK；避免输出 emoji 导致编码异常
        print("\n" + "[深度诊断流水线]" + "-" * 100)
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
        final_anchor_count = len(i_kws)
        print(
            f"【Step 2: 工业锚点】技能数: {len(i_kws)}  熔断前={melt_stats.get('before_melt', 0)} 熔断后={melt_stats.get('after_melt', 0)} "
            f"Top30后={melt_stats.get('after_top30', 0)} 最终锚点数: {final_anchor_count}"
        )

        # 通用版锚点存活简表：展示前若干个最终锚点在各阶段的流转情况，便于跨领域观察
        cleaned_sample = melt_stats.get("cleaned_terms_sample", [])
        before_melt = melt_stats.get("terms_before_melt", [])
        after_melt = melt_stats.get("terms_after_melt", [])
        after_top30 = melt_stats.get("terms_after_top30", [])
        after_sim = melt_stats.get("terms_after_sim", [])
        print("【锚点存活简表】term | cleaned | before_melt | after_melt | after_top30 | after_sim | final_anchor")
        for term in i_kws[: min(10, len(i_kws))]:
            cleaned = any(term == t for t in cleaned_sample)
            b = any(term == t for t in before_melt)
            am = any(term == t for t in after_melt)
            at = any(term == t for t in after_top30)
            s = any(term == t for t in after_sim)
            print(
                f"  {term[:24]:24s} | {cleaned!s:>6} | {b!s:>11} | {am!s:>11} | {at!s:>12} | {s!s:>9} | {True:>12}"
            )

        # Stage2 锚点-候选证据表（数据驱动 primary 打分，无硬编码词表）
        stage2_evidence = db.get("stage2_anchor_evidence_table") or []
        if stage2_evidence:
            print("【Stage2 锚点-候选证据表】anchor | candidate(tid) | edge_affinity | anchor_align | jd_align | hierarchy_cons | neighborhood_cons | isolation_penalty | primary_score")
            for row in stage2_evidence[:35]:
                anc = str(row.get("anchor") or "")[:14]
                cand = str(row.get("candidate") or "")[:22]
                tid = row.get("tid", "")
                print(f"  {anc:14s} | {cand!r}({tid}) | edge={row.get('edge_affinity', 0):.3f} | anchor_align={row.get('anchor_align', 0):.3f} | jd_align={row.get('jd_align', 0.5):.3f} | hier={row.get('hierarchy_consistency', 0):.3f} | neigh={row.get('neighborhood_consistency', 0.5):.3f} | isol={row.get('isolation_penalty', 0):.3f} | primary={row.get('primary_score', 0):.3f}")
            if len(stage2_evidence) > 35:
                print(f"  ... 共 {len(stage2_evidence)} 条")

        fcl = db.get("filter_closed_loop") or {}
        raw_tids = fcl.get("similar_to_raw_tids", [])
        pass_tids = fcl.get("similar_to_pass_tids", [])
        final_tids = fcl.get("final_term_ids_for_paper", [])
        n_final = fcl.get("final_term_count", 0)
        # Stage2 三路候选 Top20（edge / ctx / merged），统一表头：tid | term | sim_score | source/origin | degree_w | domain_span
        di = getattr(l_path, "debug_info", None)
        if di:
            for name, attr in [
                ("raw_edge Top20", "stage2_raw_edge_top20"),
                ("raw_ctx Top20", "stage2_raw_ctx_top20"),
                ("raw_merged Top20", "stage2_raw_merged_top20"),
            ]:
                rows = getattr(di, attr, None) or []
                # 降噪：空表不再打印表头，避免重复“空块”干扰诊断主线
                if not rows:
                    continue
                print(f"【Stage2 {name}】tid | term | sim_score | source/origin | degree_w | domain_span")
                for r in rows[:20]:
                    term = (r.get("term") or "")[:28]
                    sim = r.get("sim_score", 0)
                    src = r.get("source") or r.get("origin") or "-"
                    deg_w = r.get("degree_w", 0)
                    d_span = r.get("domain_span", 0)
                    print(f"  {r.get('tid')} | {term:28s} | {sim:.4f} | {src:14s} | {deg_w:>7} | {d_span:>10}")

        print("【词过滤闭环】")
        print(f"  similar_to_raw_tids 数量: {len(raw_tids)}  前30: {raw_tids[:30]}")
        print(f"  similar_to_pass_tids 数量: {len(pass_tids)}  前30: {pass_tids[:30]}")
        print(f"  final_term_ids_for_paper 数量: {n_final}  前30: {final_tids[:30]}")
        # contains_check 已在 Stage5 默认降级，避免重复输出低收益闭环信息

        top_contrib = db.get("top_terms_final_contrib") or []
        if top_contrib:
            paper_tid_set = {str(x) for x in (fcl.get("final_term_ids_for_paper") or [])}
            _show_paper_contrib_cols = os.environ.get("STAGE5_TERM_CONTRIB_DEBUG") == "1"

            def _tid_in_paper_set(tid_val) -> bool:
                if tid_val is None:
                    return False
                return str(tid_val) in paper_tid_set

            def _print_contrib_table(title: str, rows, limit: int = 20) -> None:
                if not rows:
                    return
                if _show_paper_contrib_cols:
                    print(
                        f"{title}term | tid | final_weight | main_role | role_penalty | paper_count_hit | top_paper_contrib | task_advantage"
                    )
                else:
                    print(
                        f"{title}term | tid | final_weight | main_role | role_penalty | task_advantage"
                    )
                for r in rows[:limit]:
                    ta = r.get("task_advantage")
                    ta_s = f"{ta:.3f}" if ta is not None else "-"
                    if _show_paper_contrib_cols:
                        print(
                            f"  {(r.get('term') or '')[:28]:28s} | {r.get('tid')} | {r.get('final_weight', 0):.4f} | "
                            f"{(r.get('main_role') or '')[:12]:12s} | {r.get('role_penalty') or 0:.3f} | {r.get('paper_count_hit', 0)} | "
                            f"{r.get('top_paper_contrib', 0):.6f} | {ta_s}"
                        )
                    else:
                        print(
                            f"  {(r.get('term') or '')[:28]:28s} | {r.get('tid')} | {r.get('final_weight', 0):.4f} | "
                            f"{(r.get('main_role') or '')[:12]:12s} | {r.get('role_penalty') or 0:.3f} | {ta_s}"
                        )

            in_paper = [r for r in top_contrib if _tid_in_paper_set(r.get("tid"))]
            not_in_paper = [r for r in top_contrib if not _tid_in_paper_set(r.get("tid"))]
            print("【Top term 最终贡献表】按是否进入 final_term_ids_for_paper 拆分（避免高分未入 paper 看起来像有效贡献）")
            _print_contrib_table("  [实际进论文检索] ", in_paper)
            # 批注：未进 paper 段只保留高分前 8 行，避免与主线诊断无关的长表。
            _print_contrib_table("  [Stage3/5 高分但未进论文检索] ", not_in_paper, limit=8)

            # 学术 Top term 命运表：查看这些学术词在 Stage2/Stage3 各环节的流转情况
            # 统一用 int 做 membership，避免 tid 为 str 而列表为 int（或反之）导致误判
            fcl_inner = db.get("filter_closed_loop") or {}

            def _to_int_set(lst):
                s = set()
                for x in (lst or []):
                    try:
                        s.add(int(x))
                    except (TypeError, ValueError):
                        pass
                return s

            raw_tids = _to_int_set(fcl_inner.get("similar_to_raw_tids"))
            pass_tids = _to_int_set(fcl_inner.get("similar_to_pass_tids"))
            final_tids = _to_int_set(fcl_inner.get("final_term_ids_for_paper"))
            if LABEL_DEBUG_CLI_EXTRA_TABLES:
                print("【学术词命运表】tid | term | in_similar_raw | in_similar_pass | in_final_paper")
                for r in top_contrib[:20]:
                    tid = r.get("tid")
                    term = (r.get("term") or "")[:28]
                    tid_int = int(tid) if tid is not None else None
                    in_raw = tid_int in raw_tids if tid_int is not None else False
                    in_pass = tid_int in pass_tids if tid_int is not None else False
                    in_final = tid_int in final_tids if tid_int is not None else False
                    print(
                        f"  {tid:<6} | {term:28s} | {str(in_raw):>13} | {str(in_pass):>15} | {str(in_final):>13}"
                    )

            # Stage3 来源回溯表：Top 学术词的 source + tag_purity / cos_sim / anchor_sim，便于判断坏词来自 edge 还是 ctx
            di2 = getattr(l_path, "debug_info", None)
            if di2 and LABEL_DEBUG_CLI_EXTRA_TABLES:
                exp_raw = getattr(di2, "expansion_raw_results", None) or []
                tag_debug = getattr(di2, "tag_purity_debug", None) or []
                source_by_tid = {}
                for rec in exp_raw:
                    t = rec.get("tid")
                    if t is not None:
                        source_by_tid[str(t)] = rec.get("source") or rec.get("origin") or "-"
                debug_by_tid = {}
                for row in tag_debug:
                    t = row.get("tid")
                    if t is not None:
                        debug_by_tid[str(t)] = row
                print("【Stage3 来源回溯表】term | tid | source | tag_purity | cos_sim | anchor_sim | final_weight")
                for r in top_contrib[:20]:
                    tid_s = str(r.get("tid"))
                    term = (r.get("term") or "")[:24]
                    src = source_by_tid.get(tid_s, "-")
                    row2 = debug_by_tid.get(tid_s, {})
                    tp = row2.get("capped_tag_purity") or row2.get("raw_tag_purity")
                    tp_s = f"{float(tp):.3f}" if tp is not None else "-"
                    cos = row2.get("cos_sim")
                    cos_s = f"{float(cos):.3f}" if cos is not None else "-"
                    anc = row2.get("anchor_sim") or row2.get("task_anchor_sim")
                    anc_s = f"{float(anc):.3f}" if anc is not None else "-"
                    fw = r.get("final_weight", 0)
                    fw_s = f"{float(fw):.4f}" if fw is not None else "-"
                    print(f"  {term:24s} | {tid_s:>6} | {src:14s} | {tp_s:>9} | {cos_s:>6} | {anc_s:>8} | {fw_s}")

        vocab_count = db.get("recall_vocab_count", 0)
        w_count = db.get("work_count", 0)
        a_count = db.get("author_count", 0)
        print(f"【Step 4: 召回规模】参与检索学术词数: {vocab_count}  检索论文: {w_count}  锁定作者: {a_count}")

        # 批注：合并原「Top20 作者来源」与第二块排名表，降噪且保留结构字段（struct× / st / mtp）。
        print(
            "【Top 作者榜】# | author_id | score | struct× | st | papr | mtp | "
            "top_terms(前2) | 《代表作》 | hits"
        )
        print("-" * 110)
        for i, item in enumerate(db.get("top_samples", [])[:30], 1):
            aid = item.get("aid", "")
            score = float(item.get("score", 0) or 0)
            sm = item.get("structure_mult_total")
            sm_s = f"{float(sm):.3f}" if sm is not None else "-"
            stc = item.get("strong_term_count_struct")
            stc_s = str(int(stc)) if stc is not None else "-"
            pcs = item.get("paper_count_struct")
            pcs_s = str(int(pcs)) if pcs is not None else "-"
            mtpv = item.get("multi_term_paper_count_struct")
            mtp_s = str(int(mtpv)) if mtpv is not None else "-"
            top_terms = item.get("top_terms_by_contribution", [])[:2]
            terms_s = ", ".join(f"{t}({c})" for t, c in top_terms) if top_terms else "-"
            tp = item.get("top_paper", {}) or {}
            title = (tp.get("title") or "")[:42]
            hit_tags = ", ".join((tp.get("hits") or [])[:4])
            print(
                f"  #{i:<2} | {aid:<12} | {score:.4f} | {sm_s:>7} | {stc_s:>2} | {pcs_s:>4} | {mtp_s:>3} | "
                f"{terms_s[:28]:28s} | 《{title}》 | {hit_tags}"
            )
        print("-" * 110)
        print(f"[*] 诊断完成。全链路耗时: {search_time:.2f}ms")
    else:
        # 仅人选列表（仍会在 recall 中打印 [Label 各阶段耗时] 与 S1～S5 子阶段耗时）
        print(f"\n{'排名':<6} | {'作者 ID':<14} | {'得分':<10} | {'代表作 (命中标签)'}")
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
        print(f"[*] 全链路耗时: {search_time:.2f}ms")


def run_label_debug_cli() -> None:
    try:
        domain_choice = input("\n请选择领域编号 (0跳过): ").strip() or "0"
        detail_choice = input(
            "是否启用标签路详细打印？(y=详细打印  n=仅显示最后人选列表) [n]: "
        ).strip().lower()
        verbose = detail_choice in ("y", "yes", "1", "是")
        l_path = LabelRecallPath(recall_limit=200, verbose=verbose, silent=False)

        while True:
            user_input = input("\n请输入岗位需求 (q退出): ").strip()
            if not user_input or user_input.lower() == "q":
                break
            run_single_label_debug(l_path, domain_choice, verbose, user_input)

    except Exception as e:
        print(f"运行出错: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    run_label_debug_cli()
