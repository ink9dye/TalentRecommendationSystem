# -*- coding: utf-8 -*-
"""
QueryEncoder 共振词表、LabelRecallPath 领域向量 的快照读写。
- 共振词表：主库 jobs/vocabulary 变更时随 DB mtime 自动失效重建。
- 领域向量：DOMAIN_MAP 或 SBERT 目录下 config.json 变更时失效重建。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from typing import Callable, Dict, Optional, Set

import numpy as np

_SNAPSHOT_VERSION = 1


def _safe_replace_tmp_to_final(tmp_path: str, final_path: str) -> None:
    """
    先确认临时文件存在再 os.replace；Windows 下偶发 WinError 2 时用 copy2+remove 兜底。
    """
    if not os.path.isfile(tmp_path):
        raise FileNotFoundError(f"临时文件未生成，无法落盘: {tmp_path}")
    try:
        os.replace(tmp_path, final_path)
    except OSError:
        shutil.copy2(tmp_path, final_path)
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _domain_map_signature(domain_map: Dict[str, str]) -> str:
    payload = json.dumps(sorted(domain_map.items()), ensure_ascii=False, separators=(",", ":"))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _sbert_config_mtime(sbert_dir: str) -> float:
    cfg = os.path.join(sbert_dir, "config.json")
    try:
        return float(os.path.getmtime(cfg))
    except OSError:
        return 0.0


def try_load_hardcore_lexicon(db_path: str, snapshot_path: str) -> Optional[Set[str]]:
    """若快照存在且与当前主库 mtime 一致则返回词表，否则 None。"""
    if not snapshot_path or not os.path.isfile(snapshot_path):
        return None
    if not os.path.isfile(db_path):
        return None
    try:
        db_mtime = os.path.getmtime(db_path)
        with open(snapshot_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if int(meta.get("version", 0)) != _SNAPSHOT_VERSION:
            return None
        if os.path.abspath(meta.get("db_path", "")) != os.path.abspath(db_path):
            return None
        if float(meta.get("db_mtime", -1.0)) != float(db_mtime):
            return None
        terms = meta.get("terms")
        if not isinstance(terms, list):
            return None
        return set(str(t) for t in terms)
    except Exception:
        return None


def save_hardcore_lexicon(lexicon: Set[str], db_path: str, snapshot_path: str) -> None:
    if not snapshot_path:
        return
    try:
        os.makedirs(os.path.dirname(snapshot_path) or ".", exist_ok=True)
        db_mtime = os.path.getmtime(db_path) if os.path.isfile(db_path) else 0.0
        terms = sorted(lexicon)
        payload = {
            "version": _SNAPSHOT_VERSION,
            "db_path": os.path.abspath(db_path),
            "db_mtime": float(db_mtime),
            "term_count": len(terms),
            "terms": terms,
        }
        tmp = snapshot_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        _safe_replace_tmp_to_final(tmp, snapshot_path)
    except Exception as e:
        print(f"[Warning] 共振词表快照写入失败（将不影响本次运行）: {e}", flush=True)


def try_load_domain_vectors(
    sbert_dir: str,
    domain_map: Dict[str, str],
    npz_path: str,
    meta_path: str,
) -> Optional[Dict[str, np.ndarray]]:
    if not npz_path or not meta_path or not os.path.isfile(npz_path) or not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if int(meta.get("version", 0)) != _SNAPSHOT_VERSION:
            return None
        if os.path.abspath(meta.get("sbert_dir", "")) != os.path.abspath(sbert_dir):
            return None
        if meta.get("domain_map_sig") != _domain_map_signature(domain_map):
            return None
        cfg_mt = _sbert_config_mtime(sbert_dir)
        if float(meta.get("sbert_config_mtime", -1.0)) != float(cfg_mt):
            return None
        data = np.load(npz_path, allow_pickle=False)
        out: Dict[str, np.ndarray] = {}
        for did in domain_map:
            sid = str(did)
            key_new = f"dom_{sid}"
            if key_new in data.files:
                arr = np.asarray(data[key_new], dtype=np.float32).reshape(-1)
            elif sid in data.files:
                # 旧版快照：数组名为 "1".."17"
                arr = np.asarray(data[sid], dtype=np.float32).reshape(-1)
            else:
                return None
            if arr.size == 0:
                return None
            out[sid] = arr
        if len(out) != len(domain_map):
            return None
        return out
    except Exception:
        return None


def save_domain_vectors(
    vectors: Dict[str, np.ndarray],
    sbert_dir: str,
    domain_map: Dict[str, str],
    npz_path: str,
    meta_path: str,
) -> None:
    if not npz_path or not meta_path or not vectors:
        return
    try:
        ddir = os.path.dirname(os.path.abspath(npz_path)) or "."
        os.makedirs(ddir, exist_ok=True)
        tmp_npz = os.path.abspath(npz_path + ".tmp")
        final_npz = os.path.abspath(npz_path)
        # 使用 dom_{id} 键名；经 open('wb') 写入，避免 Windows 下仅传路径时偶发不落盘
        kwargs = {
            f"dom_{str(k)}": np.asarray(v, dtype=np.float32).reshape(-1)
            for k, v in vectors.items()
        }

        def _write_npz_to_path(path: str) -> None:
            with open(path, "wb") as fp:
                np.savez_compressed(fp, **kwargs)

        try:
            _write_npz_to_path(tmp_npz)
            if not os.path.isfile(tmp_npz) or os.path.getsize(tmp_npz) <= 0:
                raise FileNotFoundError(f"npz 临时文件无效: {tmp_npz}")
            _safe_replace_tmp_to_final(tmp_npz, final_npz)
        except OSError:
            _write_npz_to_path(final_npz)
            if os.path.isfile(tmp_npz):
                try:
                    os.remove(tmp_npz)
                except OSError:
                    pass
            if not os.path.isfile(final_npz) or os.path.getsize(final_npz) <= 0:
                raise OSError(f"领域向量快照 npz 落盘失败: {final_npz}")

        meta = {
            "version": _SNAPSHOT_VERSION,
            "sbert_dir": os.path.abspath(sbert_dir),
            "sbert_config_mtime": float(_sbert_config_mtime(sbert_dir)),
            "domain_map_sig": _domain_map_signature(domain_map),
            "n_domains": len(vectors),
            "npz_layout": "dom_prefix",
        }
        tmp_meta = os.path.abspath(meta_path + ".tmp")
        final_meta = os.path.abspath(meta_path)
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))
        if not os.path.isfile(tmp_meta):
            raise FileNotFoundError(f"meta 临时文件未生成: {tmp_meta}")
        _safe_replace_tmp_to_final(tmp_meta, final_meta)
    except Exception as e:
        print(f"[Warning] 领域向量快照写入失败（将不影响本次运行）: {e}", flush=True)


def load_or_build_hardcore_lexicon(
    db_path: str,
    snapshot_path: str,
    build_fn: Callable[[], Set[str]],
) -> Set[str]:
    loaded = try_load_hardcore_lexicon(db_path, snapshot_path)
    if loaded is not None:
        print(
            f"[OK] 动态特征库已从快照加载 (核心词条: {len(loaded)})  file={snapshot_path}",
            flush=True,
        )
        return loaded
    lex = build_fn()
    save_hardcore_lexicon(lex, db_path, snapshot_path)
    print(
        f"[OK] 动态特征库已从主库构建并写入快照 (核心词条: {len(lex)})  file={snapshot_path}",
        flush=True,
    )
    return lex
