# -*- coding: utf-8 -*-
"""
标签路常驻服务：进程内只初始化一次 LabelRecallPath（含编码器、图、索引），避免每次 CLI 冷启动。

启动（在项目根目录）:
  .venv\\Scripts\\python.exe -m uvicorn src.core.recall.label_recall_service:app --host 127.0.0.1 --port 8765

请求示例:
  curl -X POST http://127.0.0.1:8765/recall -H "Content-Type: application/json" ^
    -d "{\\"query_text\\":\\"机器人控制 强化学习\\",\\"domain_id\\":\\"0\\",\\"verbose\\":false}"

环境变量:
  LABEL_PROFILE_STAGE5=1  — 在服务端进程环境中设置后，单次 recall 会打印 paper_scoring 子项累计耗时。
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional

# 项目根目录
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import faiss
import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException

_app_l_path = None


def get_label_path():
    global _app_l_path
    if _app_l_path is None:
        from src.core.recall.label_path import LabelRecallPath

        _app_l_path = LabelRecallPath(recall_limit=200, verbose=False)
    return _app_l_path


class RecallRequest(BaseModel):
    query_text: str = Field(..., description="岗位需求全文")
    domain_id: str = Field("0", description="领域编号，0 表示跳过")
    semantic_query_text: Optional[str] = Field(None, description="可选，默认与 query_text 相同")
    verbose: bool = Field(False, description="是否打开标签路 verbose（打印增多）")
    recall_limit: int = Field(200, ge=1, le=500)


class RecallResponse(BaseModel):
    ok: bool = True
    author_ids: List[str] = Field(default_factory=list)
    meta: List[Dict[str, Any]] = Field(default_factory=list)
    recall_ms: float = 0.0
    encode_ms: float = 0.0
    message: str = ""


app = FastAPI(title="Label Recall Service", version="1.0")


@app.on_event("startup")
def _startup() -> None:
    print("[*] LabelRecallService: 预加载 LabelRecallPath ...", flush=True)
    t0 = time.perf_counter()
    get_label_path()
    print(f"[OK] LabelRecallService 就绪，init={1000*(time.perf_counter()-t0):.0f}ms", flush=True)


@app.get("/health")
def health() -> Dict[str, Any]:
    lp = get_label_path()
    return {"ok": True, "graph": bool(getattr(lp, "graph", None))}


@app.post("/recall", response_model=RecallResponse)
def recall_ep(req: RecallRequest) -> RecallResponse:
    text = (req.query_text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="query_text 为空")

    lp = get_label_path()
    lp.recall_limit = int(req.recall_limit)
    lp.verbose = bool(req.verbose)

    sem = (req.semantic_query_text or req.query_text or "").strip()
    enc = lp._query_encoder
    t_enc = time.perf_counter()
    query_vec, _ = enc.encode(sem)
    encode_ms = (time.perf_counter() - t_enc) * 1000.0
    if query_vec is None or query_vec.size == 0:
        return RecallResponse(ok=False, message="encode failed", encode_ms=encode_ms)

    faiss.normalize_L2(query_vec)

    t0 = time.perf_counter()
    meta_list, recall_ms = lp.recall(
        query_vec,
        domain_id=str(req.domain_id or "0"),
        query_text=text,
        semantic_query_text=sem,
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0

    aids: List[str] = []
    for m in meta_list or []:
        if isinstance(m, dict) and m.get("author_id"):
            aids.append(str(m["author_id"]))
        elif isinstance(m, str):
            aids.append(m)

    return RecallResponse(
        author_ids=aids,
        meta=[m for m in (meta_list or []) if isinstance(m, dict)],
        recall_ms=float(recall_ms),
        encode_ms=float(encode_ms),
        message=f"wall_loop_ms={wall_ms:.1f}",
    )


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("LABEL_SERVICE_HOST", "127.0.0.1")
    port = int(os.environ.get("LABEL_SERVICE_PORT", "8765"))
    uvicorn.run(app, host=host, port=port, reload=False)
