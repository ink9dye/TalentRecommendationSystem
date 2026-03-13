import hashlib
from typing import Optional, Any


def safe_get(data: Any, path: list, default: Any = None) -> Any:
    """安全获取嵌套字典的值，防止 NoneType 报错"""
    for key in path:
        if isinstance(data, dict):
            data = data.get(key)
        else:
            return default
    return data if data is not None else default


def clean_id(raw_url: Optional[str]) -> Optional[str]:
    """从完整的 OpenAlex URL 中提取纯 ID"""
    if not raw_url:
        return None
    return raw_url.split('/')[-1]


def generate_work_id(work: dict) -> str:
    """
    统一使用 OpenAlex 官方 ID 作为主键 (例如 W4280147629)
    不再使用 doi: 或 hash: 前缀
    """
    # 直接提取 'id' 字段中的纯 ID (如从 https://api.openalex.org/W123 提取 W123)
    oa_id = clean_id(work.get('id'))
    if oa_id:
        return oa_id

    # 万一 OpenAlex ID 缺失（极少见），再尝试从 DOI 提取
    doi = work.get('doi')
    if doi:
        return clean_id(doi)

    # 最后的保底手段，生成一个标记
    title = work.get('display_name', '')
    year = str(work.get('publication_year', ''))
    hash_str = hashlib.md5(f"{title}_{year}".encode()).hexdigest()[:16]
    return f"TMP_{hash_str}"