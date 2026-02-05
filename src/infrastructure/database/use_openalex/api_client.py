import time
import requests
import logging
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from config import EMAIL, BASE_DELAY
from utils import safe_get, clean_id

logger = logging.getLogger(__name__)


class APIClient:
    def __init__(self):
        # 1. 先创建 session 对象 (必须排在第一位)
        self.session = requests.Session()

        # 2. 然后再更新请求头，加入礼貌池标识
        self.session.headers.update({
            "User-Agent": f"TalentRecSystem/1.0 (mailto:{EMAIL})",
            "From": EMAIL
        })

        # 3. 设置重试机制
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def make_request(self, url: str, params: dict) -> dict:
        """核心请求方法，自动注入 EMAIL 并处理延迟"""
        params['mailto'] = EMAIL
        time.sleep(BASE_DELAY)
        try:
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API 请求失败: {url} | 错误: {e}")
            return {}

    def fetch_authors_batch(self, author_ids: list) -> dict:
        """利用过滤器批量获取作者信息（每批最多50个）"""
        if not author_ids:
            return {}
        # 将 ID 列表拼接成 A111|A222|A333 的格式
        ids_filter = "|".join(author_ids)
        params = {
            "filter": f"openalex:{ids_filter}",
            "per-page": len(author_ids)
        }
        # 注意：这里请求的是 /authors 列表接口，而不是单个实体接口
        return self.make_request("https://api.openalex.org/authors", params)

    def fetch_author_top_works(self, author_id: str, limit: int = 20) -> list:
        """
        精准获取某位作者全球引用量最高的前 N 篇代表作
        """
        try:
            # 移除前缀，确保 ID 纯净
            pure_id = author_id.split('/')[-1] if '/' in author_id else author_id

            params = {
                'filter': f'author.id:{pure_id}',
                'sort': 'cited_by_count:desc',
                'per-page': limit,
                'cursor': '*'
            }

            logger.debug(f"请求作者 {author_id} 的 top {limit} 篇论文")

            data = self.make_request("https://api.openalex.org/works", params)

            if not data:
                logger.warning(f"获取作者 {author_id} 论文时返回空响应")
                return []

            results = data.get('results', [])

            if not results:
                logger.debug(f"作者 {author_id} 没有找到论文数据")
                return []

            # 确保返回的是列表
            if not isinstance(results, list):
                logger.warning(f"作者 {author_id} 返回的 results 不是列表: {type(results)}")
                return []

            logger.debug(f"成功获取作者 {author_id} 的 {len(results)} 篇论文")
            return results

        except Exception as e:
            logger.error(f"获取作者 {author_id} 的 top works 失败: {e}")
            return []

    def fetch_author_entity(self, author_id: str) -> dict:
        """获取作者的完整实体信息，包含 H-index"""
        pure_id = author_id.split('/')[-1] if '/' in author_id else author_id
        url = f"https://api.openalex.org/authors/{pure_id}"
        return self.make_request(url, {})