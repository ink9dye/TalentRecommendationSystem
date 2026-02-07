import time
import requests
import logging
import os
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from src.infrastructure.crawler.use_openalex.config import EMAIL, BASE_DELAY

logger = logging.getLogger(__name__)


class APIClient:
    def __init__(self):
        # --- 改进 1: 强制校验 ---
        if not EMAIL or "@" not in EMAIL:
            raise ValueError("❌ 错误：EMAIL 未正确加载或格式不正确，请检查 config.py")

        self.email = EMAIL.strip()
        self.session = requests.Session()

        # --- 改进 2: 更加标准的 Polite Pool 请求头 ---
        self.session.headers.update({
            "User-Agent": f"TalentRecSystem/1.0 (mailto:{self.email})",
            "From": self.email,
            "Accept": "application/json"
        })

        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )

        adapter = HTTPAdapter(
            pool_connections=50,
            pool_maxsize=100,  # 稍微调大连接池，适配多线程
            max_retries=retry_strategy
        )
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)

    def make_request(self, url: str, params: dict = None) -> dict:
        """核心请求方法"""
        if params is None:
            params = {}

        # 确保 mailto 始终存在于 URL 参数中（这是最稳妥的识别方式）
        params['mailto'] = self.email

        # 基础频率限制（Polite池建议不要超过10次/秒）
        time.sleep(BASE_DELAY)

        try:
            response = self.session.get(url, params=params, timeout=15)

            # --- 改进 3: 更精细的错误处理 ---
            if response.status_code == 403:
                # 有时额度耗尽会返回 403 Forbidden
                print(f"\n🛑 【权限警报】访问被拒绝！可能是 EMAIL 无效或 IP 被封。")
                os._exit(0)

            if response.status_code in [401, 402]:
                # 显式捕获你遇到的 Insufficient credits 错误
                err_data = response.json() if response.text else {}
                msg = err_data.get('message', '未知错误')
                print(f"\n🛑 【额度耗尽】OpenAlex 提示: {msg}")
                print(f"⏰ 请等待 UTC 0点（北京时间早8点）重置。")
                os._exit(0)

            if response.status_code == 429:
                logger.warning("⚠️ 触发 429 速率限制，正在休眠...")
                time.sleep(20)  # 429 建议休眠时间长一点
                return {}

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 404:
                    return {}
                logger.error(f"❌ HTTP 错误 {e.response.status_code}: {url}")
            else:
                logger.error(f"🔥 网络连接异常: {e}")
            return {}