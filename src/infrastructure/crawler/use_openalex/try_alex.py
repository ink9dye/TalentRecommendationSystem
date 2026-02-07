import logging
import json
from api_client import APIClient
from utils import clean_id
from src.infrastructure.crawler.use_openalex.config import EMAIL

# 将日志级别设为 DEBUG 以观察 APIClient 的底层动作
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DOI_Diagnostic")


def diagnostic_doi(doi_input):
    client = APIClient()

    # --- 诊断点 1: 原始输入分析 ---
    logger.info(f"📥 原始输入内容: '{doi_input}' (长度: {len(doi_input)})")

    # 彻底模拟并发脚本中的清洗逻辑
    # 检查是否包含不可见字符（如 \r, \n, \t）
    raw_doi = doi_input.strip().replace("doi:", "").replace("https://doi.org/", "").strip()
    full_doi_url = f"https://doi.org/{raw_doi}"

    logger.info(f"🧹 清洗后 DOI: {raw_doi}")
    logger.info(f"🔗 最终构造的 Filter URL: {full_doi_url}")

    # --- 诊断点 2: 构造参数 ---
    params = {
        "filter": f"doi:{full_doi_url}",
        "mailto": EMAIL
    }

    logger.info(f"📡 发送给 APIClient 的参数: {params}")

    try:
        # 发起请求
        res = client.make_request("https://api.openalex.org/works", params)

        # --- 诊断点 3: 响应全解析 ---
        if res is None:
            logger.error("❌ APIClient 返回了 None。请检查 api_client.py 是否捕获了错误但未抛出。")
            return None

        if not isinstance(res, dict):
            logger.error(f"❌ API 响应格式异常，期望 dict，实际得到: {type(res)}")
            return None

        meta = res.get('meta', {})
        results = res.get('results', [])

        logger.info(f"📊 API 响应摘要: 命中总数(count)={meta.get('count')}, 结果数(len)={len(results)}")

        if results:
            top_hit = results[0]
            logger.info("✅ 匹配成功！")
            _print_doi_info(top_hit)

            # 校验 clean_id 是否正常工作
            standard_id = top_hit.get('id', '')
            cleaned = clean_id(standard_id)
            logger.info(f"🆔 clean_id 转换结果: '{standard_id}' -> '{cleaned}'")
            if not cleaned:
                logger.error("⚠️ 警告: clean_id 返回了空值，这会导致数据库更新被跳过！")

            return top_hit
        else:
            logger.warning("⚠️ API 未报错，但 results 列表为空。说明该 DOI 在 OpenAlex 中无法通过 filter 查到。")
            # 打印 meta 信息辅助判断（比如是否被 OpenAlex 重定向或归一化了）
            logger.info(f"🔍 响应元数据: {json.dumps(meta, indent=2)}")
            return None

    except Exception as e:
        logger.error(f"🚀 捕获到未处理的异常: {type(e).__name__}: {e}", exc_info=True)
        return None


def _print_doi_info(work):
    print("\n" + "═" * 60)
    print(f"【详细数据展示】")
    print(f"标题: {work.get('display_name', 'N/A')}")
    print(f"DOI: {work.get('doi', 'N/A')}")
    # 展示 ids 字典，确认是否有其他可用的 ID
    print(f"IDs 字典: {json.dumps(work.get('ids', {}), indent=2)}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    # --- 请在此处替换一个在并发脚本中失败(Err)的 DOI ---
    test_doi = "https://doi.org/10.36227/techrxiv.176611774.47340823/v2"
    diagnostic_doi(test_doi)