#!/usr/bin/env python3
"""
OpenAlex论文链接获取工具
功能：根据OpenAlex ID自动获取论文的完整内容访问链接
作者：元宝AI助手
日期：2026-02-20
"""

import requests
import json
import sys
from typing import Optional, Dict, Any


class OpenAlexPaperFetcher:
    """OpenAlex论文链接获取器"""

    def __init__(self, timeout: int = 10):
        """
        初始化获取器

        Args:
            timeout: 请求超时时间（秒）
        """
        self.base_url = "https://api.openalex.org/works"
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'OpenAlex-Paper-Fetcher/1.0',
            'Accept': 'application/json'
        })

    def fetch_paper_data(self, openalex_id: str) -> Optional[Dict[str, Any]]:
        """
        从OpenAlex API获取论文数据

        Args:
            openalex_id: OpenAlex ID（如W2741809807）

        Returns:
            论文数据的字典，如果获取失败则返回None
        """
        try:
            # 构造API URL
            api_url = f"{self.base_url}/{openalex_id}"

            # 发送请求
            response = self.session.get(api_url, timeout=self.timeout)
            response.raise_for_status()  # 检查HTTP错误

            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"网络请求错误: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return None
        except Exception as e:
            print(f"未知错误: {e}")
            return None

    def analyze_paper_type(self, paper_data: Dict[str, Any]) -> str:
        """
        分析数据实体类型

        Args:
            paper_data: 从API获取的数据

        Returns:
            实体类型描述
        """
        entity_type = paper_data.get("type", "unknown")

        type_mapping = {
            "article": "期刊论文",
            "book": "书籍",
            "book-chapter": "书籍章节",
            "dataset": "数据集",
            "dissertation": "学位论文",
            "paratext": "期刊/书籍元数据",
            "reference-entry": "参考文献条目",
            "report": "报告",
            "review": "综述",
            "standard": "标准",
            "other": "其他类型"
        }

        return type_mapping.get(entity_type, f"未知类型: {entity_type}")

    def extract_best_paper_link(self, paper_data: Dict[str, Any]) -> Optional[str]:
        """
        提取最佳的论文访问链接

        优先级顺序：
        1. PDF直接链接（如果可获取全文）
        2. 论文主页链接
        3. DOI链接
        4. 开放获取链接

        Args:
            paper_data: 从API获取的数据

        Returns:
            最佳论文链接，如果无可用链接则返回None
        """
        # 1. 检查PDF直接链接
        pdf_url = paper_data.get("primary_location", {}).get("pdf_url")
        if pdf_url:
            return pdf_url

        # 2. 检查论文主页链接
        landing_url = paper_data.get("primary_location", {}).get("landing_page_url")
        if landing_url:
            return landing_url

        # 3. 检查DOI链接
        doi = paper_data.get("doi")
        if doi:
            # 确保DOI格式正确
            if doi.startswith("http"):
                return doi
            else:
                return f"https://doi.org/{doi}"

        # 4. 检查开放获取链接
        oa_url = paper_data.get("open_access", {}).get("oa_url")
        if oa_url:
            return oa_url

        # 5. 检查其他位置
        locations = paper_data.get("locations", [])
        for location in locations:
            if location.get("pdf_url"):
                return location["pdf_url"]
            if location.get("landing_page_url"):
                return location["landing_page_url"]

        return None

    def get_journal_info(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取期刊信息（如果是期刊元数据）

        Args:
            paper_data: 从API获取的数据

        Returns:
            期刊信息字典
        """
        source = paper_data.get("primary_location", {}).get("source", {})

        journal_info = {
            "name": source.get("display_name", "未知期刊"),
            "issn": source.get("issn", []),
            "is_oa": source.get("is_oa", False),
            "host_organization": source.get("host_organization_name", "未知出版社"),
            "homepage": None
        }

        # 尝试构造期刊主页
        if source.get("id"):
            journal_info["homepage"] = source["id"].replace("https://openalex.org/S", "https://openalex.org/sources/")

        return journal_info

    def get_paper_link(self, openalex_id: str) -> Dict[str, Any]:
        """
        主函数：根据OpenAlex ID获取论文链接

        Args:
            openalex_id: OpenAlex ID

        Returns:
            包含结果信息的字典
        """
        result = {
            "success": False,
            "openalex_id": openalex_id,
            "entity_type": "未知",
            "paper_title": "未知",
            "best_link": None,
            "message": "",
            "additional_info": {}
        }

        # 1. 验证ID格式
        if not openalex_id.startswith("W"):
            result["message"] = f"错误：OpenAlex ID应以'W'开头，您输入的是: {openalex_id}"
            return result

        # 2. 获取数据
        print(f"正在从OpenAlex获取ID为 {openalex_id} 的数据...")
        paper_data = self.fetch_paper_data(openalex_id)

        if not paper_data:
            result["message"] = f"无法获取ID为 {openalex_id} 的数据，请检查ID是否正确或网络连接"
            return result

        # 3. 提取基本信息
        result["paper_title"] = paper_data.get("title", "无标题")
        result["entity_type"] = self.analyze_paper_type(paper_data)

        # 4. 检查是否为期刊元数据
        if paper_data.get("type") == "paratext":
            result["message"] = "注意：此ID对应的是期刊元数据，不是具体论文"
            journal_info = self.get_journal_info(paper_data)
            result["additional_info"] = {
                "journal_name": journal_info["name"],
                "publisher": journal_info["host_organization"],
                "journal_homepage": journal_info["homepage"],
                "suggestion": f"请访问期刊主页查找具体论文，或使用真正的论文ID（如W2741809807）"
            }
            return result

        # 5. 提取最佳论文链接
        best_link = self.extract_best_paper_link(paper_data)

        if best_link:
            result["success"] = True
            result["best_link"] = best_link
            result["message"] = "成功获取论文链接"

            # 添加额外信息
            result["additional_info"] = {
                "publication_year": paper_data.get("publication_year"),
                "authors_count": len(paper_data.get("authorships", [])),
                "is_open_access": paper_data.get("open_access", {}).get("is_oa", False),
                "citation_count": paper_data.get("cited_by_count", 0),
                "doi": paper_data.get("doi")
            }
        else:
            result["message"] = "未找到可用的论文全文链接，可能该论文不是开放获取的"
            result["additional_info"] = {
                "suggestion": "您可以尝试通过学术搜索引擎（如Google Scholar）或联系作者获取全文"
            }

        return result

    def print_result(self, result: Dict[str, Any]):
        """
        美观地打印结果

        Args:
            result: 结果字典
        """
        print("\n" + "=" * 60)
        print("OpenAlex论文链接获取结果")
        print("=" * 60)

        print(f"OpenAlex ID: {result['openalex_id']}")
        print(f"标题: {result['paper_title']}")
        print(f"实体类型: {result['entity_type']}")
        print(f"状态: {'成功' if result['success'] else '未成功'}")

        if result['message']:
            print(f"说明: {result['message']}")

        if result['best_link']:
            print(f"\n📄 论文访问链接:")
            print(f"   {result['best_link']}")

            # 提供使用建议
            print(f"\n💡 使用建议:")
            print(f"   1. 直接点击或复制上方链接访问论文")
            print(f"   2. 如果是PDF链接，可以直接下载全文")
            print(f"   3. 如果是DOI链接，会自动跳转到论文官方页面")

        # 显示额外信息
        if result['additional_info']:
            print(f"\n📊 附加信息:")
            for key, value in result['additional_info'].items():
                if value:  # 只显示非空值
                    print(f"   • {key.replace('_', ' ').title()}: {value}")

        print("=" * 60 + "\n")


def main():
    """主函数"""
    print("🔍 OpenAlex论文链接获取工具")
    print("说明：输入OpenAlex ID（如W2741809807），获取论文完整内容访问链接")
    print("-" * 50)

    # 创建获取器实例
    fetcher = OpenAlexPaperFetcher()

    # 获取用户输入
    if len(sys.argv) > 1:
        # 从命令行参数获取ID
        openalex_id = sys.argv[1]
    else:
        # 交互式输入
        openalex_id = input("请输入OpenAlex ID（以W开头）: ").strip()

    if not openalex_id:
        print("错误：未输入OpenAlex ID")
        return

    # 获取论文链接
    result = fetcher.get_paper_link(openalex_id)

    # 打印结果
    fetcher.print_result(result)

    # 如果是期刊元数据，提供更多帮助
    if result["entity_type"] == "期刊/书籍元数据" and result["additional_info"].get("journal_homepage"):
        print("📚 期刊信息:")
        print(f"   期刊名称: {result['additional_info'].get('journal_name')}")
        print(f"   出版社: {result['additional_info'].get('publisher')}")
        print(f"   期刊主页: {result['additional_info'].get('journal_homepage')}")
        print("\n💡 建议：访问期刊主页，查找具体的论文文章")

    # 提供示例ID供测试
    if not result["success"] and result["entity_type"] != "期刊/书籍元数据":
        print("\n🔄 您可以尝试以下有效的论文ID进行测试:")
        print("   • W2741809807 - ALTEX期刊上的具体论文")
        print("   • W4391375266 - 另一篇相关论文")
        print("   • W2899084033 - 计算机科学领域论文")


if __name__ == "__main__":
    main()