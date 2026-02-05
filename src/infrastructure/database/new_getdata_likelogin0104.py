from DrissionPage import ChromiumPage, ChromiumOptions
import json
import time
import random
import pandas as pd
import os
from datetime import datetime
from DrissionPage.common import By
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
import urllib.parse
import re


# ================= 1. 全局配置 =================

class Config:
    CITY_DATA_FILE = 'txt.json'
    OUTPUT_FILE = 'jobs.csv'  # 所有数据统一存入此文件

    # 轮换城市（你可以根据科技人才分布调整顺序）
    TARGET_CITIES = ["北京", "上海", "深圳", "杭州", "广州", "成都", "武汉", "南京"]
    #
    # 爬取参数
    MAX_SCROLLS = random.randint(2, 3)  # 列表页最多滚动次数
    JOBS_PER_KEYWORD = random.randint(5, 10)  # 每个职业抓取的数量

    # 休眠策略（单位：秒）
    REST_BETWEEN_JOBS = (120, 240)  # 职业切换：2-4分钟
    REST_BETWEEN_CITIES = (2100, 2700)  # 城市切换：35-45分钟
    INNER_DETAIL_REST = (15, 25)  # 同职业内职位切换：15-25秒


# ================= 2. 数据管理（去重与保存） =================

class DataManager:
    """负责单文件实时追加与历史数据去重"""

    @staticmethod
    def load_seen_ids():
        """从 jb1.csv 中加载已抓取的职位ID，实现断点续爬和永久去重"""
        seen_ids = set()
        if os.path.exists(Config.OUTPUT_FILE):
            try:
                # 仅读取 ID 列以节省内存
                df = pd.read_csv(Config.OUTPUT_FILE, usecols=['securityId'])
                seen_ids = set(df['securityId'].astype(str).tolist())
                print(f"  [系统] 已从历史记录中加载 {len(seen_ids)} 条已存在职位。")
            except Exception as e:
                print(f"  [系统] 读取历史文件失败或文件为空: {e}")
        return seen_ids

    @staticmethod
    def clean_text(text):
        """清洗非法字符，防止影响推荐系统模型分词"""
        if not text: return ""
        return ILLEGAL_CHARACTERS_RE.sub('', str(text)).strip()

    @staticmethod
    def save_to_csv(data_dict):
        """实时追加保存到 jb1.csv"""
        df = pd.DataFrame([data_dict])
        file_exists = os.path.exists(Config.OUTPUT_FILE)
        # 使用 utf-8-sig 兼容 Excel，mode='a' 追加
        df.to_csv(Config.OUTPUT_FILE, mode='a', index=False, header=not file_exists, encoding='utf-8-sig')


# ================= 3. 核心采集引擎 =================

class TechTalentScraper:
    def __init__(self):
        self.dp = self._init_browser()
        self.city_map = self._load_city_codes()

        # --- 新增：主动创建文件逻辑 ---
        if not os.path.exists(Config.OUTPUT_FILE):
            # 预定义所有列名，确保 CSV 结构完整
            columns = [
                'securityId', 'job_name', 'salary', 'skills',
                'description', 'company', 'city', 'qualification',
                'keyword', 'crawl_time'
            ]
            pd.DataFrame(columns=columns).to_csv(
                Config.OUTPUT_FILE,
                index=False,
                encoding='utf-8-sig'
            )
            print(f"  [系统] 检测到文件不存在，已初始化空文件: {Config.OUTPUT_FILE}")
        # ----------------------------

        self.seen_ids = DataManager.load_seen_ids()

    def _init_browser(self):
        co = ChromiumOptions()
        co.set_argument('--disable-blink-features=AutomationControlled')
        page = ChromiumPage(co)
        # 注入脚本隐藏指纹
        page.run_js('Object.defineProperty(navigator, "webdriver", {get: () => undefined})')
        return page

    def _load_city_codes(self):
        with open(Config.CITY_DATA_FILE, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        mapping = {}
        for group in raw['zpData']['cityGroup']:
            for city in group['cityList']:
                mapping[city['name']] = city['code']
        return mapping

    def _check_page_stability(self):
        """检查页面是否稳定可用"""
        try:
            # 检查页面标题和基本元素
            title = self.dp.title
            if not title or "BOSS直聘" not in title:
                return False

            # 检查body元素是否存在
            body = self.dp.ele('tag:body')
            if not body:
                return False

            return True
        except:
            return False

    def _is_page_ready(self):
        """检查页面是否准备好进行操作"""
        try:
            # 执行简单的JS检查
            result = self.dp.run_js('return document.readyState === "complete"')
            return result
        except:
            return False

    def _safe_scroll(self, pixels):
        """安全的滚动操作"""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                self.dp.scroll.down(pixels)
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"    滚动失败，第{attempt + 1}次重试...")
                    time.sleep(2)
                    # 尝试恢复页面
                    try:
                        self.dp.refresh()
                        time.sleep(3)
                    except:
                        pass
                else:
                    print(f"    [!] 滚动最终失败: {e}")
                    return False
        return True

    def get_detail_wapi(self, sid, lid, search_url):
        """执行你认为最可靠的 WAPI 穿透逻辑"""
        self.dp.listen.start('/zpgeek/job/detail.json')
        wapi_url = f'https://www.zhipin.com/wapi/zpgeek/job/detail.json?securityId={sid}&lid={lid}&_={int(time.time() * 1000)}'

        self.dp.get(wapi_url)
        resp = self.dp.listen.wait(timeout=10)

        # 捕获完成后立即返回搜索页，防止 Referer 丢失导致后续 404
        self.dp.get(search_url)

        if resp and resp.response.body:
            return resp.response.body
        return None

    def run_city_keywords(self, city_name, keywords, start_keyword=None):
        """在当前城市跑完所有定义的关键词"""
        city_code = self.city_map.get(city_name, '100010000')

        # 标记是否开始抓取（如果没指定起始词，则默认开始）
        should_start = False if start_keyword else True

        for idx, kw in enumerate(keywords):
            # 中断恢复逻辑：如果还没到起始关键词，则跳过
            if not should_start:
                if kw == start_keyword:
                    should_start = True
                else:
                    continue

            print(f"\n>>> [城市: {city_name}] 正在采集职业: {kw} ({idx + 1}/{len(keywords)})")

            # 1. 拼接 URL：保持 degree 参数，尝试从检索端过滤
            encoded_kw = urllib.parse.quote(kw)
            search_url = f'https://www.zhipin.com/web/geek/jobs?city={city_code}&query={encoded_kw}&degree=204,205'

            self.dp.get(search_url)
            time.sleep(3)

            if not self._check_page_stability():
                print("    [!] 页面不稳定，重新加载...")
                self.dp.get(search_url)
                time.sleep(3)

            self.dp.listen.start('zpgeek/search/joblist.json')
            job_list_pool = []

            # 滚动加载列表
            for scroll_idx in range(Config.MAX_SCROLLS):
                try:
                    if not self._is_page_ready():
                        self.dp.get(search_url)
                        time.sleep(2)
                        continue

                    if not self._safe_scroll(random.randint(800, 1500)):
                        break

                    resp = self.dp.listen.wait(timeout=5)
                    if resp and resp.response.body:
                        job_list_pool.extend(resp.response.body['zpData'].get('jobList', []))
                        if len(job_list_pool) >= Config.JOBS_PER_KEYWORD:
                            break
                    time.sleep(random.uniform(2.0, 4.0))

                except Exception as e:
                    print(f"    [!] 滚动时出现错误: {e}")
                    continue

            # 2. 遍历详情：加入“列表页预筛”逻辑
            collected = 0
            for item in job_list_pool:
                if collected >= Config.JOBS_PER_KEYWORD:
                    break

                sid = str(item.get('securityId'))
                lid = item.get('lid')



                if sid in self.seen_ids:
                    continue

                # 只有列表页预筛通过的职位，才会进入此步，调用昂贵的详情页 API
                detail_json = self.get_detail_wapi(sid, lid, search_url)

                if detail_json and detail_json.get('code') == 0:
                    zp = detail_json['zpData']
                    job = zp['jobInfo']
                    brand = zp['brandComInfo']

                    # 详情页二次校验（防呆逻辑）
                    qualification = DataManager.clean_text(job.get('degreeName'))
                    if qualification not in ['硕士', '博士']:
                        print(f"    × 跳过非硕博职位: {qualification} (ID: {sid})")
                        self.seen_ids.add(sid)
                        time.sleep(random.uniform(*Config.INNER_DETAIL_REST))
                        continue

                    record = {
                        'securityId': sid,
                        'job_name': DataManager.clean_text(job.get('jobName')),
                        'salary': DataManager.clean_text(job.get('salaryDesc')),
                        'skills': DataManager.clean_text(', '.join(job.get('showSkills', []))),
                        'description': DataManager.clean_text(job.get('postDescription')),
                        'company': DataManager.clean_text(brand.get('brandName')),
                        'city': city_name,
                        'qualification': qualification,
                        'keyword': kw,
                        'crawl_time': datetime.now().strftime('%Y-%m-%d %H:%M')
                    }

                    DataManager.save_to_csv(record)
                    self.seen_ids.add(sid)
                    collected += 1
                    print(
                        f"     成功采集 [{collected}]: {record['job_name']} @ {record['company']} 学历:{qualification}")

                # 仅对真正请求过详情页的操作进行冷却休眠
                time.sleep(random.uniform(*Config.INNER_DETAIL_REST))

            # 3. 关键词切换休眠
            if idx < len(keywords) - 1:
                wait_time = random.randint(*Config.REST_BETWEEN_JOBS)
                print(f"    [休眠] 职业完成，休息 {wait_time // 60} 分钟...")
                time.sleep(wait_time)


# ================= 4. 程序入口 =================

if __name__ == "__main__":
    scraper = TechTalentScraper()

    TECH_KEYWORDS = [
        "数据分析", "数据科学", "数据挖掘", "统计分析", "生物统计", "流行病学",
        "行业研究", "市场研究", "战略分析", "政策研究", "宏观经济", "产业研究",
        "用户研究", "人因工程", "心理学研究", "社会科学", "实验设计",
        "行业研究", "宏观研究", "策略研究", "固定收益", "衍生品", "量化", "风控",
        "精算", "再保险", "资产管理", "投资银行", "并购", "私募", "风投", "VC",
        "管理咨询", "战略咨询", "IT咨询", "技术咨询", "专利代理", "知识产权",
        "标准研究员", "首席专利官",  # 高端标准与专利
        "新能源", "光伏", "风电", "储能", "氢能", "燃料电池", "锂电池", "钠电池",
        "智能电网", "特高压", "电力电子", "电机控制", "变频器",
        "碳达峰", "碳中和", "碳排放", "碳交易", "碳核算", "CCUS",
        "环境工程", "水处理", "大气治理", "固废处理", "土壤修复", "环评"
    ]

    scraper.dp.get('https://www.zhipin.com/web/geek/jobs?ka=open_joblist')
    print("\n" + "=" * 50)
    print("【科技人才推荐系统 - 自动化工厂模式】")
    print("1. 请在浏览器手动登录。")
    print("2. 配置起始点（直接回车表示从头开始）：")

    # 新增用户交互输入
    input_city = input("   请输入起始城市 (如: 北京): ").strip()
    input_kw = input("   请输入起始岗位 (如: 运维工程师): ").strip()

    print("=" * 50)

    if input("请输入 1 开始自动化流程: ") == '1':
        # 标记是否已经到达起始城市
        found_city = False if input_city else True

        for c_idx, city in enumerate(Config.TARGET_CITIES):
            # 城市级跳过逻辑
            if not found_city:
                if city == input_city:
                    found_city = True
                else:
                    continue

            # 只有在"当前城市"等于"用户输入的起始城市"时，才传递起始岗位参数
            current_start_kw = input_kw if (city == input_city) else None

            scraper.run_city_keywords(city, TECH_KEYWORDS, start_keyword=current_start_kw)

            # 城市切换长休眠
            if c_idx < len(Config.TARGET_CITIES) - 1:
                long_rest = random.randint(*Config.REST_BETWEEN_CITIES)
                print(f"\n 城市 {city} 采集完毕。进入降温休眠: {long_rest // 60} 分钟。")
                time.sleep(long_rest)

    print("\n[FINISH] 任务结束。")
    scraper.dp.quit()