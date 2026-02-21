import sys
from streamlit.web import cli as stcli


def main():
    # 设计启动参数：等同于在命令行输入 streamlit run src/interface/app.py
    # 我们直接调用 streamlit 内部的 cli 接口
    sys.argv = [
        "streamlit",
        "run",
        "src/interface/app.py",
        "--server.port=8501",  # 你可以固定端口
        "--server.address=127.0.0.1"
    ]

    # 顺便打印一个帅气的启动标志
    print("\n" + "=" * 50)
    print("🚀 TalentAI 专家推荐系统正在启动...")
    print("🔗 访问地址: http://127.0.0.1:8501")
    print("=" * 50 + "\n")

    # 启动
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()