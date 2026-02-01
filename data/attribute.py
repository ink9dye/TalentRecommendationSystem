import sqlite3

# 连接数据库
conn = sqlite3.connect('academic_dataset_v5.db')
cursor = conn.cursor()

# 1. 获取所有表名
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

for table_name in tables:
    name = table_name[0]
    print(f"\n--- 表名: {name} ---")

    # 2. 获取该表的所有列属性
    cursor.execute(f"PRAGMA table_info('{name}')")
    columns = cursor.fetchall()

    print(f"{'ID':<4} {'字段名':<20} {'类型':<15} {'主键':<5}")
    for col in columns:
        # col 结构: (id, name, type, notnull, default_value, pk)
        print(f"{col[0]:<4} {col[1]:<20} {col[2]:<15} {col[5]:<5}")

conn.close()