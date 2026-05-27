"""
查看 SQLite 数据库内容的简单脚本
"""

import sqlite3
import os

def view_database(db_path):
    """查看 SQLite 数据库的表和数据"""
    if not os.path.exists(db_path):
        print(f"❌ 数据库文件不存在：{db_path}")
        return

    print(f"\n{'='*70}")
    print(f"查看数据库：{os.path.basename(db_path)}")
    print(f"{'='*70}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 查看所有表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    print(f"\n📋 数据库中的表：")
    for table in tables:
        print(f"  - {table[0]}")

    # 查看每个表的数据
    for table in tables:
        table_name = table[0]
        print(f"\n📊 表 '{table_name}' 的内容：")

        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"  记录数：{count}")

            # 显示前5条记录
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
            rows = cursor.fetchall()

            if rows:
                # 获取列名
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [col[1] for col in cursor.fetchall()]
                print(f"  列：{', '.join(columns)}")

                print("\n  前5条记录：")
                for i, row in enumerate(rows, 1):
                    print(f"    [{i}] {row[:3]}...")  # 只显示前3个字段
            else:
                print("  （空表）")

        except sqlite3.Error as e:
            print(f"  ❌ 错误：{e}")

    conn.close()

def main():
    """主函数"""
    base_dir = "D:\WorkSpace_Python\Langchain1.0-Langgraph1.0-Learning\phase2_practical\\09_checkpointing"

    db_files = [
        "checkpoints.sqlite",
        "multi_user.sqlite",
        "tools.sqlite",
        "customer_service.sqlite"
    ]

    print("\n" + "="*70)
    print(" SQLite 数据库查看工具")
    print("="*70)

    for db_file in db_files:
        db_path = os.path.join(base_dir, db_file)
        view_database(db_path)

    print("\n" + "="*70)
    print(" 完成！")
    print("="*70)
    print("\n💡 提示：")
    print("  - 如果显示'数据库文件不存在'，请先运行 main.py")
    print("  - 可以使用在线工具查看完整内容：https://sqliteviewer.app/")

if __name__ == "__main__":
    main()
