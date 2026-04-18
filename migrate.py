"""
migrate.py
使用 pexpect 自动完成 scp 传输，无需手动输入密码。
"""

import pexpect
import sys

# 目标服务器配置
REMOTE_HOST = "10.135.0.63"
REMOTE_PORT = 22          # scp -P 参数（题目要求用22）
REMOTE_USER = "cjh"
REMOTE_PASS = "cjh!@#"
REMOTE_PORT_SSH = 15000   # 目标服务器 SSH 端口

# 本地项目路径
LOCAL_DIR = "/home/cjh/code/fake_news/FN_Cascaded"
# 远端目标目录（先确保父目录存在）
REMOTE_DIR = "/home/cjh/code/fake_news/"


def ssh_mkdir(host, port, user, password, remote_path):
    """在远端创建目录"""
    cmd = f"ssh -p {port} -o StrictHostKeyChecking=no -o ConnectTimeout=10 {user}@{host} 'mkdir -p {remote_path}'"
    print(f"[SSH] 创建远端目录: {remote_path}")
    child = pexpect.spawn(cmd, timeout=30, encoding="utf-8")
    idx = child.expect(["password:", "Password:", pexpect.EOF, pexpect.TIMEOUT])
    if idx in (0, 1):
        child.sendline(password)
        child.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=15)
    print(f"[SSH] 目录创建完成")


def scp_transfer(host, port, user, password, local_path, remote_path):
    """scp 传输目录"""
    cmd = (
        f"scp -P {port} "
        f"-o StrictHostKeyChecking=no "
        f"-o ConnectTimeout=15 "
        f"-r {local_path} "
        f"{user}@{host}:{remote_path}"
    )
    print(f"\n[SCP] 开始传输...")
    print(f"  本地: {local_path}")
    print(f"  远端: {user}@{host}:{remote_path}  (port={port})")
    print(f"  命令: {cmd}\n")

    child = pexpect.spawn(cmd, timeout=1800, encoding="utf-8")  # 30分钟超时
    child.logfile = sys.stdout  # 实时打印输出

    idx = child.expect(["password:", "Password:", pexpect.EOF, pexpect.TIMEOUT], timeout=30)
    if idx in (0, 1):
        child.sendline(password)
    elif idx == 2:
        print("[INFO] scp 无需密码（已有密钥）或立即结束")
        return child.exitstatus
    else:
        print("[ERROR] 等待密码提示超时，请检查网络连接")
        return 1

    # 等待传输完成（大文件可能很久）
    print("[INFO] 密码已发送，等待传输完成（1.2GB，请耐心等待）...")
    child.expect(pexpect.EOF, timeout=1800)
    child.close()
    return child.exitstatus


if __name__ == "__main__":
    print("=" * 60)
    print("  FN_Cascaded 项目迁移工具")
    print(f"  目标: {REMOTE_USER}@{REMOTE_HOST}:{REMOTE_PORT_SSH}")
    print("=" * 60)

    # 1. 在远端创建目标父目录
    ssh_mkdir(REMOTE_HOST, REMOTE_PORT_SSH, REMOTE_USER, REMOTE_PASS, REMOTE_DIR)

    # 2. 开始 scp 传输（scp 使用端口 15000）
    status = scp_transfer(REMOTE_HOST, REMOTE_PORT_SSH, REMOTE_USER, REMOTE_PASS, LOCAL_DIR, REMOTE_DIR)

    if status == 0:
        print("\n[SUCCESS] 传输完成！")
        print(f"  项目已传输至 {REMOTE_HOST}:{REMOTE_DIR}FN_Cascaded/")
        print(f"\n连接新服务器继续运行：")
        print(f"  ssh -p {REMOTE_PORT_SSH} {REMOTE_USER}@{REMOTE_HOST}")
    else:
        print(f"\n[ERROR] 传输失败，退出码: {status}")
