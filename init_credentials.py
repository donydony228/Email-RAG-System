# 初始化 credentials
# 從環境變數創建 Gmail credentials 文件
import os
import json
import base64
from pathlib import Path

def safe_b64decode(s: str) -> bytes:
    """Decode base64, handling missing padding from copy-paste or Secrets trimming."""
    s = s.strip()
    s += '=' * (-len(s) % 4)
    return base64.b64decode(s)


def init_credentials():
    """從環境變數初始化 credentials 文件"""

    # 創建 credentials 目錄
    credentials_dir = Path("credentials")
    credentials_dir.mkdir(exist_ok=True)

    # 檢查是否使用多帳號模式
    multi_account = os.getenv('GMAIL_MULTI_ACCOUNT', 'false').lower() == 'true'

    if multi_account:
        # 多帳號模式
        for i in range(1, 4):  # 支援最多 3 個帳號
            cred_key = f'GMAIL_CREDENTIALS_ACCOUNT{i}_BASE64'
            token_key = f'GMAIL_TOKEN_ACCOUNT{i}_BASE64'

            cred_base64 = os.getenv(cred_key)
            token_base64 = os.getenv(token_key)

            if cred_base64:
                try:
                    cred_data = safe_b64decode(cred_base64)
                    cred_file = credentials_dir / f'credentials_account{i}.json'
                    cred_file.write_bytes(cred_data)
                    print(f"已創建 {cred_file} ({len(cred_data)} bytes)")
                except Exception as e:
                    print(f"錯誤：無法創建 {cred_key}: {e}")

            if token_base64:
                try:
                    token_data = safe_b64decode(token_base64)
                    token_file = credentials_dir / f'token_account{i}.json'
                    token_file.write_bytes(token_data)
                    print(f"已創建 {token_file} ({len(token_data)} bytes)")

                    # 驗證 pickle 格式
                    if not token_data.startswith(b'\x80'):
                        print(f"警告：{token_file} 不是有效的 pickle 格式")
                        print(f"前 20 bytes: {token_data[:20]}")
                except Exception as e:
                    print(f"錯誤：無法創建 {token_key}: {e}")

    else:
        # 單一帳號模式
        cred_base64 = os.getenv('GMAIL_CREDENTIALS_BASE64')
        token_base64 = os.getenv('GMAIL_TOKEN_BASE64')

        if cred_base64:
            try:
                cred_data = safe_b64decode(cred_base64)
                cred_file = credentials_dir / 'credentials.json'
                cred_file.write_bytes(cred_data)
                print(f"已創建 {cred_file} ({len(cred_data)} bytes)")
            except Exception as e:
                print(f"錯誤：無法創建 GMAIL_CREDENTIALS_BASE64: {e}")

        if token_base64:
            try:
                token_data = safe_b64decode(token_base64)
                token_file = credentials_dir / 'token.json'
                token_file.write_bytes(token_data)
                print(f"已創建 {token_file} ({len(token_data)} bytes)")

                # 驗證 pickle 格式
                if not token_data.startswith(b'\x80'):
                    print(f"警告：{token_file} 不是有效的 pickle 格式")
                    print(f"前 20 bytes: {token_data[:20]}")
            except Exception as e:
                print(f"錯誤：無法創建 GMAIL_TOKEN_BASE64: {e}")

if __name__ == "__main__":
    init_credentials()
