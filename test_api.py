"""
test_api.py
测试 API 连接是否正常，发送一条简单请求并打印响应。
"""

import yaml
from openai import OpenAI


def load_config(config_path: str = "configs/api_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def test_api():
    config = load_config()
    api_cfg = config["api"]

    print(f"[INFO] API Base URL : {api_cfg['base_url']}")
    print(f"[INFO] Model        : {api_cfg['model']}")
    print(f"[INFO] API Key      : {api_cfg['api_key'][:10]}...")

    client = OpenAI(
        api_key=api_cfg["api_key"],
        base_url=api_cfg["base_url"],
    )

    print("\n[INFO] 正在发送测试请求...")
    try:
        resp = client.chat.completions.create(
            model=api_cfg["model"],
            messages=[
                {"role": "user", "content": "Say 'API test successful' in JSON: {\"status\": \"ok\"}"}
            ],
            temperature=0,
            max_tokens=50,
        )
        content = resp.choices[0].message.content
        tokens = resp.usage.total_tokens if resp.usage else "N/A"
        print(f"\n[SUCCESS] API 调用成功！")
        print(f"  响应内容 : {content}")
        print(f"  总 token : {tokens}")
    except Exception as e:
        print(f"\n[ERROR] API 调用失败: {e}")
        print("\n[尝试] 使用带 /v1 的 base_url 重试...")
        base_url_v1 = api_cfg["base_url"].rstrip("/") + "/v1"
        client2 = OpenAI(
            api_key=api_cfg["api_key"],
            base_url=base_url_v1,
        )
        try:
            resp2 = client2.chat.completions.create(
                model=api_cfg["model"],
                messages=[
                    {"role": "user", "content": "Say 'API test successful' in JSON: {\"status\": \"ok\"}"}
                ],
                temperature=0,
                max_tokens=50,
            )
            content2 = resp2.choices[0].message.content
            tokens2 = resp2.usage.total_tokens if resp2.usage else "N/A"
            print(f"\n[SUCCESS] 带 /v1 的 base_url 调用成功！")
            print(f"  响应内容 : {content2}")
            print(f"  总 token : {tokens2}")
            print(f"\n[提示] 请将 api_config.yaml 中的 base_url 修改为: {base_url_v1}")
        except Exception as e2:
            print(f"\n[ERROR] 仍然失败: {e2}")


if __name__ == "__main__":
    test_api()
