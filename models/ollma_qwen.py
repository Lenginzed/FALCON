import requests

payload = {
    "model": "qwen3:8b",
    "messages": [
        {
            "role": "system",
            "content": "You must output valid JSON only. Do not output reasoning, thinking, markdown, explanations, or code fences."
        },
        {
            "role": "user",
            "content": "只输出 JSON：{\"ok\": true}"
        }
    ],
    "temperature": 0.1,
    "top_p": 0.8,
    "max_tokens": 128,
    "stream": False,
    "reasoning_effort": "none",
    "reasoning": {
        "effort": "none"
    }
}

r = requests.post(
    "http://localhost:11434/v1/chat/completions",
    json=payload,
    timeout=120
)

print(r.status_code)
print(r.text)