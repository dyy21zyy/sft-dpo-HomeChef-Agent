import json
import math
import statistics
import time
from pathlib import Path

import requests

URL = "http://127.0.0.1:8080/v1/chat/completions"

MODEL = "Qwen3-4B-Instruct-2507"

PROMPT = """
用户想预约上门私厨。
时间：明天晚上六点。
人数：2人。
地址：上海市徐汇区。
偏好：川菜。
请按照预约系统协议给出下一步决策。
""".strip()

WARMUP = 1
RUNS = 20


def request_once():

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
            }
        ],
        "temperature": 0,
        "max_tokens": 512,
        "stream": True,
        "stream_options": {
            "include_usage": True
        },
    }

    start = time.perf_counter()

    first_content = None
    completion_tokens = None

    with requests.post(
        URL,
        json=payload,
        stream=True,
        timeout=900,
    ) as response:

        response.raise_for_status()

        for raw in response.iter_lines():

            if not raw:
                continue

            line = raw.decode("utf-8")

            if not line.startswith("data: "):
                continue

            data = line[6:]

            if data == "[DONE]":
                break

            try:
                obj = json.loads(data)
            except Exception:
                continue

            usage = obj.get("usage")

            if usage:
                completion_tokens = usage.get(
                    "completion_tokens",
                    completion_tokens,
                )

            choices = obj.get("choices") or []

            if choices:
                delta = choices[0].get("delta") or {}

                content = delta.get("content")

                if content and first_content is None:
                    first_content = time.perf_counter()

    end = time.perf_counter()

    if first_content is None:
        first_content = end

    ttft = first_content - start
    latency = end - start

    generation_time = max(
        end - first_content,
        1e-9,
    )

    throughput = None

    if completion_tokens is not None:
        throughput = (
            completion_tokens /
            generation_time
        )

    return {
        "ttft_s": ttft,
        "latency_s": latency,
        "completion_tokens": completion_tokens,
        "tokens_per_second": throughput,
    }


print("=== WARMUP ===")

for _ in range(WARMUP):
    print(request_once())


print()
print("=== MEASURED RUNS ===")

rows = []

for i in range(RUNS):

    row = request_once()

    rows.append(row)

    print(
        f"{i+1:02d}/{RUNS}",
        f"TTFT={row['ttft_s']:.3f}s",
        f"E2E={row['latency_s']:.3f}s",
        f"TPS={row['tokens_per_second']}",
    )


latencies = sorted(
    x["latency_s"]
    for x in rows
)

ttfts = [
    x["ttft_s"]
    for x in rows
]

throughputs = [
    x["tokens_per_second"]
    for x in rows
    if x["tokens_per_second"] is not None
]

p95_index = (
    math.ceil(0.95 * len(latencies)) - 1
)

summary = {
    "runs": RUNS,
    "mean_latency_s":
        statistics.mean(latencies),

    "p95_latency_s":
        latencies[p95_index],

    "mean_ttft_s":
        statistics.mean(ttfts),

    "mean_tokens_per_second":
        statistics.mean(throughputs)
        if throughputs
        else None,
}

print()
print("=== SUMMARY ===")
print(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
)

Path(
    "reports/generated/phase05/"
    "current_cpu_benchmark.json"
).write_text(
    json.dumps(
        {
            "summary": summary,
            "runs": rows,
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)
