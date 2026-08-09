import json

with open('tests/fixtures/evaluation/phase01_mock_cases.jsonl', encoding='utf-8') as f:
    for line_num, line in enumerate(f.readlines(), 1):
        case = json.loads(line)
        tools = case['input']['available_tools']
        if tools:
            print(f'Line {line_num}: id={case["id"]}')
