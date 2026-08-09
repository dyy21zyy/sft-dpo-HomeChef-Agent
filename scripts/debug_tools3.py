import json
with open('tests/fixtures/evaluation/phase01_mock_cases.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        case = json.loads(line)
        if case['id'] == 'case_valid_search_tool_call':
            props = case['input']['available_tools'][0]['function']['parameters']['properties']
            for key, val in sorted(props.items()):
                print(f"  {key}: {json.dumps(val, ensure_ascii=False)}")
            break
