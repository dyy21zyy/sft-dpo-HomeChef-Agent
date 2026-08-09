import json
with open('tests/fixtures/evaluation/phase01_mock_cases.jsonl', 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f.readlines(), 1):
        case = json.loads(line)
        tools = case['input']['available_tools']
        if tools and case['id'] == 'case_valid_search_tool_call':
            props = tools[0]['function']['parameters']['properties']
            print(f"start_time in fixture: {json.dumps(props.get('start_time'), ensure_ascii=False)}")
            # Also show canonical
            break
