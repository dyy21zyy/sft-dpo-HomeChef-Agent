"""Phase 03 v0.2.1 — Chinese Surface Realizer.

Replaces English business text in Raw v0.2 samples with natural Chinese
without altering Gold action, reply_type, tool_name, JSON keys, slot values,
Tool Result semantics, or state transitions.

Also provides difficulty assignment and diversity tracking.
"""
from __future__ import annotations

import random

# ═══════════════════════════════════════════════════════════
# 25 Style Anchors (difficulty + scenario + Chinese text)
# ═══════════════════════════════════════════════════════════

STYLE_ANCHORS = [
    # id, difficulty, scenario, text
    (1, "easy", "missing_date", "我想找个师傅上门做川菜。"),
    (2, "easy", "missing_time", "这周六想请厨师来家里做饭，4个人，在上海市徐汇区。"),
    (3, "easy", "missing_people", "明天晚上六点，地址在北京市朝阳区望京附近，想吃粤菜。"),
    (4, "easy", "missing_address", "后天晚上七点，5个人，想吃家常菜。"),
    (5, "easy", "search_complete", "8月20日晚上六点，4个人，在杭州市西湖区文三路附近，预算600元，想吃川菜。"),
    (6, "easy", "specific_chef", "我想约张师傅，周五晚上七点，3个人，地址在上海市静安区。"),
    (7, "easy", "dietary_constraint", "明晚6点，4个人，在南京市鼓楼区，其中一位客人花生过敏。"),
    (8, "easy", "confirm", "可以，就订李师傅吧。"),
    (9, "medium", "relative_time", "这周六晚上想找个做粤菜的师傅，6个人，地址在深圳市南山区。"),
    (10, "medium", "relative_budget", "下周末家里聚餐，大概8个人，预算1000元左右，想吃江浙菜。"),
    (11, "medium", "fill_slot", "地址在上海市浦东新区世纪大道附近，其他条件不变。"),
    (12, "medium", "modify_time", "六点有点早，改成晚上七点吧，其他都不变。"),
    (13, "medium", "modify_people", "刚确认了一下，不是4个人，是6个人。"),
    (14, "medium", "add_dietary", "对了，还有一位客人不吃香菜，而且要少辣。"),
    (15, "medium", "change_cuisine", "川菜先不要了，换成粤菜吧，时间和地址都不变。"),
    (16, "medium", "specific_unavailable", "王师傅没空的话，再帮我看看同一时间有没有其他做家常菜的师傅。"),
    (17, "medium", "candidate_select", "第二位陈师傅看起来不错，就选他吧。"),
    (18, "hard", "relative_correction", "不是明天，我刚说错了，是后天晚上七点。"),
    (19, "hard", "multi_slot_inherit", "地址改到杭州市滨江区，人数还是6个，预算和忌口都不要变。"),
    (20, "hard", "requery_date", "张师傅周六有空是吧？那我改成周日晚上，还能约他吗？"),
    (21, "hard", "requery_dietary", "先别订，刚知道有客人海鲜过敏。其他条件不变，重新帮我看看合适的师傅。"),
    (22, "hard", "multi_modify", "时间从六点改到七点半，人数改成8个，预算提高到1200，还是想吃粤菜。"),
    (23, "hard", "reject_requery", "这几个都不太合适，有没有其他师傅？时间和地点都不变。"),
    (24, "hard", "confirm_modify", "先别确认预约，我想把地址改到上海市长宁区，然后再确认一下这个师傅还能不能来。"),
    (25, "hard", "complex_multi", "下周六给老人过生日，7个人，在成都市武侯区，想吃清淡一点的家常菜，有人乳糖不耐，也不能吃太辣，预算控制在1000元以内。"),
]

# ═══════════════════════════════════════════════════════════
# Chinese Entity Pool
# ═══════════════════════════════════════════════════════════

CITIES = [
    ("上海市", ["徐汇区", "静安区", "浦东新区", "长宁区", "黄浦区", "杨浦区", "闵行区", "普陀区", "虹口区", "宝山区", "嘉定区", "松江区"]),
    ("北京市", ["朝阳区", "海淀区", "西城区", "东城区", "丰台区", "通州区", "大兴区", "石景山区", "顺义区", "昌平区", "房山区"]),
    ("杭州市", ["西湖区", "滨江区", "余杭区", "拱墅区", "上城区", "萧山区", "临平区", "钱塘区", "富阳区", "临安区"]),
    ("深圳市", ["南山区", "福田区", "罗湖区", "宝安区", "龙岗区", "龙华区", "盐田区", "光明区", "坪山区", "大鹏新区"]),
    ("广州市", ["天河区", "越秀区", "海珠区", "白云区", "番禺区", "黄埔区", "荔湾区", "南沙区", "增城区", "从化区"]),
    ("成都市", ["武侯区", "锦江区", "青羊区", "金牛区", "成华区", "高新区", "双流区", "郫都区", "温江区", "龙泉驿区"]),
    ("南京市", ["鼓楼区", "玄武区", "秦淮区", "建邺区", "栖霞区", "江宁区", "雨花台区", "浦口区", "六合区"]),
    ("武汉市", ["武昌区", "江岸区", "江汉区", "洪山区", "汉阳区", "硚口区", "青山区", "东西湖区", "东湖高新区"]),
    ("苏州市", ["姑苏区", "工业园区", "吴中区", "相城区", "虎丘区", "吴江区", "高新区", "昆山市", "太仓市"]),
    ("重庆市", ["渝中区", "江北区", "南岸区", "渝北区", "沙坪坝区", "九龙坡区", "北碚区", "巴南区", "大渡口区"]),
    ("天津市", ["和平区", "河西区", "南开区", "河东区", "河北区", "红桥区", "西青区", "北辰区", "滨海新区"]),
    ("西安市", ["雁塔区", "碑林区", "未央区", "莲湖区", "新城区", "长安区", "灞桥区", "曲江新区", "高新区"]),
    ("青岛市", ["市南区", "市北区", "李沧区", "崂山区", "城阳区", "黄岛区"]),
    ("厦门市", ["思明区", "湖里区", "集美区", "海沧区", "同安区", "翔安区"]),
    ("长沙市", ["岳麓区", "天心区", "雨花区", "开福区", "芙蓉区", "望城区"]),
    ("济南市", ["历下区", "市中区", "槐荫区", "天桥区", "历城区", "高新区"]),
]

# Street suffixes for natural address generation
STREET_SUFFIXES = [
    "附近", "世纪大道附近", "地铁站旁边", "商圈附近",
    "软件园", "大学城附近", "CBD区域", "居民区",
    "", "",  # some addresses just district
]

CUISINES = [
    "川菜", "粤菜", "湘菜", "鲁菜", "苏菜", "浙菜",
    "闽菜", "徽菜", "京菜", "沪菜", "东北菜", "西北菜",
    "家常菜", "淮扬菜", "本帮菜", "云南菜", "赣菜", "豫菜",
    "湖北菜", "陕西菜", "客家菜", "潮汕菜", "火锅", "烧烤",
]

CHEF_SURNAMES = [
    "张", "李", "王", "陈", "刘", "赵", "黄", "周",
    "吴", "徐", "孙", "马", "朱", "胡", "郭", "何",
    "高", "林", "罗", "郑", "梁", "谢", "宋", "唐",
]

CHEF_FIRST = ["师傅", "大厨", "老师", "主厨"]

DIETARY_SINGLE = [
    "花生过敏", "海鲜过敏", "不吃香菜", "不吃辣",
    "乳糖不耐", "不吃猪肉", "不吃牛肉", "素食",
    "不吃蒜", "不吃葱", "少油", "少盐",
]

DIETARY_COMBOS = [
    ["花生过敏", "不吃辣"],
    ["海鲜过敏", "不吃香菜"],
    ["乳糖不耐", "少油少盐"],
    ["素食", "不吃葱蒜"],
    ["不吃猪肉", "不吃牛肉"],
    ["花生过敏", "少辣"],
    ["海鲜过敏", "乳糖不耐"],
    ["不吃辣", "少油"],
    ["不吃香菜", "不吃葱"],
    ["素食", "不吃辣"],
]

OCCASIONS = [
    "家宴", "生日宴", "朋友聚会", "商务宴请",
    "家庭聚餐", "同学聚会", "节日聚餐", "纪念日",
]

_DISHES = [
    "回锅肉", "麻婆豆腐", "水煮鱼", "清蒸鲈鱼", "白切鸡", "蚝油生菜",
    "红烧肉", "糖醋排骨", "清炒时蔬", "剁椒鱼头", "小炒肉", "蒜蓉西兰花",
    "西湖醋鱼", "东坡肉", "龙井虾仁", "宫保鸡丁", "鱼香肉丝", "干煸四季豆",
    "白灼虾", "叉烧", "上汤娃娃菜", "酸菜鱼", "毛血旺", "辣子鸡",
    "葱烧海参", "九转大肠", "油焖大虾", "松鼠桂鱼", "蟹粉豆腐", "响油鳝糊",
    "佛跳墙", "荔枝肉", "沙茶牛肉", "椒盐排骨", "铁板牛肉", "清炒芥蓝",
    "香辣蟹", "蒜蓉粉丝蒸虾", "糖醋里脊", "黑椒牛柳", "口水鸡", "夫妻肺片",
    "蚂蚁上树", "干锅花菜", "番茄炒蛋", "玉米排骨汤", "冬瓜老鸭汤", "紫菜蛋花汤",
    "凉拌黄瓜", "拍黄瓜", "皮蛋豆腐", "蒜泥白肉", "红油耳丝", "口水鸡爪",
    "家常豆腐", "地三鲜", "干煸豆角", "鱼香茄子", "老干妈炒蛋", "木须肉",
]

MENU_COMBOS = [[]] + [
    [_DISHES[i % len(_DISHES)], _DISHES[(i + 7) % len(_DISHES)], _DISHES[(i + 23) % len(_DISHES)]]
    for i in range(70)
]

# ═══════════════════════════════════════════════════════════
# Relative-time expression types (≥10)
# ═══════════════════════════════════════════════════════════

# Maps a relative-time expression type → resolved date ISO string.
# base_datetime = 2026-08-12 18:00 (Wednesday, weekday=2)
# MUST match the validator's _RELATIVE_TIME_MAP in semantic_validators.py.
RELATIVE_TIME_RESOLVER: dict[str, str] = {
    "today": "2026-08-12",
    "tonight": "2026-08-12",
    "tomorrow": "2026-08-13",
    "tomorrow_evening": "2026-08-13",
    "day_after_tomorrow": "2026-08-14",
    "this_saturday": "2026-08-15",
    "this_weekend_saturday": "2026-08-15",
    "this_sunday": "2026-08-16",
    "this_weekend_sunday": "2026-08-16",
    "next_friday": "2026-08-14",
    "next_monday": "2026-08-17",
    "next_weekend": "2026-08-15",
}

RELATIVE_TIME_CHINESE: dict[str, str] = {
    "today": "今天",
    "tonight": "今晚",
    "tomorrow": "明天",
    "tomorrow_evening": "明晚",
    "day_after_tomorrow": "后天",
    "this_saturday": "这周六",
    "this_weekend_saturday": "本周末（周六）",
    "this_sunday": "这周日",
    "this_weekend_sunday": "本周末（周日）",
    "next_friday": "下周五",
    "next_monday": "下周一",
    "next_weekend": "下周末",
}

# ═══════════════════════════════════════════════════════════
# Chinese user_input template pools per scenario
# ═══════════════════════════════════════════════════════════

def _pick_city_district(rng: random.Random) -> tuple[str, str]:
    city, districts = rng.choice(CITIES)
    return city, rng.choice(districts)


# ── Clause-composition engine for high skeleton diversity ──

# Connector variants (produce different skeletons)
_CONNECTORS = ["，", "，", ",", "；", "，而且", "，然后", "；同时"]
_ENDINGS = ["。", "。", "！", "", "，谢谢", "，麻烦安排一下"]

def _compose_clauses(rng: random.Random, clauses: list[str], order_permute: bool = True) -> str:
    """Compose slot clauses into a sentence with varied order, connectors, endings.

    clauses: list of already-formatted Chinese clause strings.
    Produces many distinct skeletons via permutation + connector choice.
    """
    if order_permute and len(clauses) > 1:
        rng.shuffle(clauses)
    connector = rng.choice(_CONNECTORS)
    ending = rng.choice(_ENDINGS)
    sentence = connector.join(clauses) + ending
    return sentence


def _date_clauses(date_str: str) -> list[str]:
    return [
        f"{date_str}",
        f"{date_str}这一天",
        f"在{date_str}",
    ]


def _time_clause(time_str: str) -> str:
    return f"{time_str}"


def _people_clause(people_v: str) -> str:
    return f"{people_v}"


def _addr_clause(addr: str) -> str:
    return f"地址在{addr}"


def _addr2_clause(addr: str) -> str:
    return f"在{addr}"


def _cuisine_clause(cuisine: str) -> str:
    return f"想吃{cuisine}"


def _cuisine2_clause(cuisine: str) -> str:
    return f"{cuisine}"


def _budget_clause(budget: str) -> str:
    return f"预算{budget}"


def _occasion_clause(occ: str) -> str:
    return f"{occ}"


def _pick_chef_display(rng: random.Random) -> str:
    return rng.choice(CHEF_SURNAMES) + rng.choice(CHEF_FIRST)


def _time_variants(hour: int) -> list[str]:
    """Multiple Chinese expressions for the same time."""
    h = int(hour)
    return [
        f"{h}点",
        f"晚上{h}点" if h >= 18 else f"上午{h}点" if h < 12 else f"下午{h}点",
        f"{h}点钟",
        f"{h}点整",
        f"大概{h}点",
        f"{h}点左右",
        f"{h}点前后",
        f"{h}时",
        f"晚上{h}点整" if h >= 18 else f"下午{h}点整",
        f"{h}:00",
    ]


def _people_variants(n: int) -> list[str]:
    digits = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二"]
    chinese_num = digits[n] if n <= 12 else str(n)
    return [
        f"{n}个人",
        f"{n}人",
        f"{n}位",
        f"{n}位客人",
        f"{chinese_num}个人",
        f"{chinese_num}人",
        f"大概{n}人",
        f"一共{n}人",
        f"{n}人左右",
        f"{chinese_num}位",
        f"家里{n}口人",
    ]


# ═══════════════════════════════════════════════════════════
# Scenario-level user_input generators
# ═══════════════════════════════════════════════════════════

def _gen_missing_date_input(rng: random.Random, facts) -> str:
    cuisine = facts.booking_state.cuisine or "家常菜"
    # Many distinct skeletons for the "missing date" scenario
    templates = [
        "我想找个师傅上门做{cuisine}。",
        "帮我找个做{cuisine}的厨师。",
        "需要一位{cuisine}厨师，上门做饭。",
        "能帮忙约个{cuisine}师傅吗？",
        "帮我安排一个{cuisine}私厨。",
        "找{cuisine}厨师，上门服务。",
        "请一位会做{cuisine}的师傅到家里来。",
        "想请个{cuisine}师傅，什么时候都行。",
        "{cuisine}师傅有空吗？想上门做一桌。",
        "帮我约一个{cuisine}师傅，谢谢。",
        "家里要请客，需要{cuisine}师傅。",
        "有没有做{cuisine}的师傅能上门？",
        "想订一位{cuisine}私厨师傅。",
        "麻烦帮我找个{cuisine}厨师。",
        "需要上门做{cuisine}的师傅。",
        "{cuisine}厨师，可以约吗？",
    ]
    return rng.choice(templates).format(cuisine=cuisine)


def _gen_missing_time_input(rng: random.Random, facts) -> str:
    bs = facts.booking_state
    date_str = bs.service_date or "这周末"
    people_v = rng.choice(_people_variants(bs.people or 4))
    city, district = _pick_city_district(rng)
    addr = f"{city}{district}"
    # Frame + slot-order variations produce many skeletons
    date_expr = rng.choice([date_str, f"就是{date_str}", f"定在{date_str}"])
    people_expr = rng.choice([f"{people_v}", f"一共{people_v}"])
    templates = [
        "{date}想请厨师来家里做饭，{people}，在{addr}。",
        "{date}需要一位厨师，{people}，地址{addr}。",
        "约{date}的家宴，{people}，{addr}。",
        "{date}家里聚餐，{people}，{addr}，帮我安排一下。",
        "想约{date}的师傅，{people}，{addr}。",
        "{people}，{addr}，{date}请个厨师。",
        "{addr}，{date}，{people}，帮我找个师傅。",
        "麻烦{date}安排一位厨师，{people}，{addr}。",
        "请问{date}能约到师傅吗？{people}，{addr}。",
        "我需要{date}的厨师，{people}，{addr}。",
        "约{date}，{people}，{addr}，谢谢。",
        "家里{date}吃饭，{people}，{addr}，需要厨师。",
    ]
    t = rng.choice(templates)
    return t.format(date=date_expr, people=people_expr, addr=addr)


def _gen_missing_people_input(rng: random.Random, facts) -> str:
    bs = facts.booking_state
    time_str = rng.choice(_time_variants(int(bs.start_time.split(":")[0]) if bs.start_time else 18))
    date_str = bs.service_date or "明天"
    city, district = _pick_city_district(rng)
    cuisine = bs.cuisine or "家常菜"
    templates = [
        "{date}{time}，地址在{city}{district}，想吃{cuisine}。",
        "{date}{time}，{city}{district}，{cuisine}。",
        "想约{date}{time}的{cuisine}，在{city}{district}。",
        "{date}{time}，{cuisine}师傅，{city}{district}。",
        "{city}{district}，{date}{time}，想要{cuisine}。",
        "请{date}{time}做{cuisine}，{city}{district}。",
        "约{date}{time}，{city}{district}，{cuisine}。",
        "{date}{time}要吃{cuisine}，在{city}{district}。",
    ]
    return rng.choice(templates).format(date=date_str, time=time_str, city=city, district=district, cuisine=cuisine)


def _gen_missing_address_input(rng: random.Random, facts) -> str:
    bs = facts.booking_state
    time_str = rng.choice(_time_variants(int(bs.start_time.split(":")[0]) if bs.start_time else 19))
    date_str = bs.service_date or "后天"
    people_v = rng.choice(_people_variants(bs.people or 5))
    cuisine = bs.cuisine or "家常菜"
    templates = [
        "{date}{time}，{people}，想吃{cuisine}。",
        "{date}{time}，{people}，{cuisine}。",
        "{date}{time}约个{cuisine}师傅，{people}。",
        "{people}，{date}{time}，{cuisine}。",
        "{date}{time}需要{cuisine}，{people}。",
        "想吃{cuisine}，{date}{time}，{people}。",
        "约{date}{time}，{people}，{cuisine}。",
    ]
    return rng.choice(templates).format(date=date_str, time=time_str, people=people_v, cuisine=cuisine)


def _gen_search_complete_input(rng: random.Random, facts) -> str:
    """Combinatorial search-complete user_input with varying slot order.

    Uses clause-composition to produce many distinct skeletons.
    """
    bs = facts.booking_state
    time_str = rng.choice(_time_variants(int(bs.start_time.split(":")[0]) if bs.start_time else 18))
    people_v = rng.choice(_people_variants(bs.people or 4))
    city, district = _pick_city_district(rng)
    cuisine = bs.cuisine or "川菜"
    addr = f"{city}{district}"
    date = bs.service_date or "8月15日"
    budget_raw = bs.budget_max or bs.budget_min or ""
    budget = f"{budget_raw}元" if budget_raw else ""
    occasion = bs.occasion or "家宴"

    # Choose a sentence frame style, then compose clauses combinatorially
    style = rng.random()
    if style < 0.3:
        # Clause-composition: 5 slot clauses, shuffled
        clauses = [
            rng.choice(_date_clauses(date)),
            _time_clause(time_str),
            _people_clause(people_v),
            rng.choice([_addr_clause(addr), _addr2_clause(addr)]),
            rng.choice([_cuisine_clause(cuisine), _cuisine2_clause(cuisine)]),
        ]
        if budget:
            clauses.append(_budget_clause(budget))
        if occasion and rng.random() < 0.5:
            clauses.append(_occasion_clause(occasion))
        return _compose_clauses(rng, clauses)

    # Fixed-frame variations with different slot orders
    frames = [
        "{date}{time}，{people}，在{addr}{budget}，想吃{cuisine}{occasion}。",
        "在{addr}，{date}{time}，{people}，{cuisine}，{budget}。",
        "想吃{cuisine}，{date}{time}，{people}，{addr}{budget}。",
        "{date}{time}，{people}，{addr}，{cuisine}{budget}。",
        "帮我安排{date}{time}的{occasion}，{people}，{addr}，{cuisine}{budget}。",
        "请问{date}{time}有{cuisine}师傅吗？{people}，{addr}{budget}。",
        "{time}{date}，{people}，{addr}，{cuisine}{budget}。",
        "{budget}以内，{date}{time}，{people}，{addr}，{cuisine}。",
        "{people}的{occasion}，{date}{time}，{addr}，{cuisine}{budget}。",
        "约{date}{time}，{people}，{addr}，{cuisine}{budget}。",
        "{addr}，{date}{time}，{people}，{cuisine}{budget}。",
        "麻烦{date}{time}安排{cuisine}，{people}，{addr}{budget}。",
    ]
    frame = rng.choice(frames)
    return frame.format(
        date=date, time=time_str, people=people_v,
        addr=addr, cuisine=cuisine, budget=budget,
        occasion=occasion,
    )


def _gen_search_complete_input_old(rng: random.Random, facts) -> str:
    bs = facts.booking_state
    time_str = rng.choice(_time_variants(int(bs.start_time.split(":")[0]) if bs.start_time else 18))
    people_v = rng.choice(_people_variants(bs.people or 4))
    city, district = _pick_city_district(rng)
    cuisine = bs.cuisine or "川菜"
    budget = f"，预算{bs.budget_max}元" if bs.budget_max else ""
    occasion = f"，{bs.occasion}" if bs.occasion else ""

    styles = [
        # 正式
        "{date}{time}，{people}，在{city}{district}{budget}，想吃{cuisine}{occasion}。",
        # 口语
        "{date}{time}，{people}，{city}{district}，{cuisine}{budget}。",
        # 一次给全
        "帮我安排{date}{time}的家宴，{people}，地址{city}{district}，{cuisine}{budget}{occasion}。",
        # 礼貌询问
        "请问{date}{time}有{cuisine}师傅吗？{people}，{city}{district}{budget}。",
        # 简短
        "{date}{time}，{people}，{city}{district}，{cuisine}{budget}。",
    ]
    return rng.choice(styles).format(
        date=bs.service_date or "8月15日", time=time_str, people=people_v,
        city=city, district=district, cuisine=cuisine, budget=budget,
        occasion=occasion,
    )


def _gen_specific_chef_input(rng: random.Random, facts) -> str:
    chef = facts.requested_chef_name or _pick_chef_display(rng)
    bs = facts.booking_state
    time_str = rng.choice(_time_variants(int(bs.start_time.split(":")[0]) if bs.start_time else 19))
    people_v = rng.choice(_people_variants(bs.people or 3))
    city, district = _pick_city_district(rng)
    date_str = bs.service_date or "周五"

    templates = [
        "我想约{chef}，{date}{time}，{people}，地址在{city}{district}。",
        "帮我问问{chef}{date}有没有空，{people}，{city}{district}。",
        "{chef}能约吗？{date}{time}，{people}，{city}{district}。",
        "想找{chef}，{date}{time}，{people}，{city}{district}。",
    ]
    return rng.choice(templates).format(
        chef=chef, date=date_str, time=time_str, people=people_v, city=city, district=district,
    )


def _gen_dietary_constraint_input(rng: random.Random, facts) -> str:
    bs = facts.booking_state
    time_str = rng.choice(_time_variants(int(bs.start_time.split(":")[0]) if bs.start_time else 18))
    people_v = rng.choice(_people_variants(bs.people or 4))
    city, district = _pick_city_district(rng)
    diet = "、".join(bs.dietary_constraints) if bs.dietary_constraints else "花生过敏"

    templates = [
        "{time}，{people}，在{city}{district}，有客人{diet}。",
        "{time}，{people}，{city}{district}，需要注意{diet}。",
        "帮我找厨师，{time}，{people}，{city}{district}，其中一位{diet}。",
    ]
    return rng.choice(templates).format(time=time_str, people=people_v, city=city, district=district, diet=diet)


def _gen_confirm_input(rng: random.Random, facts) -> str:
    chef = facts.requested_chef_name or "这个师傅"
    templates = [
        f"可以，就订{chef}吧。",
        f"好的，确认{chef}。",
        f"行，就{chef}了。",
        f"没问题，帮我确认{chef}。",
        "确认预约。",
        "可以，确认吧。",
    ]
    return rng.choice(templates)


def _gen_modify_time_input(rng: random.Random, facts) -> str:
    bs = facts.booking_state
    new_time = f"{bs.start_time.split(':')[0]}点" if bs.start_time else "七点"
    templates = [
        f"改成{new_time}吧，其他都不变。",
        f"时间调整到{new_time}，别的不用改。",
        f"{new_time}行吗？其他条件不变。",
        f"改一下时间，{new_time}。",
        f"还是{new_time}比较合适，麻烦改一下。",
    ]
    return rng.choice(templates)


def _gen_modify_people_input(rng: random.Random, facts) -> str:
    bs = facts.booking_state
    people_v = rng.choice(_people_variants(bs.people or 6))
    templates = [
        f"刚确认了，不是之前说的人数，是{people_v}。",
        f"人数搞错了，是{people_v}。",
        f"改一下，{people_v}。",
        f"不好意思，人数要改成{people_v}。",
    ]
    return rng.choice(templates)


def _gen_add_dietary_input(rng: random.Random, facts) -> str:
    diet = "、".join(facts.booking_state.dietary_constraints[-2:]) if facts.booking_state.dietary_constraints else "不吃香菜"
    templates = [
        f"对了，还有客人{diet}。",
        f"补充一下，有人{diet}。",
        f"差点忘了，{diet}。",
        f"追加一个要求，{diet}。",
    ]
    return rng.choice(templates)


def _gen_change_cuisine_input(rng: random.Random, facts) -> str:
    cuisine = facts.booking_state.cuisine or "粤菜"
    templates = [
        f"还是换成{cuisine}吧。",
        f"改成{cuisine}，时间和地址都不变。",
        f"不想吃之前的菜系了，换成{cuisine}。",
        f"换{cuisine}吧，其他不变。",
    ]
    return rng.choice(templates)


def _gen_relative_correction_input(rng: random.Random, facts) -> str:
    bs = facts.booking_state
    correct = bs.service_date or "后天"
    templates = [
        f"不是明天，我说错了，是{correct}。",
        f"纠正一下，不是之前说的日期，是{correct}。",
        f"搞错了，{correct}才对。",
        f"抱歉说错了，{correct}晚上。",
    ]
    return rng.choice(templates)


def _gen_multi_slot_inherit_input(rng: random.Random, facts) -> str:
    city, district = _pick_city_district(rng)
    people_v = rng.choice(_people_variants(facts.booking_state.people or 6))
    templates = [
        f"地址改到{city}{district}，人数还是{people_v}，预算和忌口都不要变。",
        f"换地址：{city}{district}，{people_v}，其他照旧。",
        f"地址改成{city}{district}，{people_v}不变，其他条件保持。",
    ]
    return rng.choice(templates)


def _gen_requery_date_input(rng: random.Random, facts) -> str:
    chef = facts.requested_chef_name or "师傅"
    bs = facts.booking_state
    new_date = bs.service_date or "周日"
    templates = [
        f"{chef}周六有空是吧？那我改成{new_date}晚上，还能约他吗？",
        f"周六能约{chef}的话，{new_date}还能约吗？",
        f"{chef}周六可以，那{new_date}呢？",
    ]
    return rng.choice(templates)


def _gen_requery_dietary_input(rng: random.Random, facts) -> str:
    templates = [
        "先别订，刚知道有客人海鲜过敏。其他条件不变，重新帮我看看合适的师傅。",
        "等一下，有位客人海鲜过敏，重新查一下。",
        "暂停，新增海鲜过敏的约束，重新搜索。",
    ]
    return rng.choice(templates)


def _gen_multi_modify_input(rng: random.Random, facts) -> str:
    bs = facts.booking_state
    time_str = bs.start_time or "七点半"
    people_v = rng.choice(_people_variants(bs.people or 8))
    budget = bs.budget_max or 1200
    cuisine = bs.cuisine or "粤菜"
    templates = [
        f"时间改到{time_str}，人数改成{people_v}，预算提高到{budget}，还是想吃{cuisine}。",
        f"调整几个：时间{time_str}、{people_v}、预算{budget}，{cuisine}不变。",
        f"重新设定：{time_str}，{people_v}，预算{budget}，{cuisine}。",
    ]
    return rng.choice(templates)


def _gen_reject_requery_input(rng: random.Random, facts) -> str:
    templates = [
        "这几个都不太合适，有没有其他师傅？时间和地点都不变。",
        "都不满意，重新推荐一下，条件不变。",
        "换一批师傅吧，时间地点照旧。",
        "再帮我找找，其他条件都不改。",
    ]
    return rng.choice(templates)


def _gen_confirm_modify_input(rng: random.Random, facts) -> str:
    city, district = _pick_city_district(rng)
    chef = facts.requested_chef_name or "这个师傅"
    templates = [
        f"先别确认预约，我想把地址改到{city}{district}，然后再确认{chef}还能不能来。",
        f"先别确认，地址换成{city}{district}，再帮我核实{chef}。",
        f"等等，改一下地址到{city}{district}，然后确认{chef}。",
    ]
    return rng.choice(templates)


def _gen_complex_multi_input(rng: random.Random, facts) -> str:
    bs = facts.booking_state
    date_str = bs.service_date or "下周六"
    people_v = rng.choice(_people_variants(bs.people or 7))
    city, district = _pick_city_district(rng)
    occasion = bs.occasion or "过生日"
    cuisine = bs.cuisine or "家常菜"
    diet = "、".join(bs.dietary_constraints[-2:]) if bs.dietary_constraints else "乳糖不耐"
    budget = bs.budget_max or 1000
    templates = [
        f"{date_str}给老人{occasion}，{people_v}，在{city}{district}，想吃清淡一点的{cuisine}，有人{diet}，也不能吃太辣，预算控制在{budget}元以内。",
        f"{date_str}{occasion}，{people_v}，{city}{district}，{cuisine}，要清淡，{diet}，不要辣，预算{budget}以内。",
        f"安排{date_str}的{occasion}，{people_v}，{city}{district}，{cuisine}清淡口味，{diet}，少辣，预算{budget}。",
    ]
    return rng.choice(templates)


def _gen_fill_slot_input(rng: random.Random, facts) -> str:
    city, district = _pick_city_district(rng)
    templates = [
        f"地址在{city}{district}，其他条件不变。",
        f"补充地址：{city}{district}。",
        f"地址是{city}{district}，别的都不变。",
    ]
    return rng.choice(templates)


def _gen_specific_unavailable_input(rng: random.Random, facts) -> str:
    chef = facts.requested_chef_name or "王师傅"
    cuisine = facts.booking_state.cuisine or "家常菜"
    templates = [
        f"{chef}没空的话，再帮我看看同一时间有没有其他做{cuisine}的师傅。",
        f"{chef}不行就换，时间不变，找{cuisine}师傅。",
        f"{chef}约不到的话，同时段还有其他{cuisine}师傅吗？",
    ]
    return rng.choice(templates)


def _gen_candidate_select_input(rng: random.Random, facts) -> str:
    chef = "陈师傅"
    # Structurally-diverse frames (break the single 'X。' skeleton)
    templates = [
        f"第二位{chef}看起来不错，就选他吧。",
        f"就{chef}吧。",
        f"{chef}挺好的，帮我定他。",
        f"选{chef}。",
        f"那选{chef}好了。",
        f"我看{chef}可以，就他了。",
        f"帮我定{chef}。",
        f"{chef}不错，安排这位吧。",
        "就选第二位吧。",
        f"选这个{chef}。",
        f"还是选{chef}吧。",
        f"第二位的{chef}可以，就他了。",
        f"我想选第二位的{chef}，麻烦帮我安排预约。",
        f"如果{chef}有空，就约他吧。",
        f"从里面挑{chef}吧，他看起来比较合适。",
        "那就第二位吧，时间你看着安排。",
        f"帮我约{chef}，其他条件没问题。",
        f"{chef}可以，帮我确定下来。",
        f"第二位师傅挺合适的，就定{chef}了。",
        "就选这个吧，麻烦尽快安排。",
        f"我觉得{chef}不错，帮我下单预约。",
        "选第二位师傅，麻烦确认一下。",
        f"{chef}挺好的，时间也合适，就他了。",
        f"帮我选{chef}，并确认预约信息。",
    ]
    return rng.choice(templates)


def _gen_state_inheritance_input(rng: random.Random, facts) -> str:
    """Generic state inheritance: user changes something, rest stays."""
    changed = facts.booking_state
    date = changed.service_date or ""
    cuisine = changed.cuisine or ""
    budget_max = changed.budget_max
    budget = f"{budget_max}元" if budget_max else "1000元"

    # Build varied mutation utterances
    mutations = []
    if date:
        mutations += [
            f"改成{date}",
            f"日期改成{date}",
            f"时间定在{date}吧",
            f"{date}，其他不变",
            f"把日期改到{date}",
        ]
    if cuisine:
        mutations += [
            f"换{cuisine}",
            f"还是{cuisine}吧",
            f"改成{cuisine}，其他条件不变",
            f"{cuisine}就行",
            f"想换成{cuisine}",
        ]
    mutations += [
        f"预算提到{budget}",
        f"预算改成{budget}左右",
        f"预算控制到{budget}以内",
        "再查一次",
        "重新搜索一下",
        "重新帮我看看",
        "条件不变，重新查",
        "就按这个再确认一遍",
        # Longer compound variants (different skeletons, lower 'X' freq)
        f"麻烦按{date or '这个时间'}重新帮我找师傅",
        f"重新查一下{cuisine or '这个菜系'}的师傅",
        f"其他都不变，就{date or '日期'}调整一下，重新找",
        f"换个{cuisine or '菜系'}，时间和地址保持原来的",
        f"把预算提到{budget}，其余条件都不要动",
        "之前的都不算了，重新帮我筛选一遍",
        "条件我改一下，其余保持不变，重新安排",
    ]
    return rng.choice(mutations)


# ═══════════════════════════════════════════════════════════
# Chinese reply templates
# ═══════════════════════════════════════════════════════════

REPLY_TEMPLATES: dict[str, list[str]] = {
    "ask_service_date": [
        "请问您想预约哪一天呢？",
        "请告诉我您希望的服务日期。",
        "您想约哪一天？",
    ],
    "ask_start_time": [
        "请问几点开始比较合适？",
        "您希望几点开始用餐？",
        "请告诉我开始时间。",
    ],
    "ask_people": [
        "请问一共有多少人用餐？",
        "请告诉我用餐人数。",
        "几位客人呢？",
    ],
    "ask_address": [
        "请问服务地址在哪里？",
        "请提供一下您的地址。",
        "地址是哪里？",
    ],
    "ask_multiple_required_fields": [
        "请补充用餐日期、开始时间、人数和服务地址。",
        "还需要您提供日期、时间、人数和地址信息。",
        "麻烦补充一下日期、时间、人数和地址。",
    ],
    "present_chef_candidates": [
        "为您找到以下合适的厨师，请选择一位：",
        "以下是符合您要求的厨师：",
        "为您推荐以下几位师傅：",
    ],
    "confirm_specific_chef": [
        "该厨师在您指定的时间有空，需要帮您确认预约吗？",
        "这位师傅档期合适，要确认预约吗？",
        "这个时间段有档期，需要确认吗？",
    ],
    "present_alternatives": [
        "您指定的厨师暂时没空，以下是有档期的替代厨师：",
        "这位师傅没档期了，为您推荐以下替代人选：",
        "抱歉，该师傅不可用，以下是可选的师傅：",
    ],
    "inform_not_found": [
        "抱歉，没有找到该厨师的信息。",
        "系统中未查询到该厨师。",
        "没有找到您指定的师傅。",
    ],
    "inform_no_match": [
        "很遗憾，暂时没有符合您所有条件的厨师。",
        "抱歉，没有找到完全匹配的师傅。",
        "当前条件下没有合适的厨师。",
    ],
    "inform_out_of_service_area": [
        "抱歉，您所在的区域暂不支持服务。",
        "您选择的地址不在当前服务范围内。",
        "该区域暂时无法提供服务。",
    ],
    "booking_authorized": [
        "预约已确认，期待为您服务！",
        "预约成功，感谢您的信任！",
        "已为您确认预约，届时师傅会准时到达。",
    ],
    "booking_paused": [
        "好的，预约已暂停。如需继续可以随时联系我。",
        "已暂停预约，有需要随时找我。",
        "预约已取消，欢迎再次咨询。",
    ],
    "handoff": [
        "抱歉，这个问题超出了我的服务范围。请咨询其他相关服务。",
        "这不在我的能力范围内，请咨询其他渠道。",
    ],
    "acknowledge_result": [
        "好的，已了解。还有其他需要吗？",
        "收到，请问还有什么需要调整的吗？",
    ],
}

# ═══════════════════════════════════════════════════════════
# Difficulty assignment
# ═══════════════════════════════════════════════════════════

def assign_difficulty(scenario_key: str, facts) -> str:
    """Assign difficulty based on scenario type and complexity.

    Easy: single-turn, direct extraction, single-slot missing, simple confirm
    Medium: one reasoning point (relative_time, state_inheritance, specific_chef,
            tool_result, dietary, single-field modify, candidate_select)
    Hard: two+ capabilities combined (multi-turn + tool_result, requery after
          modify, stale invalidation, relative correction, multi-field modify,
          complex dietary, reject+requery, confirm+modify)
    """
    # Easy scenarios
    if scenario_key == "unrelated":
        return "easy"
    if scenario_key == "missing_required_slots":
        return "easy"

    # Medium scenarios
    if scenario_key in ("valid_search_tool_call",):
        return "medium"
    if scenario_key == "state_inheritance":
        return "medium"
    if scenario_key == "dietary_modification":
        return "medium"
    if scenario_key in ("tool_result_specific_available", "tool_result_specific_unavailable",
                        "tool_result_specific_not_found"):
        return "medium"
    if scenario_key == "tool_result_matched":
        return "medium"
    if scenario_key in ("tool_result_no_match", "tool_result_out_of_service_area"):
        return "medium"

    # Hard scenarios
    if scenario_key == "explicit_confirmation":
        return "hard"
    if scenario_key == "rejection":
        return "hard"
    if scenario_key == "tool_error":
        return "hard"

    return "medium"


def chinese_reply(reply_type: str) -> str:
    """Generate Chinese reply text for a given ReplyType."""
    templates = REPLY_TEMPLATES.get(reply_type, ["好的。"])
    return random.choice(templates)


# ═══════════════════════════════════════════════════════════
# Main Chinese surface realizer
# ═══════════════════════════════════════════════════════════

def realize_chinese_user_input(
    scenario_key: str,
    facts,
    rng: random.Random,
) -> str:
    """Generate Chinese user_input for a given scenario and ScenarioFacts.

    The returned string replaces the English user_input without changing any
    business facts, slot values, or Gold decisions.
    """
    if scenario_key == "missing_required_slots":
        # Determine which slot is missing to pick the right template
        bs = facts.booking_state
        if bs.service_date is None and bs.cuisine is None:
            return _gen_missing_date_input(rng, facts)
        elif bs.start_time is None:
            return _gen_missing_time_input(rng, facts)
        elif bs.people is None:
            return _gen_missing_people_input(rng, facts)
        elif bs.address is None:
            return _gen_missing_address_input(rng, facts)
        else:
            return _gen_missing_date_input(rng, facts)

    elif scenario_key == "valid_search_tool_call":
        return _gen_search_complete_input(rng, facts)

    elif scenario_key in ("tool_result_specific_available",):
        return _gen_specific_chef_input(rng, facts)

    elif scenario_key == "tool_result_specific_unavailable":
        return _gen_specific_unavailable_input(rng, facts)

    elif scenario_key == "tool_result_specific_not_found":
        return _gen_specific_chef_input(rng, facts)

    elif scenario_key == "tool_result_matched":
        return _gen_candidate_select_input(rng, facts)

    elif scenario_key == "tool_result_no_match":
        return rng.choice([
            "没找到合适的厨师吗？",
            "一个都没有吗？",
            "条件太苛刻了吗？怎么没合适的师傅？",
            "帮我再扩大范围找找吧，还是有别的选择？",
            "这样都找不到？能不能放宽一点条件？",
        ])

    elif scenario_key == "tool_result_out_of_service_area":
        return rng.choice([
            "这个地址能服务吗？",
            "这个区域能安排师傅吗？",
            "这地方不在服务范围内吗？",
            "这个地址可以约到师傅吗？",
        ])

    elif scenario_key == "state_inheritance":
        return _gen_state_inheritance_input(rng, facts)

    elif scenario_key == "dietary_modification":
        return _gen_requery_dietary_input(rng, facts)

    elif scenario_key == "explicit_confirmation":
        return _gen_confirm_input(rng, facts)

    elif scenario_key == "rejection":
        templates = ["算了，不订了。", "先不订了。", "取消吧。"]
        return rng.choice(templates)

    elif scenario_key == "tool_error":
        return rng.choice([
            "查询出错了，怎么办？",
            "系统报错了，能再试一次吗？",
            "刚才查询好像失败了。",
            "查询服务不可用，还有什么办法？",
            "师傅列表没查出来，麻烦重试一下。",
        ])

    elif scenario_key == "unrelated":
        templates = [
            "今天天气怎么样？",
            "帮我推荐一部电影。",
            "明天股市行情如何？",
            "附近有什么好吃的？",
            "帮我查一下快递。",
            "现在几点了？",
            "给我讲个笑话。",
            "明天要降温吗？",
            "推荐一首好听的歌。",
            "最近的新闻有什么？",
            "帮我算一下房贷。",
            "你能帮我写封邮件吗？",
            "今天限号吗？",
            "附近哪里有停车场？",
            "帮我查一下油价。",
        ]
        return rng.choice(templates)

    return "需要帮助。"


__all__ = [
    "STYLE_ANCHORS",
    "CITIES",
    "CUISINES",
    "CHEF_SURNAMES",
    "DIETARY_SINGLE",
    "DIETARY_COMBOS",
    "OCCASIONS",
    "MENU_COMBOS",
    "REPLY_TEMPLATES",
    "assign_difficulty",
    "chinese_reply",
    "realize_chinese_user_input",
    "_pick_city_district",
    "_pick_chef_display",
]
