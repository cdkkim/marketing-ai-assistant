import itertools
import json

industries = ["카페/디저트", "한식", "일식", "중식", "양식/세계요리", "주점/주류"]
franchise_types = ["프랜차이즈", "개인점포"]
store_ages = ["신규", "전환기", "오래된"]

customer_age_groups = [
    "20대 이하 고객 중심",
    "30~40대 고객 중심",
    "50대 이상 고객 중심"
]

customer_behaviors = [
    "재방문 고객 중심",
    "신규 고객 중심",
    "거주 고객 중심",
    "직장인 고객 중심",
    "유동인구 고객 중심"
]

def phase_guideline():
    return (
        "마케팅 전략은 Phase별 Action Plan(3~5단계)으로 제시해야 한다.\n"
        "- 각 Phase는 ①목표 ②실행 전략 ③다음 단계로 넘어가는 기준(KPI)을 포함한다.\n"
        "- KPI 예: 신규 방문객 수, 재방문율, 리뷰 건수·평점, SNS 반응률, 매출 성장률 등.\n"
        "- 반드시 이전 Phase KPI를 달성해야 다음 Phase로 넘어갈 수 있다."
    )

def build_prompt(industry, franchise, store_age, age_group, behavior):
    return (
        f"너는 마케팅 전략 컨설턴트다.\n"
        f"- 업종: {industry}\n"
        f"- 형태: {franchise}\n"
        f"- 점포 연령: {store_age}\n"
        f"- 고객 연령대: {age_group}\n"
        f"- 고객 행동 특성: {behavior}\n\n"
        f"맞춤형 마케팅 채널과 전략을 추천하라.\n\n"
        f"{phase_guideline()}\n\n"
        f"출력 형식은 Markdown으로:\n"
        f"# 요약\n"
        f"## 채널 우선순위\n"
        f"## 실행 전략\n"
        f"## Phase별 Action Plan\n"
        f"## KPI 및 모니터링 지표\n"
        f"## 리스크와 대응"
    )

def generate_personas(limit=10):
    personas = []
    all_combinations = itertools.product(
        industries, franchise_types, store_ages, customer_age_groups, customer_behaviors
    )
    for idx, (ind, f, age, age_group, behavior) in enumerate(all_combinations, 1):
        if limit and idx > limit:
            break
        personas.append({
            "업종": ind,
            "프랜차이즈여부": f,
            "점포연령": age,
            "고객연령대": age_group,
            "고객행동": behavior,
            "prompt": build_prompt(ind, f, age, age_group, behavior)
        })
    return personas

if __name__ == "__main__":
    personas = generate_personas(limit=1000)
    with open("personas.json", "w", encoding="utf-8") as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)
    print(f"[DONE] personas.json 저장 완료 (총 {len(personas)}개)")
