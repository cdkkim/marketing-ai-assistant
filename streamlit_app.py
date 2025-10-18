import os
import re
import json
import logging
import time
import uuid
import streamlit as st
import google.generativeai as genai

# ─────────────────────────────
# 0. 로그 및 경고 억제
# ─────────────────────────────
logging.getLogger("google.auth").setLevel(logging.ERROR)
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"

# ─────────────────────────────
# 1. Gemini API 설정
# ─────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.warning("⚠️ GOOGLE_API_KEY가 설정되어 있지 않습니다. Streamlit secrets에 추가하세요.")

DEFAULT_MODEL = "gemini-2.5-flash"
DATA_EVIDENCE_GUIDE = (
    "\n\n추가 지침:\n"
    "- 각 제안에는 데이터 근거(표/지표/규칙 등)를 함께 표기하세요.\n"
    "- 가능한 경우 간단한 표나 지표 수치를 활용해 근거를 명확히 보여주세요."
)

STRUCTURED_RESPONSE_GUIDE = (
    "\n\n응답 형식 지침(중요):\n"
    "1. 반드시 백틱이나 주석 없이 순수 JSON만 출력하세요.\n"
    "2. JSON은 아래 스키마를 따르세요.\n"
    "{\n"
    '  \"objective\": \"최우선 마케팅 목표를 1문장으로 요약\",\n'
    '  \"phase_titles\": [\"Phase 1: …\", \"Phase 2: …\", \"Phase 3: …\"],\n'
    '  \"channel_summary\": [\n'
    '    {\n'
    '      \"channel\": \"채널명\",\n'
    '      \"phase_title\": \"연결된 Phase 제목\",\n'
    '      \"reason\": \"추천 이유와 기대 효과\",\n'
    '      \"data_evidence\": \"관련 수치/규칙 등 데이터 근거\"\n'
    "    }\n"
    "  ],\n"
    '  \"phases\": [\n'
    "    {\n"
    '      \"title\": \"Phase 1: …\",\n'
    '      \"goal\": \"구체적인 목표\",\n'
    '      \"focus_channels\": [\"핵심 채널 1\", \"핵심 채널 2\"],\n'
    '      \"actions\": [\n'
    "        {\n"
    '          \"task\": \"체크박스에 들어갈 실행 항목\",\n'
    '          \"owner\": \"담당 역할(예: 점주, 스태프)\",\n'
    '          \"supporting_data\": \"선택) 관련 데이터 근거\"\n'
    "        }\n"
    "      ],\n"
    '      \"metrics\": [\"성과 KPI\"],\n'
    '      \"next_phase_criteria\": [\"다음 Phase로 넘어가기 위한 정량/정성 기준\"],\n'
    '      \"data_evidence\": [\"Phase 전략을 뒷받침하는 근거\"]\n'
    "    }\n"
    "  ],\n"
    '  \"risks\": [\"주요 리스크와 대응 요약\"],\n'
    '  \"monitoring_cadence\": \"모니터링 주기와 책임자\"\n'
    "}\n"
    "3. Phase는 시간 순서를 지키고 Phase 1의 action 항목은 최소 3개를 포함하세요.\n"
    "4. 모든 reason, supporting_data, data_evidence에는 정량 수치나 규칙적 근거를 명시하세요."
)

FOLLOWUP_RESPONSE_GUIDE = (
    "\n\n응답 형식 지침(중요):\n"
    "1. 반드시 백틱이나 주석 없이 순수 JSON만 출력하세요.\n"
    "2. JSON은 아래 스키마를 따르세요.\n"
    "{\n"
    '  "summary_points": ["핵심 요약 1", "핵심 요약 2"],\n'
    '  "detailed_guidance": "요약을 확장하는 상세 조언",\n'
    '  "evidence_mentions": ["관련 근거 또는 KPI 언급"],\n'
    '  "suggested_question": "다음으로 이어질 간단한 한 문장 질문"\n'
    "}\n"
    "3. summary_points는 최대 2개, 각 항목은 한 문장으로 작성하고 중학생이 이해할 수 있는 쉬운 한국어를 사용하세요.\n"
    "4. evidence_mentions는 최대 3개 이내의 불릿으로 작성하고, 숫자나 지표가 있다면 그대로 유지하세요.\n"
    "5. detailed_guidance는 기존 전략을 재해석하며 데이터 근거를 그대로 인용하되 쉬운 어휘로 설명하세요.\n"
    "6. suggested_question은 사용자가 바로 물어볼 수 있는 짧은 후속 질문 1개만 제안하세요."
)


TOOL_SUGGESTION_GUIDE = (
    "\n\n응답 형식 지침(중요):\n"
    "1. 반드시 백틱이나 주석 없이 순수 JSON만 출력하세요.\n"
    "2. JSON은 아래 스키마를 따르세요.\n"
    "{\n"
    '  "tools": [\n'
    "    {\n"
    '      "name": "도구 이름",\n'
    '      "category": "예: SNS 관리 / CRM / 설문",\n'
    '      "purpose": "Phase 1 목표와 연결된 활용 목적",\n'
    '      "how_to_use": ["1단계", "2단계", "3단계"],\n'
    '      "tips": ["활용 팁"],\n'
    '      "kpi": ["연관 KPI"],\n'
    '      "cost": "무료/유료 여부 및 가격 범위",\n'
    '      "korean_support": "한국어 지원 여부"\n'
    "    }\n"
    "  ],\n"
    '  "notes": ["추가 참고사항"]\n'
    "}\n"
    "3. how_to_use는 2~3단계로 구체적인 실행 순서를 작성하세요.\n"
    "4. 도구는 소상공인이 바로 활용할 수 있는 서비스 중심으로 제안하고, 가능하면 무료 또는 저비용 옵션을 우선하세요."
)


def ensure_data_evidence(prompt: str) -> str:
    """프롬프트에 데이터 근거 지침이 없으면 추가."""
    updated = prompt.rstrip()
    if "데이터 근거" not in updated:
        updated += DATA_EVIDENCE_GUIDE
    if '"phase_titles"' not in updated and "응답 형식 지침(중요)" not in updated:
        updated += STRUCTURED_RESPONSE_GUIDE
    return updated


def extract_executive_summary(markdown_text: str, max_points: int = 4):
    """생성된 전략 본문에서 요약 섹션의 핵심 불릿을 추출."""
    lines = markdown_text.splitlines()
    summary_lines = []

    def clean_bullet(line):
        stripped = line.strip()
        if not stripped:
            return None
        bullet_match = re.match(r"^[-\*\u2022]\s*(.+)", stripped)
        if bullet_match:
            return bullet_match.group(1).strip()
        numbered_match = re.match(r"^\d+[.)]\s*(.+)", stripped)
        if numbered_match:
            return numbered_match.group(1).strip()
        return None

    heading_pattern = re.compile(r"#{1,6}\s*요약")
    start_idx = next(
        (idx for idx, line in enumerate(lines) if heading_pattern.match(line.strip())),
        None,
    )

    if start_idx is not None:
        for line in lines[start_idx + 1 :]:
            stripped = line.strip()
            if stripped.startswith("#"):
                break
            cleaned = clean_bullet(stripped)
            if cleaned:
                summary_lines.append(cleaned)
            if len(summary_lines) >= max_points:
                break

    if not summary_lines:
        for line in lines:
            cleaned = clean_bullet(line)
            if cleaned:
                summary_lines.append(cleaned)
            if len(summary_lines) >= max_points:
                break

    if not summary_lines:
        collapsed = re.sub(r"\s+", " ", markdown_text)
        sentences = re.split(r"(?<=[.!?])\s", collapsed)
        for sentence in sentences:
            cleaned = sentence.strip()
            if cleaned:
                summary_lines.append(cleaned)
            if len(summary_lines) >= max_points:
                break

    return summary_lines


def strip_json_artifacts(text: str) -> str:
    """스트리밍 중 섞일 수 있는 불필요한 문자를 제거하고 순수 JSON 문자열만 남긴다."""
    cleaned = text.strip().replace("▌", "")
    fence_pattern = re.compile(r"^```(?:json)?\s*|\s*```$")
    cleaned = fence_pattern.sub("", cleaned)
    return cleaned.strip()


def parse_strategy_payload(raw_text: str):
    """JSON 응답을 안전하게 파싱. 실패 시 None."""
    candidate = strip_json_artifacts(raw_text)
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def parse_followup_payload(raw_text: str):
    """후속 질의 응답용 JSON을 파싱."""
    candidate = strip_json_artifacts(raw_text)
    if not candidate:
        return None
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def build_phase_followup_question(phase: dict) -> str:
    """Phase 1 컨텍스트로 추천 후속 질문을 생성."""
    if not isinstance(phase, dict):
        return "Phase 1 전략을 실행하면서 추가로 점검해야 할 부분을 알려줄 수 있을까요?"

    focus_channels = phase.get("focus_channels") or []
    actions = phase.get("actions") or []
    goal = (phase.get("goal") or "").strip()

    primary_channel = focus_channels[0].strip() if focus_channels else ""
    primary_action = ""
    if actions and isinstance(actions[0], dict):
        primary_action = (actions[0].get("task") or "").strip()

    if primary_channel and primary_action:
        return f"{primary_channel} 채널에서 '{primary_action}'을 실행할 때 더 준비해야 할 콘텐츠 아이디어가 있을까요?"
    if primary_channel and goal:
        return f"{primary_channel} 채널을 활용해 '{goal}' 목표를 달성하려면 추가로 어떤 실행 팁이 필요할까요?"
    if goal:
        return f"Phase 1 목표인 '{goal}'을 달성하기 위해 먼저 확인해야 할 체크포인트는 무엇일까요?"
    if primary_channel:
        return f"{primary_channel} 채널 운영 시 바로 적용할 수 있는 추가 테스트 아이디어가 있을까요?"
    return "Phase 1 전략을 실행하면서 추가로 점검해야 할 부분을 알려줄 수 있을까요?"


def build_phase_tool_prompt(strategy_payload: dict, phase: dict, store_info: dict | None) -> str:
    """Phase 1 전략을 기반으로 마케팅 도구 추천 프롬프트를 생성."""
    store_info = store_info or {}
    phase = phase if isinstance(phase, dict) else {}
    info_fields = ["상점명", "업종", "프랜차이즈여부", "점포연령", "고객연령대", "고객행동"]
    info_lines = [f"- {field}: {store_info[field]}" for field in info_fields if store_info.get(field)]
    info_block = "\n".join(info_lines) if info_lines else "- 추가 상점 정보 없음"

    objective = (strategy_payload.get("objective") or "").strip() if isinstance(strategy_payload, dict) else ""
    focus_channels = phase.get("focus_channels") or []
    focus_block = ", ".join(
        fc.strip() for fc in focus_channels if isinstance(fc, str) and fc.strip()
    ) or "미지정"

    actions = phase.get("actions") or []
    action_lines = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        label = (action.get("task") or "작업 미정").strip()
        owner = (action.get("owner") or "").strip()
        support = (action.get("supporting_data") or "").strip()
        meta_parts = []
        if owner:
            meta_parts.append(f"담당 {owner}")
        if support:
            meta_parts.append(f"근거 {support}")
        meta_text = f" ({', '.join(meta_parts)})" if meta_parts else ""
        action_lines.append(f"- {label}{meta_text}")
    actions_block = "\n".join(action_lines) if action_lines else "- 등록된 액션 없음"

    metrics = phase.get("metrics") or []
    metrics_block = (
        "\n".join(f"- {m}" for m in metrics if isinstance(m, str) and m.strip())
        if metrics
        else "- 정의된 KPI 없음"
    )

    criteria = phase.get("next_phase_criteria") or []
    criteria_block = (
        "\n".join(f"- {c}" for c in criteria if isinstance(c, str) and c.strip())
        if criteria
        else "- 기준 미정"
    )

    evidence = phase.get("data_evidence") or []
    evidence_block = (
        "\n".join(f"- {e}" for e in evidence if isinstance(e, str) and e.strip())
        if evidence
        else "- 추가 근거 없음"
    )

    prompt_lines = [
        "당신은 소상공인 마케팅을 지원하는 시니어 컨설턴트입니다.",
        "아래 Phase 1 전략을 실행할 때 도움이 되는 실무 도구(SaaS, 분석, 콘텐츠 제작, 자동화 등)를 추천하세요.",
        "도구는 무료 또는 저비용 옵션을 우선 제안하고, 각 도구별로 목적과 2~3단계 실행 가이드를 구체적으로 작성하세요.",
        "각 도구의 연관 KPI, 비용 범위, 한국어 지원 여부, 활용 시 주의점도 함께 언급하세요.",
        "",
        "=== 상점 정보 ===",
        info_block,
        "",
        "=== 전략 Objective ===",
        f"- {objective}" if objective else "- 제공된 Objective 없음",
        "",
        "=== Phase 1 개요 ===",
        f"- 제목: {phase.get('title', 'Phase 1')}",
        f"- 목표: {phase.get('goal', '미정')}",
        f"- 집중 채널: {focus_block}",
        "",
        "=== 실행 액션 ===",
        actions_block,
        "",
        "=== 주요 KPI ===",
        metrics_block,
        "",
        "=== 다음 Phase 기준 ===",
        criteria_block,
        "",
        "=== 데이터 근거 ===",
        evidence_block,
        TOOL_SUGGESTION_GUIDE,
    ]
    return "\n".join(prompt_lines)


def parse_tool_suggestions(raw_text: str) -> dict:
    """도구 추천 JSON을 파싱."""
    candidate = strip_json_artifacts(raw_text)
    if not candidate:
        return {"tools": [], "notes": [], "raw": raw_text.strip()}

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return {"tools": [], "notes": [], "raw": raw_text.strip()}

    tools_data = data.get("tools") if isinstance(data, dict) else []
    normalized_tools = []
    if isinstance(tools_data, list):
        for item in tools_data:
            if not isinstance(item, dict):
                continue
            normalized_tools.append(
                {
                    "name": (item.get("name") or "").strip(),
                    "category": (item.get("category") or "").strip(),
                    "purpose": (item.get("purpose") or "").strip(),
                    "how_to_use": [
                        step.strip()
                        for step in (item.get("how_to_use") or [])
                        if isinstance(step, str) and step.strip()
                    ],
                    "tips": [
                        tip.strip()
                        for tip in (item.get("tips") or [])
                        if isinstance(tip, str) and tip.strip()
                    ],
                    "kpi": [
                        k.strip()
                        for k in (item.get("kpi") or [])
                        if isinstance(k, str) and k.strip()
                    ],
                    "cost": (item.get("cost") or "").strip(),
                    "korean_support": (item.get("korean_support") or "").strip(),
                }
            )

    notes_data = data.get("notes") if isinstance(data, dict) else []
    normalized_notes = [
        note.strip()
        for note in notes_data
        if isinstance(note, str) and note.strip()
    ]

    return {
        "tools": normalized_tools,
        "notes": normalized_notes,
        "raw": candidate.strip(),
    }


def format_tool_suggestions(parsed: dict) -> str:
    """도구 추천 파싱 결과를 마크다운으로 변환."""
    tools = parsed.get("tools") or []
    notes = parsed.get("notes") or []
    raw = (parsed.get("raw") or "").strip()

    if not tools and not notes:
        return raw

    lines = ["### 🛠️ Phase 1 마케팅 도구 추천"]
    for tool in tools:
        name = tool.get("name") or "도구 미정"
        meta_parts = [part for part in (tool.get("category"), tool.get("cost")) if part]
        meta_text = ", ".join(meta_parts)
        header = f"- **{name}**"
        if meta_text:
            header += f" ({meta_text})"
        purpose = tool.get("purpose")
        if purpose:
            header += f": {purpose}"
        lines.append(header)

        how_steps = tool.get("how_to_use") or []
        if how_steps:
            lines.append("  - 활용 단계:")
            for step in how_steps:
                lines.append(f"    - {step}")

        tips = tool.get("tips") or []
        if tips:
            lines.append(f"  - 팁: {'; '.join(tips)}")

        kpi = tool.get("kpi") or []
        if kpi:
            lines.append(f"  - 연관 KPI: {', '.join(kpi)}")

        korean_support = tool.get("korean_support")
        if korean_support:
            lines.append(f"  - 한국어 지원: {korean_support}")

    if notes:
        lines.append("\n**추가 메모**")
        for note in notes:
            lines.append(f"- {note}")

    return "\n".join(lines)


INFO_FIELD_ORDER = ["상점명", "점포연령", "고객연령대", "고객행동"]
BUTTON_HINT = "\n\n필요한 정보가 아니어도 아래 '이대로 질문' 버튼을 누르면 지금 정보로 바로 답변을 드릴게요."
DIRECT_RESPONSE_GUIDE = (
    "\n\n답변 지침:\n"
    "- 불릿 대신 1~2개의 짧은 단락으로 설명하세요.\n"
    "- 중학생이 이해할 수 있는 쉬운 한국어를 사용하세요.\n"
    "- 가능한 경우 숫자나 규칙 같은 근거를 문장 안에 직접 녹여 주세요.\n"
    "- 실행 아이디어는 구체적인 예시와 함께 제시하세요.\n"
    "- 마지막에는 `추천 후속 질문: …` 형식으로 사용자가 이어서 물어볼 만한 질문을 한 문장으로 제시하세요."
)


def get_missing_info_fields(info: dict) -> list:
    """필수 정보 중 아직 수집되지 않은 항목을 반환."""
    missing = []
    for field in INFO_FIELD_ORDER:
        value = info.get(field)
        if not value:
            missing.append(field)
    return missing


def get_latest_strategy_message():
    """세션 기록에서 가장 최신 전략 메시지를 반환."""
    history = st.session_state.get("chat_history", [])
    for item in reversed(history):
        if item.get("type") == "strategy":
            return item
    return None


def build_followup_prompt(question: str, info: dict, strategy_payload: dict, raw_strategy: str) -> str:
    """이전 전략을 문맥으로 후속 질문에 답하기 위한 프롬프트 생성."""
    info_keys = ["상점명", "업종", "프랜차이즈여부", "점포연령", "고객연령대", "고객행동"]
    info_lines = [
        f"- {key}: {info[key]}"
        for key in info_keys
        if key in info and info[key]
    ]
    info_block = "\n".join(info_lines) if info_lines else "- 추가 상점 정보 없음"

    strategy_block = ""
    if strategy_payload:
        try:
            strategy_block = json.dumps(strategy_payload, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            strategy_block = raw_strategy or ""
    else:
        strategy_block = raw_strategy or ""

    prompt = (
        "당신은 중소상공인을 돕는 시니어 마케팅 컨설턴트입니다.\n"
        "이미 생성된 전략 내용을 기반으로 후속 질문에 답변하세요.\n"
        "전략의 Phase, 채널, 실행 항목, 데이터 근거를 우선적으로 인용하고 필요 시 간단한 추가 조언을 더하세요.\n"
        "새 전략을 새로 짜지 말고, 기존 전략을 재해석하거나 보완하는 방식으로 설명하세요.\n"
        "모든 설명은 중학생이 이해할 수 있는 쉬운 한국어로 작성하세요.\n\n"
        "=== 상점 기본 정보 ===\n"
        f"{info_block}\n\n"
        "=== 기존 전략(JSON) ===\n"
        f"{strategy_block}\n\n"
        "=== 사용자 질문 ===\n"
        f"{question}\n\n"
        "위 질문에 대해 전략 정보를 가장 신뢰할 수 있는 근거로 활용해 조언하세요.\n"
        "데이터 근거 항목이나 KPI가 있다면 그대로 언급하거나 수치로 답변에 반영하세요."
        f"{FOLLOWUP_RESPONSE_GUIDE}"
    )
    return prompt


def build_direct_question_prompt(info: dict, question: str, missing_fields=None) -> str:
    """수집된 정보만으로 직접 질문에 답하는 프롬프트 생성."""
    missing_fields = missing_fields or []
    info_lines = [
        f"- {field}: {info[field]}"
        for field in INFO_FIELD_ORDER
        if info.get(field)
    ]
    info_block = "\n".join(info_lines) if info_lines else "- 제공된 정보 없음"

    missing_note = ""
    if missing_fields:
        missing_note = (
            "\n\n주의: 아직 다음 정보가 제공되지 않았습니다. "
            + ", ".join(missing_fields)
            + "."
        )

    prompt = (
        "당신은 동네 상권을 돕는 시니어 마케팅 컨설턴트입니다.\n"
        "아래 상점 정보를 참고해 사용자의 질문에 대해 바로 실행할 수 있는 조언을 주세요.\n"
        "답변은 문단 형태로 작성하고, 필요한 경우 수치나 규칙 같은 근거를 문장 안에 녹여 설명하세요.\n"
        "새로운 가정을 만들기보다는 제공된 정보를 우선적으로 활용하세요.\n\n"
        "=== 상점 정보 ===\n"
        f"{info_block}\n\n"
        "=== 사용자 질문 ===\n"
        f"{question}\n"
        f"{missing_note}"
        f"{DIRECT_RESPONSE_GUIDE}"
    )
    return prompt


def default_suggested_question(info: dict, question: str) -> str:
    """질문이나 상점 정보를 바탕으로 기본 후속 질문을 도출."""
    text = (question or "").lower()
    if "단골" in text or "재방문" in text:
        return "단골 고객에게 줄만한 혜택 아이디어도 알려줄 수 있을까요?"
    if "매출" in text or "판매" in text or "실적" in text:
        return "매출을 더 끌어올릴 수 있는 추가 프로모션이 있을까요?"
    if "신규" in text or "새" in text:
        return "신규 손님을 늘리려면 어떤 홍보 채널이 좋을까요?"
    if "광고" in text or "홍보" in text or "마케팅" in text:
        return "광고 예산은 어느 정도로 잡으면 좋을까요?"
    age = info.get("고객연령대", "")
    if "30" in age or "40" in age:
        return "30~40대에게 반응 좋은 콘텐츠 예시를 더 알려줄 수 있을까요?"
    if "50" in age or "60" in age:
        return "50대 고객이 좋아할 이벤트나 서비스를 추천해 줄 수 있을까요?"
    return "SNS 홍보 전략도 알려줄 수 있을까요?"


def parse_direct_answer(answer_text: str, info: dict, question: str) -> tuple:
    """직접 답변에서 상세 가이드와 추천 질문을 분리."""
    if not answer_text:
        return "", ""

    guidance = answer_text.strip()
    suggested_question = ""
    match = re.search(r"추천\s*후속\s*질문\s*:\s*(.+)$", guidance, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        suggested_question = match.group(1).strip()
        guidance = guidance[: match.start()].strip()

    if not suggested_question:
        suggested_question = default_suggested_question(info, question)

    return guidance, suggested_question


def render_strategy_payload(payload: dict, container, prefix: str = "latest"):
    """구조화된 전략 응답을 Streamlit 컴포넌트로 시각화."""
    objective = payload.get("objective")
    if objective:
        container.markdown("### 🎯 Objective")
        container.markdown(objective)

    phase_titles = payload.get("phase_titles") or []
    channel_summary = payload.get("channel_summary") or []
    if channel_summary:
        container.markdown("### 📊 Recommended Channels & Phase Titles")
        summary_lines = []
        for item in channel_summary:
            channel = item.get("channel", "채널 미지정")
            phase_title = item.get("phase_title", "Phase 미지정")
            reason = item.get("reason", "")
            evidence = item.get("data_evidence", "")
            detail = f"- **{channel}** → {phase_title}: {reason}"
            if evidence:
                detail += f" _(근거: {evidence})_"
            summary_lines.append(detail)
        container.markdown("\n".join(summary_lines))
        if phase_titles:
            container.markdown("**Phase Titles:** " + ", ".join(phase_titles))
    elif phase_titles:
        container.markdown("### 📋 Phase Titles")
        container.markdown(", ".join(phase_titles))

    phases = payload.get("phases") or []
    if not phases:
        return

    # Phase 1 우선 표시
    phase1 = phases[0]
    phase1_container = container.container()
    phase1_container.markdown(f"### 🚀 {phase1.get('title', 'Phase 1')}")
    goal = phase1.get("goal")
    if goal:
        phase1_container.markdown(f"**Goal:** {goal}")
    focus_channels = phase1.get("focus_channels") or []
    if focus_channels:
        phase1_container.markdown("**Focus Channels:** " + ", ".join(focus_channels))

    actions = phase1.get("actions") or []
    if actions:
        phase1_container.markdown("**Action Checklist:**")
        for idx, action in enumerate(actions):
            label = action.get("task", f"Action {idx + 1}")
            owner = action.get("owner")
            support = action.get("supporting_data")
            help_parts = []
            if owner:
                help_parts.append(f"담당: {owner}")
            if support:
                help_parts.append(f"근거: {support}")
            help_text = " | ".join(help_parts) if help_parts else None
            checkbox_key = f"{prefix}_phase1_action_{idx}"
            phase1_container.checkbox(label, key=checkbox_key, help=help_text)

    metrics = phase1.get("metrics") or []
    if metrics:
        phase1_container.markdown("**Metrics:**")
        phase1_container.markdown("\n".join(f"- {m}" for m in metrics))

    criteria = phase1.get("next_phase_criteria") or []
    if criteria:
        phase1_container.markdown("**Criteria To Advance:**")
        phase1_container.markdown("\n".join(f"- {c}" for c in criteria))

    evidence = phase1.get("data_evidence") or []
    if evidence:
        phase1_container.markdown("**Data Evidence:**")
        phase1_container.markdown("\n".join(f"- {e}" for e in evidence))

    suggested_question = build_phase_followup_question(phase1)
    col_followup, col_tool = phase1_container.columns([3, 2])
    followup_label = f"❓ {suggested_question}"
    followup_clicked = col_followup.button(
        followup_label,
        key=f"{prefix}_phase1_followup_btn",
        use_container_width=True,
        disabled=st.session_state.get("is_generating", False),
    )
    tool_button_clicked = col_tool.button(
        "🛠️ Phase 1 도구 추천 보기",
        key=f"{prefix}_phase1_tool_btn",
        use_container_width=True,
        disabled=st.session_state.get("is_generating", False),
    )

    existing_tool_info = (st.session_state.get("phase_tool_summaries") or {}).get(prefix)
    tool_placeholder = None
    if existing_tool_info:
        display_previous = (
            existing_tool_info.get("markdown")
            or existing_tool_info.get("raw")
            or ""
        )
        if display_previous:
            tool_placeholder = phase1_container.empty()
            tool_placeholder.markdown(display_previous)

    if followup_clicked and suggested_question:
        st.session_state.followup_ui = {}
        st.session_state.auto_followup_question = suggested_question
        st.rerun()

    if tool_button_clicked:
        if tool_placeholder is None:
            tool_placeholder = phase1_container.empty()
        prompt = build_phase_tool_prompt(
            payload,
            phase1 if isinstance(phase1, dict) else {},
            st.session_state.get("info", {}),
        )
        previous_generating = st.session_state.get("is_generating", False)
        st.session_state.is_generating = True
        try:
            with phase1_container:
                tool_raw = stream_gemini(
                    prompt,
                    output_placeholder=tool_placeholder,
                    status_text="Phase 1 실행에 맞는 마케팅 도구를 정리하고 있어요... 🛠️",
                    progress_text="전략과 채널에 맞는 툴과 활용법을 모으는 중입니다...",
                    success_text="✅ 도구 추천을 정리했습니다.",
                    error_status_text="🚨 도구 추천 생성 중 오류가 발생했습니다.",
                )
        finally:
            st.session_state.is_generating = previous_generating

        if tool_raw:
            parsed_tool = parse_tool_suggestions(tool_raw)
            formatted_tool = format_tool_suggestions(parsed_tool)
            display_tool_text = formatted_tool or tool_raw
            tool_placeholder.markdown(display_tool_text)
            summaries = st.session_state.get("phase_tool_summaries") or {}
            summaries[prefix] = {
                "markdown": display_tool_text,
                "raw": tool_raw,
            }
            st.session_state.phase_tool_summaries = summaries
        else:
            tool_placeholder.warning("도구 추천을 가져오지 못했습니다. 잠시 후 다시 시도해 주세요.")

    # 나머지 Phase는 Expander로 표시
    for idx, phase in enumerate(phases[1:], start=2):
        title = phase.get("title", f"Phase {idx}")
        expander = container.expander(title, expanded=False)
        goal = phase.get("goal")
        if goal:
            expander.markdown(f"**Goal:** {goal}")
        focus_channels = phase.get("focus_channels") or []
        if focus_channels:
            expander.markdown("**Focus Channels:** " + ", ".join(focus_channels))

        actions = phase.get("actions") or []
        if actions:
            expander.markdown("**Key Actions:**")
            expander.markdown("\n".join(f"- [ ] {act.get('task', '작업 미정')}" for act in actions))

        metrics = phase.get("metrics") or []
        if metrics:
            expander.markdown("**Metrics:**")
            expander.markdown("\n".join(f"- {m}" for m in metrics))

        criteria = phase.get("next_phase_criteria") or []
        if criteria:
            expander.markdown("**Criteria To Advance:**")
            expander.markdown("\n".join(f"- {c}" for c in criteria))

        evidence = phase.get("data_evidence") or []
        if evidence:
            expander.markdown("**Data Evidence:**")
            expander.markdown("\n".join(f"- {e}" for e in evidence))

    risks = payload.get("risks") or []
    monitoring_cadence = payload.get("monitoring_cadence")
    if risks or monitoring_cadence:
        container.markdown("### ⚠️ Risks & Monitoring")
        if risks:
            container.markdown("\n".join(f"- {r}" for r in risks))
        if monitoring_cadence:
            container.markdown(f"**Monitoring Cadence:** {monitoring_cadence}")


def render_followup_panel(guidance_text: str, evidence_list, suggested_question: str, ui_key: int):
    """Follow-up 응답을 상세 가이드와 후속 질문 버튼으로 표시."""
    if not guidance_text:
        return

    st.markdown("### 📘 상세 가이드")
    st.markdown(guidance_text)

    evidence_items = evidence_list or []
    if evidence_items:
        st.markdown("**근거:**")
        st.markdown("\n".join(f"- {item}" for item in evidence_items))

    first_time = not st.session_state.get("shown_followup_suggestion", False)
    if first_time:
        st.session_state.shown_followup_suggestion = True
        if suggested_question:
            st.markdown(f"**가능한 다음 질문:** {suggested_question}")

    col1, col2 = st.columns(2)
    suggestion_clicked = False
    if suggested_question:
        suggestion_clicked = col1.button(
            suggested_question,
            key=f"followup_suggest_{ui_key}",
            use_container_width=True,
            disabled=st.session_state.get("is_generating", False),
        )
    else:
        col1.write("")
    other_clicked = col2.button(
        "다른 질문 입력",
        key=f"followup_new_{ui_key}",
        use_container_width=True,
        disabled=st.session_state.get("is_generating", False),
    )

    if suggestion_clicked:
        st.session_state.followup_ui = {}
        st.session_state.auto_followup_question = suggested_question
        st.rerun()

    if other_clicked:
        st.session_state.followup_ui = {}
        st.rerun()


# ─────────────────────────────
# 1A. 콘텐츠 및 프롬프트 생성 보조 함수
# ─────────────────────────────
COPY_LENGTH_HINTS = {
    "짧게": "문장 2~3개, 80자 내외",
    "중간": "문단 2개, 120~150자",
    "길게": "문단 3개 이상, 200자 내외",
}

DEFAULT_CHANNEL_OPTIONS = ["Instagram", "네이버 블로그", "카카오톡 채널", "오프라인 POP"]
COPY_TONE_OPTIONS = ["친근한", "트렌디한", "전문적인", "감성적인", "믿음직한"]

VISUAL_STYLE_PRESETS = [
    "밝고 친근한 평면 일러스트",
    "사진 같은 리얼리즘",
    "따뜻한 수채화 톤",
    "대비 강한 포스터 스타일",
]
VISUAL_ASPECT_OPTIONS = ["1:1 정사각형", "3:4 세로", "16:9 가로"]


def _strategy_context_lines(strategy_payload, max_lines: int = 8) -> list[str]:
    """마케팅 전략 요약을 기반으로 에셋 생성 시 참고할 핵심 포인트 추출."""
    if not strategy_payload:
        return []

    if isinstance(strategy_payload, str):
        raw_lines = [
            f"- {line.strip()}"
            for line in strategy_payload.splitlines()
            if line.strip()
        ]
        return raw_lines[:max_lines]

    if not isinstance(strategy_payload, dict):
        return []

    lines = []
    objective = strategy_payload.get("objective")
    if objective:
        lines.append(f"- Objective: {objective}")

    channel_summary = strategy_payload.get("channel_summary") or []
    for item in channel_summary[:3]:
        channel = item.get("channel", "채널 미지정")
        reason = item.get("reason", "")
        evidence = item.get("data_evidence", "")
        snippet = f"- {channel}: {reason}"
        if evidence:
            snippet += f" (근거: {evidence})"
        lines.append(snippet)

    phases = strategy_payload.get("phases") or []
    if phases:
        first_phase = phases[0]
        title = first_phase.get("title", "Phase 1")
        goal = first_phase.get("goal")
        if goal:
            lines.append(f"- {title} 목표: {goal}")
        actions = first_phase.get("actions") or []
        for action in actions[:3]:
            task = action.get("task")
            owner = action.get("owner")
            if task:
                if owner:
                    lines.append(f"- 실행: {task} (담당: {owner})")
                else:
                    lines.append(f"- 실행: {task}")

    return lines[:max_lines]


def _available_channels(strategy_payload) -> list[str]:
    """전략 데이터에서 추천 채널 목록을 뽑아 유니크하게 반환."""
    channels = []
    if isinstance(strategy_payload, dict):
        channel_summary = strategy_payload.get("channel_summary") or []
        for item in channel_summary:
            ch = item.get("channel")
            if ch:
                channels.append(ch)
        phases = strategy_payload.get("phases") or []
        for phase in phases:
            focus_channels = phase.get("focus_channels") or []
            for ch in focus_channels:
                channels.append(ch)
    unique_channels = []
    for ch in channels:
        if ch and ch not in unique_channels:
            unique_channels.append(ch)
    if not unique_channels:
        unique_channels = DEFAULT_CHANNEL_OPTIONS
    return unique_channels


def build_copy_generation_prompt(
    info: dict,
    request: dict,
    strategy_payload: dict | str | None,
) -> str:
    """카피 초안 생성을 위한 프롬프트."""
    store_name = info.get("상점명", "상점")
    industry = info.get("업종", "소상공인")
    audience = info.get("고객연령대", "")
    behavior = info.get("고객행동", "")

    tone = request.get("tone", "친근한")
    channel = request.get("channel", "SNS")
    length_label = request.get("length", "중간")
    length_hint = COPY_LENGTH_HINTS.get(length_label, "문단 2개, 120~150자")
    offer = request.get("offer") or "기본 서비스와 강점을 강조"
    extras = request.get("extras", "")

    strategy_lines = _strategy_context_lines(strategy_payload)
    strategy_block = "\n".join(strategy_lines) if strategy_lines else "• 전략 요약 없음"

    prompt = (
        "당신은 한국 중소상공인의 카피라이터입니다.\n"
        "아래 상점 정보와 전략 요약을 참고해 마케팅 카피 초안을 작성하세요.\n"
        "카피는 바로 사용할 수 있게 완성도 높은 문장으로 제공하고, 문단마다 이모지는 넣지 마세요.\n"
        "핵심 CTA 문장을 포함하고, 숫자나 구체적인 혜택이 있다면 자연스럽게 녹여주세요.\n\n"
        f"=== 상점 정보 ===\n"
        f"- 상호: {store_name}\n"
        f"- 업종: {industry}\n"
        f"- 주요 고객: {audience or '미상'} / {behavior or '특이 행동 미상'}\n\n"
        f"=== 콘텐츠 요구사항 ===\n"
        f"- 채널: {channel}\n"
        f"- 톤앤매너: {tone}\n"
        f"- 분량: {length_label} ({length_hint})\n"
        f"- 강조 포인트: {offer}\n"
        f"- 추가 요청: {extras or '없음'}\n\n"
        f"=== 전략 요약 ===\n"
        f"{strategy_block}\n\n"
        "응답 형식 지침:\n"
        "1. 아래와 같은 마크다운 섹션을 그대로 사용합니다.\n"
        "### 헤드라인\n"
        "- 한 줄 헤드라인\n\n"
        "### 본문\n"
        "- 문단 1\n"
        "- 문단 2 (필요 시)\n\n"
        "### CTA\n"
        "- 행동을 유도하는 한 문장\n\n"
        "### 채널 참고 메모\n"
        "- 채널 운영 팁 또는 게시 시 주의사항 1~2개\n"
    )
    return prompt


def build_visual_brief_prompt(
    info: dict,
    request: dict,
    strategy_payload: dict | str | None,
) -> str:
    """이미지/일러스트 콘셉트 브리프 생성을 위한 프롬프트."""
    store_name = info.get("상점명", "상점")
    industry = info.get("업종", "소상공인")

    focus = request.get("focus", "대표 메뉴와 매장 분위기")
    style = request.get("style") or VISUAL_STYLE_PRESETS[0]
    aspect = request.get("aspect", "1:1 정사각형")
    extras = request.get("extras", "")

    strategy_lines = _strategy_context_lines(strategy_payload)
    strategy_block = "\n".join(strategy_lines) if strategy_lines else "• 전략 요약 없음"

    prompt = (
        "당신은 마케팅 아트 디렉터입니다.\n"
        "아래 정보를 참고해 AI 이미지 생성 도구에 전달할 시각 콘셉트 브리프를 작성하세요.\n"
        "브리프는 장면 구성, 핵심 오브젝트, 색감, 텍스트 오버레이 가이드 등을 포함해야 합니다.\n\n"
        f"=== 상점 정보 ===\n"
        f"- 상호: {store_name}\n"
        f"- 업종: {industry}\n\n"
        f"=== 요청 사항 ===\n"
        f"- 강조 포커스: {focus}\n"
        f"- 희망 스타일: {style}\n"
        f"- 이미지 비율: {aspect}\n"
        f"- 추가 요청: {extras or '없음'}\n\n"
        f"=== 전략 요약 ===\n"
        f"{strategy_block}\n\n"
        "응답 형식 지침:\n"
        "1. 아래 마크다운 섹션 제목을 그대로 사용합니다.\n"
        "### 콘셉트 요약\n"
        "- 장면 한 줄 요약\n\n"
        "### 장면 구성\n"
        "- 전경 요소\n"
        "- 중경 요소\n"
        "- 배경 요소\n\n"
        "### 시각 톤 & 스타일\n"
        "- 컬러 팔레트 2~3개\n"
        "- 조명/질감 설명\n\n"
        "### 텍스트 오버레이\n"
        "- 포함할 문구 1~2개 (있다면)\n\n"
        "### 생성 팁\n"
        "- 이미지 생성 시 주의하거나 강조할 사항 2개 내외\n"
    )
    return prompt


PROMPT_EXAMPLE_GUIDE = (
    "\n\n응답 형식 지침(중요):\n"
    "{\n"
    '  "prompts": [\n'
    "    {\n"
    '      "title": "상황 제목",\n'
    '      "prompt": "AI 도구에 붙여넣을 프롬프트",\n'
    '      "when": "이 프롬프트가 유용한 상황 설명"\n'
    "    }\n"
    "  ],\n"
    '  "tips": ["활용 팁 1", "활용 팁 2"]\n'
    "}\n"
    "프롬프트 문장은 한 줄로 작성하고, 한국어 사용을 기본으로 하되 필요 시 영어 키워드도 병기하세요."
)


def build_prompt_example_prompt(
    asset_type: str,
    request: dict,
    generated_asset: str,
    info: dict,
) -> str:
    """에셋 결과를 바탕으로 후속 프롬프트 예시를 생성하기 위한 프롬프트."""
    asset_label = "카피" if asset_type == "copy" else "이미지"
    context_lines = []
    if request.get("channel"):
        context_lines.append(f"- 채널: {request['channel']}")
    if request.get("tone"):
        context_lines.append(f"- 톤: {request['tone']}")
    if request.get("length"):
        context_lines.append(f"- 분량: {request['length']}")
    if request.get("style"):
        context_lines.append(f"- 스타일: {request['style']}")
    if request.get("focus"):
        context_lines.append(f"- 강조 포커스: {request['focus']}")

    context_block = "\n".join(context_lines) if context_lines else "- 추가 컨텍스트 없음"
    store_name = info.get("상점명", "상점")

    prompt = (
        "당신은 마케팅 전문가입니다.\n"
        f"이미 생성된 {asset_label} 초안을 기반으로, 사용자가 다음 반복에서 활용할 프롬프트 예시를 제안하세요.\n"
        "프롬프트는 실무자가 AI 도구에 그대로 붙여넣을 수 있을 정도로 구체적이어야 합니다.\n"
        "각 프롬프트는 톤이나 목적이 서로 다르게 구성해 선택지를 제공합니다.\n\n"
        f"=== 상점명 ===\n- {store_name}\n\n"
        f"=== 컨텍스트 ===\n{context_block}\n\n"
        f"=== 현재 초안 ===\n{generated_asset.strip()}\n"
        f"{PROMPT_EXAMPLE_GUIDE}"
    )
    return prompt


def parse_prompt_examples(raw_text: str) -> dict:
    """프롬프트 예시 JSON을 파싱하고, 실패 시 원문을 함께 반환."""
    candidate = strip_json_artifacts(raw_text)
    if not candidate:
        return {"prompts": [], "tips": [], "raw": raw_text.strip()}

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return {"prompts": [], "tips": [], "raw": raw_text.strip()}

    prompts = data.get("prompts") if isinstance(data, dict) else None
    tips = data.get("tips") if isinstance(data, dict) else None

    if not isinstance(prompts, list):
        prompts = []
    normalized_prompts = []
    for item in prompts:
        if not isinstance(item, dict):
            continue
        title = item.get("title") or item.get("name") or ""
        prompt_text = item.get("prompt") or item.get("content") or ""
        when = item.get("when") or item.get("best_for") or ""
        if prompt_text:
            normalized_prompts.append(
                {
                    "title": title.strip() if isinstance(title, str) else "",
                    "prompt": prompt_text.strip() if isinstance(prompt_text, str) else "",
                    "when": when.strip() if isinstance(when, str) else "",
                }
            )

    if not isinstance(tips, list):
        tips = []
    normalized_tips = [
        tip.strip() for tip in tips if isinstance(tip, str) and tip.strip()
    ]

    return {
        "prompts": normalized_prompts,
        "tips": normalized_tips,
        "raw": candidate.strip(),
    }


def render_copy_workspace(info: dict, strategy_payload, raw_strategy: str | None):
    """카피 생성 및 프롬프트 추천 UI."""
    st.markdown("#### 📝 마케팅 카피")
    st.caption("채널별 카피 초안을 먼저 생성하고, 이어서 변형에 쓸 프롬프트를 추천받으세요.")

    available_channels = _available_channels(strategy_payload if strategy_payload else raw_strategy)
    prev_context = st.session_state.copy_context or {}

    default_channel = prev_context.get("channel", available_channels[0] if available_channels else "SNS")
    channel_index = available_channels.index(default_channel) if default_channel in available_channels else 0
    channel = st.selectbox(
        "채널 선택",
        available_channels,
        index=channel_index if available_channels else 0,
        key="copy_channel_select",
    )

    tone_default = prev_context.get("tone", COPY_TONE_OPTIONS[0])
    tone_index = COPY_TONE_OPTIONS.index(tone_default) if tone_default in COPY_TONE_OPTIONS else 0
    tone = st.selectbox("톤앤매너", COPY_TONE_OPTIONS, index=tone_index, key="copy_tone_select")

    length_options = list(COPY_LENGTH_HINTS.keys())
    length_default = prev_context.get("length", "중간")
    length_index = length_options.index(length_default) if length_default in length_options else 1
    length = st.radio("분량", length_options, index=length_index, horizontal=True, key="copy_length_radio")

    offer = st.text_input(
        "강조할 혜택/프로모션",
        value=prev_context.get("offer", ""),
        placeholder="예: 11월 한정 신메뉴 10% 할인, 오전 11시까지 아메리카노 1+1",
        key="copy_offer_input",
    )
    extras = st.text_area(
        "추가 요청 (선택)",
        value=prev_context.get("extras", ""),
        placeholder="예: 첫 문장은 질문으로 시작, 해시태그 3개 포함, 숫자 데이터 강조",
        height=80,
        key="copy_extra_input",
    )

    col_generate, col_clear = st.columns([3, 1])
    with col_generate:
        disable_generate = (
            st.session_state.copy_generation_in_progress
            or st.session_state.visual_generation_in_progress
            or st.session_state.get("is_generating", False)
        )
        if st.button(
            "AI 카피 초안 생성",
            key="copy_generate_button",
            use_container_width=True,
            disabled=disable_generate,
        ):
            request_payload = {
                "channel": channel,
                "tone": tone,
                "length": length,
                "offer": offer.strip(),
                "extras": extras.strip(),
            }
            st.session_state.copy_context = request_payload
            st.session_state.copy_request = request_payload
            st.session_state.copy_generation_in_progress = True
            st.session_state.copy_draft = ""
            st.session_state.copy_prompt_examples = []
            st.session_state.copy_prompt_tips = []
            st.session_state.copy_prompts_raw = ""
            st.rerun()

    with col_clear:
        if st.button(
            "초안 초기화",
            key="copy_clear_button",
            use_container_width=True,
            disabled=st.session_state.copy_generation_in_progress,
        ):
            st.session_state.copy_draft = ""
            st.session_state.copy_prompt_examples = []
            st.session_state.copy_prompt_tips = []
            st.session_state.copy_prompts_raw = ""
            st.rerun()

    if st.session_state.copy_generation_in_progress and st.session_state.copy_request:
        generation_container = st.container()
        with generation_container:
            placeholder = st.empty()
            prompt = build_copy_generation_prompt(
                info,
                st.session_state.copy_request,
                strategy_payload or raw_strategy,
            )
            copy_result = stream_gemini(
                prompt,
                output_placeholder=placeholder,
                status_text="카피 초안을 정리하고 있어요... ✍️",
                progress_text="전략과 요청을 반영해 문구를 다듬는 중입니다...",
                success_text="✅ 카피 초안이 준비되었습니다.",
                error_status_text="🚨 카피 생성 중 오류가 발생했습니다.",
            )

        st.session_state.copy_generation_in_progress = False
        st.session_state.copy_request = None

        if copy_result:
            st.session_state.copy_draft = copy_result.strip()
            prompt_container = st.container()
            with prompt_container:
                prompt_placeholder = st.empty()
                prompt_request = build_prompt_example_prompt(
                    "copy",
                    st.session_state.copy_context,
                    st.session_state.copy_draft,
                    info,
                )
                prompt_result = stream_gemini(
                    prompt_request,
                    output_placeholder=prompt_placeholder,
                    status_text="추천 프롬프트를 정리하고 있어요... 💡",
                    progress_text="다음 반복에 쓸 프롬프트 변형을 모으는 중입니다...",
                    success_text="✅ 프롬프트 제안이 준비되었습니다.",
                    error_status_text="🚨 프롬프트 추천 생성 중 오류가 발생했습니다.",
                )
            if prompt_result:
                parsed = parse_prompt_examples(prompt_result)
                st.session_state.copy_prompt_examples = parsed.get("prompts", [])
                st.session_state.copy_prompt_tips = parsed.get("tips", [])
                st.session_state.copy_prompts_raw = parsed.get("raw", "")
            else:
                st.session_state.copy_prompt_examples = []
                st.session_state.copy_prompt_tips = []
                st.session_state.copy_prompts_raw = ""
        else:
            st.session_state.copy_draft = ""
            st.session_state.copy_prompt_examples = []
            st.session_state.copy_prompt_tips = []
            st.session_state.copy_prompts_raw = ""

        st.rerun()

    if st.session_state.copy_draft:
        st.markdown("##### 생성된 카피")
        st.markdown(st.session_state.copy_draft)

        with st.expander("🔧 프롬프트 추천", expanded=False):
            if st.session_state.copy_prompt_examples:
                for idx, item in enumerate(st.session_state.copy_prompt_examples, start=1):
                    title = item.get("title") or f"프롬프트 {idx}"
                    prompt_text = item.get("prompt", "")
                    when_text = item.get("when", "")
                    st.markdown(f"**{title}**")
                    if prompt_text:
                        st.code(prompt_text, language="text")
                    if when_text:
                        st.caption(when_text)
            elif st.session_state.copy_prompts_raw:
                st.markdown(st.session_state.copy_prompts_raw)

            if st.session_state.copy_prompt_tips:
                st.markdown("**활용 팁**")
                tips_markdown = "\n".join(f"- {tip}" for tip in st.session_state.copy_prompt_tips)
                st.markdown(tips_markdown)
    else:
        st.info("생성 버튼을 눌러 채널에 맞는 카피 초안을 받아보세요.")


def render_visual_workspace(info: dict, strategy_payload, raw_strategy: str | None):
    """비주얼 브리프 생성 및 프롬프트 추천 UI."""
    st.markdown("#### 🎨 이미지/일러스트 브리프")
    st.caption("AI가 먼저 콘셉트 브리프를 제안하고, 이어서 이미지 생성용 프롬프트를 추천합니다.")

    prev_context = st.session_state.visual_context or {}

    focus = st.text_input(
        "강조 포커스",
        value=prev_context.get("focus", "대표 상품과 매장 분위기"),
        placeholder="예: 겨울 한정 메뉴, 배달 서비스, 테이크아웃 존",
        key="visual_focus_input",
    )

    style_default = prev_context.get("style", VISUAL_STYLE_PRESETS[0])
    style_index = VISUAL_STYLE_PRESETS.index(style_default) if style_default in VISUAL_STYLE_PRESETS else 0
    style = st.selectbox(
        "희망 스타일",
        VISUAL_STYLE_PRESETS,
        index=style_index,
        key="visual_style_select",
    )

    aspect_default = prev_context.get("aspect", VISUAL_ASPECT_OPTIONS[0])
    aspect_index = VISUAL_ASPECT_OPTIONS.index(aspect_default) if aspect_default in VISUAL_ASPECT_OPTIONS else 0
    aspect = st.selectbox(
        "이미지 비율",
        VISUAL_ASPECT_OPTIONS,
        index=aspect_index,
        key="visual_aspect_select",
    )

    extras = st.text_area(
        "추가 요청 (선택)",
        value=prev_context.get("extras", ""),
        placeholder="예: 따뜻한 조명감, 매장 외부 전경 포함, 텍스트는 한국어만 사용",
        height=80,
        key="visual_extra_input",
    )

    col_generate, col_clear = st.columns([3, 1])
    with col_generate:
        disable_generate = (
            st.session_state.visual_generation_in_progress
            or st.session_state.copy_generation_in_progress
            or st.session_state.get("is_generating", False)
        )
        if st.button(
            "AI 브리프 생성",
            key="visual_generate_button",
            use_container_width=True,
            disabled=disable_generate,
        ):
            request_payload = {
                "focus": focus.strip(),
                "style": style,
                "aspect": aspect,
                "extras": extras.strip(),
            }
            st.session_state.visual_context = request_payload
            st.session_state.visual_request = request_payload
            st.session_state.visual_generation_in_progress = True
            st.session_state.visual_brief = ""
            st.session_state.visual_prompt_examples = []
            st.session_state.visual_prompt_tips = []
            st.session_state.visual_prompts_raw = ""
            st.rerun()

    with col_clear:
        if st.button(
            "브리프 초기화",
            key="visual_clear_button",
            use_container_width=True,
            disabled=st.session_state.visual_generation_in_progress,
        ):
            st.session_state.visual_brief = ""
            st.session_state.visual_prompt_examples = []
            st.session_state.visual_prompt_tips = []
            st.session_state.visual_prompts_raw = ""
            st.rerun()

    if st.session_state.visual_generation_in_progress and st.session_state.visual_request:
        generation_container = st.container()
        with generation_container:
            placeholder = st.empty()
            prompt = build_visual_brief_prompt(
                info,
                st.session_state.visual_request,
                strategy_payload or raw_strategy,
            )
            brief_result = stream_gemini(
                prompt,
                output_placeholder=placeholder,
                status_text="이미지 콘셉트를 정리하고 있어요... 🎨",
                progress_text="요청한 스타일을 반영해 장면을 구성하는 중입니다...",
                success_text="✅ 브리프가 준비되었습니다.",
                error_status_text="🚨 브리프 생성 중 오류가 발생했습니다.",
            )

        st.session_state.visual_generation_in_progress = False
        st.session_state.visual_request = None

        if brief_result:
            st.session_state.visual_brief = brief_result.strip()
            prompt_container = st.container()
            with prompt_container:
                prompt_placeholder = st.empty()
                prompt_request = build_prompt_example_prompt(
                    "visual",
                    st.session_state.visual_context,
                    st.session_state.visual_brief,
                    info,
                )
                prompt_result = stream_gemini(
                    prompt_request,
                    output_placeholder=prompt_placeholder,
                    status_text="이미지 프롬프트를 정리하고 있어요... 💡",
                    progress_text="다음 변형에 쓸 프롬프트 옵션을 모으는 중입니다...",
                    success_text="✅ 프롬프트 제안이 준비되었습니다.",
                    error_status_text="🚨 프롬프트 추천 생성 중 오류가 발생했습니다.",
                )

            if prompt_result:
                parsed = parse_prompt_examples(prompt_result)
                st.session_state.visual_prompt_examples = parsed.get("prompts", [])
                st.session_state.visual_prompt_tips = parsed.get("tips", [])
                st.session_state.visual_prompts_raw = parsed.get("raw", "")
            else:
                st.session_state.visual_prompt_examples = []
                st.session_state.visual_prompt_tips = []
                st.session_state.visual_prompts_raw = ""
        else:
            st.session_state.visual_brief = ""
            st.session_state.visual_prompt_examples = []
            st.session_state.visual_prompt_tips = []
            st.session_state.visual_prompts_raw = ""

        st.rerun()

    if st.session_state.visual_brief:
        st.markdown("##### 생성된 브리프")
        st.markdown(st.session_state.visual_brief)

        with st.expander("🔧 프롬프트 추천", expanded=False):
            if st.session_state.visual_prompt_examples:
                for idx, item in enumerate(st.session_state.visual_prompt_examples, start=1):
                    title = item.get("title") or f"프롬프트 {idx}"
                    prompt_text = item.get("prompt", "")
                    when_text = item.get("when", "")
                    st.markdown(f"**{title}**")
                    if prompt_text:
                        st.code(prompt_text, language="text")
                    if when_text:
                        st.caption(when_text)
            elif st.session_state.visual_prompts_raw:
                st.markdown(st.session_state.visual_prompts_raw)

            if st.session_state.visual_prompt_tips:
                st.markdown("**활용 팁**")
                tips_markdown = "\n".join(f"- {tip}" for tip in st.session_state.visual_prompt_tips)
                st.markdown(tips_markdown)
    else:
        st.info("생성 버튼을 눌러 이미지 제작에 활용할 브리프를 받아보세요.")


def render_asset_workshop(strategy_payload, raw_strategy: str | None, info: dict):
    """콘텐츠/이미지 생성 워크스페이스를 렌더링."""
    st.markdown("### ✨ 콘텐츠 & 이미지 생성 워크스페이스")
    st.caption("AI가 먼저 초안을 제시하고, 이어서 활용할 프롬프트 예시를 제공합니다.")

    copy_tab, visual_tab = st.tabs(["카피 생성", "이미지 브리프"])
    with copy_tab:
        render_copy_workspace(info, strategy_payload, raw_strategy)
    with visual_tab:
        render_visual_workspace(info, strategy_payload, raw_strategy)


# ─────────────────────────────
# 2. Persona 데이터 로드
# ─────────────────────────────
@st.cache_data
def load_personas(path="personas.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        st.error("⚠️ personas.json 파일을 찾을 수 없습니다. prompt_generator.py를 먼저 실행하세요.")
        return []

personas = load_personas()

# ─────────────────────────────
# 3. 업종 분류
# ─────────────────────────────
BRAND_KEYWORDS_BY_CATEGORY = {
    "카페/디저트": {
        "파리",
        "뚜레",
        "배스",
        "베스",
        "베스킨",
        "던킨",
        "크리스피",
        "투썸",
        "이디",
        "빽다",
        "메가",
        "메머",
        "매머",
        "컴포",
        "컴포즈",
        "할리",
        "스타벅",
        "탐앤",
        "공차",
        "요거",
        "와플대",
        "와플",
        "폴바셋",
    },
    "한식": {
        "교촌",
        "네네",
        "호식",
        "둘둘",
        "처갓",
        "굽네",
        "bbq",
        "bhc",
        "맘스",
        "맘스터치",
        "죠스",
        "신전",
        "국대",
        "명랑",
        "두끼",
        "땅스",
        "명륜",
        "하남",
        "등촌",
        "봉추",
        "원할",
        "본죽",
        "원할머니",
        "한촌",
        "백채",
        "한솥",
        "바르",
    },
    "양식/세계요리": {
        "도미",
        "피자헛",
        "파파",
        "파파존스",
        "롯데",
        "버거킹",
        "써브",
        "서브웨이",
        "이삭",
        "프랭",
        "프랭크",
        "피자스쿨",
    },
    "주점/주류": {
        "한신",
    },
}


def classify_hpsn_mct(name: str) -> str:
    normalized = _normalize_name(name)
    for category, keywords in BRAND_KEYWORDS_BY_CATEGORY.items():
        if any(k in normalized for k in keywords):
            return category

    nm = name.strip().lower()
    if any(k in nm for k in ["카페", "커피", "디저트", "도너츠", "빙수", "와플", "마카롱"]):
        return "카페/디저트"
    if any(k in nm for k in ["한식", "국밥", "백반", "찌개", "감자탕", "분식", "치킨", "한정식", "죽"]):
        return "한식"
    if any(k in nm for k in ["일식", "초밥", "돈가스", "라멘", "덮밥", "소바", "이자카야"]):
        return "일식"
    if any(k in nm for k in ["중식", "짬뽕", "짜장", "마라", "훠궈", "딤섬"]):
        return "중식"
    if any(k in nm for k in ["양식", "스테이크", "피자", "파스타", "햄버거", "샌드위치", "토스트", "버거"]):
        return "양식/세계요리"
    if any(k in nm for k in ["주점", "호프", "맥주", "와인바", "소주", "요리주점", "이자카야"]):
        return "주점/주류"
    return "기타"

# ─────────────────────────────
# 4. 프랜차이즈 판별
# ─────────────────────────────
BRAND_KEYWORDS = {kw for kws in BRAND_KEYWORDS_BY_CATEGORY.values() for kw in kws}

# 2) 오검출을 줄이기 위한 예외/모호 토큰 (상황 보고 추가)
AMBIGUOUS_NEGATIVES = {
    # 너무 일반적이거나 지역/지명/업종성 토큰
    "카페", "커피", "왕십", "성수", "행당", "종로", "전주", "춘천", "와인", "치킨", "피자", "분식",
    "국수", "초밥", "스시", "곱창", "돼지", "한우", "막창", "수산", "축산", "베이", "브레", "브레드",
}

def _normalize_name(name: str) -> str:
    """
    - 마스킹(*), 공백, 특수문자 제거
    - 소문자 변환(영문 대비)
    """
    n = name.lower()
    n = re.sub(r"[\s\*\-\(\)\[\]{}_/\\.|,!?&^%$#@~`+=:;\"']", "", n)
    return n

def is_franchise(name: str) -> bool:
    """
    데이터셋2의 '상호명' 문자열만으로 프랜차이즈 여부를 휴리스틱으로 판정.
    - 1차: 브랜드 키워드 포함 여부
    - 2차: 모호/일반 토큰만으로 이루어진 경우 제외
    - 3차(선택): '점' 패턴(지점명) 보정
    """
    n = _normalize_name(name)
    if not n:
        return False
    has_branch_marker = "점" in name
    hit = any(k in n for k in BRAND_KEYWORDS)
    if hit:
        if any(bad in n for bad in AMBIGUOUS_NEGATIVES):
            short_hits = [k for k in BRAND_KEYWORDS if k in n and len(k) <= 2]
            if short_hits and not has_branch_marker:
                return False
        return True
    return False


def _store_age_label_from_months(months: int) -> str:
    if months <= 12:
        return "신규"
    if months <= 24:
        return "전환기"
    return "오래된"


def extract_initial_store_info(text: str) -> tuple:
    """복합 문장에서 상점 정보와 질문을 분리해 추출."""
    info_updates = {}
    question = None
    if not text:
        return info_updates, question

    sentences = re.split(r"(?<=[?.!])\s+", text.strip())
    info_sentences = []
    question_sentences = []

    for sentence in sentences:
        stripped = sentence.strip()
        if not stripped:
            continue
        if "?" in stripped or stripped.endswith("까요") or stripped.endswith("할 수 있을까요") or "어떻게" in stripped:
            question_sentences.append(stripped)
        else:
            info_sentences.append(stripped)

    if not question_sentences and info_sentences:
        # 마지막 문장을 질문으로 간주
        question_sentences.append(info_sentences.pop())

    context_text = " ".join(info_sentences) if info_sentences else text.strip()
    question = " ".join(question_sentences).strip() if question_sentences else text.strip()

    normalized_no_space = _normalize_name(context_text)
    brand_hits = [kw for kw in BRAND_KEYWORDS if kw in normalized_no_space]
    if brand_hits:
        store_name = max(brand_hits, key=len)
        info_updates["상점명"] = store_name
    else:
        name_match = re.search(r"([가-힣A-Za-z0-9]+)(?:점)?(?:입니다|이에요|예요|에요)", context_text)
        if name_match:
            info_updates["상점명"] = name_match.group(1)

    age_months = None
    year_match = re.search(r"(\d+)\s*(?:년|년차|년째)", text)
    month_match = re.search(r"(\d+)\s*(?:개월|달)", text)
    if year_match:
        age_months = int(year_match.group(1)) * 12
    elif month_match:
        age_months = int(month_match.group(1))

    if age_months is not None:
        info_updates["점포연령"] = _store_age_label_from_months(age_months)

    if re.search(r"20\s*대", text):
        info_updates["고객연령대"] = "20대 이하 고객 중심"
    elif re.search(r"(?:30|40)\s*대", text):
        info_updates["고객연령대"] = "30~40대 고객 중심"
    elif re.search(r"(?:50|60)\s*대", text):
        info_updates["고객연령대"] = "50대 이상 고객 중심"

    behaviors = []
    if "단골" in text or "재방문" in text:
        behaviors.append("재방문 고객")
    if "신규" in text or "새손님" in text or "새 손님" in text:
        behaviors.append("신규 고객")
    if "거주" in text or "주민" in text:
        behaviors.append("거주 고객")
    if "직장" in text or "오피스" in text or "회사" in text:
        behaviors.append("직장인 고객")
    if "유동" in text or "지나가는" in text or "관광" in text:
        behaviors.append("유동 고객")

    if behaviors:
        info_updates["고객행동"] = " + ".join(sorted(set(behaviors)))

    return info_updates, question


# ─────────────────────────────
# 5. Gemini Streaming 호출
# ─────────────────────────────
DEFAULT_MODEL = "gemini-2.5-flash"

# https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash
def stream_gemini(
    prompt,
    model=DEFAULT_MODEL,
    temperature=0.6,
    max_tokens=65535,
    output_placeholder=None,
    status_text="전략을 생성중입니다... ⏳",
    progress_text="AI가 전략을 정리하고 있어요... 📋",
    success_text="✅ 전략 생성이 완료되었습니다.",
    error_status_text="🚨 전략 생성 중 오류가 발생했습니다.",
):
    """안정적인 스트리밍 + 완료사유 점검 + 친절한 에러"""
    status_placeholder = st.empty()
    status_placeholder.info(status_text)
    step_interval = 2.0
    step_state = {"idx": 0, "next_time": time.time() + step_interval}

    try:
        gmodel = genai.GenerativeModel(model)
        cfg = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": 0.9,
            "top_k": 40,
        }

        stream = gmodel.generate_content(prompt, generation_config=cfg, stream=True)

        placeholder = output_placeholder or st.empty()
        placeholder.info(progress_text)
        full_text = ""

        # 1) 스트리밍 수집 (chunk.text가 없을 수도 있으니 candidates도 확인)
        for event in stream:
            piece = ""
            if getattr(event, "text", None):
                piece = event.text
            elif getattr(event, "candidates", None):
                # 일부 이벤트는 delta 형태로 들어와서 text가 비어 있음
                for c in event.candidates:
                    # 각 candidate의 content에서 추가 텍스트를 안전하게 추출
                    try:
                        piece += "".join([p.text or "" for p in c.content.parts])
                    except Exception:
                        pass

            if piece:
                full_text += piece

        # 2) 최종 해석 (finish_reason/blocked 여부 확인)
        try:
            stream.resolve()  # 최종 상태/메타 확보
        except Exception:
            # resolve에서 오류가 나도 본문이 있으면 계속 진행
            pass

        if not full_text:
            placeholder.warning("응답이 비어 있습니다. 다시 시도해 주세요.")

        # finish_reason/blocked 안내
        try:
            cand0 = stream.candidates[0]
            fr = getattr(cand0, "finish_reason", None)
            block = getattr(cand0, "safety_ratings", None)
        except Exception:
            fr, block = None, None

        # 3) 잘림/차단 안내 + 이어쓰기 버튼
        if fr == "MAX_TOKENS":
            st.info("ℹ️ 응답이 길어 중간에 잘렸어요. 아래 버튼으로 이어서 생성할 수 있어요.")
            if st.button("➕ 이어서 더 생성"):
                continue_from(full_text, prompt, gmodel, cfg)
        elif fr == "SAFETY":
            st.warning("⚠️ 안전 필터로 일부 내용이 숨겨졌을 수 있어요. 표현을 다듬어 다시 시도해보세요.")

        status_placeholder.success(success_text)
        return full_text

    except Exception as e:
        status_placeholder.error(error_status_text)
        st.error(
            "🚨 Gemini 응답 생성 중 오류가 발생했습니다.\n\n"
            f"**에러 유형**: {type(e).__name__}\n"
            f"**메시지**: {e}\n\n"
            "• API Key/모델 이름을 확인하세요.\n"
            "• 일시적인 네트워크/서비스 이슈일 수 있습니다. 잠시 후 다시 시도해 주세요."
        )
        return None


def continue_from(previous_text: str, original_prompt: str, gmodel, cfg):
    """
    MAX_TOKENS로 잘렸을 때 이어쓰기. 원본 문맥을 간단히 요약·복원해 연속성 유지.
    """
    followup_prompt = (
        "아래 초안의 이어지는 내용을 같은 톤/서식으로 계속 작성하세요. "
        "불필요한 반복 없이 Phase 나머지와 KPI, 실행 체크리스트를 마저 채워주세요.\n\n"
        "=== 지금까지 생성된 초안 ===\n"
        f"{previous_text}\n"
        "=== 원래의 요구사항 ===\n"
        f"{original_prompt}\n"
    )

    try:
        stream2 = gmodel.generate_content(followup_prompt, generation_config=cfg, stream=True)
        placeholder = st.empty()
        full2 = ""
        for ev in stream2:
            if getattr(ev, "text", None):
                full2 += ev.text
                placeholder.markdown(full2 + "▌")
        placeholder.markdown(full2)
        st.session_state.chat_history.append({"role": "assistant", "content": full2})
    except Exception as e:
        st.error(
            "🚨 이어쓰기 중 오류가 발생했습니다.\n\n"
            f"**에러 유형**: {type(e).__name__}\n"
            f"**메시지**: {e}"
        )

# ─────────────────────────────
# 6. 페르소나 매칭
# ─────────────────────────────
def find_persona(업종, 프랜차이즈, 점포연령="미상", 고객연령대="미상", 고객행동="미상"):
    for p in personas:
        if p["업종"] == 업종 and p["프랜차이즈여부"] == 프랜차이즈:
            return p
    return None  # None 그대로 반환

# ─────────────────────────────
# 7. Streamlit UI 설정
# ─────────────────────────────
st.set_page_config(page_title="AI 마케팅 컨설턴트", layout="wide")
st.title("💬 AI 마케팅 컨설턴트")

if st.button("🔄 새 상담 시작"):
    st.session_state.clear()
    st.rerun()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "info" not in st.session_state:
    st.session_state.info = {}
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "latest_strategy" not in st.session_state:
    st.session_state.latest_strategy = {}
if "phase_tool_summaries" not in st.session_state:
    st.session_state.phase_tool_summaries = {}
if "followup_ui" not in st.session_state:
    st.session_state.followup_ui = {}
if "followup_ui_key" not in st.session_state:
    st.session_state.followup_ui_key = 0
if "auto_followup_question" not in st.session_state:
    st.session_state.auto_followup_question = None
if "shown_followup_suggestion" not in st.session_state:
    st.session_state.shown_followup_suggestion = False
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "use_pending_question" not in st.session_state:
    st.session_state.use_pending_question = False
if "pending_question_button_key" not in st.session_state:
    st.session_state.pending_question_button_key = 0
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "copy_request" not in st.session_state:
    st.session_state.copy_request = None
if "copy_generation_in_progress" not in st.session_state:
    st.session_state.copy_generation_in_progress = False
if "copy_draft" not in st.session_state:
    st.session_state.copy_draft = ""
if "copy_context" not in st.session_state:
    st.session_state.copy_context = {}
if "copy_prompt_examples" not in st.session_state:
    st.session_state.copy_prompt_examples = []
if "copy_prompt_tips" not in st.session_state:
    st.session_state.copy_prompt_tips = []
if "copy_prompts_raw" not in st.session_state:
    st.session_state.copy_prompts_raw = ""
if "visual_request" not in st.session_state:
    st.session_state.visual_request = None
if "visual_generation_in_progress" not in st.session_state:
    st.session_state.visual_generation_in_progress = False
if "visual_brief" not in st.session_state:
    st.session_state.visual_brief = ""
if "visual_context" not in st.session_state:
    st.session_state.visual_context = {}
if "visual_prompt_examples" not in st.session_state:
    st.session_state.visual_prompt_examples = []
if "visual_prompt_tips" not in st.session_state:
    st.session_state.visual_prompt_tips = []
if "visual_prompts_raw" not in st.session_state:
    st.session_state.visual_prompts_raw = ""

if not st.session_state.initialized:
    with st.chat_message("assistant"):
        st.markdown(
            "안녕하세요 👋 저는 **AI 마케팅 컨설턴트**입니다.\n\n"
            "상점명을 입력해주시면 업종과 프랜차이즈 여부를 분석하고, "
            "몇 가지 질문을 통해 맞춤형 마케팅 전략을 제안드릴게요.\n\n"
            "예: `교촌치킨`, `파리바게뜨`, `카페행당점`, `왕십리돼지국밥`"
        )
    st.session_state.initialized = True

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "strategy":
            message_container = st.container()
            render_strategy_payload(msg.get("data", {}), message_container, prefix=msg.get("id", "history"))
        else:
            st.markdown(msg["content"])

pending_question = st.session_state.get("pending_question")
missing_info = get_missing_info_fields(st.session_state.info)
if pending_question and missing_info:
    with st.container():
        st.info(
            "조금만 더 정보를 알려주시면 맞춤 전략을 더 정확히 만들 수 있어요. "
            "지금 정보만으로도 바로 답변을 받고 싶다면 아래 버튼을 눌러주세요."
        )
        if st.button(
            "이대로 질문",
            key=f"use_question_{st.session_state.pending_question_button_key}",
            use_container_width=True,
            disabled=st.session_state.get("is_generating", False),
        ):
            st.session_state.use_pending_question = True
            st.session_state.auto_followup_question = pending_question
            st.session_state.pending_question_button_key += 1
            st.rerun()

latest_strategy_state = st.session_state.get("latest_strategy", {})
strategy_payload_for_assets = latest_strategy_state.get("payload")
raw_strategy_for_assets = latest_strategy_state.get("raw")
info_state_for_assets = st.session_state.get("info", {})

if not missing_info and (strategy_payload_for_assets or raw_strategy_for_assets):
    with st.container():
        render_asset_workshop(strategy_payload_for_assets, raw_strategy_for_assets, info_state_for_assets)

auto_followup_question = st.session_state.pop("auto_followup_question", None)
chat_box_value = st.chat_input("상점명을 입력하거나 질문에 답해주세요...")
user_input = auto_followup_question or chat_box_value

def add_message(role, content=None, **kwargs):
    message = {"role": role}
    if kwargs.get("type") == "strategy":
        message["type"] = "strategy"
        message["data"] = kwargs.get("data", {})
        message["id"] = kwargs.get("id", str(uuid.uuid4()))
        message["raw"] = kwargs.get("raw")
    else:
        message["content"] = content if content is not None else ""
    st.session_state.chat_history.append(message)


def answer_question_with_current_info(question_text: str):
    """수집된 정보만으로 사용자의 질문에 답변."""
    info = st.session_state.get("info", {})
    missing_fields = get_missing_info_fields(info)
    prompt = build_direct_question_prompt(info, question_text, missing_fields)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        st.session_state.is_generating = True
        try:
            answer = stream_gemini(
                prompt,
                output_placeholder=placeholder,
                status_text="질문에 대한 조언을 정리하고 있어요... 💡",
                progress_text="제공된 정보를 바탕으로 실행 조언을 모으고 있습니다... 🧭",
                success_text="✅ 답변이 준비되었습니다.",
                error_status_text="🚨 답변 생성 중 오류가 발생했습니다.",
            )
        finally:
            st.session_state.is_generating = False
        if answer:
            guidance_text, suggested_question = parse_direct_answer(answer, info, question_text)
            guidance_text = guidance_text or answer.strip()
            ui_key = st.session_state.get("followup_ui_key", 0) + 1
            st.session_state.followup_ui_key = ui_key
            st.session_state.shown_followup_suggestion = False
            st.session_state.followup_ui = {
                "guidance": guidance_text,
                "evidence": [],
                "suggested_question": suggested_question,
                "key": ui_key,
            }

            placeholder.empty()
            with st.container():
                render_followup_panel(guidance_text, [], suggested_question, ui_key)

            log_message = "### 📘 상세 가이드\n\n" + guidance_text
            if suggested_question:
                log_message += f"\n\n**가능한 다음 질문:** {suggested_question}"
            add_message("assistant", log_message)

            st.session_state.latest_strategy = {
                "payload": None,
                "raw": guidance_text,
            }
        else:
            warning_text = "답변을 생성하지 못했습니다. 질문을 조금 다르게 해보시면 도움이 될 수 있어요."
            placeholder.warning(warning_text)
            add_message("assistant", warning_text)
            st.session_state.latest_strategy = {
                "payload": None,
                "raw": "",
            }

    if get_missing_info_fields(info):
        st.session_state.pending_question = question_text
    else:
        st.session_state.pending_question = None
    st.session_state.use_pending_question = False
    st.session_state.shown_followup_suggestion = False

# ─────────────────────────────
# 8. 대화 로직
# ─────────────────────────────
if user_input:
    use_pending = st.session_state.pop("use_pending_question", False)
    st.session_state.followup_ui = {}
    add_message("user", user_input)

    if use_pending:
        answer_question_with_current_info(user_input)
        st.stop()

    # ① 상점명
    if "상점명" not in st.session_state.info:
        info_updates, detected_question = extract_initial_store_info(user_input)
        name = info_updates.get("상점명") or user_input.strip()
        st.session_state.info["상점명"] = name
        st.session_state.info["업종"] = classify_hpsn_mct(name)
        st.session_state.info["프랜차이즈여부"] = "프랜차이즈" if is_franchise(name) else "개인점포"

        for field in ("점포연령", "고객연령대", "고객행동"):
            if info_updates.get(field):
                st.session_state.info[field] = info_updates[field]

        st.session_state.pending_question = detected_question or user_input
        st.session_state.shown_followup_suggestion = False

        missing_fields = get_missing_info_fields(st.session_state.info)
        if missing_fields:
            next_field = missing_fields[0]
            if next_field == "점포연령":
                prompt_text = (
                    f"'{name}'은(는) **{st.session_state.info['업종']} 업종**이며 "
                    f"**{st.session_state.info['프랜차이즈여부']}**로 추정됩니다. 🏪\n\n"
                    "개업 시기가 언제인가요? (예: 6개월 전, 2년 전)"
                )
            elif next_field == "고객연령대":
                prompt_text = "좋아요 👍 주요 고객층은 어떤 연령대인가요? (20대 / 30~40대 / 50대 이상)"
            else:
                prompt_text = "마지막으로, 고객 유형은 어떤 편인가요? (쉼표로 구분 가능: 재방문, 신규, 직장인, 유동, 거주)"

            add_message("assistant", prompt_text + BUTTON_HINT)
            st.rerun()
        else:
            st.session_state.pending_question_button_key += 1
            st.session_state.pending_question = st.session_state.pending_question or user_input
            st.session_state.auto_followup_question = st.session_state.pending_question
            st.session_state.use_pending_question = True
            st.rerun()

    # ② 개업 시기
    elif "점포연령" not in st.session_state.info:
        months = re.findall(r"\d+", user_input)
        months = int(months[0]) if months else 0
        if months <= 12:
            st.session_state.info["점포연령"] = "신규"
        elif months <= 24:
            st.session_state.info["점포연령"] = "전환기"
        else:
            st.session_state.info["점포연령"] = "오래된"

        add_message("assistant", "좋아요 👍 주요 고객층은 어떤 연령대인가요? (20대 / 30~40대 / 50대 이상)" + BUTTON_HINT)
        st.rerun()

    # ③ 고객 연령대
    elif "고객연령대" not in st.session_state.info:
        txt = user_input
        if "20" in txt:
            st.session_state.info["고객연령대"] = "20대 이하 고객 중심"
        elif "30" in txt or "40" in txt:
            st.session_state.info["고객연령대"] = "30~40대 고객 중심"
        else:
            st.session_state.info["고객연령대"] = "50대 이상 고객 중심"

        add_message("assistant", "마지막으로, 고객 유형은 어떤 편인가요? (쉼표로 구분 가능: 재방문, 신규, 직장인, 유동, 거주)" + BUTTON_HINT)
        st.rerun()

    # ④ 고객행동 (다중 입력 유연 파싱)
    elif "고객행동" not in st.session_state.info:
        txt = user_input.lower()
        parts = re.split(r"[,/+\s]*(?:및|와|그리고)?[,/+\s]*", txt)
        parts = [p for p in parts if p]

        behaviors = []
        for p in parts:
            if "재" in p or "단골" in p:
                behaviors.append("재방문 고객")
            if "신" in p or "새" in p:
                behaviors.append("신규 고객")
            if "거주" in p or "주민" in p:
                behaviors.append("거주 고객")
            if "직장" in p or "오피스" in p or "회사" in p:
                behaviors.append("직장인 고객")
            if "유동" in p or "지나" in p or "관광" in p:
                behaviors.append("유동 고객")

        behaviors = list(set(behaviors)) or ["일반 고객"]
        st.session_state.info["고객행동"] = " + ".join(behaviors)

        # ... 고객행동까지 수집된 뒤:
        info = st.session_state.info
        persona = find_persona(
            info["업종"], info["프랜차이즈여부"],
            info["점포연령"], info["고객연령대"], info["고객행동"]
        )

        # ① persona prompt 또는 ② fallback prompt
        if persona and "prompt" in persona:
            prompt = ensure_data_evidence(persona["prompt"])
        else:
            prompt = ensure_data_evidence(
                "다음 상점 정보를 기반으로 3~5단계 Phase별 맞춤형 마케팅 전략을 제안하세요.\n"
                "각 Phase는 목표, 핵심 액션(채널·컨텐츠·오퍼), 예산범위, 예상 KPI, 다음 Phase로 넘어가는 기준을 포함하세요.\n\n"
                f"- 업종: {info['업종']}\n"
                f"- 형태: {info['프랜차이즈여부']}\n"
                f"- 점포연령: {info['점포연령']}\n"
                f"- 주요 고객연령대: {info['고객연령대']}\n"
                f"- 고객행동 특성: {info['고객행동']}\n"
                "응답은 불릿과 표를 적절히 섞어 간결하게 작성하세요."
            )

        #with st.expander("📜 프롬프트 보기"):
        #    st.code(prompt, language="markdown")

        add_message("assistant", "이제 AI 상담사가 맞춤형 마케팅 전략을 생성합니다... ⏳")

        with st.chat_message("assistant"):
            st.markdown("### 📈 생성된 마케팅 전략 결과")
            content_placeholder = st.empty()
            st.session_state.is_generating = True
            try:
                result = stream_gemini(prompt, output_placeholder=content_placeholder)  # ⬅️ 스트리밍 출력
            finally:
                st.session_state.is_generating = False
            if result:
                payload = parse_strategy_payload(result)
                if payload:
                    message_id = str(uuid.uuid4())
                    content_placeholder.empty()
                    strategy_container = st.container()
                    render_strategy_payload(payload, strategy_container, prefix=message_id)
                    st.session_state["latest_strategy"] = {
                        "payload": payload,
                        "raw": result,
                    }
                    st.session_state.shown_followup_suggestion = False
                    add_message(
                        "assistant",
                        type="strategy",
                        data=payload,
                        id=message_id,
                        raw=result,
                    )
                else:
                    summary_points = extract_executive_summary(result)
                    if summary_points:
                        summary_markdown = "#### ⚡ 핵심 요약\n\n" + "\n".join(
                            f"- {point}" for point in summary_points
                        )
                        content_placeholder.markdown(summary_markdown)
                        st.session_state["latest_strategy"] = {
                            "payload": None,
                            "raw": result,
                        }
                        st.session_state.shown_followup_suggestion = False
                        add_message("assistant", summary_markdown)
                    else:
                        fallback_notice = (
                            "구조화된 응답을 표시하지 못했습니다. 다시 시도하거나 프롬프트를 조정해 주세요."
                        )
                        content_placeholder.warning(fallback_notice)
                        st.session_state["latest_strategy"] = {
                            "payload": None,
                            "raw": result,
                        }
                        st.session_state.shown_followup_suggestion = False
                        add_message("assistant", fallback_notice)
    else:
        latest_strategy_msg = get_latest_strategy_message()
        stored_strategy = st.session_state.get("latest_strategy", {})
        strategy_payload = None
        raw_strategy = ""

        if latest_strategy_msg:
            strategy_payload = latest_strategy_msg.get("data") or stored_strategy.get("payload")
            raw_strategy = latest_strategy_msg.get("raw") or stored_strategy.get("raw") or ""
        elif stored_strategy:
            strategy_payload = stored_strategy.get("payload")
            raw_strategy = stored_strategy.get("raw") or ""

        if not (strategy_payload or raw_strategy):
            pending_q = st.session_state.get("pending_question")
            info_state = st.session_state.get("info", {})
            if pending_q and info_state:
                answer_question_with_current_info(pending_q)
                st.rerun()

            fallback_notice = (
                "아직 참고할 전략이 없습니다. 상점 정보를 입력해 맞춤 전략을 생성하거나 "
                "지금까지의 정보로 '이대로 질문' 버튼을 눌러 바로 조언을 받아보세요."
            )
            add_message("assistant", fallback_notice)
            st.rerun()

        followup_prompt = build_followup_prompt(
            user_input,
            st.session_state.get("info", {}),
            strategy_payload,
            raw_strategy,
        )

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            st.session_state.is_generating = True
            try:
                followup_answer = stream_gemini(
                    followup_prompt,
                    output_placeholder=response_placeholder,
                    status_text="질문에 대한 답변을 정리하고 있어요... 💡",
                    progress_text="기존 전략을 바탕으로 가이드를 준비하고 있습니다... 🧭",
                    success_text="✅ 답변이 준비되었습니다.",
                    error_status_text="🚨 답변 생성 중 오류가 발생했습니다.",
                )
            finally:
                st.session_state.is_generating = False
            if followup_answer:
                parsed_followup = parse_followup_payload(followup_answer)
                if (
                    parsed_followup
                    and (
                        parsed_followup.get("summary_points")
                        or parsed_followup.get("detailed_guidance")
                    )
                ):
                    summary_points = (parsed_followup.get("summary_points") or [])[:2]
                    evidence_mentions = (parsed_followup.get("evidence_mentions") or [])[:3]
                    detail_text = parsed_followup.get("detailed_guidance", "")

                    guidance_parts = []
                    if summary_points:
                        guidance_parts.append("\n".join(point for point in summary_points))
                    if detail_text:
                        guidance_parts.append(detail_text)
                    guidance_text = "\n\n".join(part.strip() for part in guidance_parts if part.strip())
                    guidance_text = guidance_text or detail_text or followup_answer
                    suggested_question = (parsed_followup.get("suggested_question") or "").strip()
                    if not suggested_question:
                        suggested_question = default_suggested_question(
                            st.session_state.get("info", {}),
                            user_input,
                        )

                    ui_key = st.session_state.get("followup_ui_key", 0) + 1
                    st.session_state.followup_ui_key = ui_key
                    st.session_state.followup_ui = {
                        "guidance": guidance_text,
                        "evidence": evidence_mentions,
                        "suggested_question": suggested_question,
                        "key": ui_key,
                    }

                    log_sections = ["### 📘 상세 가이드", guidance_text]
                    if evidence_mentions:
                        log_sections.append("**근거:**\n" + "\n".join(f"- {item}" for item in evidence_mentions))
                    log_message = "\n\n".join(section for section in log_sections if section.strip())
                    if log_message:
                        add_message("assistant", log_message)

                    response_placeholder.empty()
                    with st.container():
                        render_followup_panel(guidance_text, evidence_mentions, suggested_question, ui_key)
                else:
                    ui_key = st.session_state.get("followup_ui_key", 0) + 1
                    st.session_state.followup_ui_key = ui_key
                    fallback_suggestion = default_suggested_question(
                        st.session_state.get("info", {}),
                        user_input,
                    )
                    st.session_state.followup_ui = {
                        "guidance": followup_answer,
                        "evidence": [],
                        "suggested_question": fallback_suggestion,
                        "key": ui_key,
                    }

                    clean_answer = (followup_answer or "").strip()
                    if clean_answer:
                        log_message = "### 📘 상세 가이드\n\n" + clean_answer
                        if fallback_suggestion:
                            log_message += f"\n\n**가능한 다음 질문:** {fallback_suggestion}"
                        add_message("assistant", log_message)

                    response_placeholder.empty()
                    with st.container():
                        render_followup_panel(followup_answer, [], fallback_suggestion, ui_key)
            else:
                warning_text = "답변을 생성하지 못했습니다. 질문을 조금 다르게 해보시면 도움이 될 수 있어요."
                response_placeholder.warning(warning_text)
else:
    active_followup_ui = st.session_state.get("followup_ui", {})
    if active_followup_ui.get("guidance"):
        guidance_text = active_followup_ui.get("guidance", "")
        evidence_items = active_followup_ui.get("evidence", [])
        suggested_question = active_followup_ui.get("suggested_question", "")
        ui_key = active_followup_ui.get("key", st.session_state.get("followup_ui_key", 0))
        with st.chat_message("assistant"):
            render_followup_panel(guidance_text, evidence_items, suggested_question, ui_key)
