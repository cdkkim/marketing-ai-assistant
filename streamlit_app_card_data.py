# -*- coding: utf-8 -*-
import os
import io
import csv
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
# 2. Persona 데이터 로드
# ─────────────────────────────
@st.cache_data
def load_personas(path="personas.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        st.error("⚠️ personas.json 파일을 찾을 수 없습니다. persona_generator.py를 먼저 실행하세요.")
        return []

personas = load_personas()

# ─────────────────────────────
# 3. 업종 분류 & 프랜차이즈 판별
# ─────────────────────────────
BRAND_KEYWORDS_BY_CATEGORY = {
    "카페/디저트": {
        "파리","뚜레","배스","베스","베스킨","던킨","크리스피","투썸","이디","빽다","메가",
        "메머","매머","컴포","컴포즈","할리","스타벅","탐앤","공차","요거","와플대","와플","폴바셋",
    },
    "한식": {
        "교촌","네네","호식","둘둘","처갓","굽네","bbq","bhc","맘스","맘스터치","죠스","신전","국대",
        "명랑","두끼","땅스","명륜","하남","등촌","봉추","원할","본죽","원할머니","한촌","백채","한솥","바르",
    },
    "양식/세계요리": {"도미","피자헛","파파","파파존스","롯데","버거킹","써브","서브웨이","이삭","프랭","프랭크","피자스쿨"},
    "주점/주류": {"한신"},
}

def _normalize_name(name: str) -> str:
    n = name.lower()
    n = re.sub(r"[\s\*\-\(\)\[\]{}_/\\.|,!?&^%$#@~`+=:;\"']", "", n)
    return n

def classify_hpsn_mct(name: str) -> str:
    normalized = _normalize_name(name)
    for category, keywords in BRAND_KEYWORDS_BY_CATEGORY.items():
        if any(k in normalized for k in keywords):
            return category

    nm = name.strip().lower()
    if any(k in nm for k in ["카페","커피","디저트","도너츠","빙수","와플","마카롱"]): return "카페/디저트"
    if any(k in nm for k in ["한식","국밥","백반","찌개","감자탕","분식","치킨","한정식","죽"]): return "한식"
    if any(k in nm for k in ["일식","초밥","돈가스","라멘","덮밥","소바","이자카야"]): return "일식"
    if any(k in nm for k in ["중식","짬뽕","짜장","마라","훠궈","딤섬"]): return "중식"
    if any(k in nm for k in ["양식","스테이크","피자","파스타","햄버거","샌드위치","토스트","버거"]): return "양식/세계요리"
    if any(k in nm for k in ["주점","호프","맥주","와인바","소주","요리주점","이자카야"]): return "주점/주류"
    return "기타"

BRAND_KEYWORDS = {kw for kws in BRAND_KEYWORDS_BY_CATEGORY.values() for kw in kws}
AMBIGUOUS_NEGATIVES = {"카페","커피","왕십","성수","행당","종로","전주","춘천","와인","치킨","피자","분식","국수","초밥","스시","곱창","돼지","한우","막창","수산","축산","베이","브레","브레드"}

def is_franchise(name: str) -> bool:
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
    if months <= 12: return "신규"
    if months <= 24: return "전환기"
    return "오래된"

def extract_initial_store_info(text: str) -> tuple:
    """복합 문장에서 상점 정보와 질문을 분리해 추출."""
    info_updates = {}
    question = None
    if not text:
        return info_updates, question

    sentences = re.split(r"(?<=[?.!])\s+", text.strip())
    info_sentences, question_sentences = [], []
    for sentence in sentences:
        stripped = sentence.strip()
        if not stripped:
            continue
        if "?" in stripped or stripped.endswith("까요") or stripped.endswith("할 수 있을까요") or "어떻게" in stripped:
            question_sentences.append(stripped)
        else:
            info_sentences.append(stripped)

    if not question_sentences and info_sentences:
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
    if "단골" in text or "재방문" in text: behaviors.append("재방문 고객")
    if "신규" in text or "새손님" in text or "새 손님" in text: behaviors.append("신규 고객")
    if "거주" in text or "주민" in text: behaviors.append("거주 고객")
    if "직장" in text or "오피스" in text or "회사" in text: behaviors.append("직장인 고객")
    if "유동" in text or "지나가는" in text or "관광" in text: behaviors.append("유동 고객")
    if behaviors:
        info_updates["고객행동"] = " + ".join(sorted(set(behaviors)))

    return info_updates, question

# ─────────────────────────────
# 4. Gemini Streaming 호출
# ─────────────────────────────
DEFAULT_MODEL = "gemini-2.5-flash"

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

    try:
        gmodel = genai.GenerativeModel(model)
        cfg = {"temperature": temperature, "max_output_tokens": max_tokens, "top_p": 0.9, "top_k": 40}
        stream = gmodel.generate_content(prompt, generation_config=cfg, stream=True)

        placeholder = output_placeholder or st.empty()
        placeholder.info(progress_text)
        full_text = ""

        for event in stream:
            piece = ""
            if getattr(event, "text", None):
                piece = event.text
            elif getattr(event, "candidates", None):
                for c in event.candidates:
                    try:
                        piece += "".join([p.text or "" for p in c.content.parts])
                    except Exception:
                        pass
            if piece:
                full_text += piece

        try:
            stream.resolve()
        except Exception:
            pass

        if not full_text:
            placeholder.warning("응답이 비어 있습니다. 다시 시도해 주세요.")

        try:
            cand0 = stream.candidates[0]
            fr = getattr(cand0, "finish_reason", None)
        except Exception:
            fr = None

        if fr == "MAX_TOKENS":
            st.info("ℹ️ 응답이 길어 중간에 잘렸어요. 아래 버튼으로 이어서 생성할 수 있어요.")
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

# ─────────────────────────────
# 5. 페르소나 매칭
# ─────────────────────────────
def find_persona(업종, 프랜차이즈, 점포연령="미상", 고객연령대="미상", 고객행동="미상"):
    for p in personas:
        if p["업종"] == 업종 and p["프랜차이즈여부"] == 프랜차이즈:
            return p
    return None

# ─────────────────────────────
# 6. ENCODED_MCT 전용 모드 (새로 추가)
# ─────────────────────────────
@st.cache_data
def load_mct_prompts(default_path="store_scores_with_clusterlabel_v2_with_targets_updown.csv", uploaded_file=None):
    """
    ENCODED_MCT → {'prompt_str', 'analysis_prompt_updown'} 매핑 로드.
    - 업로드 파일이 있으면 그걸 우선 사용
    - 없으면 기본 경로를 시도
    """
    text = ""
    src = ""
    mapping = {}
    try:
        if uploaded_file is not None:
            text = uploaded_file.getvalue().decode("utf-8-sig")
            src = "uploaded"
        else:
            with open(default_path, "r", encoding="utf-8-sig") as f:
                text = f.read()
            src = default_path
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            k = (row.get("ENCODED_MCT") or "").strip()
            if not k:
                continue
            mapping[k] = {
                "prompt_str": (row.get("prompt_str") or "").strip(),
                "analysis_prompt_updown": (row.get("analysis_prompt_updown") or "").strip(),
            }
        return mapping, src, None
    except Exception as e:
        return {}, src, str(e)

def build_mct_consult_prompt(info: dict, encoded_mct: str, p_main: str, p_updn: str) -> str:
    """
    신한카드 ENCODED_MCT 기반 전문 컨설팅 프롬프트.
    - 기존 JSON 구조 가이드를 그대로 활용(ensure_data_evidence)
    """
    name = info.get("상점명") or "-"
    industry = info.get("업종") or "-"
    franchise = info.get("프랜차이즈여부") or "-"
    store_age = info.get("점포연령") or "-"
    customer_age = info.get("고객연령대") or "-"
    behavior = info.get("고객행동") or "-"

    base = (
        "당신은 소상공인 식당/리테일의 운영·마케팅 컨설턴트입니다.\n"
        "신한카드 ENCODED_MCT를 기반으로 **데이터 근거 중심의 실행 전략**만 제시하세요.\n"
        "모든 전략은 가능한 한 **정량 지표(%, %p, 건수, 원)**로 근거를 제시하세요.\n\n"
        "=== 상점 카드 ===\n"
        f"- 상점명: {name}\n"
        f"- 업종: {industry}\n"
        f"- 형태: {franchise}\n"
        f"- 점포연령: {store_age}\n"
        f"- 고객연령대: {customer_age}\n"
        f"- 고객행동: {behavior}\n\n"
        "=== ENCODED_MCT ===\n"
        f"- 코드: {encoded_mct}\n"
        f"- 외부 프롬프트 요약:\n{p_main or '-'}\n\n"
        f"- 목표 업/다운 지시문:\n{p_updn or '-'}\n\n"
        "정렬 규칙(필수): 제안한 각 전략은 위 '목표 업/다운 지시문' 중 어떤 지표(↑/↓/유지)를 겨냥하는지 명확히 매핑하세요.\n"
    )
    return ensure_data_evidence(base)

def render_mct_tab():
    """사이드바 전환형: ENCODED_MCT 전용 컨설턴트 화면"""
    st.header("💳 신한카드 ENCODED_MCT 컨설턴트")
    st.markdown(
        "안녕하세요 👋 저는 **AI 마케팅 컨설턴트**입니다.\n\n"
        "상점명을 입력해주시면 업종과 프랜차이즈 여부를 추정하고, "
        "ENCODED_MCT(상점 세부 코드)를 기반으로 **신한카드 세부 정보**에 정렬된 전문 솔루션을 제공합니다.\n\n"
        "예: `교촌치킨`, `파리바게뜨`, `카페행당점`, `왕십리돼지국밥`"
    )

    # 전용 세션 상태
    if "mct_history" not in st.session_state:
        st.session_state.mct_history = []
    if "mct_info" not in st.session_state:
        st.session_state.mct_info = {}
    if "mct_latest_strategy" not in st.session_state:
        st.session_state.mct_latest_strategy = {}

    with st.expander("📄 ENCODED_MCT 소스 CSV (선택 업로드)", expanded=False):
        mct_csv_file = st.file_uploader("프롬프트 CSV 업로드", type=["csv"], key="mct_csv_uploader")
        st.caption("기본 파일명: store_scores_with_clusterlabel_v2_with_targets_updown.csv (프로젝트 루트)")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        store_name = st.text_input("상점명", key="mct_store_name", placeholder="예: 교촌치킨 행당점")
        encoded_mct = st.text_input("상점 세부 코드 (ENCODED_MCT)", key="mct_code", placeholder="예: SEG_KR_05")
    with col_b:
        if store_name:
            guess_industry = classify_hpsn_mct(store_name)
            guess_fr = "프랜차이즈" if is_franchise(store_name) else "개인점포"
            st.metric("예상 업종", guess_industry)
            st.metric("예상 형태", guess_fr)

    # 이전 대화 표시
    for msg in st.session_state.mct_history:
        with st.chat_message(msg["role"]):
            if msg.get("type") == "strategy":
                c = st.container()
                render_strategy_payload(msg.get("data", {}), c, prefix=msg.get("id", "mct_hist"))
            else:
                st.markdown(msg["content"])

    # 생성 버튼
    generate = st.button("🚀 전문 솔루션 생성", use_container_width=True, disabled=not encoded_mct)
    if generate:
        info = {
            "상점명": store_name or "",
            "업종": classify_hpsn_mct(store_name) if store_name else "",
            "프랜차이즈여부": ("프랜차이즈" if (store_name and is_franchise(store_name)) else "개인점포") if store_name else "",
            "점포연령": st.session_state.mct_info.get("점포연령", ""),
            "고객연령대": st.session_state.mct_info.get("고객연령대", ""),
            "고객행동": st.session_state.mct_info.get("고객행동", ""),
        }
        st.session_state.mct_info.update({k: v for k, v in info.items() if v})

        # CSV 로드 & 스니펫 추출
        mapping, src, err = load_mct_prompts(uploaded_file=mct_csv_file)
        p_main, p_updn = "", ""
        if err:
            st.warning(f"CSV 로드 실패: {err}")
        else:
            data = mapping.get(encoded_mct.strip())
            if data:
                p_main = data.get("prompt_str", "")
                p_updn = data.get("analysis_prompt_updown", "")
            else:
                st.info("해당 ENCODED_MCT에 대한 외부 프롬프트가 없어 **기본 로직**으로 진행합니다.")

        prompt = build_mct_consult_prompt(st.session_state.mct_info, encoded_mct, p_main, p_updn)
        with st.chat_message("assistant"):
            st.markdown("### 📈 생성된 마케팅 전략 결과")
            ph = st.empty()
            result = stream_gemini(
                prompt,
                output_placeholder=ph,
                status_text="ENCODED_MCT 전문 전략을 생성 중입니다... ⏳",
                progress_text="외부 프롬프트와 상점 정보를 정렬 중... 📋",
                success_text="✅ 전략 생성이 완료되었습니다."
            )
            if result:
                payload = parse_strategy_payload(result)
                if payload:
                    ph.empty()
                    cid = str(uuid.uuid4())
                    box = st.container()
                    render_strategy_payload(payload, box, prefix=cid)
                    st.session_state.mct_latest_strategy = {"payload": payload, "raw": result}
                    st.session_state.mct_history.append({"role": "assistant", "type": "strategy", "data": payload, "id": cid, "raw": result})
                else:
                    st.markdown(result)
                    st.session_state.mct_latest_strategy = {"payload": None, "raw": result}
                    st.session_state.mct_history.append({"role": "assistant", "content": result})

    # 후속 질문
    mct_q = st.chat_input("ENCODED_MCT 기반 추가 질문을 입력하세요...", key="mct_chat_input")
    if mct_q:
        st.session_state.mct_history.append({"role": "user", "content": mct_q})
        latest = st.session_state.get("mct_latest_strategy", {})
        payload = latest.get("payload")
        raw = latest.get("raw", "")

        base_follow = build_followup_prompt(mct_q, st.session_state.mct_info, payload, raw)
        if encoded_mct:
            base_follow += (
                "\n\n[ENCODED_MCT 힌트]\n"
                f"- 코드: {encoded_mct}\n"
                "- 위 전략의 각 항목을 '목표 지표(↑/↓/유지)'와 계속 매핑하세요.\n"
            )
        with st.chat_message("assistant"):
            ph2 = st.empty()
            ans = stream_gemini(
                base_follow,
                output_placeholder=ph2,
                status_text="질문에 대한 답변을 정리하고 있어요... 💡",
                progress_text="기존 전략과 ENCODED_MCT 정보를 바탕으로 가이드를 준비 중... 🧭",
                success_text="✅ 답변이 준비되었습니다."
            )
            if ans:
                parsed = parse_followup_payload(ans)
                if parsed and (parsed.get("summary_points") or parsed.get("detailed_guidance")):
                    pts = (parsed.get("summary_points") or [])[:2]
                    ev = (parsed.get("evidence_mentions") or [])[:3]
                    txt = parsed.get("detailed_guidance", "")
                    guidance = "\n\n".join([s for s in ["\n".join(pts), txt] if s.strip()])
                    render_followup_panel(guidance, ev, (parsed.get("suggested_question") or ""), ui_key=1)
                    st.session_state.mct_history.append({"role": "assistant", "content": guidance})
                else:
                    st.markdown(ans)
                    st.session_state.mct_history.append({"role": "assistant", "content": ans})

# ─────────────────────────────
# 7. Streamlit UI 설정 (모드 전환 추가)
# ─────────────────────────────
st.set_page_config(page_title="AI 마케팅 컨설턴트", layout="wide")
st.title("💬 AI 마케팅 컨설턴트")

# 👉 사이드바 모드 선택: 기존 상담 / ENCODED_MCT 컨설턴트
mode = st.sidebar.radio("모드", ["기존 상담", "ENCODED_MCT 컨설턴트"], index=0)
if mode == "ENCODED_MCT 컨설턴트":
    render_mct_tab()
    st.stop()

# ── 이하: 기존 상담 화면(원본 로직 유지) ──────────────────────────

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
            st.session_state.latest_strategy = {"payload": None, "raw": ""}

    if get_missing_info_fields(info):
        st.session_state.pending_question = question_text
    else:
        st.session_state.pending_question = None
    st.session_state.use_pending_question = False
    st.session_state.shown_followup_suggestion = False

# ─────────────────────────────
# 8. 대화 로직 (기존)
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
            if "재" in p or "단골" in p: behaviors.append("재방문 고객")
            if "신" in p or "새" in p: behaviors.append("신규 고객")
            if "거주" in p or "주민" in p: behaviors.append("거주 고객")
            if "직장" in p or "오피스" in p or "회사" in p: behaviors.append("직장인 고객")
            if "유동" in p or "지나" in p or "관광" in p: behaviors.append("유동 고객")

        behaviors = list(set(behaviors)) or ["일반 고객"]
        st.session_state.info["고객행동"] = " + ".join(behaviors)

        # persona 기반 or fallback 전략 생성
        info = st.session_state.info
        persona = find_persona(
            info["업종"], info["프랜차이즈여부"],
            info["점포연령"], info["고객연령대"], info["고객행동"]
        )

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

        add_message("assistant", "이제 AI 상담사가 맞춤형 마케팅 전략을 생성합니다... ⏳")

        with st.chat_message("assistant"):
            st.markdown("### 📈 생성된 마케팅 전략 결과")
            content_placeholder = st.empty()
            st.session_state.is_generating = True
            try:
                result = stream_gemini(prompt, output_placeholder=content_placeholder)
            finally:
                st.session_state.is_generating = False
            if result:
                payload = parse_strategy_payload(result)
                if payload:
                    message_id = str(uuid.uuid4())
                    content_placeholder.empty()
                    strategy_container = st.container()
                    render_strategy_payload(payload, strategy_container, prefix=message_id)
                    st.session_state["latest_strategy"] = {"payload": payload, "raw": result}
                    st.session_state.shown_followup_suggestion = False
                    add_message("assistant", type="strategy", data=payload, id=message_id, raw=result)
                else:
                    summary_points = extract_executive_summary(result)
                    if summary_points:
                        summary_markdown = "#### ⚡ 핵심 요약\n\n" + "\n".join(f"- {point}" for point in summary_points)
                        content_placeholder.markdown(summary_markdown)
                        st.session_state["latest_strategy"] = {"payload": None, "raw": result}
                        st.session_state.shown_followup_suggestion = False
                        add_message("assistant", summary_markdown)
                    else:
                        fallback_notice = "구조화된 응답을 표시하지 못했습니다. 다시 시도하거나 프롬프트를 조정해 주세요."
                        content_placeholder.warning(fallback_notice)
                        st.session_state["latest_strategy"] = {"payload": None, "raw": result}
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
                if parsed_followup and (parsed_followup.get("summary_points") or parsed_followup.get("detailed_guidance")):
                    summary_points = (parsed_followup.get("summary_points") or [])[:2]
                    evidence_mentions = (parsed_followup.get("evidence_mentions") or [])[:3]
                    detail_text = parsed_followup.get("detailed_guidance", "")
                    guidance_parts = []
                    if summary_points: guidance_parts.append("\n".join(point for point in summary_points))
                    if detail_text: guidance_parts.append(detail_text)
                    guidance_text = "\n\n".join(part.strip() for part in guidance_parts if part.strip())
                    guidance_text = guidance_text or detail_text or followup_answer
                    suggested_question = (parsed_followup.get("suggested_question") or "").strip()
                    if not suggested_question:
                        suggested_question = default_suggested_question(st.session_state.get("info", {}), user_input)

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
                    fallback_suggestion = default_suggested_question(st.session_state.get("info", {}), user_input)
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
