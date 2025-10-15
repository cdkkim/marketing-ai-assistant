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
def classify_hpsn_mct(name: str) -> str:
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
BRAND_KEYWORDS = {
    "파리","뚜레","배스","던킨","투썸","이디","빽다","메가","컴포","할리",
    "스타벅","탐앤","공차","요거","와플","교촌","네네","호식","둘둘","처갓",
    "굽네","bbq","bhc","맘스","죠스","신전","명랑","두끼","땅스",
    "도미","파파","롯데","버거킹","써브","이삭","명륜","하남","한신",
    "등촌","봉추","원할","본죽","한촌","백채","프랭","바르","한솥","베스"
}
AMBIGUOUS_NEGATIVES = {
    "카페","커피","왕십","성수","행당","종로","전주","춘천","와인","치킨","피자",
    "분식","국수","초밥","곱창","돼지","한우"
}

def _normalize_name(name: str) -> str:
    n = name.lower()
    n = re.sub(r"[\s\*\-\(\)\[\]{}_/\\.|,!?&^%$#@~`+=:;\"']", "", n)
    return n

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
):
    """안정적인 스트리밍 + 완료사유 점검 + 친절한 에러"""
    status_placeholder = st.empty()
    status_placeholder.info("전략을 생성중입니다... ⏳")
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
        placeholder.info("AI가 전략을 정리하고 있어요... 📋")
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

        status_placeholder.success("✅ 전략 생성이 완료되었습니다.")
        return full_text

    except Exception as e:
        status_placeholder.error("🚨 전략 생성 중 오류가 발생했습니다.")
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

user_input = st.chat_input("상점명을 입력하거나 질문에 답해주세요...")

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

# ─────────────────────────────
# 8. 대화 로직
# ─────────────────────────────
if user_input:
    add_message("user", user_input)

    # ① 상점명
    if "상점명" not in st.session_state.info:
        name = user_input.strip()
        st.session_state.info["상점명"] = name
        st.session_state.info["업종"] = classify_hpsn_mct(name)
        st.session_state.info["프랜차이즈여부"] = "프랜차이즈" if is_franchise(name) else "개인점포"

        add_message(
            "assistant",
            f"'{name}'은(는) **{st.session_state.info['업종']} 업종**이며 "
            f"**{st.session_state.info['프랜차이즈여부']}**로 추정됩니다. 🏪\n\n"
            "개업 시기가 언제인가요? (예: 6개월 전, 2년 전)"
        )
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

        add_message("assistant", "좋아요 👍 주요 고객층은 어떤 연령대인가요? (20대 / 30~40대 / 50대 이상)")
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

        add_message("assistant", "마지막으로, 고객 유형은 어떤 편인가요? (쉼표로 구분 가능: 재방문, 신규, 직장인, 유동, 거주)")
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
            result = stream_gemini(prompt, output_placeholder=content_placeholder)  # ⬅️ 스트리밍 출력
            if result:
                payload = parse_strategy_payload(result)
                if payload:
                    message_id = str(uuid.uuid4())
                    content_placeholder.empty()
                    strategy_container = st.container()
                    render_strategy_payload(payload, strategy_container, prefix=message_id)
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
                        add_message("assistant", summary_markdown)
                    else:
                        fallback_notice = (
                            "구조화된 응답을 표시하지 못했습니다. 다시 시도하거나 프롬프트를 조정해 주세요."
                        )
                        content_placeholder.warning(fallback_notice)
                        add_message("assistant", fallback_notice)
