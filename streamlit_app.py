# -*- coding: utf-8 -*-
import os
import io
import csv
import re
import json
import glob
import time
import uuid
import logging
import numpy as np
import streamlit as st
import google.generativeai as genai

# (선택) FAISS가 없으면 NumPy로 폴백
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

# ─────────────────────────────
# 0) 로그/경고 억제
# ─────────────────────────────
logging.getLogger("google.auth").setLevel(logging.ERROR)
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "3")

# ─────────────────────────────
# 1) Gemini API 설정
# ─────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.warning("⚠️ GOOGLE_API_KEY가 설정되어 있지 않습니다. (export GOOGLE_API_KEY=...)")

DEFAULT_MODEL = "gemini-2.5-flash"

# ─────────────────────────────
# 2) 프롬프트 가이드(삼중따옴표로 안전하게)
# ─────────────────────────────
DATA_EVIDENCE_GUIDE = """
추가 지침:
- 각 제안에는 데이터 근거(표/지표/규칙 등)를 함께 표기하세요.
- 가능한 경우 간단한 표나 지표 수치를 활용해 근거를 명확히 보여주세요.
"""

STRUCTURED_RESPONSE_GUIDE = """
응답 형식 지침(중요):
1. 반드시 백틱이나 주석 없이 순수 JSON만 출력하세요.
2. JSON은 아래 스키마를 따르세요.
{
  "objective": "최우선 마케팅 목표를 1문장으로 요약",
  "phase_titles": ["Phase 1: …", "Phase 2: …", "Phase 3: …"],
  "channel_summary": [
    {
      "channel": "채널명",
      "phase_title": "연결된 Phase 제목",
      "reason": "추천 이유와 기대 효과",
      "data_evidence": "관련 수치/규칙 등 데이터 근거"
    }
  ],
  "phases": [
    {
      "title": "Phase 1: …",
      "goal": "구체적인 목표",
      "focus_channels": ["핵심 채널 1", "핵심 채널 2"],
      "actions": [
        {
          "task": "체크박스에 들어갈 실행 항목",
          "owner": "담당 역할(예: 점주, 스태프)",
          "supporting_data": "선택) 관련 데이터 근거"
        }
      ],
      "metrics": ["성과 KPI"],
      "next_phase_criteria": ["다음 Phase로 넘어가기 위한 정량/정성 기준"],
      "data_evidence": ["Phase 전략을 뒷받침하는 근거"]
    }
  ],
  "risks": ["주요 리스크와 대응 요약"],
  "monitoring_cadence": "모니터링 주기와 책임자"
}
3. Phase는 시간 순서를 지키고 Phase 1의 action 항목은 최소 3개를 포함하세요.
4. 모든 reason, supporting_data, data_evidence에는 정량 수치나 규칙적 근거를 명시하세요.
"""

FOLLOWUP_RESPONSE_GUIDE = """
응답 형식 지침(중요):
1. 반드시 백틱이나 주석 없이 순수 JSON만 출력하세요.
2. JSON은 아래 스키마를 따르세요.
{
  "summary_points": ["핵심 요약 1", "핵심 요약 2"],
  "detailed_guidance": "요약을 확장하는 상세 조언",
  "evidence_mentions": ["관련 근거 또는 KPI 언급"],
  "suggested_question": "다음으로 이어질 간단한 한 문장 질문"
}
3. summary_points는 최대 2개, 각 항목은 한 문장으로 작성하고 중학생이 이해할 수 있는 쉬운 한국어를 사용하세요.
4. evidence_mentions는 최대 3개 이내의 불릿으로 작성하고, 숫자나 지표가 있다면 그대로 유지하세요.
5. detailed_guidance는 기존 전략을 재해석하며 데이터 근거를 그대로 인용하되 쉬운 어휘로 설명하세요.
6. suggested_question은 사용자가 바로 물어볼 수 있는 짧은 후속 질문 1개만 제안하세요.
"""

def ensure_data_evidence(prompt: str) -> str:
    """프롬프트에 데이터 근거/구조 가이드가 없으면 추가."""
    updated = prompt.rstrip()
    if "데이터 근거" not in updated:
        updated += "\n\n" + DATA_EVIDENCE_GUIDE.strip()
    if '"phase_titles"' not in updated and "응답 형식 지침(중요)" not in updated:
        updated += "\n\n" + STRUCTURED_RESPONSE_GUIDE.strip()
    return updated

# ─────────────────────────────
# 3) 공통 유틸
# ─────────────────────────────
def _extract_bullet_content(text_line: str) -> str | None:
    s = text_line.strip()
    if not s:
        return None
    m1 = re.match(r"^[-\*\u2022]\s*(.+)", s)
    if m1:
        return m1.group(1).strip()
    m2 = re.match(r"^\d+[.)]\s*(.+)", s)
    if m2:
        return m2.group(1).strip()
    return None

def _looks_like_evidence_line(text: str) -> bool:
    """간단한 휴리스틱으로 근거 전용 라인을 식별."""
    if not text:
        return False
    # 불릿/번호 접두 제거 후 비교
    normalized = re.sub(r"^[\-\*\u2022\d\.\)\(\s]+", "", text).strip()
    normalized = normalized.replace(":", " ").strip()
    lower = normalized.lower()
    evidence_prefixes = (
        "근거",
        "데이터 근거",
        "evidence",
        "증빙",
        "supporting data",
    )
    return any(lower.startswith(prefix) for prefix in evidence_prefixes)

def extract_executive_summary(markdown_text: str, max_points: int = 4):
    """비구조 응답에서 핵심 불릿을 추출."""
    lines = markdown_text.splitlines()
    summary = []

    # '# 요약' 섹션 우선 탐색
    start = None
    for i, line in enumerate(lines):
        if re.match(r"#{1,6}\s*요약", line.strip()):
            start = i; break
    if start is not None:
        for line in lines[start+1:]:
            if line.strip().startswith("#"):
                break
            v = _extract_bullet_content(line)
            if v and not _looks_like_evidence_line(v) and v not in summary:
                summary.append(v)
                if len(summary) >= max_points:
                    break

    # 없으면 전체에서 불릿 추출
    if not summary:
        for line in lines:
            v = _extract_bullet_content(line)
            if v and not _looks_like_evidence_line(v) and v not in summary:
                summary.append(v)
                if len(summary) >= max_points:
                    break

    # 그래도 없으면 문장 몇 개
    if not summary:
        sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", markdown_text))
        for s in sentences:
            s = s.strip()
            if s and not _looks_like_evidence_line(s) and s not in summary:
                summary.append(s)
                if len(summary) >= max_points:
                    break
    return summary

def get_action_guidelines_from_session(max_items: int = 2) -> list[str]:
    """세션에 저장된 최신 전략에서 액션 가이드를 추출."""
    guidelines: list[str] = []
    owner_guidelines: list[str] = []
    owner_keywords = ("점주", "사장", "오너", "대표", "매장주", "운영자")

    def harvest(payload):
        if not payload:
            return
        for phase in payload.get("phases") or []:
            for act in phase.get("actions") or []:
                task = (act.get("task") or "").strip()
                if not task or task in guidelines or task in owner_guidelines:
                    continue
                owner = (act.get("owner") or "").strip()
                if owner and any(k in owner for k in owner_keywords):
                    owner_guidelines.append(task)
                else:
                    guidelines.append(task)
                if len(owner_guidelines) + len(guidelines) >= max_items:
                    return

    payload_candidates = []
    if "latest_strategy" in st.session_state:
        latest = st.session_state.get("latest_strategy") or {}
        payload_candidates.append(latest.get("payload"))
    if "mct_latest_strategy" in st.session_state:
        latest_mct = st.session_state.get("mct_latest_strategy") or {}
        payload_candidates.append(latest_mct.get("payload"))

    for payload in payload_candidates:
        harvest(payload)
        if len(owner_guidelines) + len(guidelines) >= max_items:
            break

    ordered = []
    for task in owner_guidelines:
        if task not in ordered:
            ordered.append(task)
        if len(ordered) >= max_items:
            break
    if len(ordered) < max_items:
        for task in guidelines:
            if task not in ordered:
                ordered.append(task)
            if len(ordered) >= max_items:
                break

    return ordered[:max_items]

def _extract_guidelines_from_text(text: str, max_items: int = 2) -> list[str]:
    """상세 가이드 텍스트에서 휴리스틱으로 실행 지침 문장을 추출."""
    if not text:
        return []

    keywords = (
        "하세요", "하십시오", "해보세요", "해 보세요", "해 주세요", "해주", "실행", "도입",
        "만드", "유도", "운영", "제공", "준비", "구축", "강화", "업데이트", "확실히",
        "활용", "등록", "설정", "점검", "관리", "증정", "혜택", "쿠폰", "이벤트",
        "서비스", "기획", "추천", "권장", "유지", "테스트", "연결", "확장", "유치"
    )

    candidates: list[str] = []
    for line in text.splitlines():
        cand = _extract_bullet_content(line)
        if cand:
            candidates.append(cand)

    normalized = re.sub(r"\s+", " ", text).strip()
    if normalized:
        for sentence in re.split(r"(?<=[.!?])\s+", normalized):
            sentence = sentence.strip()
            if sentence:
                candidates.append(sentence)

    action_guidelines: list[str] = []
    for cand in candidates:
        if any(keyword in cand for keyword in keywords):
            cleaned = cand.strip()
            if cleaned and cleaned not in action_guidelines:
                action_guidelines.append(cleaned)
            if len(action_guidelines) >= max_items:
                break

    return action_guidelines[:max_items]

def extract_action_guidelines_with_gemini(text: str, max_items: int = 2) -> list[str]:
    """Gemini 모델로 실행 지침을 추출."""
    if not text or not GOOGLE_API_KEY:
        return []

    def _gather_response_text(response) -> str:
        """응답 객체에서 텍스트 파트를 안전하게 추출."""
        if not response:
            return ""
        pieces: list[str] = []
        candidates = getattr(response, "candidates", None)
        if candidates:
            for cand in candidates:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None) if content else None
                if parts:
                    for part in parts:
                        part_text = getattr(part, "text", None)
                        if part_text:
                            pieces.append(part_text)
            if pieces:
                return "".join(pieces)
        try:
            text_attr = response.text  # type: ignore[attr-defined]
            return text_attr if isinstance(text_attr, str) else ""
        except Exception:
            return ""

    prompt = (
        "당신은 로컬 상권 점주를 돕는 시니어 마케팅 컨설턴트입니다.\n"
        "아래 상세 가이드를 읽고, 매장 점주가 직접 실행할 수 있는 설득력 있는 행동 제안을 "
        f"최대 {max_items}개 도출하세요.\n"
        "- 이미 상세 가이드에 있는 문장을 그대로 사용하거나 자연스럽게 다듬어도 됩니다.\n"
        "- 본사나 외부 대행사가 아닌 점주가 바로 실행할 수 있는 항목만 선택하세요.\n"
        "- 각 항목은 제안/권장 어조로 1~2문장으로 작성하고, 첫 문장에는 실행 행동을, 두 번째 문장에는 기대 효과나 팁을 덧붙이세요.\n"
        "- 같은 내용을 반복하지 마세요.\n\n"
        "=== 상세 가이드 ===\n"
        f"{text.strip()}\n\n"
        "=== 응답 형식 ===\n"
        "{\n"
        '  \"action_guidelines\": [\"점주 실행 지침 1\", \"점주 실행 지침 2\"]\n'
        "}\n"
        "- JSON 외의 텍스트는 출력하지 마세요.\n"
        "- 필요한 항목만 포함하세요."
    )

    try:
        gmodel = genai.GenerativeModel(DEFAULT_MODEL)
        cfg = {"temperature": 0.2, "max_output_tokens": 65000, "top_p": 0.8, "top_k": 32}
        response = gmodel.generate_content(prompt, generation_config=cfg)
        raw = _gather_response_text(response)
        cleaned = strip_json_artifacts(raw)
        if not cleaned:
            return []
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            values = parsed.get("action_guidelines")
            if isinstance(values, list):
                return [str(v).strip() for v in values[:max_items] if str(v).strip()]
    except json.JSONDecodeError as err:
        logging.info("Gemini action guideline JSON 파싱 실패: %s", err)
    except ValueError as err:
        # finish_reason이 2(SAFETY) 등일 때 response.text 접근 등에서 ValueError가 발생 가능
        logging.info("Gemini action guideline 텍스트를 추출하지 못했습니다: %s", err)
    except Exception as err:
        logging.warning("Gemini action guideline 추출 실패: %s", err)
    return []

def extract_action_guidelines(text: str, max_items: int = 2) -> list[str]:
    """상세 가이드에서 실행 지침을 추출 (Gemini 우선, 휴리스틱 보조)."""
    guidelines = extract_action_guidelines_with_gemini(text, max_items=max_items)
    if not guidelines:
        guidelines = _extract_guidelines_from_text(text, max_items=max_items)
    return guidelines[:max_items]

def strip_json_artifacts(text: str) -> str:
    cleaned = (text or "").strip().replace("▌", "")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    return cleaned.strip()

def parse_strategy_payload(raw_text: str):
    cand = strip_json_artifacts(raw_text)
    if not cand:
        return None
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        return None

def parse_followup_payload(raw_text: str):
    cand = strip_json_artifacts(raw_text)
    if not cand:
        return None
    try:
        obj = json.loads(cand)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None

INFO_FIELD_ORDER = ["상점명", "점포연령", "고객연령대", "고객행동"]
BUTTON_HINT = "\n\n필요한 정보가 아니어도 아래 '이대로 질문' 버튼을 누르면 지금 정보로 바로 답변을 드릴게요."
DIRECT_RESPONSE_GUIDE = """
답변 지침:
- 불릿 대신 1~2개의 짧은 단락으로 설명하세요.
- 중학생이 이해할 수 있는 쉬운 한국어를 사용하세요.
- 가능한 경우 숫자나 규칙 같은 근거를 문장 안에 직접 녹여 주세요.
- 실행 아이디어는 구체적인 예시와 함께 제시하세요.
- 마지막에는 `추천 후속 질문: …` 형식으로 사용자가 이어서 물어볼 만한 질문을 한 문장으로 제시하세요.
"""

def get_missing_info_fields(info: dict) -> list:
    missing = [f for f in INFO_FIELD_ORDER if not info.get(f)]
    return missing

def default_suggested_question(info: dict, question: str) -> str:
    text = (question or "").lower()
    if "단골" in text or "재방문" in text:
        return "단골 고객에게 줄만한 혜택 아이디어도 알려줄 수 있을까요?"
    if any(k in text for k in ["매출","판매","실적"]):
        return "매출을 더 끌어올릴 수 있는 추가 프로모션이 있을까요?"
    if any(k in text for k in ["신규","새"]):
        return "신규 손님을 늘리려면 어떤 홍보 채널이 좋을까요?"
    if any(k in text for k in ["광고","홍보","마케팅"]):
        return "광고 예산은 어느 정도로 잡으면 좋을까요?"
    age = info.get("고객연령대","")
    if any(k in age for k in ["30","40"]):
        return "30~40대에게 반응 좋은 콘텐츠 예시를 더 알려줄 수 있을까요?"
    if any(k in age for k in ["50","60"]):
        return "50대 고객이 좋아할 이벤트나 서비스를 추천해 줄 수 있을까요?"
    return "SNS 홍보 전략도 알려줄 수 있을까요?"

def parse_direct_answer(answer_text: str, info: dict, question: str) -> tuple[str, str]:
    if not answer_text:
        return "", ""
    guidance = answer_text.strip()
    suggested = ""
    m = re.search(r"추천\s*후속\s*질문\s*:\s*(.+)$", guidance, flags=re.I | re.M)
    if m:
        suggested = m.group(1).strip()
        guidance = guidance[:m.start()].strip()
    if not suggested:
        suggested = default_suggested_question(info, question)
    return guidance, suggested

# ─────────────────────────────
# 4) 시각화 컴포넌트
# ─────────────────────────────
def render_strategy_payload(payload: dict, container, prefix: str = "latest"):
    objective = payload.get("objective")
    if objective:
        container.markdown("### 🎯 Objective")
        container.markdown(objective)

    phase_titles = payload.get("phase_titles") or []
    channel_summary = payload.get("channel_summary") or []
    if channel_summary:
        container.markdown("### 📊 Recommended Channels & Phase Titles")
        lines = []
        for item in channel_summary:
            channel = item.get("channel","채널 미지정")
            phase_title = item.get("phase_title","Phase 미지정")
            reason = item.get("reason","")
            evidence = item.get("data_evidence","")
            s = f"- **{channel}** → {phase_title}: {reason}"
            if evidence:
                s += f" _(근거: {evidence})_"
            lines.append(s)
        container.markdown("\n".join(lines))
        if phase_titles:
            container.markdown("**Phase Titles:** " + ", ".join(phase_titles))
    elif phase_titles:
        container.markdown("### 📋 Phase Titles")
        container.markdown(", ".join(phase_titles))

    phases = payload.get("phases") or []
    if not phases:
        return

    # Phase 1
    p1 = phases[0]
    c1 = container.container()
    c1.markdown(f"### 🚀 {p1.get('title','Phase 1')}")
    if p1.get("goal"):
        c1.markdown(f"**Goal:** {p1['goal']}")
    if p1.get("focus_channels"):
        c1.markdown("**Focus Channels:** " + ", ".join(p1["focus_channels"]))
    actions = p1.get("actions") or []
    if actions:
        c1.markdown("**Action Checklist:**")
        for i, act in enumerate(actions):
            label = act.get("task", f"Action {i+1}")
            owner = act.get("owner")
            support = act.get("supporting_data")
            help_parts = []
            if owner: help_parts.append(f"담당: {owner}")
            if support: help_parts.append(f"근거: {support}")
            help_txt = " | ".join(help_parts) if help_parts else None
            c1.checkbox(label, key=f"{prefix}_p1_{i}", help=help_txt)
    if p1.get("metrics"):
        c1.markdown("**Metrics:**\n" + "\n".join(f"- {m}" for m in p1["metrics"]))
    if p1.get("next_phase_criteria"):
        c1.markdown("**Criteria To Advance:**\n" + "\n".join(f"- {x}" for x in p1["next_phase_criteria"]))
    if p1.get("data_evidence"):
        c1.markdown("**Data Evidence:**\n" + "\n".join(f"- {e}" for e in p1["data_evidence"]))

    # 나머지 Phase는 Expander
    for idx, ph in enumerate(phases[1:], start=2):
        ex = container.expander(ph.get("title", f"Phase {idx}"), expanded=False)
        if ph.get("goal"):
            ex.markdown(f"**Goal:** {ph['goal']}")
        if ph.get("focus_channels"):
            ex.markdown("**Focus Channels:** " + ", ".join(ph["focus_channels"]))
        if ph.get("actions"):
            ex.markdown("**Key Actions:**\n" + "\n".join(f"- [ ] {a.get('task','작업 미정')}" for a in ph["actions"]))
        if ph.get("metrics"):
            ex.markdown("**Metrics:**\n" + "\n".join(f"- {m}" for m in ph["metrics"]))
        if ph.get("next_phase_criteria"):
            ex.markdown("**Criteria To Advance:**\n" + "\n".join(f"- {x}" for x in ph["next_phase_criteria"]))
        if ph.get("data_evidence"):
            ex.markdown("**Data Evidence:**\n" + "\n".join(f"- {e}" for e in ph["data_evidence"]))

    risks = payload.get("risks") or []
    cadence = payload.get("monitoring_cadence")
    if risks or cadence:
        container.markdown("### ⚠️ Risks & Monitoring")
        if risks: container.markdown("\n".join(f"- {r}" for r in risks))
        if cadence: container.markdown(f"**Monitoring Cadence:** {cadence}")

def render_followup_panel(guidance_text: str, evidence_list, suggested_question: str, ui_key: int):
    if not guidance_text:
        return
    st.markdown("### 📘 상세 가이드")
    summary_points = [
        p for p in extract_executive_summary(guidance_text, max_points=3)
        if not _looks_like_evidence_line(p)
    ]
    action_guidelines = extract_action_guidelines(guidance_text, max_items=2)
    if len(action_guidelines) < 2:
        fallback = get_action_guidelines_from_session(max_items=2)
        for item in fallback:
            if item not in action_guidelines:
                action_guidelines.append(item)
            if len(action_guidelines) >= 2:
                break
    action_guidelines = action_guidelines[:2]
    clean_guidelines: list[str] = []
    for guideline in action_guidelines:
        formatted = guideline.strip()
        if not formatted:
            continue
        if "action guideline" not in formatted.lower():
            formatted = f"{formatted}"
        if formatted not in clean_guidelines:
            clean_guidelines.append(formatted)
        if len(clean_guidelines) >= 2:
            break

    clean_summary: list[str] = []
    for point in summary_points:
        if point and point not in clean_guidelines and point not in clean_summary:
            clean_summary.append(point)

    max_quick_items = 3
    summary_slots = max(0, max_quick_items - len(clean_guidelines))
    display_points: list[str] = clean_summary[:summary_slots]
    if not display_points and not clean_summary and clean_guidelines:
        # 요약이 없으면 행동 지침만 사용
        display_points = []
    for guideline in clean_guidelines:
        if len(display_points) >= max_quick_items:
            break
        display_points.append(guideline)
    if display_points:
        st.markdown("**빠른 요약:**")
        st.markdown("\n".join(f"- {item}" for item in display_points))
    st.markdown(guidance_text)
    if evidence_list:
        st.markdown("**근거:**")
        st.markdown("\n".join(f"- {item}" for item in (evidence_list or [])))
    first_time = not st.session_state.get("shown_followup_suggestion", False)
    if first_time:
        st.session_state.shown_followup_suggestion = True
        if suggested_question:
            st.markdown(f"**가능한 다음 질문:** {suggested_question}")
    col1, col2 = st.columns(2)
    suggestion_clicked = False
    if suggested_question:
        suggestion_clicked = col1.button(
            suggested_question, key=f"followup_suggest_{ui_key}",
            use_container_width=True,
            disabled=st.session_state.get("is_generating", False),
        )
    else:
        col1.write("")
    other_clicked = col2.button(
        "다른 질문 입력", key=f"followup_new_{ui_key}",
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
# 5) personas.json 로딩
# ─────────────────────────────
@st.cache_data
def load_personas(path="personas.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        st.error("⚠️ personas.json 파일을 찾을 수 없습니다. (persona_generator.py로 먼저 생성하세요)")
        return []

personas = load_personas()

# ─────────────────────────────
# 6) 업종/프랜차이즈 추정
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
BRAND_KEYWORDS = {kw for kws in BRAND_KEYWORDS_BY_CATEGORY.values() for kw in kws}
AMBIGUOUS_NEGATIVES = {"카페","커피","왕십","성수","행당","종로","전주","춘천","와인","치킨","피자","분식","국수","초밥","스시","곱창","돼지","한우","막창","수산","축산","베이","브레","브레드"}

def _normalize_name(name: str) -> str:
    n = name.lower()
    n = re.sub(r"[\s\*\-\(\)\[\]{}_/\\.|,!?&^%$#@~`+=:;\"']", "", n)
    return n

def classify_hpsn_mct(name: str) -> str:
    normalized = _normalize_name(name or "")
    for category, keywords in BRAND_KEYWORDS_BY_CATEGORY.items():
        if any(k in normalized for k in keywords):
            return category
    nm = (name or "").strip().lower()
    if any(k in nm for k in ["카페","커피","디저트","도너츠","빙수","와플","마카롱"]): return "카페/디저트"
    if any(k in nm for k in ["한식","국밥","백반","찌개","감자탕","분식","치킨","한정식","죽"]): return "한식"
    if any(k in nm for k in ["일식","초밥","돈가스","라멘","덮밥","소바","이자카야"]): return "일식"
    if any(k in nm for k in ["중식","짬뽕","짜장","마라","훠궈","딤섬"]): return "중식"
    if any(k in nm for k in ["양식","스테이크","피자","파스타","햄버거","샌드위치","토스트","버거"]): return "양식/세계요리"
    if any(k in nm for k in ["주점","호프","맥주","와인바","소주","요리주점","이자카야"]): return "주점/주류"
    return "기타"

def is_franchise(name: str) -> bool:
    n = _normalize_name(name or "")
    if not n:
        return False
    has_branch_marker = "점" in (name or "")
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
def _to_pct(value_str):
    try:
        x = float(str(value_str).replace("%","").strip())
        if x <= 1.0:  # 0~1 스케일일 때
            return int(round(x * 100))
        return int(round(min(x, 100)))
    except Exception:
        return None

def render_mct_kpi(perf_score_global: str | float | None, success_label: str | None):
    perf_pct = _to_pct(perf_score_global) if perf_score_global not in (None, "") else None
    label = (success_label or "").strip() or "데이터 없음"

    st.subheader("📊 MCT 성과 요약")
    c1, c2 = st.columns(2)
    if perf_pct is not None:
        c1.metric("Perf Score (Global)", f"{perf_pct}%")
    else:
        c1.metric("Perf Score (Global)", "데이터 없음")

    # 간단한 뱃지 느낌
    badge = "✅ 성공" if label.lower().startswith(("succ","성공")) else ("⚠️ 주의" if "warn" in label.lower() else f"ℹ️ {label}")
    c2.metric("Success Level", badge)

    # 진행바(있을 때만)
    if perf_pct is not None:
        st.progress(perf_pct)

def extract_initial_store_info(text: str) -> tuple[dict, str | None]:
    """복합 문장에서 상점 정보/질문을 분리 추출."""
    info_updates, question = {}, None
    if not text: return info_updates, question
    sents = re.split(r"(?<=[?.!])\s+", text.strip())
    info_sents, q_sents = [], []
    for s in sents:
        s = s.strip()
        if not s: continue
        if "?" in s or s.endswith("까요") or "어떻게" in s:
            q_sents.append(s)
        else:
            info_sents.append(s)
    if not q_sents and info_sents:
        q_sents.append(info_sents.pop())
    context_text = " ".join(info_sents) if info_sents else text.strip()
    question = " ".join(q_sents).strip() if q_sents else text.strip()

    normalized_no_space = _normalize_name(context_text)
    brand_hits = [kw for kw in BRAND_KEYWORDS if kw in normalized_no_space]
    if brand_hits:
        info_updates["상점명"] = max(brand_hits, key=len)
    else:
        m = re.search(r"([가-힣A-Za-z0-9]+)(?:점)?(?:입니다|이에요|예요|에요)", context_text)
        if m: info_updates["상점명"] = m.group(1)

    age_months = None
    y = re.search(r"(\d+)\s*(?:년|년차|년째)", text)
    m = re.search(r"(\d+)\s*(?:개월|달)", text)
    if y: age_months = int(y.group(1)) * 12
    elif m: age_months = int(m.group(1))
    if age_months is not None:
        info_updates["점포연령"] = _store_age_label_from_months(age_months)

    if re.search(r"20\s*대", text): info_updates["고객연령대"] = "20대 이하 고객 중심"
    elif re.search(r"(?:30|40)\s*대", text): info_updates["고객연령대"] = "30~40대 고객 중심"
    elif re.search(r"(?:50|60)\s*대", text): info_updates["고객연령대"] = "50대 이상 고객 중심"

    behaviors = []
    if "단골" in text or "재방문" in text: behaviors.append("재방문 고객")
    if "신규" in text or "새손님" in text: behaviors.append("신규 고객")
    if "거주" in text or "주민" in text: behaviors.append("거주 고객")
    if any(k in text for k in ["직장","오피스","회사"]): behaviors.append("직장인 고객")
    if any(k in text for k in ["유동","지나가는","관광"]): behaviors.append("유동 고객")
    if behaviors:
        info_updates["고객행동"] = " + ".join(sorted(set(behaviors)))
    return info_updates, question

# ─────────────────────────────
# 7) Gemini 스트리밍 호출
# ─────────────────────────────
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
    status_ph = st.empty()
    status_ph.info(status_text)
    try:
        gmodel = genai.GenerativeModel(model)
        cfg = {"temperature": temperature, "max_output_tokens": max_tokens, "top_p": 0.9, "top_k": 40}
        stream = gmodel.generate_content(prompt, generation_config=cfg, stream=True)

        ph = output_placeholder or st.empty()
        ph.info(progress_text)
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

        status_ph.success(success_text)
        if not full_text:
            ph.warning("응답이 비어 있습니다. 다시 시도해 주세요.")
        return full_text
    except Exception as e:
        status_ph.error(error_status_text)
        st.error(
            "🚨 Gemini 응답 생성 중 오류가 발생했습니다.\n\n"
            f"**에러 유형**: {type(e).__name__}\n"
            f"**메시지**: {e}\n\n"
            "• API Key/모델 이름을 확인하세요.\n"
            "• 일시적인 네트워크/서비스 이슈일 수 있습니다."
        )
        return None

# ─────────────────────────────
# 8) 페르소나 매칭
# ─────────────────────────────
def find_persona(업종, 프랜차이즈, 점포연령="미상", 고객연령대="미상", 고객행동="미상"):
    for p in personas:
        if p.get("업종") == 업종 and p.get("프랜차이즈여부") == 프랜차이즈:
            return p
    return None

# ─────────────────────────────
# 9) 외부 지식베이스(subtitle_summary) RAG
# ─────────────────────────────
@st.cache_resource(show_spinner=False)
def load_external_kb(dir_path: str, embed_model: str | None = None):
    """subtitle_summary/summary 폴더의 JSON들을 임베딩/색인. 없으면 None."""
    files = glob.glob(os.path.join(dir_path, "*.json"))
    docs, metas = [], []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            for i, d in enumerate(data):
                ctx = d.get("context", {})
                prob = d.get("problem", [])
                sol = d.get("solution", [])
                text = f"상황: {ctx}\n문제점: {prob}\n해결: {sol}"
                docs.append(text)
                metas.append((os.path.basename(fp), i))
        except Exception:
            continue
    if not docs:
        return None

    # 임베딩 모델 import (미설치 환경에서도 앱 실행되도록 내부 import)
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        st.warning("sentence-transformers가 설치되지 않아 외부 지식 RAG를 비활성화합니다.")
        return None

    model_name = os.getenv("EMBED_MODEL_NAME", embed_model or "jhgan/ko-sbert-nli")
    model = SentenceTransformer(model_name)
    doc_emb = model.encode(docs, normalize_embeddings=True)
    doc_emb = np.asarray(doc_emb, dtype=np.float32)

    if faiss is not None:
        index = faiss.IndexFlatIP(doc_emb.shape[1])
        index.add(doc_emb)
    else:
        index = None

    def search(query: str, top_k: int = 3) -> str:
        if not query:
            return ""
        q = model.encode([query], normalize_embeddings=True)
        q = np.asarray(q, dtype=np.float32)
        if index is not None:
            D, I = index.search(q, top_k)
            idxs = I[0]
        else:
            sims = doc_emb @ q[0]
            idxs = np.argsort(-sims)[:top_k]
        lines = ["[외부 지식베이스 상위 근거]"]
        for rank, idx in enumerate(idxs, 1):
            if idx < 0 or idx >= len(docs):
                continue
            file, j = metas[idx]
            snippet = docs[idx].replace("\n", " ")[:200]
            lines.append(f"{rank}. 파일:{file} #{j} | {snippet}")
        return "\n".join(lines)

    return {"search": search}

def build_kb_query(info: dict, extra: str = "") -> str:
    keys = ["상점명", "업종", "프랜차이즈여부", "점포연령", "고객연령대", "고객행동"]
    parts = [f"{k}:{info[k]}" for k in keys if info.get(k)]
    if extra: parts.append(str(extra))
    return " ".join(parts)


# ─────────────────────────────
# 10) ENCODED_MCT 전용 (CSV 연동 + 전용 화면)
# ─────────────────────────────
# @st.cache_data
# def load_mct_prompts(default_path="store_scores_with_clusterlabel_v2_with_targets_updown.csv", uploaded_file=None):
#     """ENCODED_MCT → {'prompt_str','analysis_prompt_updown'} 매핑 로드."""
#     text = ""
#     src = ""
#     mapping = {}
#     try:
#         if uploaded_file is not None:
#             text = uploaded_file.getvalue().decode("utf-8-sig")
#             src = "uploaded"
#         else:
#             with open(default_path, "r", encoding="utf-8-sig") as f:
#                 text = f.read()
#             src = default_path
#         reader = csv.DictReader(io.StringIO(text))
#         for row in reader:
#             k = (row.get("ENCODED_MCT") or "").strip()
#             if not k: continue
#             mapping[k] = {
#                 "prompt_str": (row.get("prompt_str") or "").strip(),
#                 "analysis_prompt_updown": (row.get("analysis_prompt_updown") or "").strip(),
#             }
#         return mapping, src, None
#     except Exception as e:
#         return {}, src, str(e)
@st.cache_data
def load_mct_prompts(default_path="store_scores_with_clusterlabel_v2_with_targets_updown.csv", uploaded_file=None):
    """
    ENCODED_MCT → {'prompt_str','analysis_prompt_updown','perf_score_global','success_label'} 매핑 로드.
    - 업로드 파일이 있으면 그걸 우선 사용, 없으면 기본 경로를 시도
    - 헤더 이름이 애매하게 다를 수 있어 '포함 토큰'으로 유연 탐색
    """
    import io, csv, re
    text = ""; src = ""; mapping = {}

    def _find_key(row_keys, *tokens):
        # 헤더에서 토큰 전부를 포함하는 첫 키 반환 (공백/밑줄/대소문자 무시)
        for k in row_keys:
            norm = re.sub(r"[^a-z0-9]", "", k.lower())
            if all(t in norm for t in tokens):
                return k
        return None

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
            keys = list(row.keys())
            # 필수 키들 위치 찾기
            key_mct = _find_key(keys, "encoded", "mct") or "ENCODED_MCT"
            if not (row.get(key_mct) or "").strip():
                continue

            key_prompt  = _find_key(keys, "prompt", "str") or "prompt_str"
            key_updown  = _find_key(keys, "analysis", "updown") or "analysis_prompt_updown"

            # 새로 추가: perf_score_global / success_label (이름 변형도 커버)
            key_perf    = (_find_key(keys, "perf", "score", "global")
                           or _find_key(keys, "perf", "score")
                           or "perf_score_global")
            key_success = (_find_key(keys, "success", "label")
                           or _find_key(keys, "success")
                           or "success_label")

            mct = (row.get(key_mct) or "").strip()
            mapping[mct] = {
                "prompt_str": (row.get(key_prompt) or "").strip(),
                "analysis_prompt_updown": (row.get(key_updown) or "").strip(),
                "perf_score_global": (row.get(key_perf) or "").strip(),
                "success_label": (row.get(key_success) or "").strip(),
            }
        return mapping, src, None
    except Exception as e:
        return {}, src, str(e)

def build_mct_consult_prompt(info: dict, encoded_mct: str, p_main: str, p_updn: str) -> str:
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
        """[비판적 의견 제시 방안]
- detailed_guidance의 첫 2문장은 '반대 시각 요약'으로 시작하라(간결하게).
- evidence_mentions 중 1개는 '근거의 한계'를 설명하라(표본 크기/계절성/역인과/측정 편향 등).
- 제안이 유지되는 최소 조건 1개와, 중단해야 하는 신호 1개를 명시하라(수치나 규칙 형태).
- 어조는 직설적이고 검증 중심으로, 과장/모호한 표현을 피하라."""
    )
    return ensure_data_evidence(base)

def render_mct_tab():
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
    # --- (입력 박스 다음) ENCODED_MCT KPI 미리보기 -----------------
    if encoded_mct:
        mapping_preview, src_preview, err_preview = load_mct_prompts(uploaded_file=mct_csv_file)
        if err_preview:
            st.warning(f"CSV 로드 실패: {err_preview}")
        else:
            row_preview = mapping_preview.get(encoded_mct.strip())
            if row_preview:
                render_mct_kpi(
                    row_preview.get("perf_score_global"),
                    row_preview.get("success_label")
                )
            else:
                st.info("해당 ENCODED_MCT에 대한 KPI 데이터가 없습니다.")
    # ----------------------------------------------------------------
    # 이전 대화 표시
    for msg in st.session_state.mct_history:
        with st.chat_message(msg["role"]):
            if msg.get("type") == "strategy":
                c = st.container()
                render_strategy_payload(msg.get("data", {}), c, prefix=msg.get("id", "mct_hist"))
            else:
                st.markdown(msg.get("content",""))

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

        # CSV 로딩
        mapping, src, err = load_mct_prompts(uploaded_file=mct_csv_file)
        p_main, p_updn = "", ""
        if err:
            st.warning(f"CSV 로드 실패: {err}")
        else:
            data = mapping.get((encoded_mct or "").strip())
            if data:
                p_main = data.get("prompt_str", "")
                p_updn = data.get("analysis_prompt_updown", "")
            else:
                st.info("해당 ENCODED_MCT에 대한 외부 프롬프트가 없어 기본 로직으로 진행합니다.")

        prompt = build_mct_consult_prompt(st.session_state.mct_info, encoded_mct, p_main, p_updn)

        # 외부 KB 근거 주입 (옵션)
        kb_opts = st.session_state.get("_kb_opts", {})
        if kb_opts.get("use"):
            kb = load_external_kb(kb_opts.get("dir", "./subtitle_summary/summary"))
            if kb:
                kb_q = build_kb_query(st.session_state.mct_info, encoded_mct)
                kb_ev = kb["search"](kb_q, top_k=int(kb_opts.get("topk", 3)))
                if kb_ev:
                    prompt += f"\n\n{kb_ev}\n"

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
        # 외부 KB 근거 주입 (옵션)
        kb_opts = st.session_state.get("_kb_opts", {})
        if kb_opts.get("use"):
            kb = load_external_kb(kb_opts.get("dir", "./subtitle_summary/summary"))
            print("kb",kb)
            if kb:
                base_follow += "\n\n" + kb["search"](build_kb_query(st.session_state.mct_info, mct_q),
                                                     top_k=int(kb_opts.get("topk", 3)))

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
# 11) 공통 프롬프트 빌더 (직접 답/후속)
# ─────────────────────────────
def build_followup_prompt(question: str, info: dict, strategy_payload: dict | None, raw_strategy: str) -> str:
    info_keys = ["상점명","업종","프랜차이즈여부","점포연령","고객연령대","고객행동"]
    info_lines = [f"- {k}: {info[k]}" for k in info_keys if info.get(k)]
    info_block = "\n".join(info_lines) if info_lines else "- 추가 상점 정보 없음"
    if strategy_payload:
        try:
            strategy_block = json.dumps(strategy_payload, ensure_ascii=False, indent=2)
        except Exception:
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
        "데이터 근거 항목이나 KPI가 있다면 그대로 언급하거나 수치로 답변에 반영하세요.\n"
        f"{FOLLOWUP_RESPONSE_GUIDE}"
    )
    return prompt

def build_direct_question_prompt(info: dict, question: str, missing_fields=None) -> str:
    missing_fields = missing_fields or []
    info_lines = [f"- {f}: {info[f]}" for f in INFO_FIELD_ORDER if info.get(f)]
    info_block = "\n".join(info_lines) if info_lines else "- 제공된 정보 없음"
    missing_note = ""
    if missing_fields:
        missing_note = "\n\n주의: 아직 다음 정보가 제공되지 않았습니다. " + ", ".join(missing_fields) + "."
    prompt = (
        "당신은 동네 상권을 돕는 시니어 마케팅 컨설턴트입니다.\n"
        "아래 상점 정보를 참고해 사용자의 질문에 대해 바로 실행할 수 있는 조언을 주세요.\n"
        "답변은 문단 형태로 작성하고, 필요한 경우 수치나 규칙 같은 근거를 문장 안에 녹여 설명하세요.\n"
        "새로운 가정을 만들기보다는 제공된 정보를 우선적으로 활용하세요.\n\n"
        "=== 상점 정보 ===\n"
        f"{info_block}\n\n"
        "=== 사용자 질문 ===\n"
        f"{question}\n"
        f"{missing_note}\n"
        f"{DIRECT_RESPONSE_GUIDE}"
    )
    return prompt

# ─────────────────────────────
# 12) Streamlit UI (모드 전환 + 사이드바 KB 옵션)
# ─────────────────────────────
st.set_page_config(page_title="AI 마케팅 컨설턴트", layout="wide")
st.title("💬 AI 마케팅 컨설턴트")

mode = st.sidebar.radio("모드", ["기존 상담", "ENCODED_MCT 컨설턴트"], index=0)
st.sidebar.divider()
st.sidebar.markdown("**외부 지식베이스(subtitle_summary) 사용**")
use_kb = st.sidebar.checkbox("근거 주입 사용", value=False)
kb_dir = st.sidebar.text_input("KB 폴더 경로", "./subtitle_summary/summary", disabled=not use_kb)
kb_topk = st.sidebar.slider("근거 개수", 1, 5, 3, disabled=not use_kb)
st.session_state["_kb_opts"] = {"use": use_kb, "dir": kb_dir, "topk": kb_topk}

if mode == "ENCODED_MCT 컨설턴트":
    render_mct_tab()
    st.stop()

# ─────────────────────────────
# 13) 기존 상담 플로우 (원형 유지)
# ─────────────────────────────
if st.button("🔄 새 상담 시작"):
    st.session_state.clear()
    st.rerun()

# 상태 초기화
for k, v in {
    "chat_history": [],
    "info": {},
    "initialized": False,
    "latest_strategy": {},
    "followup_ui": {},
    "followup_ui_key": 0,
    "auto_followup_question": None,
    "shown_followup_suggestion": False,
    "pending_question": None,
    "use_pending_question": False,
    "pending_question_button_key": 0,
    "is_generating": False,
}.items():
    st.session_state.setdefault(k, v)

# 첫 안내
if not st.session_state.initialized:
    with st.chat_message("assistant"):
        st.markdown(
            "안녕하세요 👋 저는 **AI 마케팅 컨설턴트**입니다.\n\n"
            "상점명을 입력해주시면 업종과 프랜차이즈 여부를 분석하고, "
            "몇 가지 질문을 통해 맞춤형 마케팅 전략을 제안드릴게요.\n\n"
            "예: `교촌치킨`, `파리바게뜨`, `카페행당점`, `왕십리돼지국밥`"
        )
    st.session_state.initialized = True

def add_message(role, content=None, **kwargs):
    msg = {"role": role}
    if kwargs.get("type") == "strategy":
        msg["type"] = "strategy"
        msg["data"] = kwargs.get("data", {})
        msg["id"] = kwargs.get("id", str(uuid.uuid4()))
        msg["raw"] = kwargs.get("raw")
    else:
        msg["content"] = content if content is not None else ""
    st.session_state.chat_history.append(msg)

# 히스토리 렌더
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "strategy":
            box = st.container()
            render_strategy_payload(msg.get("data", {}), box, prefix=msg.get("id", "history"))
        else:
            st.markdown(msg.get("content",""))

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

def answer_question_with_current_info(question_text: str):
    info = st.session_state.get("info", {})
    missing_fields = get_missing_info_fields(info)
    prompt = build_direct_question_prompt(info, question_text, missing_fields)

    # 외부 KB 근거 주입 (옵션)
    kb_opts = st.session_state.get("_kb_opts", {})
    if kb_opts.get("use"):
        kb = load_external_kb(kb_opts.get("dir", "./subtitle_summary/summary"))
        if kb:
            prompt += "\n\n" + kb["search"](build_kb_query(info, question_text), top_k=int(kb_opts.get("topk", 3)))

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
            guidance_text, suggested = parse_direct_answer(answer, info, question_text)
            guidance_text = guidance_text or answer.strip()
            ui_key = st.session_state.get("followup_ui_key", 0) + 1
            st.session_state.followup_ui_key = ui_key
            st.session_state.shown_followup_suggestion = False
            st.session_state.followup_ui = {
                "guidance": guidance_text,
                "evidence": [],
                "suggested_question": suggested,
                "key": ui_key,
            }
            placeholder.empty()
            with st.container():
                render_followup_panel(guidance_text, [], suggested, ui_key)
            log = "### 📘 상세 가이드\n\n" + guidance_text
            if suggested:
                log += f"\n\n**가능한 다음 질문:** {suggested}"
            add_message("assistant", log)
            st.session_state.latest_strategy = {"payload": None, "raw": guidance_text}
        else:
            warn = "답변을 생성하지 못했습니다. 질문을 조금 다르게 해보시면 도움이 될 수 있어요."
            placeholder.warning(warn)
            add_message("assistant", warn)
            st.session_state.latest_strategy = {"payload": None, "raw": ""}

    if get_missing_info_fields(info):
        st.session_state.pending_question = question_text
    else:
        st.session_state.pending_question = None
    st.session_state.use_pending_question = False
    st.session_state.shown_followup_suggestion = False

# 입력 처리
if user_input:
    use_pending = st.session_state.pop("use_pending_question", False)
    st.session_state.followup_ui = {}
    add_message("user", user_input)

    if use_pending:
        answer_question_with_current_info(user_input)
        st.stop()

    # ① 상점명 수집
    if "상점명" not in st.session_state.info:
        info_updates, detected_q = extract_initial_store_info(user_input)
        name = info_updates.get("상점명") or user_input.strip()
        st.session_state.info["상점명"] = name
        st.session_state.info["업종"] = classify_hpsn_mct(name)
        st.session_state.info["프랜차이즈여부"] = "프랜차이즈" if is_franchise(name) else "개인점포"
        for f in ("점포연령","고객연령대","고객행동"):
            if info_updates.get(f): st.session_state.info[f] = info_updates[f]
        st.session_state.pending_question = detected_q or user_input
        st.session_state.shown_followup_suggestion = False

        missing = get_missing_info_fields(st.session_state.info)
        if missing:
            nf = missing[0]
            if nf == "점포연령":
                msg = (
                    f"'{name}'은(는) **{st.session_state.info['업종']} 업종**이며 "
                    f"**{st.session_state.info['프랜차이즈여부']}**로 추정됩니다. 🏪\n\n"
                    "개업 시기가 언제인가요? (예: 6개월 전, 2년 전)"
                )
            elif nf == "고객연령대":
                msg = "좋아요 👍 주요 고객층은 어떤 연령대인가요? (20대 / 30~40대 / 50대 이상)"
            else:
                msg = "마지막으로, 고객 유형은 어떤 편인가요? (쉼표로 구분: 재방문, 신규, 직장인, 유동, 거주)"
            add_message("assistant", msg + BUTTON_HINT)
            st.rerun()
        else:
            st.session_state.pending_question_button_key += 1
            st.session_state.pending_question = st.session_state.pending_question or user_input
            st.session_state.auto_followup_question = st.session_state.pending_question
            st.session_state.use_pending_question = True
            st.rerun()

    # ② 점포연령
    elif "점포연령" not in st.session_state.info:
        nums = re.findall(r"\d+", user_input)
        months = int(nums[0]) if nums else 0
        st.session_state.info["점포연령"] = _store_age_label_from_months(months)
        add_message("assistant", "좋아요 👍 주요 고객층은 어떤 연령대인가요? (20대 / 30~40대 / 50대 이상)" + BUTTON_HINT)
        st.rerun()

    # ③ 고객연령대
    elif "고객연령대" not in st.session_state.info:
        txt = user_input
        if "20" in txt:
            st.session_state.info["고객연령대"] = "20대 이하 고객 중심"
        elif "30" in txt or "40" in txt:
            st.session_state.info["고객연령대"] = "30~40대 고객 중심"
        else:
            st.session_state.info["고객연령대"] = "50대 이상 고객 중심"
        add_message("assistant", "마지막으로, 고객 유형은 어떤 편인가요? (쉼표로 구분: 재방문, 신규, 직장인, 유동, 거주)" + BUTTON_HINT)
        st.rerun()

    # ④ 고객행동
    elif "고객행동" not in st.session_state.info:
        parts = re.split(r"[,/+\s]*(?:및|와|그리고)?[,/+\s]*", (user_input or "").lower())
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

        # persona 기반 or fallback 전략
        info = st.session_state.info
        persona = find_persona(info["업종"], info["프랜차이즈여부"], info["점포연령"], info["고객연령대"], info["고객행동"])
        if persona and persona.get("prompt"):
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

        # 외부 KB 근거 주입 (옵션)
        kb_opts = st.session_state.get("_kb_opts", {})
        if kb_opts.get("use"):
            kb = load_external_kb(kb_opts.get("dir", "./subtitle_summary/summary"))
            if kb:
                kb_q = build_kb_query(info, st.session_state.get("pending_question") or "")
                kb_ev = kb["search"](kb_q, top_k=int(kb_opts.get("topk", 3)))
                if kb_ev:
                    prompt += f"\n\n{kb_ev}\n"

        add_message("assistant", "이제 AI 상담사가 맞춤형 마케팅 전략을 생성합니다... ⏳")
        with st.chat_message("assistant"):
            st.markdown("### 📈 생성된 마케팅 전략 결과")
            ph = st.empty()
            st.session_state.is_generating = True
            try:
                result = stream_gemini(prompt, output_placeholder=ph)
            finally:
                st.session_state.is_generating = False
            if result:
                payload = parse_strategy_payload(result)
                if payload:
                    mid = str(uuid.uuid4())
                    ph.empty()
                    box = st.container()
                    render_strategy_payload(payload, box, prefix=mid)
                    st.session_state["latest_strategy"] = {"payload": payload, "raw": result}
                    st.session_state.shown_followup_suggestion = False
                    add_message("assistant", type="strategy", data=payload, id=mid, raw=result)
                else:
                    summary_points = extract_executive_summary(result)
                    if summary_points:
                        summary_md = "#### ⚡ 핵심 요약\n\n" + "\n".join(f"- {p}" for p in summary_points)
                        ph.markdown(summary_md)
                        st.session_state["latest_strategy"] = {"payload": None, "raw": result}
                        st.session_state.shown_followup_suggestion = False
                        add_message("assistant", summary_md)
                    else:
                        fb = "구조화된 응답을 표시하지 못했습니다. 다시 시도하거나 프롬프트를 조정해 주세요."
                        ph.warning(fb)
                        st.session_state["latest_strategy"] = {"payload": None, "raw": result}
                        st.session_state.shown_followup_suggestion = False
                        add_message("assistant", fb)

    # ⑤ 후속 질의
    else:
        # 최신 전략/원문 확보
        latest_msg = None
        for item in reversed(st.session_state.get("chat_history", [])):
            if item.get("type") == "strategy":
                latest_msg = item; break
        stored = st.session_state.get("latest_strategy", {})
        strategy_payload = latest_msg.get("data") if latest_msg else stored.get("payload")
        raw_strategy = (latest_msg.get("raw") if latest_msg else None) or stored.get("raw") or ""

        if not (strategy_payload or raw_strategy):
            pending_q = st.session_state.get("pending_question")
            info_state = st.session_state.get("info", {})
            if pending_q and info_state:
                answer_question_with_current_info(pending_q)
                st.rerun()
            fb = "아직 참고할 전략이 없습니다. 상점 정보를 입력해 맞춤 전략을 생성하거나 '이대로 질문' 버튼으로 바로 조언을 받아보세요."
            add_message("assistant", fb)
            st.rerun()

        followup_prompt = build_followup_prompt(user_input, st.session_state.get("info", {}), strategy_payload, raw_strategy)

        # 외부 KB 근거 주입 (옵션)
        kb_opts = st.session_state.get("_kb_opts", {})
        if kb_opts.get("use"):
            kb = load_external_kb(kb_opts.get("dir", "./subtitle_summary/summary"))
            if kb:
                followup_prompt += "\n\n" + kb["search"](
                    build_kb_query(st.session_state.get("info", {}), user_input),
                    top_k=int(kb_opts.get("topk", 3))
                )

        with st.chat_message("assistant"):
            ph2 = st.empty()
            st.session_state.is_generating = True
            try:
                ans = stream_gemini(
                    followup_prompt,
                    output_placeholder=ph2,
                    status_text="질문에 대한 답변을 정리하고 있어요... 💡",
                    progress_text="기존 전략을 바탕으로 가이드를 준비하고 있습니다... 🧭",
                    success_text="✅ 답변이 준비되었습니다.",
                    error_status_text="🚨 답변 생성 중 오류가 발생했습니다.",
                )
            finally:
                st.session_state.is_generating = False
            if ans:
                parsed = parse_followup_payload(ans)
                if parsed and (parsed.get("summary_points") or parsed.get("detailed_guidance")):
                    pts = (parsed.get("summary_points") or [])[:2]
                    ev = (parsed.get("evidence_mentions") or [])[:3]
                    txt = parsed.get("detailed_guidance", "")
                    guidance = "\n\n".join([s for s in ["\n".join(pts), txt] if s.strip()])
                    guidance = guidance or txt or ans
                    suggested = (parsed.get("suggested_question") or "").strip() or default_suggested_question(st.session_state.get("info", {}), user_input)

                    ui_key = st.session_state.get("followup_ui_key", 0) + 1
                    st.session_state.followup_ui_key = ui_key
                    st.session_state.followup_ui = {
                        "guidance": guidance,
                        "evidence": ev,
                        "suggested_question": suggested,
                        "key": ui_key,
                    }
                    log = "### 📘 상세 가이드\n\n" + guidance
                    if ev:
                        log += "\n\n**근거:**\n" + "\n".join(f"- {x}" for x in ev)
                    add_message("assistant", log)
                    ph2.empty()
                    with st.container():
                        render_followup_panel(guidance, ev, suggested, ui_key)
                else:
                    ui_key = st.session_state.get("followup_ui_key", 0) + 1
                    st.session_state.followup_ui_key = ui_key
                    fallback_suggest = default_suggested_question(st.session_state.get("info", {}), user_input)
                    st.session_state.followup_ui = {
                        "guidance": ans,
                        "evidence": [],
                        "suggested_question": fallback_suggest,
                        "key": ui_key,
                    }
                    clean = (ans or "").strip()
                    if clean:
                        log = "### 📘 상세 가이드\n\n" + clean
                        if fallback_suggest:
                            log += f"\n\n**가능한 다음 질문:** {fallback_suggest}"
                        add_message("assistant", log)
                    ph2.empty()
                    with st.container():
                        render_followup_panel(ans, [], fallback_suggest, ui_key)
            else:
                ph2.warning("답변을 생성하지 못했습니다. 질문을 조금 다르게 해보시면 도움이 될 수 있어요.")
else:
    # 기존 세션의 followup 패널 재표시
    ui = st.session_state.get("followup_ui", {})
    if ui.get("guidance"):
        with st.chat_message("assistant"):
            render_followup_panel(ui.get("guidance",""), ui.get("evidence", []), ui.get("suggested_question",""), ui.get("key", 0))
