
# 내 가게를 살리는 AI 비밀상담사 - 가맹점별 찰떡 마케팅 전략을 찾아라

## Objective

- 우리 주변 음식 가맹점에게 진짜 필요한 ‘맞춤 마케팅 전략’을 제공하는 AI 에이전트 제안
- 제공된 데이터 및 다양한 외부 데이터를 활용하여 매장별 특징/고객층/상권에 맞게 홍보, 이벤트, 할인, SNS 활용 꿀팁 등
최고의 마케팅 방법을 자동으로 추천하는‘AI비서’를 기획하고, 실제 점주가 바로 쓸 수 있는 서비스 아이디어 제안
(예시) - 매출, 고객, 주변 정보 분석/반영하여 마케팅 채널 추천
- 사장님에게 카카오톡/앱 알림/인스타그램 홍보안 작성해주는 컨설팅 챗봇 만들기
- 계절별, 단골 고객 데이터 활용, ‘신제품 출시 타이밍’ 등 추천 시나리오 제안

## Sample code 
- [예시 코드](https://github.com/thjeong/shcard_2025_bigcontest.git)

## Strategy

1. 사용자 프로필 입력
- 업종
- 프랜차이즈 여부
- 신규 상점 여부
- 매장 규모(배달 전용 구분)
- 고객 연령대
- 고객 분류

2. 프로필 매칭 후 프롬프트 생성
3. 프롬프트에 따른 채널 & 전략 & 콘텐츠 가이드 생성

## Development

### Setup

#### Google AI Studio API KEY 생성 방법

https://aistudio.google.com/apikey 접속 후 (Google 로그인 필요) Get API KEY 메뉴에서 생성하면 됩니다.

```
export GOOGLE_API_KEY=
```

#### 환경 설정
```
python3 -m venv ./.venv
source ./.venv/bin/activate

pip install -r requirements.txt

mkdir .streamlit
echo "GOOGLE_API_KEY='${GOOGLE_API_KEY}'" > .streamlit/secrets.toml
```

#### Project Structure
```
root/
├── streamlit_app.py        # 마케팅 상담사 AI Agent Streamlit 앱
├── persona_generator.py    # 페르소나 프롬프트 생성 (→ personas.json)
├── personas.json           # persona_generator가 생성하는 페르소나 파일
└── requirements.txt
```

#### 실행
```
# Prepare persona.json
python persona_generator.py

# Run app
streamlit run streamlit_app.py
```

