import streamlit as st
import time

# --- 페이지 설정 ---
st.set_page_config(
    page_title="정책 큐레이터",
    page_icon="🤖",
    layout="wide"
)

# --- [개선] 커스텀 CSS 스타일 ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'IBM Plex Sans KR', sans-serif;
}
.stApp { background-color: #F0F2F6; }
[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E0E0E0;
}
.stButton > button {
    border: 1px solid #E0E0E0; border-radius: 10px; color: #31333F;
    background-color: #FFFFFF; transition: all 0.2s ease-in-out;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.stButton > button:hover {
    border-color: #0068C9; color: #0068C9;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
div[data-testid="stSidebar"] .stButton > button {
    background-color: #0068C9; color: white; border: none;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #0055A3; color: white;
}
/* 정책 카드 컨테이너 */
div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] [data-testid="stVerticalBlock"] .st-emotion-cache-12w0qpk {
    background-color: #FFFFFF; border-radius: 10px;
    padding: 1.2rem 1rem 1rem 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
/* st.metric 스타일링 */
div[data-testid="stMetric"] {
    background-color: #F8F9FA;
    border: 1px solid #E0E0E0;
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)


# --- 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "profile" not in st.session_state:
    st.session_state.profile = {}
if "selected_question" not in st.session_state:
    st.session_state.selected_question = None
# [개선] 비교함을 위한 세션 상태 추가 (정책 제목을 key로 사용)
if "compare_basket" not in st.session_state:
    st.session_state.compare_basket = {}

# --- 좌측 사이드바 ---
with st.sidebar:
    st.header("🎯 나의 맞춤 조건 설정")
    st.markdown("AI가 더 정확한 정책을 추천하도록 정보를 입력해주세요.")

    age = st.number_input("나이(만)", min_value=18, max_value=100, value=st.session_state.profile.get("age", 25))
    location = st.text_input("현재 거주지 또는 희망 지역", placeholder="예: 서울시", value=st.session_state.profile.get("location", ""))
    interests = st.multiselect(
        "주요 관심 분야",
        ['주거 지원', '일자리/창업', '금융/자산 형성', '생활/복지'],
        default=st.session_state.profile.get("interests", [])
    )

    with st.expander("상세 조건 입력하기"):
        income = st.number_input("연 소득 (만원 단위)", min_value=0, value=st.session_state.profile.get("income", 3000))
        household_type = st.selectbox("가구 형태", ['1인 가구', '2인 이상 가구', '신혼부부'], index=["1인 가구", "2인 이상 가구", "신혼부부"].index(st.session_state.profile.get("household_type", "1인 가구")))

    if st.button("✅ 조건 저장 및 반영", type="primary", use_container_width=True):
        st.session_state.profile = { "age": age, "location": location, "interests": interests, "income": income, "household_type": household_type }
        st.success("맞춤 조건이 저장되었습니다!")
        time.sleep(1)
        st.rerun()

# --- 메인 화면 ---
st.title("🤖 정책 큐레이터")
st.caption("AI 기반 맞춤형 정책 탐색기 (UI 프로토타입)")

# 추천 질문 데이터
recommended_questions_db = {
    "주거 지원": ["청년 월세 지원 자격 알려줘", "신혼부부 전세 대출 조건", "생애최초 주택 구입 혜택"],
    "일자리/창업": ["개발자 신입 채용 공고 찾아줘", "창업 지원금 종류 알려줘", "내일채움공제 신청 방법"],
}

# 컨텍스트 기반 추천 질문 생성
st.markdown("##### 무엇을 도와드릴까요?")
profile_interests = st.session_state.get("profile", {}).get("interests", [])
questions_to_show = recommended_questions_db.get(profile_interests[0], ["청년 월세 지원 자격 알려줘", "내일채움공제 신청 방법"]) if profile_interests else ["청년 월세 지원 자격 알려줘", "내일채움공제 신청 방법", "귀농 지원 정책 찾아줘"]

cols = st.columns(len(questions_to_show))
for i, question in enumerate(questions_to_show):
    if cols[i].button(question, use_container_width=True, key=f"rec_q_{i}"):
        st.session_state.selected_question = question

# --- 채팅 인터페이스 ---

# 동적 온보딩 메시지
if not st.session_state.messages:
    profile = st.session_state.get("profile", {})
    welcome_message = f"안녕하세요! {profile['age']}세, {profile['interests'][0]} 분야에 관심이 있으시군요." if profile.get("age") and profile.get("interests") else "안녕하세요! 어떤 정책이 궁금하신가요?"
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})

# 이전 대화 기록 표시
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "cards" in message:
            for card in message["cards"]:
                with st.container(border=True):
                    card_id = card["title"]
                    # [개선] 비교함 체크박스 추가
                    is_in_basket = card_id in st.session_state.compare_basket
                    if st.checkbox("비교함에 담기", value=is_in_basket, key=f"compare_{card_id}"):
                        if not is_in_basket:
                            st.session_state.compare_basket[card_id] = card
                            st.rerun()
                    else:
                        if is_in_basket:
                            del st.session_state.compare_basket[card_id]
                            st.rerun()

                    st.subheader(card["title"])
                    st.write(card["summary"])
                    # [개선] st.metric을 사용한 핵심 지표 시각화
                    st.metric(label="나와 일치도", value=f"{card['match_rate']}%")
                    with st.expander("자세히 보기 및 출처 확인"):
                        st.markdown(card["details"])
                        st.caption(f"출처: {card['source']}")


# 사용자 입력 처리
prompt = st.chat_input("궁금한 정책에 대해 질문해보세요.")
if st.session_state.selected_question:
    prompt = st.session_state.selected_question
    st.session_state.selected_question = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    simulated_response_text = "네, 입력해주신 조건에 맞는 정책들을 찾았어요."
    card1 = { "title": "서울시 청년월세지원", "summary": "월 최대 20만원, 12개월간 지원", "match_rate": 85, "details": "- **지원대상**: 만 19세~39세\n- **소득기준**: 기준중위소득 150% 이하", "source": "2024년 서울시 청년월세지원 모집 공고문" }
    card2 = { "title": "국토교통부 청년월세 한시 특별지원", "summary": "월 최대 20만원, 12개월간 지원", "match_rate": 70, "details": "- **지원대상**: 만 19세~34세\n- **소득기준**: 청년가구 중위소득 60% 이하", "source": "국토교통부 2차 청년월세 한시 특별지원 보도자료" }
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": simulated_response_text,
        "cards": [card1, card2]
    })
    st.rerun()
