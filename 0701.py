import streamlit as st
import time

# --- 페이지 설정 ---
st.set_page_config(
    page_title="정책 큐레이터",
    page_icon="🤖",
    layout="centered"
)

# --- 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "profile" not in st.session_state:
    st.session_state.profile = {}
if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

# --- 좌측 사이드바: '나의 맞춤 조건' 설정 ---
with st.sidebar:
    st.header("🎯 나의 맞춤 조건 설정")
    st.markdown("AI가 더 정확한 정책을 추천하도록 정보를 입력해주세요.")

    age = st.number_input("나이(만)", min_value=18, max_value=100, value=st.session_state.profile.get("age", 25))
    location = st.text_input("현재 거주지 또는 희망 지역", placeholder="예: 서울시, 전라남도 담양군", value=st.session_state.profile.get("location", ""))
    interests = st.multiselect(
        "주요 관심 분야",
        ['주거 지원', '일자리/창업', '금융/자산 형성', '생활/복지'],
        default=st.session_state.profile.get("interests", [])
    )

    with st.expander("상세 조건 입력하기"):
        income = st.number_input("연 소득 (만원 단위)", min_value=0, value=st.session_state.profile.get("income", 3000))
        household_type = st.selectbox("가구 형태", ['1인 가구', '2인 이상 가구', '신혼부부'], index=["1인 가구", "2인 이상 가구", "신혼부부"].index(st.session_state.profile.get("household_type", "1인 가구")))
        keywords = st.text_input("관심 키워드 입력", placeholder="예: 스마트팜, 전세 대출", value=st.session_state.profile.get("keywords", ""))

    # [개선] '조건 저장' 버튼 클릭 시 초록색 성공 메시지를 표시하여 명확한 피드백 제공
    if st.button("✅ 조건 저장 및 반영", type="primary"): # 'primary' 타입으로 버튼 강조
        st.session_state.profile = {
            "age": age,
            "location": location,
            "interests": interests,
            "income": income,
            "household_type": household_type,
            "keywords": keywords
        }
        st.success("맞춤 조건이 저장되었습니다!")
        time.sleep(1)
        st.rerun()

# --- 메인 화면: 대화형 정보 탐색 공간 ---
st.title("🤖 정책 큐레이터")
st.caption("AI 기반 맞춤형 정책 탐색기 (UI 프로토타입)")

# 추천 질문 데이터
recommended_questions_db = {
    "주거 지원": ["청년 월세 지원 자격 알려줘", "신혼부부 전세 대출 조건", "생애최초 주택 구입 혜택"],
    "일자리/창업": ["개발자 신입 채용 공고 찾아줘", "창업 지원금 종류 알려줘", "내일채움공제 신청 방법"],
    "금융/자산 형성": ["청년희망적금 만기 후 비과세 혜택", "개인종합자산관리계좌(ISA)란?", "신용점수 올리는 방법"],
    "생활/복지": ["K-패스 신청 방법 알려줘", "육아휴직 급여 신청하기", "문화누리카드 사용처"]
}

# 컨텍스트 기반 추천 질문 생성
st.markdown("##### 무엇을 도와드릴까요?")
profile_interests = st.session_state.get("profile", {}).get("interests", [])
if profile_interests:
    main_interest = profile_interests[0]
    questions_to_show = recommended_questions_db.get(main_interest, [])
else:
    questions_to_show = ["청년 월세 지원 자격 알려줘", "내일채움공제 신청 방법", "귀농 지원 정책 찾아줘"]

cols = st.columns(len(questions_to_show))
for i, question in enumerate(questions_to_show):
    if cols[i].button(question, use_container_width=True, key=f"rec_q_{i}"):
        st.session_state.selected_question = question

# --- 지능형 채팅 인터페이스 ---

# 동적 온보딩 메시지 생성 (대화 기록이 비어있을 때만)
if not st.session_state.messages:
    profile = st.session_state.get("profile", {})
    if profile.get("age") and profile.get("interests"):
        age = profile["age"]
        interest_str = ", ".join(f"'{i}'" for i in profile["interests"])
        welcome_message = f"안녕하세요! {age}세이시군요. {interest_str} 분야에 관심이 있으시네요. 무엇을 도와드릴까요?"
    else:
        welcome_message = "안녕하세요! 어떤 정책이 궁금하신가요? 좌측 사이드바에서 맞춤 조건을 설정하면 더 정확한 추천을 받을 수 있어요."
    
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})

# 이전 대화 기록 표시
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        # [개선] AI 답변은 st.info()를 사용해 시각적으로 구분
        if message["role"] == "assistant":
            st.info(message["content"])
        else:
            st.markdown(message["content"])

        if "cards" in message:
            for card in message["cards"]:
                # [개선] 정책 카드는 st.container(border=True)를 사용해 중립적인 UI로 표시
                with st.container(border=True):
                    st.subheader(card["title"])
                    st.write(card["summary"])
                    st.markdown("**나와 일치도**")
                    st.progress(card["match_rate"], text=f"{card['match_rate']}%")
                    with st.expander("자세히 보기 및 출처 확인"):
                        st.markdown(card["details"])
                        st.caption(f"출처: {card['source']}")
        
        if message["role"] == "assistant" and "feedback" in message:
             feedback_cols = st.columns([1, 1, 8])
             if feedback_cols[0].button("👍", key=f"thumb_up_{i}"):
                 st.toast("피드백 감사합니다!")
             if feedback_cols[1].button("👎", key=f"thumb_down_{i}"):
                 st.toast("개선에 참고하겠습니다.")

# 사용자 입력 또는 추천 질문 처리
prompt = st.chat_input("궁금한 정책에 대해 질문해보세요.")
if st.session_state.selected_question:
    prompt = st.session_state.selected_question
    st.session_state.selected_question = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        # [개선] AI 응답 시뮬레이션도 st.info()를 사용
        simulated_response_text = "네, 입력해주신 조건에 맞는 정책들을 찾았어요."
        st.info(simulated_response_text)
        
        card1 = { "title": "서울시 청년월세지원", "summary": "월 최대 20만원, 12개월간 지원", "match_rate": 85, "details": "- **지원대상**: 서울에 거주하는 만 19세~39세 무주택 청년 1인 가구...", "source": "2024년 서울시 청년월세지원 모집 공고문" }
        card2 = { "title": "국토교통부 청년월세 한시 특별지원", "summary": "월 최대 20만원, 12개월간 지원 (2차 사업)", "match_rate": 70, "details": "- **지원대상**: 부모와 별도 거주하는 만 19세~34세 무주택 청년...", "source": "국토교통부 2차 청년월세 한시 특별지원 보도자료" }

        with st.container(border=True):
            st.subheader(card1["title"])
            st.write(card1["summary"])
            st.markdown("**나와 일치도**"); st.progress(card1["match_rate"], text=f"{card1['match_rate']}%")
            with st.expander("자세히 보기 및 출처 확인"):
                st.markdown(card1["details"]); st.caption(f"출처: {card1['source']}")

        with st.container(border=True):
            st.subheader(card2["title"])
            st.write(card2["summary"])
            st.markdown("**나와 일치도**"); st.progress(card2["match_rate"], text=f"{card2['match_rate']}%")
            with st.expander("자세히 보기 및 출처 확인"):
                st.markdown(card2["details"]); st.caption(f"출처: {card2['source']}")
        
        feedback_cols = st.columns([1, 1, 8])
        if feedback_cols[0].button("👍", key="thumb_up_new"): st.toast("피드백 감사합니다!")
        if feedback_cols[1].button("👎", key="thumb_down_new"): st.toast("개선에 참고하겠습니다.")

    st.session_state.messages.append({
        "role": "assistant",
        "content": simulated_response_text,
        "cards": [card1, card2],
        "feedback": True
    })
    st.rerun()
