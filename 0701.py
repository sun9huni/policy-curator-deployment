import streamlit as st
import time

# --- 페이지 설정 ---
st.set_page_config(
    page_title="정책 큐레이터",
    page_icon="🤖",
    layout="centered"
)

# --- 세션 상태 초기화 ---
# 대화 기록을 저장할 리스트
if "messages" not in st.session_state:
    st.session_state.messages = []
# 사용자 프로필 정보를 저장할 딕셔너리
if "profile" not in st.session_state:
    st.session_state.profile = {}
# 추천 질문 버튼 클릭 시 처리할 변수
if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

# --- 좌측 사이드바: '나의 맞춤 조건' 설정 ---
with st.sidebar:
    st.header("🎯 나의 맞춤 조건 설정")
    st.markdown("AI가 더 정확한 정책을 추천하도록 정보를 입력해주세요.")

    # 핵심 프로필 입력
    age = st.number_input("나이(만)", min_value=18, max_value=100, value=st.session_state.profile.get("age", 25))
    location = st.text_input("현재 거주지 또는 희망 지역", placeholder="예: 서울시, 전라남도 담양군", value=st.session_state.profile.get("location", ""))
    interests = st.multiselect(
        "주요 관심 분야",
        ['주거 지원', '일자리/창업', '금융/자산 형성', '생활/복지'],
        default=st.session_state.profile.get("interests", [])
    )

    # 상세 조건 입력 (확장 가능)
    with st.expander("상세 조건 입력하기"):
        income = st.number_input("연 소득 (만원 단위)", min_value=0, value=st.session_state.profile.get("income", 3000))
        household_type = st.selectbox("가구 형태", ['1인 가구', '2인 이상 가구', '신혼부부'], index=["1인 가구", "2인 이상 가구", "신혼부부"].index(st.session_state.profile.get("household_type", "1인 가구")))
        keywords = st.text_input("관심 키워드 입력", placeholder="예: 스마트팜, 전세 대출", value=st.session_state.profile.get("keywords", ""))

    # 적용 버튼
    if st.button("✅ 조건 저장 및 반영"):
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

# 추천 질문 버튼
st.markdown("##### 무엇을 도와드릴까요?")
cols = st.columns(3)
if cols[0].button("청년 월세 지원 자격 알려줘", use_container_width=True):
    st.session_state.selected_question = "청년 월세 지원 자격 알려줘"
if cols[1].button("내일채움공제 신청 방법", use_container_width=True):
    st.session_state.selected_question = "내일채움공제 신청 방법"
if cols[2].button("귀농 지원 정책 찾아줘", use_container_width=True):
    st.session_state.selected_question = "귀농 지원 정책 찾아줘"


# --- 지능형 채팅 인터페이스 ---

# 이전 대화 기록 표시
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        # 일반 메시지 표시
        st.markdown(message["content"])

        # AI의 답변에 '정책 카드'가 포함된 경우
        if "cards" in message:
            for card in message["cards"]:
                with st.container(border=True):
                    st.subheader(card["title"])
                    st.write(card["summary"])
                    st.markdown("**나와 일치도**")
                    st.progress(card["match_rate"], text=f"{card['match_rate']}%")
                    with st.expander("자세히 보기 및 출처 확인"):
                        st.markdown(card["details"])
                        st.caption(f"출처: {card['source']}")
        
        # AI의 답변에 '피드백 버튼'이 포함된 경우
        if message["role"] == "assistant" and "feedback" in message:
             feedback_cols = st.columns([1, 1, 8])
             # 각 버튼에 고유한 키를 부여하여 오류 방지
             if feedback_cols[0].button("👍", key=f"thumb_up_{i}"):
                 st.toast("피드백 감사합니다!")
             if feedback_cols[1].button("👎", key=f"thumb_down_{i}"):
                 st.toast("개선에 참고하겠습니다.")

# 사용자 입력 또는 추천 질문 처리
prompt = st.chat_input("궁금한 정책에 대해 질문해보세요.")
if st.session_state.selected_question:
    prompt = st.session_state.selected_question
    st.session_state.selected_question = None # 처리 후 초기화

if prompt:
    # 사용자 메시지 기록 및 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # AI 응답 시뮬레이션
    with st.chat_message("assistant"):
        st.markdown("네, 입력해주신 조건에 맞는 정책들을 찾았어요.")
        
        # 정책 카드 데이터 예시 (시뮬레이션)
        card1 = {
            "title": "서울시 청년월세지원",
            "summary": "월 최대 20만원, 12개월간 지원",
            "match_rate": 85,
            "details": """
            - **지원대상**: 서울에 거주하는 만 19세~39세 무주택 청년 1인 가구
            - **소득기준**: 기준중위소득 150% 이하
            - **주요내용**: 임차보증금 8천만원 이하 및 월세 60만원 이하 건물에 거주 시 지원
            """,
            "source": "2024년 서울시 청년월세지원 모집 공고문"
        }
        card2 = {
            "title": "국토교통부 청년월세 한시 특별지원",
            "summary": "월 최대 20만원, 12개월간 지원 (2차 사업)",
            "match_rate": 70,
            "details": """
            - **지원대상**: 부모와 별도 거주하는 만 19세~34세 무주택 청년
            - **소득기준**: 청년가구 기준중위소득 60% 이하, 원가구 기준중위소득 100% 이하
            - **주요내용**: 보증금 5천만원 이하 및 월세 70만원 이하 주택 거주 시 지원
            """,
            "source": "국토교통부 2차 청년월세 한시 특별지원 보도자료"
        }

        # 정책 카드 UI 표시
        with st.container(border=True):
            st.subheader(card1["title"])
            st.write(card1["summary"])
            st.markdown("**나와 일치도**")
            st.progress(card1["match_rate"], text=f"{card1['match_rate']}%")
            with st.expander("자세히 보기 및 출처 확인"):
                st.markdown(card1["details"])
                st.caption(f"출처: {card1['source']}")

        with st.container(border=True):
            st.subheader(card2["title"])
            st.write(card2["summary"])
            st.markdown("**나와 일치도**")
            st.progress(card2["match_rate"], text=f"{card2['match_rate']}%")
            with st.expander("자세히 보기 및 출처 확인"):
                st.markdown(card2["details"])
                st.caption(f"출처: {card2['source']}")
        
        # 피드백 버튼 표시
        feedback_cols = st.columns([1, 1, 8])
        if feedback_cols[0].button("👍", key="thumb_up_new"):
            st.toast("피드백 감사합니다!")
        if feedback_cols[1].button("👎", key="thumb_down_new"):
            st.toast("개선에 참고하겠습니다.")

    # AI 응답을 대화 기록에 추가
    st.session_state.messages.append({
        "role": "assistant",
        "content": "네, 입력해주신 조건에 맞는 정책들을 찾았어요.",
        "cards": [card1, card2],
        "feedback": True
    })

    # 화면을 다시 그려서 모든 위젯이 정상적으로 표시되도록 함
    st.rerun()
