import streamlit as st

# -----------------------
# PAGE CONFIG
# -----------------------

st.set_page_config(
    page_title="정책 큐레이터",
    page_icon="🤖",
    layout="wide",
)

# -----------------------
# SECRETS
# -----------------------

try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    st.error("오류: OpenAI API 키를 찾을 수 없습니다. Streamlit Cloud의 Secrets 설정을 확인해주세요.")
    st.stop()

# -----------------------
# IMPORTS
# -----------------------

import time
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

# -----------------------
# CSS
# -----------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"]  { font-family: 'IBM Plex Sans KR', sans-serif; }
.stApp { background-color: #F0F2F6; }
[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E0E0E0; }
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
</style>
""", unsafe_allow_html=True)

# -----------------------
# RAG COMPONENTS
# -----------------------

DATA_PATH = "./data"

@st.cache_resource
def get_rag_components():
    if not os.path.exists(DATA_PATH) or not any(f.endswith('.pdf') for f in os.listdir(DATA_PATH)):
        st.error(f"오류: '{DATA_PATH}' 폴더를 찾을 수 없거나 PDF가 없습니다.")
        st.stop()

    documents = []
    for file in os.listdir(DATA_PATH):
        if file.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = Qdrant.from_documents(
        chunks, embeddings, location=":memory:", collection_name="policy_documents"
    )
    retriever = vectorstore.as_retriever(search_kwargs={'k': 20})

    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0.1)

    prompt_template = PromptTemplate.from_template(
        """당신은 대한민국 정부 정책 전문가입니다. 사용자의 질문에 대해 아래의 '문서 내용'을 바탕으로, 명확하고 친절하게 답변해주세요.
        답변은 항상 한국어로 작성해야 합니다. 만약 '문서 내용'에 질문과 정확히 일치하는 정보가 없다면, 그 사실을 먼저 언급한 후, 관련성이 매우 높은 정보가 있다면 그 정보를 바탕으로 유연하게 답변해주세요.
        문서 내용에 전혀 관련 없는 정보만 있다면, "죄송합니다. 제공된 문서에서는 관련 정보를 찾을 수 없습니다." 라고만 답변하세요.
        [문서 내용]
        {context}
        [사용자 질문]
        {question}
        [답변]
        """
    )

    reranker_model = CrossEncoder('bongsoo/kpf-cross-encoder-v1')

    return retriever, llm, prompt_template, reranker_model

try:
    retriever, llm, prompt_template, reranker_model = get_rag_components()
except Exception as e:
    st.error(f"RAG 구성 요소 설정 중 오류 발생: {e}")
    st.stop()

# -----------------------
# KEYWORD DICT
# -----------------------

keyword_dict = {
    # 1. 주거 관련 정책
    "신혼부부·청년 임차보증금 이자 지원사업": ["청년 전세대출 이자", "신혼부부 버팀목 대출", "주거 보증금 대출", "전월세 이자 지원"],
    "청년 부동산 중개보수 및 이사비 지원": ["부동산 중개비 지원", "복비 지원", "청년 이사 비용 지원", "이사비 40만원"],
    "1인가구 전월세 안심계약 도움서비스": ["전월세 안심계약", "집보기 동행 서비스", "부동산 계약 상담", "주거안심매니저"],
    
    # 2. 일자리/창업 관련 정책
    "청년내일채움공제": ["내일채움공제", "내채공", "중소기업 2년 적금", "1200만원 통장"],
    "미래 청년 일자리 사업": ["서울시 인턴", "미래일자리", "AI 일자리 인턴", "소셜벤처 인턴십"],
    "취업날개 서비스": ["면접 정장 무료 대여", "취준생 정장 대여", "무료 정장 대여", "면접 복장 지원"],
    "서울시 일자리카페 운영": ["일자리카페", "취업 스터디룸", "무료 이력서 사진 촬영", "취업 상담 카페"],
    "청년창업기업 보증": ["창업 보증 지원", "기술보증기금 청년 창업", "청년 사업자 보증", "창업 대출 보증"],
    "청년전용창업자금": ["청년창업대출", "창업 융자 지원", "중소벤처기업부 창업 자금"],
    "서울 청년 취업 멘토링 페스타": ["취업 멘토링", "직무 멘토링", "현직자 상담", "서울시 취업 박람회"],
    "서울 청년 밀키트 창업지원": ["밀키트 창업", "요식업 창업 지원", "F&B 창업 교육"],
    "AI 면접체험실 운영": ["AI 면접 연습", "AI 역량검사 체험", "가상 면접 시뮬레이션", "취업 면접 준비"],

    # 3. 금융/자산 관련 정책
    "희망두배 청년통장": ["청년통장", "희두청", "서울시 청년 적금", "청년 목돈 마련 지원"],
    "청년내일저축계좌": ["청년저축계좌", "저소득 청년 자산형성", "차상위계층 청년 통장"],
    "청년층 신용회복 지원사업": ["청년 신용불량 지원", "학자금 대출 연체 해결", "신용회복 상담", "청년 부채 조정"],
    "학자금 대출이자 지원사업": ["학자금 이자 지원", "대출 이자 감면", "한국장학재단 이자 지원"],
    "서울 영테크": ["영테크", "청년 재무설계", "무료 재무상담", "청년 금융교육"],

    # 4. 복지/문화 관련 정책
    "기후동행카드": ["기동카", "서울 교통카드 할인", "청년 교통비 지원", "알뜰교통카드 청년"],
    "서울시 청년수당": ["청년수당", "미취업 청년 지원금", "서울시 생활비 지원", "구직활동지원금"],
    "서울시 고립·은둔청년 지원사업": ["고립청년 지원", "은둔청년 상담", "히키코모리 지원", "사회성 회복 프로그램"],
    "자립준비청년 자립수당 지원": ["보호종료아동 지원", "자립수당", "자립정착금", "보육원 퇴소 청년 지원"],
    "서울 청년 마음건강 지원사업": ["청년 심리상담", "마음건강 바우처", "청년 우울증 상담", "정신건강 지원"],
    "청년예술지원사업": ["청년 예술인 지원", "신진 예술가 지원", "창작지원금", "첫 작품 발표 지원"],
    "서울청년문화패스 지원": ["청년문화패스", "공연 관람비 지원", "문화이용권", "20만원 문화생활비"],
    "서울 러닝크루": ["7979 러닝크루", "서울 달리기 모임", "목요일 달리기", "러닝 동호회"],
    "청년예술청 운영": ["청년예술청", "예술가 공간 지원", "연습실 대관", "예술인 공유 오피스"]
}

def expand_keywords(user_query):
    expanded = [user_query]
    for key, synonyms in keyword_dict.items():
        if key in user_query:
            expanded += synonyms
    return list(set(expanded))

# -----------------------
# SIDEBAR UI
# -----------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "profile" not in st.session_state:
    st.session_state.profile = {}
if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

with st.sidebar:
    st.header("🎯 나의 맞춤 조건 설정")
    st.markdown("AI가 더 정확한 정책을 추천하도록 정보를 입력해주세요.")
    age = st.number_input("나이(만)", min_value=19, max_value=39, value=st.session_state.profile.get("age", 25))
    interests = st.multiselect(
        "주요 관심 분야",
        ['주거', '일자리/창업', '금융/자산', '복지/문화'],
        default=st.session_state.profile.get("interests", [])
    )
    # [개선] 버튼 클릭 시 프로필 저장과 함께 동적 메시지를 생성합니다.
    if st.button("✅ 조건 저장 및 반영", type="primary", use_container_width=True):
        st.session_state.profile = { "age": age, "interests": interests }
        st.success("맞춤 조건이 저장되었습니다!")

        # [개선] 저장된 프로필을 기반으로 동적 환영 메시지를 생성하고 대화 기록에 추가합니다.
        if interests:
             welcome_message = f"안녕하세요! {age}세, '{interests[0]}' 분야에 관심이 있으시군요. 이제부터 맞춤형으로 답변해 드릴게요!"
        else:
             welcome_message = f"안녕하세요! {age}세이시군요. 관심 분야를 선택하시면 더 정확한 추천을 받을 수 있어요."
        
        st.session_state.messages.append({"role": "assistant", "content": welcome_message})
        
        time.sleep(1)
        st.rerun()

# -----------------------
# MAIN UI
# -----------------------

st.title("🤖 청년 정책 큐레이터")
st.caption("AI 기반 맞춤형 정책 탐색기")

recommended_questions_db = {
    "주거": [
        "임차보증금 알려줘",
        "역세권 청년주택 알려줘?"
    ],
    "일자리/창업": [
        "청년내일채움공제 지금도 신청할 수 있나?",
        "서울시 청년수당으로 무엇에 쓸 수 있는지 알려줘"
    ],
    "금융/자산": [
        "희망두배 청년통장은 어떤 혜택이 있나?",
        "청년내일저축계좌 지원금은 얼마까지 받을 수 있나?"
    ],
    "복지/문화": [
        "고립은둔청년 지원사업은 어떤 내용이야?",
        "자립준비청년 자립수당은 어떻게 신청하나?"
    ]
}

st.markdown("##### 무엇을 도와드릴까요?")
profile_interests = st.session_state.get("profile", {}).get("interests", [])
questions_to_show = recommended_questions_db.get(
    profile_interests[0],
    ["자립준비청년 자립수당 알려줘?", "희망두배 청년통장 신청 조건 알려줘"]
) if profile_interests else ["자립준비청년 자립수당 알려줘?", "희망두배 청년통장 신청 조건 알려줘"]

cols = st.columns(len(questions_to_show))
for i, question in enumerate(questions_to_show):
    if cols[i].button(question, use_container_width=True, key=f"rec_q_{i}"):
        st.session_state.selected_question = question

# -----------------------
# CHAT UI
# -----------------------

if not st.session_state.messages:
    profile = st.session_state.get("profile", {})
    welcome_message = f"안녕하세요! {profile['age']}세, '{profile['interests'][0]}' 분야에 관심이 있으시군요." if profile.get("age") and profile.get("interests") else "안녕하세요! 어떤 정책이 궁금하신가요?"
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 근거 자료 확인하기"):
                for source in message["sources"]:
                    st.info(f"출처: {source.metadata.get('source', 'N/A')} (페이지: {source.metadata.get('page', 'N/A')})")
                    st.write(source.page_content)

prompt = st.chat_input("궁금한 정책에 대해 질문해보세요.")
if st.session_state.selected_question:
    prompt = st.session_state.selected_question
    st.session_state.selected_question = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("질문을 분석하고 관련 정보를 찾는 중입니다..."):
            try:
                expanded_queries = expand_keywords(prompt)

                expansion_prompt = PromptTemplate.from_template(
                    """당신은 한국 청년 정책 검색어 전문가입니다.
                    사용자의 질문을 보고, 관련성이 높은 정책명, 제도명, 혹은 프로그램명을 최대 3개 생성해주세요.
                    특히 다른 이름으로 불릴 가능성이 있다면 반드시 포함해주세요.
                    쉼표로 구분된 하나의 문자열로 응답해주세요.
                    질문: {question}"""
                )
                query_expansion_chain = expansion_prompt | llm | StrOutputParser()
                expanded_queries_str = query_expansion_chain.invoke({"question": prompt})
                expanded_queries += [q.strip() for q in expanded_queries_str.split(',') if q.strip()]
                expanded_queries = list(set(expanded_queries))

                all_retrieved_docs = []
                for q in expanded_queries:
                    all_retrieved_docs.extend(retriever.invoke(q))

                unique_docs = list({doc.page_content: doc for doc in all_retrieved_docs}.values())

                final_docs = []
                if unique_docs:
                    pairs = [[prompt, doc.page_content] for doc in unique_docs]
                    scores = reranker_model.predict(pairs)
                    doc_scores = sorted(zip(scores, unique_docs), key=lambda x: x[0], reverse=True)
                    final_docs = [doc for score, doc in doc_scores[:5]]

                if final_docs:
                    context = "\n\n".join(doc.page_content for doc in final_docs)
                    final_prompt = prompt_template.format(context=context, question=prompt)
                    response = llm.invoke(final_prompt).content
                else:
                    response = "죄송합니다. PDF 문서에서도 정보를 찾을 수 없습니다. 다른 질문을 시도해보세요!"

                st.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": final_docs
                })

            except Exception as e:
                error_message = f"답변 생성 중 오류가 발생했습니다: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

    st.rerun()
