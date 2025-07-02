# ======================================================================
# 파일 1: app.py
# 지능형 쿼리 확장 기능이 추가되고, RAG 파이프라인 로직이 개선되었습니다.
# ======================================================================
import streamlit as st
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
    st.sidebar.info("문서를 로드하고 있습니다...")
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

    st.sidebar.info("데이터베이스를 구축하고 있습니다...")
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = Qdrant.from_documents(
        chunks, embeddings, location=":memory:", collection_name="policy_documents"
    )
    retriever = vectorstore.as_retriever(search_kwargs={'k': 20})
    st.sidebar.success("데이터베이스 구축 완료!")

    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0.1)

    prompt_template = PromptTemplate.from_template(
        """당신은 대한민국 정부 정책 전문가입니다. 사용자의 질문에 대해 아래의 '문서 내용'을 바탕으로, 명확하고 친절하게 답변해주세요.
        답변은 항상 한국어로 작성해야 합니다. 문서 내용에 없는 정보는 답변에 포함하지 마세요.
        [문서 내용]
        {context}
        [사용자 질문]
        {question}
        [답변]
        """
    )

    st.sidebar.info("Re-ranker 모델을 로드하고 있습니다...")
    reranker_model = CrossEncoder('bongsoo/kpf-cross-encoder-v1')
    st.sidebar.success("모든 컴포넌트 로드 완료!")

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
    "청년 월세": ["월세 지원", "임대료 지원", "청년 주거"],
    "청년수당": ["청년 지원금", "서울시 청년수당", "생활비 지원"],
    "청년AI취업캠프": ["AI취업캠프", "청년 취업 프로그램"],
    "청년도전": ["청년도전 지원사업", "심리 지원 프로그램"],
    "역세권 청년주택": ["청년 주거", "역세권 주택"],
    "서울형 청년인턴 직무캠프": ["청년 인턴십", "청년 직무 체험", "서울 청년 인턴"],
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
    age = st.number_input("나이(만)", min_value=18, max_value=100, value=st.session_state.profile.get("age", 25))
    interests = st.multiselect(
        "주요 관심 분야",
        ['주거', '일자리/창업', '금융/자산', '복지/문화'],
        default=st.session_state.profile.get("interests", [])
    )
    if st.button("✅ 조건 저장 및 반영", type="primary", use_container_width=True):
        st.session_state.profile = { "age": age, "interests": interests }
        st.success("맞춤 조건이 저장되었습니다!")
        time.sleep(1)
        st.rerun()

# -----------------------
# MAIN UI
# -----------------------

st.title("🤖 청년 정책 큐레이터")
st.caption("AI 기반 맞춤형 정책 탐색기")

recommended_questions_db = {
    "주거": ["임차보증금 알려줘", "역세권 청년주택 알려줘?"],
    "일자리/창업": ["청년내일채움공제 지금도 신청할 수 있나?", "서울시 청년수당으로 무엇에 쓸 수 있는지 알려줘"],
    "금융/자산": ["희망두배 청년통장은 어떤 혜택이 있나?", "청년내일저축계좌 지원금은 얼마까지 받을 수 있나?"],
    "복지/문화": ["고립은둔청년 지원사업은 어떤 내용이야?", "자립준비청년 자립수당은 어떻게 신청하나?"]
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
                # --- [개선] 1. 초기 검색 및 관련성 평가 ---
                initial_docs = retriever.invoke(prompt)
                unique_initial_docs = list({doc.page_content: doc for doc in initial_docs}.values())
                
                relevance_score = 0.0
                final_docs = []

                if unique_initial_docs:
                    pairs = [[prompt, doc.page_content] for doc in unique_initial_docs]
                    scores = reranker_model.predict(pairs)
                    if scores.any():
                        relevance_score = max(scores)
                
                RELEVANCE_THRESHOLD = 0.1 # 관련성 임계값 설정

                # --- [개선] 2. 조건부 쿼리 확장 실행 ---
                if relevance_score < RELEVANCE_THRESHOLD:
                    with st.spinner("초기 검색 결과가 낮아, 추가적인 검색을 수행합니다..."):
                        # 키워드 기반 확장
                        expanded_queries = expand_keywords(prompt)

                        # LLM 기반 확장
                        expansion_prompt = PromptTemplate.from_template(
                            """당신은 한국 청년 정책 검색어 전문가입니다.
                            사용자의 질문을 보고, 관련성이 높은 정책명, 제도명, 혹은 프로그램명을 최대 3개 생성해주세요.
                            쉼표로 구분된 하나의 문자열로 응답해주세요. 질문: {question}"""
                        )
                        query_expansion_chain = expansion_prompt | llm | StrOutputParser()
                        expanded_queries_str = query_expansion_chain.invoke({"question": prompt})
                        expanded_queries += [q.strip() for q in expanded_queries_str.split(',') if q.strip()]
                        expanded_queries = list(set(expanded_queries))

                        # 확장된 쿼리로 재검색
                        all_retrieved_docs = []
                        for q in expanded_queries:
                            all_retrieved_docs.extend(retriever.invoke(q))
                        
                        unique_docs = list({doc.page_content: doc for doc in all_retrieved_docs}.values())
                        
                        # 최종 재순위화
                        if unique_docs:
                            pairs = [[prompt, doc.page_content] for doc in unique_docs]
                            scores = reranker_model.predict(pairs)
                            doc_scores = sorted(zip(scores, unique_docs), key=lambda x: x[0], reverse=True)
                            final_docs = [doc for score, doc in doc_scores[:3]]
                else:
                    # 관련성이 높으면 초기 검색 결과 사용
                    doc_scores = sorted(zip(scores, unique_initial_docs), key=lambda x: x[0], reverse=True)
                    final_docs = [doc for score, doc in doc_scores[:3]]

                # --- 3. 최종 답변 생성 ---
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
