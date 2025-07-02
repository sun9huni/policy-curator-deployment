import streamlit as st
import os
import time
import re
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableConfig
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="정책 큐레이터 v3",
    page_icon="🎯",
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

# --- 1. 모델 및 프롬프트 정의 ---
@st.cache_resource
def get_models_and_prompts():
    """LLM, Reranker, Prompts 등 모델 관련 객체들을 로드하고 캐싱합니다."""
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0.1)
    reranker_model = CrossEncoder('bongsoo/kpf-cross-encoder-v1')
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""당신은 사용자의 질문을 다양한 관점에서 유사한 질문들로 다시 생성하는 AI입니다.
        생성된 질문들은 벡터 데이터베이스에서 관련 문서를 찾는 데 사용됩니다.
        사용자의 질문에 대해 3개의 다른 버전의 질문을 생성해주세요.
        질문은 반드시 한국어로 작성되어야 합니다. 각 질문은 다음 줄로 구분해주세요.
        원본 질문: {question}""",
    )
    response_prompt = ChatPromptTemplate.from_template(
        """당신은 대한민국 정부의 청년 정책 전문가입니다. 사용자의 질문에 대해 아래 '문서 내용'을 바탕으로, 명확하고 친절하게 답변해주세요.
        답변은 항상 한국어로 작성해야 합니다. 단계별로 알기 쉽게 설명해주세요.
        만약 '문서 내용'에 질문과 정확히 일치하는 정보가 없다면, 그 사실을 먼저 언급한 후, 관련성이 매우 높은 정보가 있다면 그 정보를 바탕으로 유연하게 답변해주세요.
        문서 내용에 전혀 관련 없는 정보만 있다면, "죄송합니다. 제공된 문서에서는 관련 정보를 찾을 수 없습니다." 라고만 답변하세요.

        [문서 내용]
        {context}

        [사용자 질문]
        {question}

        [답변]
        """
    )
    return llm, reranker_model, QUERY_PROMPT, response_prompt

# --- 2. 벡터 저장소 생성 (✨ 1단계 적용) ---
@st.cache_resource
def create_vector_store():
    """PDF 문서를 로드, 분할하고 '정책 유형' 메타데이터를 추가하여 벡터 저장소를 생성합니다."""
    if not os.path.exists(DATA_PATH) or not any(f.endswith('.pdf') for f in os.listdir(DATA_PATH)):
        st.error(f"오류: '{DATA_PATH}' 폴더를 찾을 수 없거나 PDF 파일이 없습니다.")
        st.stop()

    with st.spinner("정책 문서를 읽고 분석하여 데이터베이스를 구축하는 중입니다..."):
        documents = []
        for file in os.listdir(DATA_PATH):
            if file.endswith('.pdf'):
                loader = PyPDFLoader(os.path.join(DATA_PATH, file))
                documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        # ✨ [1단계] 각 chunk에 '정책 유형' 메타데이터 추가
        current_policy_types = []
        for chunk in chunks:
            # 정규식을 사용하여 '정책 유형: ...' 패턴을 찾습니다.
            match = re.search(r"정책 유형:\s*(.*)", chunk.page_content)
            if match:
                # 유형이 여러 개일 수 있으므로(예: 복지/문화, 금융/자산), 쉼표로 분리합니다.
                types_str = match.group(1).strip()
                current_policy_types = [t.strip() for t in types_str.split(',')]

            # chunk의 메타데이터에 'policy_type' 리스트를 추가합니다.
            chunk.metadata['policy_type'] = current_policy_types

        embedding_model = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sbert-nli",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vectorstore = Qdrant.from_documents(
            chunks, embedding_model, location=":memory:", collection_name="policy_documents_filtered"
        )
    return vectorstore

# --- 3. RAG 체인 구성 (✨ 2단계 적용) ---
def setup_rag_chain(vectorstore, reranker, llm, response_prompt, query_prompt):
    """사용자 관심사에 따라 동적으로 필터링되는 RAG 체인을 구성합니다."""

    def rerank_documents(inputs):
        query = inputs['question']
        retrieved_docs = inputs['documents']
        unique_docs = list({doc.page_content: doc for doc in retrieved_docs}.values())
        if not unique_docs:
            return []
        pairs = [[query, doc.page_content] for doc in unique_docs]
        scores = reranker.predict(pairs)
        doc_scores = sorted(zip(scores, unique_docs), key=lambda x: x[0], reverse=True)
        return [doc for score, doc in doc_scores[:5]]

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # ✨ [2단계] 사용자 관심 분야(interests)에 따라 동적으로 Retriever를 생성하는 함수
    def get_dynamic_retriever(inputs: dict):
        interests = inputs.get("interests", [])
        search_kwargs = {'k': 20}
        
        # 관심 분야가 설정된 경우, 메타데이터 필터를 추가합니다.
        if interests:
            # Qdrant는 $or 조건을 지원하지 않으므로, should 조건을 사용합니다.
            # 하지만 간단한 구현을 위해 여기서는 첫번째 관심사만 필터링합니다.
            # 고급 필터링은 Qdrant의 필터링 문법을 따라야 합니다.
            # 여기서는 간단하게 'must' 조건으로 첫번째 관심사를 필터링합니다.
            # 실제로는 여러 관심사에 대해 'should' 조건을 구성해야 합니다.
            search_kwargs['filter'] = {
                "must": [{"key": "metadata.policy_type", "match": {"any": interests}}]
            }
            
        base_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
        
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever, llm=llm, prompt=query_prompt
        )
        return multi_query_retriever

    # RAG 체인 구성
    # 1. get_dynamic_retriever를 호출하여 필터링된 retriever를 가져옵니다.
    # 2. 해당 retriever로 문서를 검색합니다.
    # 3. 검색된 문서를 rerank하고 포맷팅하여 최종 답변을 생성합니다.
    rag_chain = (
        {
            "documents": RunnableLambda(lambda inputs: get_dynamic_retriever(inputs).invoke(inputs["question"], config=RunnableConfig(run_name="retrieval"))),
            "question": lambda x: x["question"]
        }
        | RunnableLambda(rerank_documents).with_config(run_name="reranking")
        | {
            "answer": (
                RunnablePassthrough.assign(context=lambda docs: format_docs(docs))
                | response_prompt
                | llm
                | StrOutputParser()
            ),
            "sources": lambda docs: docs
        }
    )
    return rag_chain

# --- 컴포넌트 로드 ---
try:
    llm, reranker_model, query_prompt, response_prompt_template = get_models_and_prompts()
    vectorstore = create_vector_store()
    rag_chain_with_source = setup_rag_chain(vectorstore, reranker_model, llm, response_prompt_template, query_prompt)
except Exception as e:
    st.error(f"RAG 구성 요소 설정 중 오류 발생: {e}")
    st.exception(e)
    st.stop()

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
    if st.button("✅ 조건 저장 및 반영", type="primary", use_container_width=True):
        st.session_state.profile = { "age": age, "interests": interests }
        st.success("맞춤 조건이 저장되었습니다!")

        if interests:
            welcome_message = f"안녕하세요! {age}세, '{', '.join(interests)}' 분야에 관심이 있으시군요. 이제부터 관련 정책 위주로 찾아드릴게요!"
        else:
            welcome_message = f"안녕하세요! {age}세이시군요. 관심 분야를 선택하시면 더 정확한 추천을 받을 수 있어요."
        
        st.session_state.messages.append({"role": "assistant", "content": welcome_message})
        time.sleep(1)
        st.rerun()

# -----------------------
# MAIN UI
# -----------------------
st.title("🤖 청년 정책 큐레이터 v3")
st.caption("관심 분야 필터링 기능이 적용된 맞춤형 탐색기")

recommended_questions_db = {
    "주거": ["전세보증금 이자 지원 정책 알려줘", "역세권 청년주택 신청 자격은?"],
    "일자리/창업": ["취업 준비생인데 면접 정장 빌릴 수 있어?", "서울시에서 인턴십 할 수 있는 프로그램 찾아줘"],
    "금융/자산": ["희망두배 청년통장 가입 조건이 뭐야?", "학자금 대출 이자 지원 사업에 대해 설명해줘"],
    "복지/문화": ["서울시 청년수당 신청 방법 알려줘", "청년들이 문화생활 즐길 수 있게 지원해주는 정책 있어?"]
}

st.markdown("##### 👇 이런 질문은 어떠세요?")
profile_interests = st.session_state.get("profile", {}).get("interests", [])
if profile_interests:
    questions_to_show = recommended_questions_db.get(profile_interests[0], [])
else:
    questions_to_show = ["전세보증금 이자 지원 정책 알려줘", "취업 준비생인데 면접 정장 빌릴 수 있어?"]

cols = st.columns(len(questions_to_show))
for i, question in enumerate(questions_to_show):
    if cols[i].button(question, use_container_width=True, key=f"rec_q_{i}"):
        st.session_state.selected_question = question
        st.rerun()

# -----------------------
# CHAT UI
# -----------------------
if not st.session_state.messages:
    profile = st.session_state.get("profile", {})
    if profile.get("age") and profile.get("interests"):
         welcome_message = f"안녕하세요! {profile['age']}세, '{', '.join(profile['interests'])}' 분야에 관심이 있으시군요. 무엇이든 물어보세요!"
    else:
        welcome_message = "안녕하세요! 어떤 정책이 궁금하신가요? 왼쪽 사이드바에서 맞춤 정보를 설정할 수 있습니다."
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 근거 자료 확인하기"):
                for source in message["sources"]:
                    st.info(f"출처: {source.metadata.get('source', 'N/A')} (페이지: {source.metadata.get('page', 'N/A')}) | 유형: {source.metadata.get('policy_type', 'N/A')}")
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
        with st.spinner("AI가 맞춤 정책 정보를 찾고 있습니다..."):
            try:
                # ✨ [3단계] RAG 체인 호출 시 사용자 관심 분야 전달
                profile_interests = st.session_state.get("profile", {}).get("interests", [])
                
                result = rag_chain_with_source.invoke({
                    "question": prompt,
                    "interests": profile_interests
                })
                
                response = result.get("answer", "오류: 답변을 생성하지 못했습니다.")
                final_docs = result.get("sources", [])

                if not final_docs:
                     response = "죄송합니다. 선택하신 관심 분야에서는 관련 정보를 찾을 수 없습니다. 다른 분야를 선택하시거나 질문을 바꿔보세요."

            except Exception as e:
                response = f"답변 생성 중 오류가 발생했습니다: {e}"
                final_docs = []
                st.error(response)
                st.exception(e)

        st.markdown(response)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": final_docs
        })

    st.rerun()
