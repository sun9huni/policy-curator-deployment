import streamlit as st
import time
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
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

# Streamlit Cloud의 Secrets에서 API 키를 안전하게 로드합니다.
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
    """
    RAG 파이프라인의 핵심 구성 요소들을 설정하고 반환합니다.
    이 함수는 앱 실행 시 한 번만 호출되며, 결과는 캐시에 저장됩니다.
    """
    st.sidebar.info("문서를 로드하고 있습니다...")
    if not os.path.exists(DATA_PATH) or not any(f.endswith('.pdf') for f in os.listdir(DATA_PATH)):
        st.error(f"오류: '{DATA_PATH}' 폴더를 찾을 수 없거나 PDF가 없습니다.")
        st.stop()

    documents = []
    for file in os.listdir(DATA_PATH):
        if file.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    st.sidebar.info("데이터베이스를 구축하고 있습니다...")
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = Qdrant.from_documents(
        chunks, embeddings, location=":memory:", collection_name="policy_documents"
    )
    retriever = vectorstore.as_retriever(search_kwargs={'k': 20})
    st.sidebar.success("데이터베이스 구축 완료!")

    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0.1)

    st.sidebar.info("Re-ranker 모델을 로드하고 있습니다...")
    reranker_model = CrossEncoder('bongsoo/kpf-cross-encoder-v1')
    st.sidebar.success("모든 컴포넌트 로드 완료!")

    return retriever, llm, reranker_model

try:
    retriever, llm, reranker_model = get_rag_components()
except Exception as e:
    st.error(f"RAG 구성 요소 설정 중 오류 발생: {e}")
    st.stop()

# -----------------------
# CONVERSATIONAL RAG CHAIN
# -----------------------

# [개선] 대화 맥락을 이해하는 RAG 체인 구성
contextualize_q_system_prompt = """
Given a chat history and the latest user question which might reference context in the chat history, 
formulate a standalone question which can be understood without the chat history. 
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = """당신은 대한민국 정부 정책 전문가입니다. 
사용자의 질문에 대해 아래의 '문서 내용'을 바탕으로, 명확하고 친절하게 답변해주세요.
답변은 항상 한국어로 작성해야 합니다. 문서 내용에 없는 정보는 답변에 포함하지 마세요.
\n\n
[문서 내용]
{context}
"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# -----------------------
# SESSION STATE & SIDEBAR
# -----------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 기록을 LangChain이 이해할 수 있는 형태로 변환
from langchain_core.messages import AIMessage, HumanMessage
def get_chat_history(messages):
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))
    return history

# 대화형 RAG 체인에 메모리 추가
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: get_chat_history(st.session_state.messages),
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

with st.sidebar:
    st.header("🎯 나의 맞춤 조건 설정")
    st.markdown("AI가 더 정확한 정책을 추천하도록 정보를 입력해주세요.")
    age = st.number_input("나이(만)", min_value=18, max_value=100)
    interests = st.multiselect("주요 관심 분야", ['주거', '일자리/창업', '금융/자산', '복지/문화'])

# -----------------------
# MAIN UI & CHAT LOGIC
# -----------------------

st.title("🤖 청년 정책 큐레이터")
st.caption("AI 기반 맞춤형 정책 탐색기")

# 초기 환영 메시지
if not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": "안녕하세요! 어떤 정책이 궁금하신가요?"})

# 이전 대화 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("궁금한 정책에 대해 질문해보세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # [개선] 스트리밍 응답을 위한 UI 준비
            response_placeholder = st.empty()
            full_response = ""

            # [개선] 스트리밍 방식으로 체인 실행
            stream_handler = conversational_rag_chain.stream(
                {"input": prompt},
                {"configurable": {"session_id": "any_session_id"}},
            )

            for chunk in stream_handler:
                if "answer" in chunk:
                    full_response += chunk["answer"]
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
            # 전체 응답이 완료된 후, 대화 기록에 저장
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            error_message = f"답변 생성 중 오류가 발생했습니다: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
