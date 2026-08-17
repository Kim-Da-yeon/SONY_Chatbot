import os
import glob
import warnings
import torch
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

warnings.filterwarnings("ignore")

# 경로 설정 — 환경변수로 덮어쓸 수 있다 (기본값은 저장소 기준 상대경로)
DATA_DIR = os.environ.get("SONY_DATA_DIR", "data")
VECTORSTORE_DIR = os.environ.get("SONY_VECTORSTORE_DIR", os.path.join(DATA_DIR, "vectorstore"))

# PDF 파일 목록 — DATA_DIR 안의 모든 PDF를 대상으로 한다.
# 설명서 PDF는 저작권상 저장소에 포함하지 않으므로 소니 공식 지원 사이트에서 직접 내려받아 넣을 것.
pdf_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
if not pdf_files:
    raise SystemExit(
        f"{DATA_DIR}/ 에 PDF가 없습니다. "
        "소니 공식 지원 사이트에서 제품 설명서를 내려받아 이 폴더에 넣으세요."
    )

# 여러 PDF 파일을 로드하여 문서 리스트 생성
docs = []
for pdf_file in pdf_files:
    loader = PyMuPDFLoader(pdf_file)
    pages = loader.load()
    for page_num, page in enumerate(pages):
        # 파일명에서 확장자를 제거하여 간단한 형식으로 처리
        base_name = os.path.splitext(os.path.basename(pdf_file))[0]
        # 각 페이지에 파일명과 페이지 번호 메타데이터 추가
        page.metadata["file_name"] = base_name
        page.metadata["page_number"] = page_num + 1
        docs.append(page)

# 메타데이터 확인
for doc in docs:
    print(f"파일명: {doc.metadata['file_name']}, 페이지 번호: {doc.metadata['page_number']}")

# 문서를 문장으로 분리
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
split_docs = text_splitter.split_documents(docs)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 문장을 임베딩으로 변환하고 벡터 저장소에 저장
embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True},
)


# 벡터 저장소 생성 및 저장
vectorstore_path = VECTORSTORE_DIR
os.makedirs(vectorstore_path, exist_ok=True)
vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory=vectorstore_path)
vectorstore.persist()
print("Vectorstore created and persisted")

