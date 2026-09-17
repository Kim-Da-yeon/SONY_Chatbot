# SONY Product Manual RAG Chatbot

제품 설명서 PDF 대상 한국어 검색증강 질의응답. 페이지 단위 출처 표기, 문서 검색 실패 시 웹 검색 대체.

---

## 구성

```
manual PDFs → PyMuPDFLoader → 청크 500/overlap 50 → bge-m3 → Chroma
질문 + 제품 → top-5 검색 → 제품명 필터(cos ≥ 0.7) → KoAlpaca-7b → 답변 + 출처
                                     └ 통과 문서 없음 → Tavily 웹 검색
```

| 단계 | 구성 | 설정 |
|---|---|---|
| 청킹 | RecursiveCharacterTextSplitter | size 500, overlap 50 |
| 임베딩 | BAAI/bge-m3 | normalize |
| 벡터 저장소 | Chroma | 디스크 persist |
| 검색 | similarity | top k=5 |
| 제품 필터 | all-MiniLM-L6-v2 cosine | 임계 0.7 |
| 생성 | KoAlpaca-llama-1-7b | max_new_tokens 200, fp16 |
| 대체 | Tavily search | k=2 |

답변마다 근거 청크의 `file_name`과 `page_number` 반환.

| 파일 | 내용 |
|---|---|
| `build_vectorstore.py` | PDF → 청크 → 임베딩 → Chroma |
| `app.py` | Streamlit UI (제품 선택 + 질문) |
| `query_test.py` | 단일 질의 headless 테스트 |

## 실행

설명서 PDF는 소니 저작물이므로 저장소에 포함하지 않음. [공식 지원 사이트](https://www.sony.com/electronics/support)에서 내려받아 `data/`에 배치.

```bash
pip install -r requirements.txt
export TAVILY_API_KEY="..."                          # https://tavily.com
export SONY_DATA_DIR="data"                          # 기본값
export SONY_VECTORSTORE_DIR="data/vectorstore"
export SONY_MODEL_PATH="beomi/KoAlpaca-llama-1-7b"

python build_vectorstore.py
streamlit run app.py
```

`app.py`의 제품 드롭다운은 하드코딩. 다른 설명서를 색인하면 목록도 수정 필요. 7B 생성 모델은 GPU 권장.

## 한계

- 평가 없음. 검색 품질(recall@k, MRR)·답변 품질 미측정, 테스트셋 없음
- 제품 필터가 한국어 문자열을 영어 중심 모델(all-MiniLM-L6-v2)로 임베딩하고 임계 0.7은 미조정. 이미 로드된 다국어 `bge-m3`를 쓰거나 파일명 직접 매칭이 타당
- 검색 후 필터 순서라 5개 청크 중 통과분이 없으면 정답 설명서가 색인돼 있어도 빈손. Chroma 메타데이터 필터로 검색 단계에 반영하는 편이 나음
- 웹 검색 대체가 `result.get('snippet')`을 읽는데 현행 LangChain `TavilySearchResults`는 `content` 키 반환. 설치 버전 확인 필요

## 참고

bge-m3 (Chen et al. 2024) · [KoAlpaca](https://github.com/Beomi/KoAlpaca) · [Chroma](https://github.com/chroma-core/chroma) · [Tavily](https://tavily.com)


