import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sentence_transformers import SentenceTransformer, util
from pytrends.request import TrendReq
import datetime
import requests
import json

# ===== 폰트 설정 =====
def set_korean_font():
    font_path = os.path.join(os.getcwd(), "NanumGothic-Regular.ttf")  # 프로젝트 폴더 내 폰트 파일
    if os.path.exists(font_path):
        # Matplotlib 폰트 등록
        fm.fontManager.addfont(font_path)
        plt.rc("font", family="NanumGothic")

        # ✅ 마이너스 기호 깨짐 방지 설정
        plt.rcParams['axes.unicode_minus'] = False

        # Streamlit HTML 마크다운으로 한글 폰트 적용
        st.markdown(
            f"""
            <style>
            @font-face {{
                font-family: 'NanumGothic';
                src: url('file://{font_path}') format('truetype');
            }}
            html, body, [class*="css"] {{
                font-family: 'NanumGothic';
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("❗ NanumGothic-Regular.ttf 파일이 프로젝트 경로에 존재하지 않습니다.")

# ===== 데이터 로딩 및 전처리 =====
@st.cache_data
def load_and_preprocess_data(uploaded_file, code_name_file):
    import os
    filename = uploaded_file.name
    _, ext = os.path.splitext(filename)

    # 매매기록 데이터 불러오기
    if ext == ".csv":
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    elif ext == ".xlsx":
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        raise ValueError("CSV 또는 XLSX 형식만 지원됩니다.")

    # 날짜 전처리
    df["매수일"] = pd.to_datetime(df["매수일"])
    df["매도일"] = pd.to_datetime(df["매도일"])
    df["보유일수"] = (df["매도일"] - df["매수일"]).dt.days

    # ✅ 종목코드 → 종목명 매핑
    _, ext2 = os.path.splitext(code_name_file.name)
    if ext2 == ".csv":
        code_df = pd.read_csv(code_name_file, dtype={"종목코드": str})
    elif ext2 == ".xlsx":
        code_df = pd.read_excel(code_name_file, dtype={"종목코드": str})
    else:
        raise ValueError("종목코드 매핑 파일은 CSV 또는 XLSX만 지원됩니다.")

    df["종목코드"] = df["종목코드"].astype(str)
    code_df["종목코드"] = code_df["종목코드"].astype(str)

    df = df.merge(code_df, on="종목코드", how="left")

    # ✅ 문장 생성용 컬럼 추가 (종목명 포함)
    df["문장"] = df.apply(
        lambda row: f"{row['종목명']}의 RSI는 {row['RSI']}이며, 수익률은 {row['수익률']}%입니다.",
        axis=1,
    )

    return df
    
# ===== Top5 종목 시각화 =====
import streamlit as st
import matplotlib.pyplot as plt

def visualize_top5_bar(df, start_date, end_date):
    df["매수일"] = pd.to_datetime(df["매수일"], errors='coerce')
    sub_df = df[(df["매수일"] >= start_date) & (df["매수일"] <= end_date)]

    top5 = sub_df["종목명"].value_counts().nlargest(5)

    st.subheader("📈 분석기간 Top 5 매매 종목")
    fig, ax = plt.subplots()
    top5.plot(kind="bar", ax=ax)

    ax.set_title("Top 5 종목별 매매 횟수")
    ax.set_xlabel("종목명")
    ax.set_ylabel("매매 횟수")
    ax.tick_params(axis='x', labelrotation=45)

    st.pyplot(fig)
    return top5.index.tolist()

# ===== Clova 요약 =====
import requests

def clova_analysis(prompt: str) -> str:
    api_key = "nv-b21936503dc049a488669d28299ad294s0i2"
    invoke_url = "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-005"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "messages": [
            {"role": "system", "content": "당신은 투자 분석 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
        "maxTokens": 800,
        "temperature": 0.7,
        "topP": 0.9
    }

    try:
        res = requests.post(invoke_url, headers=headers, json=payload)
        if res.status_code == 200:
            return res.json()['result']['message']['content']
        else:
            return f"❌ Clova 오류: {res.status_code} - {res.text}"
    except Exception as e:
        return f"❌ Clova 예외 발생: {str(e)}"
        
# ===== Pytrends 키워드 검색 =====
def plot_keyword_trend(keyword):
    pytrends = TrendReq(hl="ko", tz=540)
    today = datetime.date.today()
    past_date = today - datetime.timedelta(days=90)
    pytrends.build_payload([keyword], cat=0, timeframe=f"{past_date} {today}", geo="KR")
    df = pytrends.interest_over_time()
    if not df.empty:
        st.subheader("🔍 키워드 검색량 추이")
        st.line_chart(df[keyword])
    else:
        st.warning("검색 데이터가 충분하지 않습니다.")

# ===== RAG 기반 유사 문장 검색 =====
def search_similar_sentences(query, corpus, top_k=3):
    model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    query_embedding = model.encode(query, convert_to_tensor=True)
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]
    return [corpus[hit["corpus_id"]] for hit in hits]

set_korean_font()

# ===== main 함수 =====
def main():
    st.title("📊 투자 복기 분석 시스템")

    uploaded_file = st.file_uploader("📁 매매기록 CSV/XLSX 파일", type=["csv", "xlsx"])
    code_name_file = st.file_uploader("📄 종목코드-이름 매핑 파일", type=["csv", "xlsx"])

    if uploaded_file and code_name_file:
        df = load_and_preprocess_data(uploaded_file, code_name_file)

        # "매수일" 기준 날짜 범위
        start_date = st.date_input("분석 시작일", df["매수일"].min().date())
        end_date = st.date_input("분석 종료일", df["매수일"].max().date())

        keyword = st.text_input("📌 분석 키워드 입력")
        top5 = visualize_top5_bar(df, pd.to_datetime(start_date), pd.to_datetime(end_date))


        # 문장 생성 (Clova 요약과 RAG 검색용)
        df["문장"] = df.apply(
            lambda row: f"{row['매수일'].strftime('%Y-%m-%d')}에 {row['종목명']} 종목을 매수하여 "
                        f"{row['매도일'].strftime('%Y-%m-%d')}에 매도, 수익률 {row['수익률']}%, "
                        f"RSI {row['RSI']}, MA5: {row['MA5']}, 보유일수: {row['보유일수']}일",
            axis=1
        )

        # ✅ RAG 기반 유사 문장 검색
        st.subheader("🗃 RAG 기반 유사 문장 검색")
        user_query = st.text_input("🔍 검색할 문장을 입력하세요")
        if user_query:
            corpus = df["문장"].tolist()
            results = search_similar_sentences(user_query, corpus)
            for idx, r in enumerate(results):
                st.write(f"{idx+1}. {r}")

        # ✅ 키워드 트렌드 시각화
        if keyword:
            plot_keyword_trend(keyword)

        # ✅ Clova 요약 분석
        st.subheader("🧠 Clova 요약 분석")
        if st.button("Clova로 분석 시작"):
            # 통계 요약
            rsi_overheated = df[df["RSI"] >= 70].shape[0]
            short_term = df[df["보유일수"] <= 3].shape[0]
            loss_cut = df[(df["수익률"] <= -3) & (df["보유일수"] <= 5)].shape[0]
            summary = (
                f"📌 **요약 통계**\n"
                f"- RSI 70 이상 매수: **{rsi_overheated}회**\n"
                f"- 3일 이하 단기 매매: **{short_term}회**\n"
                f"- -3% 이상 손실 + 5일 이하 보유 매도: **{loss_cut}회**\n\n"
            )

            combined_text = "\n".join(df["문장"].tolist())

            prompt = (
                f"{summary}"
                f"📌 아래 매매기록에 나오는 종목들에 대해 분석해주세요:\n\n"
                f"{combined_text}\n\n"
                f"🧠 이 기록들을 바탕으로 다음을 분석해주세요:\n"
                f"1. 반복된 손실 원인\n"
                f"2. 보완할 점\n"
                f"3. 수익 전략의 공통점\n"
                f"간결하고 실용적인 투자 조언을 중심으로 답해주세요."
            )

            result = clova_analysis(prompt)
            st.markdown("#### ✅ 분석 결과")
            st.markdown(result)

if __name__ == "__main__":
    main()