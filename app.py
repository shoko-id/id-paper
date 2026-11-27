import datetime
import json
import time
import urllib.parse
import xml.etree.ElementTree as ET
import requests
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(
    page_title="ID論文トレンド × コーチング壁打ち",
    page_icon="🧭",
    layout="wide",
)

MODEL_NAME = "gemini-3.8-flash"

COACH_SYSTEM_PROMPT = """
あなたは、鈴木克明先生や熊本大学教授システム学専攻（GSIS）のインストラクショナルデザイン（学習者中心、ARCSモデル、ID第一原理、入口・出口・構造など）の知見を背景に持つ、熟練の教育コーチです。

【あなたの役割と厳守ルール】
1. いきなり「正解」「改善案」「理論の解説」「ダメ出し」を教えないでください（ティーチング禁止）。
2. 1回の返答は「相手の発言の受け止め・言語化の承認（1〜2文）」＋「相手自身の現場や気づきを引き出すオープンクエスチョン（1つだけ）」に絞ってください。
3. 全体で3〜4文以内の簡潔で温かいトーンを保ってください。
4. 相手が明示的に「IDの視点で整理して」「ヒントやアイデアが欲しい」と求めたときだけ、控えめにIDの観点や論文の示唆を補助線として提案してください。
""".strip()


def get_client(api_key: str) -> genai.Client:
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=20000),
    )


def reconstruct_abstract(inv_index: dict | None) -> str:
    if not inv_index:
        return ""
    words = {}
    for word, positions in inv_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words[i] for i in sorted(words))


def generate_with_retry(client: genai.Client, **kwargs):
    for attempt in range(2):
        try:
            return client.models.generate_content(**kwargs)
        except Exception as e:
            if "503" in str(e) and attempt == 0:
                time.sleep(1.5)
                continue
            raise


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_openalex_papers(query: str, days: int, max_results: int = 5) -> list[dict]:
    today = datetime.date.today()
    since_date = (today - datetime.timedelta(days=days)).isoformat()
    today_str = today.isoformat()
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "filter": (
            f"has_abstract:true,type:article,"
            f"from_publication_date:{since_date},to_publication_date:{today_str},"
            f"title_and_abstract.search:{query}"
        ),
        "sort": "relevance_score:desc",
        "per-page": max_results,
        "select": "id,title,publication_date,doi,primary_location,abstract_inverted_index",
    }
    res = requests.get(url, params=params, timeout=15)
    res.raise_for_status()
    data = res.json()

    papers = []
    for item in data.get("results", []):
        abstract = reconstruct_abstract(item.get("abstract_inverted_index"))
        if not abstract:
            continue
        source = (item.get("primary_location") or {}).get("source") or {}
        papers.append(
            {
                "id": item.get("id", ""),
                "title": item.get("title", "No Title"),
                "date": item.get("publication_date", ""),
                "journal": source.get("display_name", "Unknown Journal"),
                "url": item.get("doi") or "",
                "abstract": abstract[:350],
            }
        )
    return papers


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pubmed_papers(query: str, days: int, max_results: int = 5) -> list[dict]:
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": f"({query}[Title/Abstract])",
        "reldate": days,
        "datetype": "pdat",
        "retmax": max_results,
        "sort": "relevance",
        "retmode": "json",
    }
    s_res = requests.get(search_url, params=search_params, timeout=15)
    s_res.raise_for_status()
    id_list = s_res.json().get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return []

    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "xml",
    }
    f_res = requests.get(fetch_url, params=fetch_params, timeout=15)
    f_res.raise_for_status()

    root = ET.fromstring(f_res.text)
    papers = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//MedlineCitation/PMID", "")
        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else "No Title"
        journal = article.findtext(".//Journal/Title", "PubMed")

        year = article.findtext(".//JournalIssue/PubDate/Year", "")
        month = article.findtext(".//JournalIssue/PubDate/Month", "")
        day = article.findtext(".//JournalIssue/PubDate/Day", "")
        date_str = "-".join([p for p in (year, month, day) if p]) or "日付不明"

        abs_els = article.findall(".//Abstract/AbstractText")
        abstract = " ".join("".join(el.itertext()).strip() for el in abs_els if el is not None)
        if not abstract:
            abstract = "(抄録なし)"

        papers.append(
            {
                "id": pmid,
                "title": title,
                "date": date_str,
                "journal": journal,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                "abstract": abstract[:350],
            }
        )
    return papers


@st.cache_data(ttl=3600, show_spinner=False)
def summarize_papers_with_gemini(api_key: str, papers: list[dict]) -> dict:
    client = get_client(api_key)
    compact_inputs = [
        {"title": p["title"], "abstract": p["abstract"]}
        for p in papers
    ]
    prompt = f"""
以下の論文リストを次のJSON形式のみで超簡潔にまとめてください。余計な説明は不要です。
{{
  "trends": [{{"topic": "注目トピック名", "description": "1文で簡潔に"}}],
  "items": [{{"ja_title": "自然な日本語タイトル", "summary": "要点を1〜2文で簡潔に"}}]
}}
trendsは3つ、itemsは入力論文と同じ順番・同じ件数で出力してください。

論文データ:
{json.dumps(compact_inputs, ensure_ascii=False)}
"""
    response = generate_with_retry(
        client,
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    return json.loads(response.text)


def ask_coach(api_key: str, paper_context: dict, history: list[dict]) -> str:
    client = get_client(api_key)
    system_text = (
        f"{COACH_SYSTEM_PROMPT}\n\n"
        f"【今回壁打ちのきっかけにしている論文】\n"
        f"タイトル: {paper_context.get('ja_title', paper_context['title'])}\n"
        f"要約: {paper_context.get('summary', paper_context['abstract'])}"
    )
    contents = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part.from_text(text=m["content"])],
        )
        for m in history
    ]
    response = generate_with_retry(
        client,
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_text,
            temperature=0.7,
        ),
    )
    return response.text


# --- セッション状態の初期化 ---
for key, default in {
    "papers": [],
    "summary_data": None,
    "selected_paper": None,
    "messages": [],
    "last_source": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- サイドバー ---
with st.sidebar:
    st.header("🔍 論文検索設定")
    db_source = st.radio(
        "検索データベース",
        ["PubMed", "OpenAlex"],
        horizontal=True,
    )
    st.caption(
        "・**PubMed**: 医学・医療教育・臨床指導の論文に特化（ノイズが少なめ）\n"
        "・**OpenAlex**: 教育工学・企業研修・大学教育などID全般を広く網羅\n"
        "・**Google Scholar**: 公式APIがないため下のボタンから直接検索できます"
    )

    query = st.text_input("検索キーワード（英語推奨）", value="medical education")
    period_label = st.radio("対象期間", ["直近6ヶ月", "直近1年"], index=1, horizontal=True)
    days = 180 if period_label == "直近6ヶ月" else 365

    fetch_btn = st.button("論文を取得してまとめる", type="primary", use_container_width=True)

    gs_url = f"https://scholar.google.com/scholar?q={urllib.parse.quote(query)}"
    st.link_button("Google Scholarで同じ語を検索 ↗", gs_url, use_container_width=True)

    if st.button("壁打ちをリセット", use_container_width=True):
        st.session_state.selected_paper = None
        st.session_state.messages = []
        st.rerun()

    st.caption("キーワード例: `instructional design`, `ARCS model`, `medical education`")

# --- APIキー確認 ---
api_key = st.secrets.get("GEMINI_API_KEY", "")
if not api_key:
    st.warning("Streamlit Cloudの「Settings → Secrets」に `GEMINI_API_KEY` を設定してください。")
    st.stop()

st.title("🧭 ID論文トレンド × コーチング壁打ち")
st.caption("最新論文からトレンドを掴み、自分の教育実践に引き寄せて壁打ちします。")

if fetch_btn:
    with st.status(f"{db_source} から論文を取得しています...", expanded=True) as status:
        try:
            if db_source == "PubMed":
                papers = fetch_pubmed_papers(query=query, days=days, max_results=5)
            else:
                papers = fetch_openalex_papers(query=query, days=days, max_results=5)

            st.session_state.papers = papers
            st.session_state.last_source = db_source

            if not papers:
                st.session_state.summary_data = None
                status.update(label="該当する論文が見つかりませんでした", state="complete")
            else:
                st.write(f"{len(papers)}件取得しました。AIで簡潔に整理中...")
                try:
                    summary_data = summarize_papers_with_gemini(api_key, papers)
                    st.session_state.summary_data = summary_data
                    status.update(label="完了しました！", state="complete")
                except Exception as e:
                    st.session_state.summary_data = None
                    status.update(label="論文は取得できましたが、AI要約がタイムアウトしました", state="error")
                    st.warning(f"Gemini側が混雑しているため英語の論文一覧のみ表示します（詳細: {e}）")
        except Exception as e:
            status.update(label="論文の取得でエラーが発生しました", state="error")
            st.error(f"エラー: {e}")

col_left, col_right = st.columns([1.1, 1.0], gap="large")

# --- 左カラム：論文トピックまとめ ---
with col_left:
    source_label = f"（{st.session_state.last_source}）" if st.session_state.last_source else ""
    st.subheader(f"📚 最近のトピックと論文 {source_label}")
    if not st.session_state.papers:
        st.info("左のサイドバーから「論文を取得してまとめる」を押してください。")
    else:
        items = (st.session_state.summary_data or {}).get("items", [])
        trends = (st.session_state.summary_data or {}).get("trends", [])
        for t in trends:
            st.markdown(f"**・{t['topic']}**：{t['description']}")

        st.markdown("#### 取得した論文")
        for idx, paper in enumerate(st.session_state.papers):
            item = items[idx] if idx < len(items) else {}
            display_title = item.get("ja_title") or paper["title"]
            summary_text = item.get("summary") or paper["abstract"]

            with st.expander(f"{display_title}（{paper['date']}）"):
                st.caption(f"原題: {paper['title']} / {paper['journal']}")
                st.write(summary_text)
                if paper["url"]:
                    st.markdown(f"[論文リンク]({paper['url']})")
                if st.button("この論文で壁打ちする", key=f"coach_{idx}"):
                    merged = {**paper, **item}
                    st.session_state.selected_paper = merged
                    first_q = (
                        f"「{display_title}」をテーマに壁打ちですね。\n\n"
                        f"この論文の内容を聞いたとき、ご自身の担当している授業や研修のどんな場面・誰の顔が浮かびましたか？"
                    )
                    st.session_state.messages = [{"role": "assistant", "content": first_q}]
                    st.rerun()

# --- 右カラム：コーチング壁打ち ---
with col_right:
    st.subheader("💬 コーチング壁打ち")
    if not st.session_state.selected_paper:
        st.info("左の論文リストから「この論文で壁打ちする」を押すと、対話が始まります。")
    else:
        sp = st.session_state.selected_paper
        st.success(f"テーマ：{sp.get('ja_title', sp['title'])}")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if user_input := st.chat_input("思ったことや現場の状況を書いてみてください..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                with st.spinner("考えています..."):
                    try:
                        reply = ask_coach(api_key, sp, st.session_state.messages)
                        st.write(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                    except Exception as e:
                        st.error(f"応答の生成に失敗しました（時間を置いて再度お試しください）: {e}")
