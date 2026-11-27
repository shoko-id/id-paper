import datetime
import json
import requests
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(
    page_title="ID論文トレンド × コーチング壁打ち",
    page_icon="🧭",
    layout="wide",
)

COACH_SYSTEM_PROMPT = """
あなたは、鈴木克明先生や熊本大学教授システム学専攻（GSIS）のインストラクショナルデザイン（学習者中心、ARCSモデル、ID第一原理、入口・出口・構造など）の知見を背景に持つ、熟練の教育コーチです。

【あなたの役割と厳守ルール】
1. いきなり「正解」「改善案」「理論の解説」「ダメ出し」を教えないでください（ティーチング禁止）。
2. 1回の返答は「相手の発言の受け止め・言語化の承認（1〜2文）」＋「相手自身の現場や気づきを引き出すオープンクエスチョン（1つだけ）」に絞ってください。
3. 全体で3〜4文以内の簡潔で温かいトーンを保ってください。
4. 相手が明示的に「IDの視点で整理して」「ヒントやアイデアが欲しい」と求めたときだけ、控えめにIDの観点や論文の示唆を補助線として提案してください。
""".strip()


def reconstruct_abstract(inv_index: dict | None) -> str:
    if not inv_index:
        return ""
    words = {}
    for word, positions in inv_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words[i] for i in sorted(words))


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_openalex_papers(query: str, days: int, max_results: int = 8) -> list[dict]:
    since_date = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "filter": f"has_abstract:true,from_publication_date:{since_date}",
        "sort": "publication_date:desc",
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
                "doi": item.get("doi") or "",
                "abstract": abstract[:1500],
            }
        )
    return papers


@st.cache_data(ttl=3600, show_spinner=False)
def summarize_papers_with_gemini(api_key: str, papers: list[dict]) -> dict:
    client = genai.Client(api_key=api_key)
    papers_text = json.dumps(papers, ensure_ascii=False)
    prompt = f"""
以下はOpenAlexから取得した最新のインストラクショナルデザイン関連論文のリストです。
JSON形式で以下を出力してください。

1. "trends": 全体の傾向から見える「最近の注目トピック」を3つ（各トピック名と1〜2文の説明）
2. "items": 各論文について、日本語タイトル("ja_title")・3行要約("summary")・教育実践への問いかけヒント("hint")

入力論文データ:
{papers_text}
"""
    schema = {
        "type": "OBJECT",
        "properties": {
            "trends": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "topic": {"type": "STRING"},
                        "description": {"type": "STRING"},
                    },
                    "required": ["topic", "description"],
                },
            },
            "items": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "ja_title": {"type": "STRING"},
                        "summary": {"type": "STRING"},
                        "hint": {"type": "STRING"},
                    },
                    "required": ["ja_title", "summary", "hint"],
                },
            },
        },
        "required": ["trends", "items"],
    }

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.3,
        ),
    )
    return json.loads(response.text)


def ask_coach(api_key: str, paper_context: dict, history: list[dict]) -> str:
    client = genai.Client(api_key=api_key)
    system_text = (
        f"{COACH_SYSTEM_PROMPT}\n\n"
        f"【今回壁打ちのきっかけにしている論文】\n"
        f"タイトル: {paper_context['ja_title']} ({paper_context['title']})\n"
        f"要約: {paper_context['summary']}\n"
        f"ヒント: {paper_context['hint']}"
    )
    contents = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part.from_text(text=m["content"])],
        )
        for m in history
    ]
    response = client.models.generate_content(
        model="gemini-3.8-flash",
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
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- サイドバー ---
with st.sidebar:
    st.header("🔍 論文検索設定")
    query = st.text_input("検索キーワード（英語推奨）", value="instructional design")
    period_label = st.radio("対象期間", ["直近6ヶ月", "直近1年"], index=1)
    days = 180 if period_label == "直近6ヶ月" else 365

    fetch_btn = st.button("論文を取得してまとめる", type="primary", use_container_width=True)
    if st.button("壁打ちをリセット", use_container_width=True):
        st.session_state.selected_paper = None
        st.session_state.messages = []
        st.rerun()

    st.caption("キーワード例: `ARCS model`, `self-regulated learning`, `medical education`")

# --- APIキー確認 ---
api_key = st.secrets.get("GEMINI_API_KEY", "")
if not api_key:
    st.warning("Streamlit Cloudの「Settings → Secrets」に `GEMINI_API_KEY` を設定してください。")
    st.stop()

st.title("🧭 ID論文トレンド × コーチング壁打ち")
st.caption("OpenAlexの最新論文からトレンドを掴み、自分の教育実践に引き寄せて壁打ちします。")

if fetch_btn:
    with st.spinner("OpenAlexから論文を取得し、トピックを整理しています..."):
        try:
            papers = fetch_openalex_papers(query=query, days=days, max_results=8)
            if not papers:
                st.warning("該当する論文が見つかりませんでした。キーワードや期間を変えてみてください。")
            else:
                summary_data = summarize_papers_with_gemini(api_key, papers)
                st.session_state.papers = papers
                st.session_state.summary_data = summary_data
        except Exception as e:
            st.error(f"取得または要約中にエラーが発生しました: {e}")

col_left, col_right = st.columns([1.1, 1.0], gap="large")

# --- 左カラム：論文トピックまとめ ---
with col_left:
    st.subheader("📚 最近のトピックと論文")
    if not st.session_state.summary_data:
        st.info("左のサイドバーから「論文を取得してまとめる」を押してください。")
    else:
        trends = st.session_state.summary_data.get("trends", [])
        for t in trends:
            st.markdown(f"**・{t['topic']}**：{t['description']}")

        st.markdown("#### 取得した論文")
        items = st.session_state.summary_data.get("items", [])
        for idx, (paper, item) in enumerate(zip(st.session_state.papers, items)):
            with st.expander(f"{item['ja_title']}（{paper['date']}）"):
                st.caption(f"原題: {paper['title']} / {paper['journal']}")
                st.write(item["summary"])
                if paper["doi"]:
                    st.markdown(f"[論文リンク (DOI)]({paper['doi']})")
                if st.button("この論文で壁打ちする", key=f"coach_{idx}"):
                    merged = {**paper, **item}
                    st.session_state.selected_paper = merged
                    first_q = (
                        f"「{item['ja_title']}」をテーマに壁打ちですね。\n\n"
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
        st.success(f"テーマ：{sp['ja_title']}")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if user_input := st.chat_input("思ったことや現場の状況を書いてみてください..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                with st.spinner("考えています..."):
                    reply = ask_coach(api_key, sp, st.session_state.messages)
                    st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
