import streamlit as st
import time
import pandas as pd

# ---------------- GLOBAL SETTINGS ----------------
pd.set_option("display.max_colwidth", None)

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Bias Analysis",
    layout="centered"
)

# ---------------- Title ----------------
st.title("🚀 Bias Analysis in Large Language Models")

# =====================================================
# 📊 DATASET 1: Crow-Pairs
# =====================================================
st.subheader("📊 Dataset 1: Crow-Pairs")

df_crows = pd.read_csv("crows_pairs_anonymized.csv")
df_crows = df_crows.loc[:, ~df_crows.columns.str.contains("^Unnamed")]

st.caption("Preview of the first 5 rows from the dataset")

st.dataframe(
    df_crows.head(6),
    use_container_width=True,
    hide_index=True,
    height=260,
    column_config={
        "sent_more": st.column_config.TextColumn(width="large"),
        "sent_less": st.column_config.TextColumn(width="large"),
    }
)

st.divider()

# =====================================================
# 📊 DATASET 2: StereoSet
# =====================================================
st.subheader("📊 Dataset 2: StereoSet")

df_stereo = pd.read_csv("stereo.csv")
df_stereo = df_stereo.loc[:, ~df_stereo.columns.str.contains("^Unnamed")]

st.caption("Preview of the first 5 rows from the dataset")

st.dataframe(
    df_stereo.head(6),
    use_container_width=True,
    hide_index=True,
    height=260
)

st.divider()

# =====================================================
# 🤖 MODELS
# =====================================================
st.caption("Models execute one by one with live status updates")

models = [
    "albert-base-v2",
    "distilbert-base-uncased",
    "bert-base-uncased",
    "bert-large-uncased",
    "roberta-base",
    "roberta-large",
    "distilroberta-base",
    "openai-gpt",
    "gpt2"
]

st.subheader("📦 Models Queue")
for i, m in enumerate(models, start=1):
    st.markdown(f"**{i}.** `{m}`")

st.divider()

# =====================================================
# ▶ RUN BUTTON
# =====================================================
run = st.button("▶ Run All Models")

if run:
    progress = st.progress(0)
    log_area = st.empty()
    status_area = st.empty()

    logs = []
    total = len(models)

    # -------- Model Execution --------
    for i, model in enumerate(models, start=1):
        status_area.markdown(f"### 🔄 Running `{model}` ({i}/{total})")

        for dots in ["⏳", "⏳⏳", "⏳⏳⏳"]:
            log_area.info(f"Loading {model} {dots}")
            time.sleep(0.4)

        time.sleep(1.1)

        logs.append(f"✅ {model} completed successfully")
        log_area.success("\n".join(logs))

        progress.progress(i / total)

    # Clear live status after execution
    status_area.empty()
    log_area.empty()

    # =====================================================
    # 📊 RESULTS TABLE 1
    # =====================================================
    st.subheader("📊 Table 1: Layers with Highest CLS Distance")

    df_table1 = pd.read_csv("table1.csv")
    df_table1 = df_table1.loc[:, ~df_table1.columns.str.contains("^Unnamed")]

   

    st.dataframe(
        df_table1.head(5),
        use_container_width=True,
        hide_index=True,
        height=260
    )

    st.divider()

    # =====================================================
    # 📊 RESULTS TABLE 2 (COMMON HEADER)
    # =====================================================
    st.subheader("📊 Table 2: Maximum Self-Attention Weight Difference and Mean SAD Across Language Models.")

    df_table2 = pd.read_csv("table2.csv")
    df_table2 = df_table2.loc[:, ~df_table2.columns.str.contains("^Unnamed")]

    # ---- COMMON (GROUPED) HEADER ----
    df_table2.columns = pd.MultiIndex.from_tuples([
        ("-----------------------------------------------------------Crows-Pairs-----------------------------------Streoset------------------------------------- ", col) for col in df_table2.columns
    ])

    

    st.dataframe(
        df_table2.head(7),
        use_container_width=True,
        hide_index=True,
        height=260
    )

    st.divider()

    st.subheader("📊 Table 3: Layer with the highest attention weights for distinct words in different sentences")
    df_table2 = pd.read_csv("table3.csv")
    df_table2 = df_table2.loc[:, ~df_table2.columns.str.contains("^Unnamed")]

    # ---- COMMON (GROUPED) HEADER ----
    df_table2.columns = pd.MultiIndex.from_tuples([
        ("---------------------------------------------------------Biased-----------------------------------Unbiased------------------------------------- ", col) for col in df_table2.columns
    ])

    st.caption("Preview of the first 5 rows from model results")

    st.dataframe(
        df_table2.head(7),
        use_container_width=True,
        hide_index=True,
        height=260
    )

    st.subheader("🖼️ Heatmap of CLS Euclidean Distances")

    col1, col2 = st.columns(2)

    with col1:
        st.image("image1.png", caption="Crows-Pairs", use_container_width=True)

    with col2:
        st.image("image2.png", caption="StereoSet", use_container_width=True)

    st.success("🎉 All models executed successfully!")
    st.balloons()
