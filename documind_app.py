
import streamlit as st
import tempfile
from pathlib import Path
from core.search_engine import (
    load_document_chunks,
    build_bm25_index,
    load_semantic_model,
    build_semantic_index,
    bm25_search,
    semantic_search,
    hybrid_search
)

from core.database import (
    initialize_database,
    get_all_documents,
    save_processed_document
)

from core.document_processor import process_document
# DATABASE
initialize_database()

@st.cache_resource(show_spinner=False)
def get_search_resources():

    df = load_document_chunks()

    if df.empty:
        return df, None, None, None

    texts = df["text"].tolist()

    bm25 = build_bm25_index(texts)

    model = load_semantic_model()

    embeddings = build_semantic_index(texts, model)

    return df, bm25, model, embeddings

# PAGE CONFIGURATION
st.set_page_config(
    page_title="DocSearch | Intelligent Document Search",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)
# PROFESSIONAL UI DESIGN
st.markdown("""
<style>
.stApp {
    background: #F5F7FB;
    color: #172B4D;
}
.block-container {
    max-width: 1350px;
    padding: 2rem 2.5rem 3rem;
}
[data-testid="stSidebar"] {
    background: #14243D;
    border-right: 1px solid #263B59;
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] hr {
    border-color: #344761;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 0.35rem 0;
}
/* Clickable workflow navigation */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    min-height: 43px;
    text-align: left;
    justify-content: flex-start;
    background: transparent;
    color: #DCE6F2 !important;
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 0.55rem 0.75rem;
    margin: 3px 0;
    font-weight: 600;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #203B5B;
    border-color: #3D7890;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #167D91 !important;
    border: 1px solid #39A6B7 !important;
    color: #FFFFFF !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.16);
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: #116477 !important;
}
h1, h2, h3 {
    color: #172B4D !important;
    font-weight: 700 !important;
}
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E3EAF2;
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0 4px 15px rgba(20, 36, 61, 0.05);
}
[data-testid="stMetricLabel"] {
    color: #64748B !important;
}
[data-testid="stMetricValue"] {
    color: #14243D !important;
    font-weight: 750;
}
.stButton > button {
    background: #167D91;
    color: #FFFFFF;
    border: 1px solid #167D91;
    border-radius: 9px;
    padding: 0.6rem 1.2rem;
    font-weight: 650;
    transition: 0.2s ease;
}
.stButton > button:hover {
    background: #116477;
    color: #FFFFFF;
    border-color: #116477;
}
.stTextInput input, .stSelectbox, .stSlider {
    border-radius: 9px;
}
[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #A9B9CC;
    border-radius: 12px;
    padding: 14px;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid #E4EAF2;
    border-radius: 13px;
}
[data-testid="stAlert"] {
    border-radius: 10px;
}
hr {
    border-color: #E5EAF2;
}
.workflow-side-step {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 11px 12px;
    margin: 5px 0;
    border-radius: 9px;
    font-size: 14px;
    border: 1px solid transparent;
}
.workflow-side-step.active {
    background: #23415F;
    border-color: #3D7890;
    font-weight: 700;
}
.workflow-side-step.completed {
    color: #B8F2D8 !important;
}
.workflow-side-step.pending {
    opacity: 0.78;
}
.workflow-side-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 23px;
    height: 23px;
    border-radius: 50%;
    background: rgba(255,255,255,0.13);
    font-size: 12px;
    font-weight: 800;
    flex-shrink: 0;
}
.workflow-side-step.completed .workflow-side-icon {
    background: #167D68;
    color: #FFFFFF !important;
}
.workflow-side-step.active .workflow-side-icon {
    background: #167D91;
    color: #FFFFFF !important;
}
.workflow-progress-label {
    color: #64748B;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
div[data-testid="stProgressBar"] > div > div {
    background-color: #167D91;
}
.workflow-navigation {
    margin-top: 2.5rem;
    padding-top: 1.2rem;
    border-top: 1px solid #E5EAF2;
}
.workflow-navigation .stButton > button {
    min-height: 44px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)


# GUIDED WORKFLOW STATE
WORKFLOW_STEPS = [
    "Dashboard",
    "Upload Documents",
    "Search Documents",
    "Document Library",
]

if "workflow_step" not in st.session_state:
    st.session_state.workflow_step = 0
if "completed_steps" not in st.session_state:
    st.session_state.completed_steps = set()

# Keep the state safe if the app is rerun after a code update.
st.session_state.workflow_step = max(
    0, min(int(st.session_state.workflow_step), len(WORKFLOW_STEPS) - 1)
)
current_step = st.session_state.workflow_step
page = WORKFLOW_STEPS[current_step]

# SIDEBAR — clickable workflow navigation with a distinct active state.
with st.sidebar:
    st.markdown("# 📄 DocSearch")
    st.caption("INTELLIGENT DOCUMENT PLATFORM")
    st.divider()
    st.markdown("**WORKFLOW · CLICK ANY STEP**")

    for step_index, step_name in enumerate(WORKFLOW_STEPS):
        if step_index in st.session_state.completed_steps:
            icon = "✓"
            status = "Completed"
        elif step_index == current_step:
            icon = "●"
            status = "Current step"
        else:
            icon = str(step_index + 1)
            status = "Not visited"

        button_type = "primary" if step_index == current_step else "secondary"
        if st.button(
            f"{icon}   {step_name}",
            key=f"sidebar_workflow_step_{step_index}",
            type=button_type,
            use_container_width=True,
            help=f"Open {step_name} · {status}",
        ):
            st.session_state.workflow_step = step_index
            st.rerun()

    st.divider()
    st.caption("Smart Document Search & Retrieval")
    st.caption("Version 1.0")


# MAIN HEADER
st.markdown("""
<h1 style="margin-bottom: 5px;">DocSearch</h1>

<p style="color: #64748B; font-size: 16px;">
Intelligent Document Search & Retrieval System
</p>
""", unsafe_allow_html=True)

st.divider()

# WORKFLOW PROGRESS
progress_col, count_col = st.columns([5, 1])
with progress_col:
    st.markdown(
        f"<div class='workflow-progress-label'>Step {current_step + 1} of "
        f"{len(WORKFLOW_STEPS)} · {page}</div>",
        unsafe_allow_html=True,
    )
    st.progress((current_step + 1) / len(WORKFLOW_STEPS))
with count_col:
    st.markdown(
        f"<div style='text-align:right;color:#64748B;font-weight:700;"
        f"padding-top:5px;'>{current_step + 1}/{len(WORKFLOW_STEPS)}</div>",
        unsafe_allow_html=True,
    )


# DASHBOARD
if page == "Dashboard":

    st.subheader("Dashboard Overview")

    documents = get_all_documents()

    total_documents = len(documents)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Documents", total_documents)

    with col2:
        st.metric("Search Methods", 3)

    with col3:
        st.metric("Platform Status", "Active")

    st.divider()

    st.subheader("Search Capabilities")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### BM25 Search")
        st.write("Keyword-based document retrieval.")

    with col2:
        st.markdown("### Semantic Search")
        st.write("Meaning-based intelligent retrieval.")

    with col3:
        st.markdown("### Hybrid Search")
        st.write("Combines keyword and semantic search.")


# UPLOAD DOCUMENTS
elif page == "Upload Documents":

    st.subheader("Upload Documents")

    st.write(
        "Upload your documents to make their content searchable."
    )

    st.markdown("### Select Files")

    uploaded_files = st.file_uploader(
        "Choose documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX and TXT"
    )

    if uploaded_files:

        st.markdown("### Selected Documents")

        for file in uploaded_files:
            st.write(f"📄 {file.name}")

        st.info(f"{len(uploaded_files)} document(s) selected.")

        if st.button("Upload Documents"):

            for uploaded_file in uploaded_files:

                try:
                    with tempfile.TemporaryDirectory() as temp_dir:

                        temp_path = (
                            Path(temp_dir) / Path(uploaded_file.name).name
                        )

                        temp_path.write_bytes(uploaded_file.getvalue())

                        chunks = process_document(temp_path)

                        if not chunks:
                            st.warning(
                                f"No text extracted from {uploaded_file.name}"
                            )
                            continue

                        document_id = save_processed_document(
                            temp_path,
                            chunks
                        )

                        get_search_resources.clear()

                    st.success(
                        f"{uploaded_file.name} uploaded successfully! "
                        f"Document ID: {document_id} | "
                        f"Chunks: {len(chunks)}"
                    )

                except Exception as e:

                    st.error(
                        f"Error uploading {uploaded_file.name}: {e}"
                    )

    else:

        st.info("Please select PDF, DOCX or TXT files to upload.")


# SEARCH DOCUMENTS
elif page == "Search Documents":

    st.subheader("Search Documents")
    st.write(
        "Discover relevant information across your uploaded documents."
    )
    st.divider()

    # Use document metadata for filter options without loading the model.
    stored_documents = get_all_documents()

    if not stored_documents:
        st.info("Upload documents first to start searching.")
    else:
        search_query = st.text_input(
            "Search query",
            placeholder="Example: What is deep learning?",
            help="Enter a question, phrase, or keywords to find relevant document sections."
        )

        control_col1, control_col2 = st.columns([2, 1])

        with control_col1:
            search_method = st.selectbox(
                "Search method",
                ["Hybrid Search", "Semantic Search", "BM25 Search"]
            )

        with control_col2:
            top_k = st.selectbox(
                "Results to show",
                [5, 10, 15, 20],
                index=0
            )

        filter_col1, filter_col2 = st.columns(2)

        available_types = sorted({
            str(document[2]).lower() for document in stored_documents
            if document[2]
        })
        document_names = sorted({
            str(document[1]) for document in stored_documents
            if document[1]
        })

        with filter_col1:
            selected_file_type = st.selectbox(
                "Filter by file type",
                ["All types"] + available_types
            )

        with filter_col2:
            selected_document = st.selectbox(
                "Filter by document",
                ["All documents"] + document_names
            )

        search_clicked = st.button(
            "Search documents",
            use_container_width=True,
            type="primary"
        )

        if search_clicked:
            if not search_query.strip():
                st.warning("Please enter a search query.")
            else:
                try:
                    with st.spinner("Preparing search and finding relevant passages..."):
                        df, bm25, model, embeddings = get_search_resources()

                        if df.empty:
                            st.warning("No indexed document content is available.")
                        else:
                            # Keep the original row positions so cached embeddings
                            # remain aligned with the selected document subset.
                            mask = [True] * len(df)

                            if selected_file_type != "All types":
                                mask = [
                                    keep and str(file_type).lower() == selected_file_type
                                    for keep, file_type in zip(mask, df["file_type"])
                                ]

                            if selected_document != "All documents":
                                mask = [
                                    keep and str(name) == selected_document
                                    for keep, name in zip(mask, df["document_name"])
                                ]

                            selected_positions = [
                                position for position, keep in enumerate(mask) if keep
                            ]

                            if not selected_positions:
                                st.info("No indexed content matches the selected filters.")
                            else:
                                filtered_df = df.iloc[selected_positions].reset_index(drop=True)
                                filtered_embeddings = embeddings[selected_positions]

                                # Rebuild the lightweight BM25 index only for the
                                # filtered subset, keeping all cached embeddings.
                                filtered_bm25 = build_bm25_index(
                                    filtered_df["text"].tolist()
                                )

                                if search_method == "BM25 Search":
                                    results = bm25_search(
                                        search_query, filtered_df,
                                        filtered_bm25, top_k
                                    )
                                elif search_method == "Semantic Search":
                                    results = semantic_search(
                                        search_query, filtered_df,
                                        model, filtered_embeddings, top_k
                                    )
                                else:
                                    results = hybrid_search(
                                        search_query, filtered_df,
                                        filtered_bm25, model,
                                        filtered_embeddings, top_k
                                    )

                                st.divider()
                                st.subheader("Search Results")
                                st.caption(
                                    f"{len(results)} result(s) · {search_method} · "
                                    f"{len(filtered_df)} searchable chunks in the selected scope"
                                )

                                if results.empty:
                                    st.info("No relevant results found for this query.")
                                else:
                                    for index, row in results.iterrows():
                                        with st.container(border=True):
                                            title_col, type_col = st.columns([5, 1])

                                            with title_col:
                                                st.markdown(
                                                    f"**{index + 1}. {row['document_name']}**"
                                                )

                                            with type_col:
                                                st.markdown(
                                                    f"<div style='background:#E0F2F1;"
                                                    f"color:#087E75;padding:5px 10px;"
                                                    f"border-radius:18px;text-align:center;"
                                                    f"font-size:12px;font-weight:700;'>"
                                                    f"{str(row['file_type']).upper()}</div>",
                                                    unsafe_allow_html=True
                                                )

                                            page_value = (
                                                row["page_number"]
                                                if row["page_number"] is not None
                                                else "N/A"
                                            )
                                            st.caption(
                                                f"Page: {page_value}  ·  "
                                                f"Chunk: {row['chunk_number']}"
                                            )
                                            st.markdown(
                                                f"**Search score:** `{float(row['score']):.4f}`"
                                            )

                                            with st.expander("View matching content"):
                                                st.write(row["text"])

                except Exception as e:
                    st.error(f"Search failed: {e}")

# DOCUMENT LIBRARY
elif page == "Document Library":

    st.subheader("Document Library")

    documents = get_all_documents()

    if documents:

        for document in documents:

            with st.container(border=True):

                col1, col2 = st.columns([4, 1])

                with col1:
                    st.markdown(f"**{document[1]}**")
                    st.caption(
                        f"File Type: {document[2]} | ID: {document[0]}"
                    )

                with col2:
                    st.success("Stored")

    else:

        st.info("No documents available.")


# WORKFLOW NAVIGATION — shared footer for all four steps.
st.markdown("<div class='workflow-navigation'></div>", unsafe_allow_html=True)
back_col, spacer_col, next_col = st.columns([1, 4, 1])

with back_col:
    if current_step > 0:
        if st.button("← Back", key=f"workflow_back_{current_step}", use_container_width=True):
            st.session_state.workflow_step = current_step - 1
            st.rerun()

with next_col:
    if current_step < len(WORKFLOW_STEPS) - 1:
        if st.button("Next →", key=f"workflow_next_{current_step}", use_container_width=True, type="primary"):
            st.session_state.completed_steps.add(current_step)
            st.session_state.workflow_step = current_step + 1
            st.rerun()
    else:
        if st.button("Finish ✓", key="workflow_finish", use_container_width=True, type="primary"):
            st.session_state.completed_steps = set()
            st.session_state.workflow_step = 0
            st.rerun()

