"""
Streamlit app for Multi-Document Abstractive Summarization.

Users can upload up to 10 .txt documents and generate abstractive summaries
using the trained MARL-MDS model.

Requirements:
- checkpoints/marl_mds_multinews.pt (trained model)
- If missing, run: python train_multinews.py

Usage:
    streamlit run app.py
"""
import os
import sys
import torch
import streamlit as st

project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
sys.path.insert(0, project_root)

from marl_trainer import MARLMdsTrainer


def load_trainer(checkpoint_dir, specific_checkpoint=None):
    """Load the MARL trainer with the latest checkpoint if available."""
    if specific_checkpoint:
        checkpoint_path = os.path.join(checkpoint_dir, specific_checkpoint)
        if os.path.exists(checkpoint_path):
            st.success(f"Loaded trained checkpoint from {checkpoint_path}")
            return MARLMdsTrainer(checkpoint_path=checkpoint_path), True, checkpoint_path
        else:
            st.warning(f"Specified checkpoint not found: {specific_checkpoint}")
            return MARLMdsTrainer(), False, None

    # Find all checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("marl_mds_multinews") and f.endswith(".pt")]

    if checkpoint_files:
        # Sort by modification time to get the latest
        checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[0])
        st.success(f"Loaded trained checkpoint from {latest_checkpoint}")
        return MARLMdsTrainer(checkpoint_path=latest_checkpoint), True, latest_checkpoint
    else:
        st.warning(f"No trained checkpoint found in {checkpoint_dir}")
        st.info("Falling back to untrained model (extractive summarization)")
        st.info("To use the trained abstractive model, run: `python train_multinews.py`")
        return MARLMdsTrainer(), False, None


def main():
    st.set_page_config(
        page_title="Multi-Document Abstractive Summarization",
        page_icon="📝",
        layout="wide"
    )

    st.title("Multi-Document Abstractive Summarization")
    st.markdown("""
    Upload up to 10 text documents to generate an abstractive summary using the trained MARL-MDS model.
    """)

    # Sidebar for parameters
    st.sidebar.header("Parameters")

    # Summary mode selection
    summary_mode = st.sidebar.radio(
        "Summary Mode",
        ["Abstractive (Trained Model)", "Extractive (Fallback)"],
        help="Abstractive uses the trained T5 model for creative summaries. Extractive selects sentences directly for factual accuracy."
    )

    # Summary lines parameter
    max_summary_lines = st.sidebar.slider(
        "Maximum Summary Lines",
        min_value=1,
        max_value=20,
        value=5,
        help="Maximum number of lines in the generated summary (capped at half of total document lines)"
    )

    # Compression ratio parameter
    compression_ratio = st.sidebar.slider(
        "Compression Ratio",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help="Ratio of sentences to select from the input documents (0.1 = 10%, 0.5 = 50%)"
    )

    # File upload section
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload text documents (up to 10 files)",
        type=["txt"],
        accept_multiple_files=True,
        help="Upload .txt files containing the documents you want to summarize"
    )

    if uploaded_files:
        if len(uploaded_files) > 10:
            st.error("❌ Maximum 10 documents allowed. Please upload fewer files.")
            return

        # Read documents
        documents = []
        for file in uploaded_files:
            content = file.read().decode("utf-8")
            if content.strip():
                documents.append(content.strip())

        if not documents:
            st.error("❌ No valid documents found in uploaded files.")
            return

        st.success(f"Successfully loaded {len(documents)} documents")

        # Show document preview
        with st.expander("Preview Uploaded Documents"):
            for i, doc in enumerate(documents):
                st.text(f"Document {i+1}:")
                st.text(doc[:500] + "..." if len(doc) > 500 else doc)
                st.divider()

        # Calculate actual max summary lines (capped at half of total sentences)
        from marl_trainer import split_sentences
        all_sentences = split_sentences(documents)
        total_sentences = len(all_sentences)
        actual_max_lines = min(max_summary_lines, total_sentences // 2)
        if actual_max_lines < 1:
            actual_max_lines = 1

        st.info(f"Total document sentences: {total_sentences} | Max summary lines: {actual_max_lines}")

        # Generate summary button
        if st.button("Generate Summary", type="primary"):
            with st.spinner("Generating summary... This may take a few minutes."):
                try:
                    # Load trainer with the specific new checkpoint
                    checkpoint_dir = os.path.join(project_root, "checkpoints")
                    specific_checkpoint = "marl_mds_multinews_20260711_205245.pt"
                    trainer, using_trained, checkpoint_path = load_trainer(checkpoint_dir, specific_checkpoint)

                    # Run inference with max_length based on user preference
                    # Convert lines to approximate tokens (roughly 15 tokens per line)
                    max_length_tokens = actual_max_lines * 20

                    # Determine summary mode
                    mode = "extractive" if "Extractive" in summary_mode else "abstractive"

                    loss, metrics = trainer.run_episode(
                        documents,
                        reference_summary=None,
                        compression_ratio=compression_ratio,
                        max_length=max_length_tokens,
                        summary_mode=mode
                    )

                    # Display results
                    st.header("Results")

                    # Model used
                    if using_trained:
                        st.success(f"Used trained abstractive model (T5-base)")
                    else:
                        st.warning(f"Used untrained abstractive model (may hallucinate)")

                    # Summary
                    st.subheader("Generated Summary")
                    summary = metrics["summary"]
                    st.text_area("Summary", summary, height=200)

                    # Selected sentences
                    st.subheader("Selected Sentences")
                    selected_sentences = metrics["selected_sentences"]
                    for i, sentence in enumerate(selected_sentences):
                        st.text(f"{i+1}. {sentence}")

                    # Statistics
                    st.subheader("Statistics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Input Documents", len(documents))
                    col2.metric("Selected Sentences", len(selected_sentences))
                    col3.metric("Summary Lines", len(summary.split('\n')))

                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
                    st.info("Please ensure all dependencies are installed and the checkpoint is available.")

    # Instructions section
    with st.expander("Instructions"):
        st.markdown("""
        ### How to Use
        1. Upload up to 10 text documents (.txt files)
        2. Adjust parameters in the sidebar (optional)
        3. Click "Generate Summary"
        4. View the generated summary and selected sentences

        ### Model Information
        - **Trained Model:** Uses T5-base for abstractive summarization
        - **Fallback:** Extractive summarization if trained model is unavailable
        - **Checkpoint:** `checkpoints/marl_mds_multinews.pt`

        ### Training the Model
        If the trained checkpoint is missing, you can train it yourself:

        ```bash
        conda activate GPU-pytorch
        python train_multinews.py
        ```

        This will train on the XSUM dataset and save the checkpoint to `checkpoints/marl_mds_multinews.pt`.

        ### Requirements
        - Python 3.8+
        - PyTorch with GPU support (recommended)
        - All dependencies in requirements.txt
        """)


if __name__ == "__main__":
    main()
