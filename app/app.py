import asyncio
from io import BytesIO

import streamlit as st

from ingestion.pipeline import ProcessDocumentPDF
from retrieval.process_question import ProcessQuestion

st.set_page_config(page_title='Rag custom', page_icon='📚', layout='centered')

st.title('Upload a PDF file.')
input_file: BytesIO = st.file_uploader('Choose a PDF', type=['pdf'])

if input_file is not None:
    with st.spinner('Processing file...'):
        obj = ProcessDocumentPDF()
        doc_id: str = asyncio.run(obj.process_document(input_file))

    st.success(f"file successfully processed.")
    st.caption("You can upload another PDF if you like.")

st.divider()
st.title("Ask a question about your PDFs")

with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=7, value=4, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Sending event and generating answer..."):
            obj = ProcessQuestion()
            answer: str = obj.process_pipeline_sync(question)

        st.subheader("Answer")
        st.write(answer or "(No answer)")
        # if sources:
        #     st.caption("Sources")
        #     for s in sources:
        #         st.write(f"- {s}")

