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
        # TODO переписать на sync адаптер
        doc_id: str = asyncio.run(obj.process_document(input_file))

    st.success('file successfully processed.')
    st.caption('You can upload another PDF if you like.')

st.divider()
st.title('Enter your question.')

with st.form('rag_query_form'):
    question = st.text_input('Your question')
    submitted = st.form_submit_button('Ask')

    if submitted and question.strip():
        with st.spinner('Generating answer...'):
            result: dict = ProcessQuestion().process_pipeline_sync(question)
            answer: str = result['llm_answer']
            sources: list[dict] = result['sources']

        st.subheader('Answer', text_alignment='center')
        st.markdown(answer, unsafe_allow_html=True)

        if sources:
            st.caption('Sources', text_alignment='center')
            for num, data in enumerate(sources, start=1):
                st.write(f'{num}) {data["document_name"]}, page = {data["page"]}')
