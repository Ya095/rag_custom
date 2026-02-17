IMAGE_SUMMARY_PROMPT = """Describe the visual content precisely.

    If the content is a diagram:
    - Name visible components.
    - Describe visible connections, arrows, and data flow direction.

    If the content is a chart:
    - Describe axes, labels, units, and legend (if present).
    - Describe visible comparisons (higher/lower), increases/decreases, peaks, and outliers.

    Rules:
    - Use at most 170 tokens.
    - Describe only what is visible.
    - No assumptions, explanations, or conclusions.
    - Do not mention that this is an image.
    - If something is unclear or unreadable, say it is not visible.

    Output:
    One concise factual paragraph."""

TEXT_SUMMARY_PROMPT = """You summarize a technical text fragment.

    Goal:
    - Capture the main idea.
    - Preserve key technical terms, acronyms, identifiers, and numbers.
    - Remove redundancy.

    Constraints:
    - Use at most 160 tokens.
    - Write factual statements only.
    - No introductions or conclusions.
    - Do not mention that this is a summary.
    - If the fragment is incomplete, summarize only what is present.

    Output:
    A concise paragraph with key searchable terms.
    
    Text fragment: {element}"""

TABLE_SUMMARY_PROMPT = """You describe structured data from a technical table.

    Goal:
    - Explain what is being compared or measured.
    - Describe columns (headers, units) and important rows/groups.
    - Highlight key numeric differences and notable values.

    Constraints:
    - Use at most 180 tokens.
    - Do not reproduce table formatting.
    - No interpretation beyond the data.
    - Do not mention that this is a table.
    - If some cells are missing or unreadable, say they are not visible.

    Output:
    One concise factual paragraph.
    
    Technical table: {element}"""

RAG_ANSWER_PROMPT = """You are an assistant for answering questions.
    You need to answer the question using only the context provided and rules below.

    The context may contain special image placeholders in the form: [[IMG:ds123dgh_fds3fg21a]].
    Image placeholders are critical structural elements.

    Strict rules:
    - Image placeholders are part of the document structure.
    - You must copy every [[IMG:ds123dgh_fds3fg21a]] exactly as it appears.
    - You must not remove, modify, reformat, or relocate image placeholders.
    - Preserve the original order of image placeholders exactly as in the context.
    - Never move image placeholders to the end of the answer unless they appear at the end of the context.
    - If an image appears between two paragraphs, it must remain between those paragraphs.
    - Never group image placeholders together unless they are grouped in the context.

    General rules:
    - You are not generating a new summary.
    - You are restructuring the existing context into an answer.
    - Do not include disclaimers like "the context provide...".
    - If the answer cannot be found in the context, say exactly: "I don't know".

    Context:
    {context}

    Question:
    {question}

    Answer:"""
