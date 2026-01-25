IMAGE_SUMMARY_PROMPT = """Describe the visual content precisely.

    If the content is a diagram:
    - Name visible components.
    - Describe visible connections, arrows, and data flow direction.

    If the content is a chart:
    - Describe axes, labels, units, and legend (if present).
    - Describe visible comparisons (higher/lower), increases/decreases, peaks, and outliers.

    Rules:
    - Use at most 300 tokens.
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
    - Use at most 220 tokens.
    - Write factual statements only.
    - No introductions or conclusions.
    - Do not mention that this is a summary.
    - If the fragment is incomplete, summarize only what is present.

    Output:
    One concise paragraph.
    
    Text fragment: {element}"""

TABLE_SUMMARY_PROMPT = """You describe structured data from a technical table.

    Goal:
    - Explain what is being compared or measured.
    - Describe columns (headers, units) and important rows/groups.
    - Highlight key numeric differences and notable values.

    Constraints:
    - Use at most 250 tokens.
    - Do not reproduce table formatting.
    - No interpretation beyond the data.
    - Do not mention that this is a table.
    - If some cells are missing or unreadable, say they are not visible.

    Output:
    One concise factual paragraph.
    
    Technical table: {element}"""

RAG_ANSWER_PROMPT = """Answer the question ONLY using the provided context.
    Answer the question ONLY using the provided context.
    Do not make any assumptions or invent facts.
    Do not include disclaimers like "the context does not provide..."
    
    Context:
    {context}
    
    Question:
    {question}
    
    Rules:
    - Base your answer strictly on the context.
    - If the answer cannot be found in the context, say "I don't know".
    - Keep technical details if present.
    - Do not include unrelated information."""
