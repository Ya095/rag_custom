IMAGE_SUMMARY_PROMPT = """Describe the visual content precisely.

    If the content is a diagram:
    - Name components.
    - Describe connections or data flow.

    If the content is a chart:
    - Describe axes and units.
    - Describe main trends or comparisons.

    Rules:
    - Use at most 150 tokens.
    - Describe only what is visible.
    - No assumptions or conclusions.
    - Do not mention that this is an image.

    Output:
    One concise factual paragraph."""

TEXT_SUMMARY_PROMPT = """You summarize a technical text fragment.

    Goal:
    - Capture the main idea.
    - Preserve key technical terms.
    - Remove redundancy.

    Constraints:
    - Use at most 120 tokens.
    - Write factual statements only.
    - No introductions or conclusions.
    - Do not mention that this is a summary.

    Output:
    One concise paragraph.
    
    Text fragment: {element}"""

TABLE_SUMMARY_PROMPT = """You describe structured data from a technical table.

    Goal:
    - Explain what is being compared or measured.
    - Describe columns and important rows.
    - Highlight key numeric patterns or differences.

    Constraints:
    - Use at most 120 tokens.
    - Do not reproduce table formatting.
    - No interpretation beyond the data.
    - Do not mention that this is a table.

    Output:
    One concise factual paragraph.
    
    Technical table: {element}"""
