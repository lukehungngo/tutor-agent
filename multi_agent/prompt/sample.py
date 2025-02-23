STRUCTURED_ANALYSIS_PROMPT = """Given the following content, provide a detailed structured analysis:

CONTENT TO ANALYZE:
{content}

Analyze this using the following structure:

1. INFORMATION EXTRACTION
- Key Titles and Sources
- URLs and References
- Core Content Elements
- Author Information

2. CLASSIFICATION
- Difficulty Level Assessment
- Relevance Evaluation
- Confidence Scoring
- Source Type Identification

3. ORGANIZATION
- Content Categorization
- Priority Ordering
- Relationship Mapping

4. STRUCTURED OUTPUT
- Title (preserve original)
- URL (if present)
- Content Summary
- Difficulty Level
- Author/Source
- Relevance Score (0-1)
- Confidence Score (0-1)
"""