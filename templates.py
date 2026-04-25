from langchain_core.prompts import ChatPromptTemplate

CHUNK_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """ You are summarizing a segment of a meeting transcript.
        Extract and retain:
        - Key decisions made
        - Action items or tasks assigned
        - Important discussion points

        Be concise. Output only the summary, no preamble.

        TRANSCRIPT SEGMENT:
        {chunk}

        SUMMARY:"""
)

FINAL_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """ You are an expert meeting summarizer. Below are partial summaries extracted from a meeting transcript.

        Your task is to synthesize these into a single, polished final summary.

        PARTIAL SUMMARIES:
        {combined}

        INSTRUCTIONS:
        - Merge overlapping or repeated information into unified points
        - Preserve all unique decisions, action items, and key discussions
        - Maintain a professional, neutral tone
        - Be concise but comprehensive — do not omit critical details
        - Do NOT include phrases like "this summary covers..." or "the meeting discussed..."
        - Output ONLY the final summary text, no headers, no preamble, no meta-commentary

        FINAL SUMMARY:"""
)

QUERY_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant answering questions about a meeting.

        Use ONLY the context below to answer. Do not add information from outside the context.
        If the answer is not in the context, say: "I don't have enough information from this meeting to answer that."

        Context:
        {context}

        Question:
        {question}

        Answer:"""
)

ORCHESTRATOR_PROMPT = ChatPromptTemplate.from_template(
    """
    You are an intelligent orchestrator synthesizing information from multiple meetings.
    Below are answers from different meeting sub-agents based on the user's query.

    Combine these partial answers into a single, comprehensive, and professional response.
    If none of the meetings contained the answer, explicitly state that none of the meetings provided the necessary information.

    PARTIAL ANSWERS:
    {combined}

    User Question: {query}

    FINAL RESPONSE:"""
)

ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """
    A user asked a question about their meetings.
    Based on the question, identify which meetings are relevant.
    
    Today: {today}
    
    Available meetings:
    {meeting_list}
    
    User question: {query}
    
    Rules:
    - "today" means return meetings from today's date
    - "yesterday" means return meetings from yesterday's date
    - If they mention a meeting name or topic, match it by title
    - If the user provides a general query without any reference to dates, specific names, or topics, return an empty array `[]`. Do NOT assume all meetings.
    - IMPORTANT: Your ONLY output must be a single, valid JSON array of strings containing the EXACT `room_id` values from the available meetings.
    - DO NOT modify or prefix the `room_id` strings in any way.
    - DO NOT write any Python code, scripts, or explanations. 
    - DO NOT wrap your response in markdown code blocks.
    
    Example valid output: ["exact_room_id_here"]"""
)
