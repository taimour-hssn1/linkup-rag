import json
from concurrent.futures import ThreadPoolExecutor

from templates import QUERY_SUMMARY_PROMPT, ORCHESTRATOR_PROMPT, ROUTER_PROMPT
from models import QueryRequest

from database import index
from llm_config import embeddings_model, groq_chat, output_parser
from db_service import get_all_meetings_for_user, get_today

def query_summary(req: QueryRequest):
    # 1. Embed query
    query_embedding = embeddings_model.embed_query(req.query)

    # 2. Query pinecone directly using the same 'index' object
    res = index.query(
        vector=query_embedding,
        top_k=3,
        filter={"room_id": req.room_id},
        include_metadata=True
    )

    # 3. Extract the text chunks from the response metadata
    context_chunks = [match["metadata"]["chunk_text"] for match in res["matches"] if "chunk_text" in match["metadata"]]
    context = "\n\n".join(context_chunks)
    print("Retrieved context:\n", context)

    # 4. Query LLM
    chain = QUERY_SUMMARY_PROMPT | groq_chat | output_parser
    response = chain.invoke({"context": context, "question": req.query})
    print("LLM Response:\n", response)

    return {
        "response": response
    }   

def run_parallel_subagents(room_ids: list, query: str) -> dict:
    with ThreadPoolExecutor() as executor:
        futures = {
            room_id: executor.submit(query_summary, QueryRequest(room_id=room_id, query=query))
            for room_id in room_ids
        }
        return {
            room_id: future.result()
            for room_id, future in futures.items()
        }

def orchestrate(results: dict, query: str) -> str:
    """
    Synthesizes multiple sub-agent answers into a single coherent response.
    """
    combined_context = []
    for room_id, response in results.items():
        if isinstance(response, dict) and "response" in response:
            response_text = response["response"]
        else:
            response_text = str(response)
        combined_context.append(f"Meeting (Room {room_id}):\n{response_text}\n")
    
    combined_text = "\n".join(combined_context)

    chain = ORCHESTRATOR_PROMPT | groq_chat | output_parser
    final_response = chain.invoke({"combined": combined_text, "query": query})
    return final_response

def smart_router(query: str, user_id: str) -> list:
    """
    Determines which room_ids are relevant based on the user's query and their available meetings.
    """
    all_meetings = get_all_meetings_for_user(user_id)
    print(all_meetings)
    if not all_meetings:
        return []

    meeting_list = "\n".join([
        f"- room_id: {m['room_id']}, title: {m['title']}, date: {m['date']}, time: {m['time']}"
        for m in all_meetings
    ])
    print(meeting_list)
    chain = ROUTER_PROMPT | groq_chat | output_parser
    result = chain.invoke({
        "query": query,
        "meeting_list": meeting_list,
        "today": get_today()
    })
    print("Result:", result)
    try:
        import re
        # Find all JSON arrays of strings in the output to bypass any yapping or generated code
        matches = re.findall(r'\[\s*(?:"[^"]*"\s*(?:,\s*"[^"]*"\s*)*)?\]', result)
        if matches:
            # Take the last match as it's most likely the final output
            clean_result = matches[-1]
        else:
            # Fallback to simple stripping
            clean_result = result.replace('```json', '').replace('```', '').strip()
            
        room_ids = json.loads(clean_result)
        
        if not isinstance(room_ids, list):
            return []

        # Validate against real meetings to prevent hallucinations
        valid_ids = [m["room_id"] for m in all_meetings]
        final_ids = []
        for r in room_ids:
            # Check for exact match or if the LLM hallucinated a prefix like "room-id-"
            for v in valid_ids:
                if r == v or r.endswith(v) or v.endswith(r):
                    if v not in final_ids:
                        final_ids.append(v)
                    break
        return final_ids
    except Exception as e:
        print(f"Failed to parse LLM router response: {e}\nResponse was: {result}")
        return []
