# Vercel API handler for the agent service
import os
import json
from dotenv import load_dotenv
load_dotenv()

from rag_agent.agent.rag_agent import RagAgent

# Initialize agent once (will be reused across invocations)
agent = None

def get_agent():
    global agent
    if agent is None:
        agent = RagAgent()
    return agent

def app(request):
    """
    Vercel serverless function handler
    """
    try:
        # Get the path and method
        path = request.get('path', '/')
        method = request.get('method', 'GET')
        
        # Health check endpoint
        if path == '/health' and method == 'GET':
            retrieval_api_url = os.getenv("RAG_RETRIEVAL_API_URL")
            return {
                'statusCode': 200,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    "status": "ok",
                    "retrieval_api_url": retrieval_api_url,
                    "retrieval_api_configured": bool(retrieval_api_url)
                })
            }
        
        # Ask endpoint
        elif path == '/ask' and method == 'POST':
            body = json.loads(request.get('body', '{}'))
            query = body.get('query')
            
            if not query:
                return {
                    'statusCode': 400,
                    'headers': {'Content-Type': 'application/json'},
                    'body': json.dumps({"error": "query is required"})
                }
            
            rag_agent = get_agent()
            result = rag_agent.ask(query)
            
            return {
                'statusCode': 200,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    "domain": result.domain,
                    "config_name": result.config_name,
                    "answer": result.answer,
                    "retrieved_docs": [{"text": doc.text, "metadata": doc.metadata, "score": doc.score, "rerank_score": doc.rerank_score} for doc in result.retrieved_docs],
                    "scores": result.scores,
                    "rgb_scores": result.rgb_scores,
                    "latencies": result.latencies
                })
            }
        else:
            return {
                'statusCode': 404,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({"error": "Not found"})
            }
            
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({"error": str(e)})
        }
