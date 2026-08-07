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

def handler(request):
    """
    Vercel serverless function handler
    """
    try:
        # Get the method and body
        method = request.get('method', 'POST')
        
        if method == 'POST':
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
                'statusCode': 405,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({"error": "Method not allowed"})
            }
            
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({"error": str(e)})
        }
