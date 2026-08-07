# Lightweight agent for Vercel deployment (uses remote retrieval API)
import os
import json
from dotenv import load_dotenv
load_dotenv()

class LightweightAgent:
    def __init__(self):
        self.retrieval_api_url = os.getenv("RAG_RETRIEVAL_API_URL")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
    def health_check(self):
        return {
            "status": "ok",
            "retrieval_api_url": self.retrieval_api_url,
            "retrieval_api_configured": bool(self.retrieval_api_url),
            "groq_api_configured": bool(self.groq_api_key)
        }
    
    def ask(self, query):
        # Placeholder implementation - will be enhanced with actual agent logic
        # when retrieval API is available
        if not self.retrieval_api_url:
            return {
                "domain": "pending",
                "config_name": "pending",
                "answer": "Retrieval API not configured. Please set RAG_RETRIEVAL_API_URL environment variable.",
                "retrieved_docs": [],
                "scores": {},
                "rgb_scores": {},
                "latencies": {}
            }
        
        # This would call the retrieval API and process the response
        # For now, return a simple response
        return {
            "domain": "pending",
            "config_name": "pending",
            "answer": f"Retrieval API configured at {self.retrieval_api_url}. Full agent implementation pending.",
            "retrieved_docs": [],
            "scores": {},
            "rgb_scores": {},
            "latencies": {}
        }

# Initialize agent once (will be reused across invocations)
agent = None

def get_agent():
    global agent
    if agent is None:
        agent = LightweightAgent()
    return agent

def handler(request):
    """
    Vercel serverless function handler
    """
    try:
        # Get the path and method from the request
        path = request.get('path', '/')
        method = request.get('method', 'GET')
        
        # Health check endpoint
        if path == '/health' and method == 'GET':
            lightweight_agent = get_agent()
            health_info = lightweight_agent.health_check()
            return {
                'statusCode': 200,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps(health_info)
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
            
            lightweight_agent = get_agent()
            result = lightweight_agent.ask(query)
            
            return {
                'statusCode': 200,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps(result)
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
