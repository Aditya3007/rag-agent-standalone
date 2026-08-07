# Vercel serverless function for health check
import os
import json
from dotenv import load_dotenv
load_dotenv()

def handler(request):
    """
    Vercel serverless function handler for health check
    """
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
