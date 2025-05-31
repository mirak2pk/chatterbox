import runpod
import json

def handler(event):
    """
    Handle incoming requests for the chatbot
    """
    try:
        # Get input from the request
        input_data = event.get("input", {})
        
        # Extract the user message
        user_message = input_data.get("message", "Hello!")
        
        # Simple echo response (replace with your actual chatbot logic)
        response = f"Echo: {user_message}"
        
        # Return the response
        return {
            "response": response,
            "status": "success"
        }
        
    except Exception as e:
        # Return error if something goes wrong
        return {
            "error": str(e),
            "status": "failed"
        }

if __name__ == "__main__":
    # Start the RunPod serverless worker
    runpod.serverless.start({"handler": handler})
