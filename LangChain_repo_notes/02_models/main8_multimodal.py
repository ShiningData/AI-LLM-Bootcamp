from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
import requests
import base64
import os

load_dotenv()

model = init_chat_model(
    "google_genai:gemini-2.5-flash-lite",
    temperature=0.3,
)

# Download a simple image
print("Downloading test image...")
image_url = "https://httpbin.org/image/png"
response = requests.get(image_url)

if response.status_code == 200:
    # Save the image
    with open("test_image.png", "wb") as f:
        f.write(response.content)
    
    # Convert to base64
    with open("test_image.png", "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    print("Image downloaded successfully!")
    print("Analyzing image with AI...")
    
    # Analyze the image
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe what you see in this image in detail."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
        ]
    )
    
    ai_response = model.invoke([message])
    print("\nAI Analysis:")
    print(ai_response.content)
    
    # Clean up
    os.remove("test_image.png")
    
else:
    print(f"Failed to download image. Status code: {response.status_code}")