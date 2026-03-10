## Multimodal
- Certain models can process and return non-textual data such as images, audio, and video. You can pass non-textual data to a model by providing content blocks.
- Following example downloads an image then describes it.
- main8_multimodal.py
```python
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
```

- Run
```bash
uv --project uv_env/ run python week_05_langchain/01_core_components/02_models/main8_multimodal.py
```
- Output
```
Downloading test image...
Image downloaded successfully!
Analyzing image with AI...

AI Analysis:
This image displays a cheerful and friendly cartoon pig's face. It is a two-dimensional illustration characterized by clean lines and a simple design.

Here's a detailed description:

*   **Overall Shape and Color:** The pig's head is predominantly a large, light, pastel pink circle, outlined entirely in black.
*   **Ears:** At the top of the head, there are two small, rounded, pink ears, also outlined in black. They are positioned symmetrically on either side.
*   **Eyes:** The face features two large, round eyes. Each eye consists of a white circle with a smaller, perfectly round black pupil in the center. The eyes are outlined in black and are positioned relatively high on the face.
*   **Cheeks/Blush:** Below and slightly to the sides of the eyes, there are two prominent, circular, rosy-red blush marks. These marks have a soft, gradient effect, making them appear to glow gently.
*   **Snout:** In the center of the face, below the eyes and between the blush marks, is the pig's snout. It's an oval shape, the same light pink as the face, and contains two small, black, circular nostrils. The snout is also outlined in black.
*   **Mouth:** Below the snout, a simple, upward-curving black line forms a wide, happy smile.

The overall impression is one of cuteness, happiness, and simplicity, typical of a children's illustration or an emoji-style character. The strong black outlines give it a clear, graphic quality.
```