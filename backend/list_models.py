from google import genai
import os

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY 未設定")
else:
    client = genai.Client(api_key=api_key)
    print("可用模型清單：")
    for m in client.models.list():
        print(" -", m.name)
