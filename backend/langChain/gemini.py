# import os
# os.environ["GEMINI_API_KEY"] = "AIzaSyAuGtDLgtYIGX6tctzGjjYMAp0X-hGUeYY"
# from google import genai
#
# # The client gets the API key from the environment variable `GEMINI_API_KEY`.
# client = genai.Client()
#
# response = client.models.generate_content(
#     model="gemini-2.5-flash", contents="Explain how AI works in a few words"
# )
# print(response.text)


from openai import OpenAI

client = OpenAI(
    api_key="495336b5c3a44a85b7b97b64da809573.Hbr3XNqsR3KKNxWO",
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

response = client.chat.completions.create(
    model="glm-4.5-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Explain to me how AI works"
        }
    ]
)

print(response.choices[0].message)