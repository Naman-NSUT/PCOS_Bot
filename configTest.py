import google.generativeai as genai

genai.configure(api_key='AIzaSyAi_hrOsQaXl03gRMJWPCgjph9lzpgJ9Cs'
) 

for m in genai.list_models():
    if "embedContent" in m.supported_generation_methods:
        print(m.name)