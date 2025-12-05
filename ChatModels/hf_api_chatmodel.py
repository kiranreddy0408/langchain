from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Run below snippet once to add api
# import getpass
# import os

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass(
#     "Enter your Hugging Face API key: "
# )
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto"
)

chat_model = ChatHuggingFace(llm=llm)
res=chat_model.invoke("generate code for prime number in python")
print(res.content)