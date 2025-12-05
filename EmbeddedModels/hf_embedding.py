from langchain_huggingface import HuggingFaceEmbeddings
# from dotenv import load_dotenv

# load_dotenv()
model = "sentence-transformers/all-mpnet-base-v2"
hf = HuggingFaceEmbeddings(
    model_name=model
   
)

res=hf.embed_query("india")
print(str(res))