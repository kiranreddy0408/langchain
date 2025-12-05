from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

hf=HuggingFaceEmbeddings(
model_name= "sentence-transformers/all-mpnet-base-v2"
)

# documents=[
#   "Mumbai, the city of dreams, pulses with a vibrant energy and Bollywood glamour.",
#   "Delhi, a historic capital, showcases a captivating blend of ancient Mughal architecture and modern dynamism.",
#   "Kolkata, the cultural heart of India, resonates with artistic heritage and intellectual fervor.",
#   "Bengaluru, the Silicon Valley of India, thrives as a hub for technology and innovation.",
#   "Chennai, a gateway to South India, proudly preserves its rich Dravidian traditions and temple grandeur."
# ]

documents=[
  "Mumbai, a vibrant city, showcases its Bollywood film industry.",
  "Delhi, a historic city, boasts numerous Mughal monuments.",
  "Kolkata, the cultural city, resonates with artistic heritage.",
  "Bengaluru, a major city, thrives as a technology hub, it industry.",
  "Chennai, a vibrant city, preserves its Dravidian traditions."
]


# query="Bengaluru is also know as ___"
query="a vibrant city, industry"

doc_embeddings=hf.embed_documents(documents)
query_embeddings=hf.embed_query(query)

scores=cosine_similarity([query_embeddings],doc_embeddings)[0]
print(scores)
index,score=sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]
print(f'{query}\n Answer : {documents[index]}\nsimilarity : {score}')