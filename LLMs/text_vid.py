from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="replicate",
    api_key="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",
)

video = client.text_to_video(
    "A young man walking on the street",
    model="Wan-AI/Wan2.1-T2V-14B",
)
with open("output_video.mp4", "wb") as f:
    f.write(video)