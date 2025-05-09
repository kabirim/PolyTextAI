from transformers import pipeline
import asyncio

#  Input text cannot be bigger than 1024 tokens (more or less equal to 800 words)
# TODO => reduce input text to 400 words
async def summairizing(text):
    # Used Transformer-based models like Bart Large CNN
    pipe = pipeline("summarization", model="facebook/bart-large-cnn")
    # The min_length and max_length parameters indicate the minimum and maximum sizes of your summary, do_sample, our summarized text will be the same and not random each time
    summary = pipe(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']  # ← Afficher le résumé

if __name__ == '__main__':
    text = """
    Machine learning is a method of data analysis that automates analytical model building.
    It is a branch of artificial intelligence based on the idea that systems can learn from data,
    identify patterns and make decisions with minimal human intervention.
    """
    asyncio.run(summairizing(text))