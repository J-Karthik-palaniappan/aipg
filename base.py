from utils.text_processing import load_pdf_and_prepare_chunks
from utils.embed_store import create_or_load_vectorstore, get_retriever
from utils.rag_pipeline import get_qa_pipeline

# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Explain the importance of fast language models",
#         }
#     ],
#     model="llama-3.3-70b-versatile",
# )

# print(chat_completion.choices[0].message.content)

chunks = load_pdf_and_prepare_chunks("data.pdf")
vectorstore = create_or_load_vectorstore(chunks)
retriever = get_retriever(vectorstore)
qa = get_qa_pipeline(retriever)

while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break
    result = qa(query)
    print("\n📌 Answer:", result["result"])
    print("\n📚 Source snippet(s):")
    for doc in result["source_documents"]:
        print("-", doc.page_content[:200], "...\n")