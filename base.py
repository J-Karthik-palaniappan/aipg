from utils.text_processing import load_pdf_and_prepare_chunks
from utils.embed_store import create_or_load_vectorstore, get_retriever
from utils.rag_pipeline import get_qa_pipeline
from groq import Groq
from pathlib import Path
import json

# ---------- API Setup ----------
client = Groq(api_key="your_api_key_here")

# ---------- Load and Prepare Document ----------
chunks = load_pdf_and_prepare_chunks("data.pdf")
vectorstore = create_or_load_vectorstore(chunks)
retriever = get_retriever(vectorstore)
qa = get_qa_pipeline(retriever)

# ---------- Question Generator Prompt ----------
def generate_application_questions(chunk):
    prompt = f"""
You are a scientific educator designing application-based questions for students based on a given text. Your task is to extract meaningful, real-world, or conceptual application-based questions from the following content.

Guidelines:
1. Focus on practical applications, reasoning, or implications of the concept.
2. Avoid simple factual or definitional questions.
3. Prefer open-ended or scenario-based questions that require understanding.
4. Return 3-5 well-formed questions.

Content:
{chunk}

Output Format (JSON):
[
    "Question 1",
    "Question 2",
    ...
]
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a scientific question generation assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_completion_tokens=512,
    )

    try:
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print("⚠️ Failed to parse questions:", e)
        return []

# ---------- Simulated/Manual Student Response ----------
def get_student_response(q):
    # OPTIONAL: You can collect input or simulate a student answer
    # return input(f"📝 Student Response to: {q}\n> ")
    return ""  # Placeholder for automated runs

# ---------- Main Pipeline ----------
qa_dataset = []
print("\n📘 Generating and Answering Questions...\n" + "-"*50)

# You can limit chunks for testing (e.g., chunks[:2])
for chunk in chunks[:3]:
    questions = generate_application_questions(chunk.page_content)
    for q in questions:
        print(f"\n❓ Q: {q}")
        result = qa(q)
        answer = result['result']
        print(f"📌 A: {answer}")

        student_response = get_student_response(q)

        qa_dataset.append({
            "question": q,
            "answer_key": answer,
            "student_response": student_response
        })

# ---------- Save to JSON ----------
output_path = Path("generated_qa_dataset.json")
with output_path.open("w", encoding="utf-8") as f:
    json.dump(qa_dataset, f, indent=4, ensure_ascii=False)

print(f"\n✅ Saved Q/A dataset to {output_path.resolve()}")
