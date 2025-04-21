from groq import Groq
import json
from pathlib import Path

client = Groq(api_key="your_api_key_here")

# Load saved Q/A dataset
with open("generated_qa_dataset.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# Simulated second set of student answers (can be replaced with real data)
student_2_responses = {
    q["question"]: "This is a second student's answer. It may be similar or different." 
    for q in qa_data
}

# Function to build and run plagiarism prompt
def check_plagiarism(question, answer1, answer2):
    prompt = f"""
You are a scientific plagiarism detection assistant. Your job is to compare two student responses to the same question and determine if there is evidence of plagiarism.

Consider:
1. *Lexical similarity* – Are the word choices and sentence structures unusually similar?
2. *Semantic similarity* – Are the core ideas copied or paraphrased without genuine reinterpretation?
3. *Originality* – Does each student contribute distinct reasoning or examples?
4. *Intentionality* – Is the similarity likely coincidental, influenced by the same source, or copied?

Return:
- A detailed breakdown under each category.
- A final verdict: 'Likely Plagiarized', 'Possibly Similar due to Topic', or 'Original and Independent'.
---
Question: {question}

Student 1 Answer: {answer1}

Student 2 Answer: {answer2}
---
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a scientific plagiarism detection assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_completion_tokens=512,
        top_p=1,
        stream=False,
    )
    return response.choices[0].message.content

# Run plagiarism check for each question
plagiarism_reports = []
for entry in qa_data:
    q = entry["question"]
    student_1 = entry.get("student_response", "").strip()
    student_2 = student_2_responses[q]

    if student_1 and student_2:
        print(f"\n🔍 Checking Plagiarism for Question:\n{q}\n")
        result = check_plagiarism(q, student_1, student_2)
        print(result)

        plagiarism_reports.append({
            "question": q,
            "student_1": student_1,
            "student_2": student_2,
            "plagiarism_report": result
        })

# Optional: Save plagiarism results
with open("plagiarism_results.json", "w", encoding="utf-8") as f:
    json.dump(plagiarism_reports, f, indent=4, ensure_ascii=False)

print("\n✅ Plagiarism check completed and saved to plagiarism_results.json")
