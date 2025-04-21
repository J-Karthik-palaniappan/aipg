from groq import Groq
import json
from pathlib import Path

client = Groq(api_key="your_api_key_here")  # Replace with your API key

# Load Q/A dataset
with open("generated_qa_dataset.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# Simulated second responses for demo (replace with real data if available)
for entry in qa_data:
    if "student_response_2" not in entry:
        entry["student_response_2"] = "This is another student's version of the answer."  # Replace or randomize if needed

# Function to run plagiarism detection
def detect_plagiarism(question, student_1, student_2):
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

Student 1 Answer: {student_1}

Student 2 Answer: {student_2}
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

# Run plagiarism checks
results = []

for entry in qa_data:
    q = entry["question"]
    s1 = entry.get("student_response", "") or entry.get("student_response_1", "")
    s2 = entry.get("student_response_2", "")

    if not (s1 and s2):
        continue

    print(f"\n🔍 Checking for question:\n{q}\n")
    result = detect_plagiarism(q, s1, s2)
    print(result)

    results.append({
        "question": q,
        "student_1": s1,
        "student_2": s2,
        "plagiarism_analysis": result
    })

# Save output
with open("plagiarism_report.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("\n✅ All plagiarism checks complete. Results saved to plagiarism_report.json.")
