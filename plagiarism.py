from groq import Groq

client = Groq(api_key="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")

# Define inputs
question = "Explain the greenhouse effect."
student_1 = "The greenhouse effect is when gases like carbon dioxide trap heat in the atmosphere, warming the Earth."
student_2 = "Heat gets trapped in Earth’s atmosphere because of gases like CO2. This causes the planet to heat up. It’s called the greenhouse effect."

# Enhanced plagiarism detection prompt
plagiarism_prompt = f"""
You are a scientific plagiarism detection assistant. Compare two student responses to the same question and determine if there is evidence of plagiarism.

Evaluate the following dimensions:
1. *Lexical Similarity (0-25 points)* – Similarity in phrasing, word choice, and syntax.
2. *Semantic Similarity (0-25 points)* – Do the ideas significantly overlap even if phrased differently?
3. *Originality (0-25 points)* – Does the response show independent thinking, reasoning, or added value?
4. *Intentionality (0-25 points)* – Based on structure and flow, does the similarity appear to be copied or simply due to topic?

Output the following:
- A breakdown score out of 25 for each category.
- A total plagiarism similarity score (0 = completely independent, 100 = high likelihood of plagiarism).
- A scientific reasoning summary justifying the score.
- Final verdict: 'Likely Plagiarized', 'Possibly Similar due to Topic', or 'Original and Independent'.

---
Question: {question}

Student 1 Answer: {student_1}

Student 2 Answer: {student_2}
---
"""

chat_completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are a scientific plagiarism detection assistant."},
        {"role": "user", "content": plagiarism_prompt}
    ],
    temperature=0.3,
    max_completion_tokens=512,
    top_p=1,
    stream=False,
)

print(chat_completion.choices[0].message.content)
