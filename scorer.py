from groq import Groq

client = Groq(api_key="apikey")

# Define inputs
question = "Explain the greenhouse effect."
student_1 = "The greenhouse effect is when gases like carbon dioxide trap heat in the atmosphere, warming the Earth."
student_2 = "Heat gets trapped in Earth’s atmosphere because of gases like CO2. This causes the planet to heat up. It’s called the greenhouse effect."

# Prompt for scientific plagiarism detection
plagiarism_prompt = f"""
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
