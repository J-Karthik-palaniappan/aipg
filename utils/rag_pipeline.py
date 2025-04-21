# utils/rag_pipeline.py

from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import List, Optional, Any
import os
from groq import Groq

class GroqLLM(LLM):
    model: str = "llama3-70b-8192"  # correct model name
    temperature: float = 0.0
    client: Any = None

    def __init__(self, model="llama3-70b-8192", temperature=0.0):
        super().__init__()
        api_key = os.environ.get("GROQ_API_KEY")
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
        )
        return chat_completion.choices[0].message.content


def get_qa_pipeline(retriever):
    llm = GroqLLM(model="llama3-70b-8192", temperature=0.0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
