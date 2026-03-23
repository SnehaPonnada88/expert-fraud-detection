import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) or st.secrets.get("OPENAI_API_KEY")

def extract_signals(answer):
    prompt = f"""
    You are an expert interviewer evaluating a candidate.

    Extract structured signals from the answer below.

    Answer:
    "{answer}"

    Return STRICT JSON:

    {{
      "tools": [],
      "concepts": [],
      "has_reasoning": true/false,
      "has_failure": true/false,
      "specificity": number (0 to 1),
      "project_anchor": ""
    }}

    Be generous and infer missing details.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content

        print("\nLLM RAW OUTPUT:\n", content)

        return json.loads(content)

    except Exception as e:
        print("LLM ERROR:", e)

        return {
            "tools": [],
            "concepts": [],
            "has_reasoning": False,
            "has_failure": False,
            "specificity": 0.3,
            "project_anchor": ""
        }
