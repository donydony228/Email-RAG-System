"""
OpenAI API service for generating responses based on retrieved context.

This service handles:
1. Building prompts with retrieved email context
2. Calling OpenAI API
3. Parsing and formatting responses
"""

import os
from typing import List, Dict, Optional
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Model configuration
DEFAULT_MODEL = "gpt-4o"  # or "gpt-4-turbo", "gpt-3.5-turbo"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7


def build_context_from_emails(matches: List[Dict], max_emails: int = 5) -> str:
    """
    Build context string from retrieved email matches.

    Args:
        matches: List of search results from Pinecone
        max_emails: Maximum number of emails to include in context

    Returns:
        Formatted context string
    """
    context_parts = []

    for i, match in enumerate(matches[:max_emails], 1):
        metadata = match.get('metadata', {})
        score = match.get('score', 0)

        # Format email information
        email_context = f"""
### Email {i} (Similarity: {score:.2%})
**Subject:** {metadata.get('subject', 'N/A')}
**From:** {metadata.get('from', 'N/A')}
**Date:** {metadata.get('date', 'N/A')}
**Account:** {metadata.get('account', 'N/A')}

**Content:**
{metadata.get('content', 'N/A')}
"""
        context_parts.append(email_context.strip())

    return "\n\n---\n\n".join(context_parts)


def build_prompt(query: str, context: str, language: str = "auto") -> str:
    """
    Build the complete prompt for Claude.

    Args:
        query: User's question
        context: Retrieved email context
        language: Response language ("zh" for Chinese, "en" for English, "auto" for auto-detect)

    Returns:
        Complete prompt string
    """
    # Language instruction
    lang_instruction = ""
    if language == "zh":
        lang_instruction = "請用繁體中文回答。"
    elif language == "en":
        lang_instruction = "Please respond in English."
    else:
        lang_instruction = "請根據問題的語言來決定回答的語言（中文問題用中文回答，英文問題用英文回答）。"

    prompt = f"""你是一個郵件助手，專門幫助用戶從他們的郵件中找到信息並回答問題。

我已經從郵件資料庫中檢索到以下相關的郵件：

{context}

---

**用戶問題：** {query}

**回答指南：**
1. 基於上述檢索到的郵件內容來回答問題
2. 如果郵件中沒有相關信息，請明確說明「在檢索到的郵件中找不到相關信息」
3. 引用具體的郵件時，請註明是哪一封（例如：「根據 Email 1...」）
4. 保持回答簡潔、準確
5. {lang_instruction}

請開始回答："""

    return prompt


def generate_response(
    query: str,
    context: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    language: str = "auto",
    stream: bool = False
) -> str:
    """
    Generate a response using OpenAI API.

    Args:
        query: User's question
        context: Retrieved email context
        model: OpenAI model to use (e.g., "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo")
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0-1)
        language: Response language preference
        stream: Enable streaming responses

    Returns:
        Generated response text
    """
    # Build the prompt
    prompt = build_prompt(query, context, language)

    try:
        if stream:
            # Streaming response
            response_text = ""
            stream_response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                stream=True
            )

            for chunk in stream_response:
                if chunk.choices[0].delta.content is not None:
                    text = chunk.choices[0].delta.content
                    print(text, end="", flush=True)
                    response_text += text
            print()  # Newline after streaming
            return response_text
        else:
            # Regular response
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return response.choices[0].message.content

    except Exception as e:
        raise Exception(f"OpenAI API error: {e}")


def ask_with_emails(
    query: str,
    search_results: Dict,
    max_context_emails: int = 5,
    language: str = "auto",
    stream: bool = False
) -> Dict:
    """
    Convenience function to generate answer from search results.

    Args:
        query: User's question
        search_results: Results from Pinecone search
        max_context_emails: Max emails to include in context
        language: Response language preference
        stream: Enable streaming

    Returns:
        Dictionary with answer and metadata
    """
    matches = search_results.get('matches', [])

    if not matches:
        return {
            'answer': '沒有找到相關的郵件來回答這個問題。',
            'sources': [],
            'context_used': 0
        }

    # Build context from matches
    context = build_context_from_emails(matches, max_context_emails)

    # Generate response
    answer = generate_response(
        query=query,
        context=context,
        language=language,
        stream=stream
    )

    # Extract source information
    sources = []
    for match in matches[:max_context_emails]:
        metadata = match.get('metadata', {})
        sources.append({
            'subject': metadata.get('subject', 'N/A'),
            'from': metadata.get('from', 'N/A'),
            'date': metadata.get('date', 'N/A'),
            'similarity': match.get('score', 0)
        })

    return {
        'answer': answer,
        'sources': sources,
        'context_used': len(sources)
    }
