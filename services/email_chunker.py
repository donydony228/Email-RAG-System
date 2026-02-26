# Email chunker service

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

# tiktoken cl100k_base is used by text-embedding-3-small
_tokenizer = tiktoken.get_encoding("cl100k_base")

# Configuration — sized for text-embedding-3-small (max 8192 tokens per input).
# Larger than the old 128-token limit: more context per retrieval unit.
MAX_TOKENS = 300
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30


def count_tokens(text: str) -> int:
    """
    Count tokens using the tiktoken cl100k_base encoder
    (matches text-embedding-3-small's tokenizer).

    Args:
        text (str): Input text
    Returns:
        int: Number of tokens
    """
    return len(_tokenizer.encode(text))

def split_text_for_sentence_transformer(
    content: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """
    Split email content into chunks suitable for text-embedding-3-small.

    Args:
        content (str): Email content
        chunk_size (int): Desired chunk size in tokens
        overlap (int): Desired overlap in tokens
    Returns:
        list[str]: List of text chunks
    """
    # Find Subject line
    lines = content.split('\n')
    subject_line = lines[0] if 'Subject:' in lines[0] else ''

    # Remove Subject from body
    body_start = 1 if subject_line else 0
    body_content = '\n'.join(lines[body_start:])

    # Approximate: 1 token ≈ 4 characters (for English; rough guide only)
    chunk_size_chars = chunk_size * 4
    overlap_chars = overlap * 4

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=overlap_chars,
        separators=["\n\n", "\n", ". ", " "],
        length_function=lambda x: count_tokens(x)
    )

    body_chunks = splitter.split_text(body_content)

    # Add Subject line back to each chunk if exists
    final_chunks = []
    for chunk in body_chunks:
        if subject_line:
            final_chunk = f"{subject_line}\n\n{chunk}"
        else:
            final_chunk = chunk

        # Check if adding subject exceeds chunk size
        if count_tokens(final_chunk) <= CHUNK_SIZE:
            final_chunks.append(final_chunk)
        else:
            # If too long with subject, use chunk without subject
            final_chunks.append(chunk)

    return final_chunks



def chunk_emails(formatted_emails: list[dict]) -> list[dict]:
    """
    Chunk formatted email documents for text-embedding-3-small.

    Emails that exceed the token limit are split into multiple chunks.
    Each chunk preserves the original metadata and adds chunking information.

    Args:
        formatted_emails (list[dict]): List of formatted email documents
    Returns:
        list[dict]: List of chunked email documents
    """
    chunked_docs = []

    for email_doc in formatted_emails:
        content = email_doc['content']
        token_count = count_tokens(content)

        if token_count <= CHUNK_SIZE:
            # Email is short enough, no chunking needed
            email_doc['metadata']['is_chunked'] = False
            email_doc['metadata']['chunk_index'] = 0
            email_doc['metadata']['total_chunks'] = 1
            chunked_docs.append(email_doc)
        else:
            # Email is too long, needs chunking
            chunks = split_text_for_sentence_transformer(content)

            for i, chunk_text in enumerate(chunks):
                chunked_docs.append({
                    'id': f"{email_doc['id']}_chunk_{i}",
                    'content': chunk_text,
                    'metadata': {
                        **email_doc['metadata'],
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'is_chunked': True,
                        'original_email_id': email_doc['id']
                    }
                })

    return chunked_docs

