"""
Preprocess email content by cleaning and formatting.
"""

import re
from email.utils import parsedate_to_datetime


def _parse_timestamp(date_str: str) -> int:
    """Convert RFC 2822 date string to Unix epoch seconds (int). Returns 0 on failure."""
    try:
        return int(parsedate_to_datetime(date_str).timestamp())
    except Exception:
        return 0


def clean(text: str) -> str:
    """
    Clean the email content by removing unwanted elements.

    Args:
        text (str): The raw email content.
    Returns:
        str: The cleaned email content.
    """
    # Remove carriage returns and excessive newlines
    text = re.sub(r'\r\n+', '\n', text)
    text = re.sub(r'\n+', '\n', text)
    # Remove tracking codes
    text = re.sub(r'%opentrack%', '', text)
    # Remove special whitespace characters
    text = text.replace('\xa0', ' ')
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    return text

def format_email(emails: list[dict]) -> list[dict]:
    """
    Format and clean the email content.

    Args:
        emails (list[dict]): A list of dictionaries containing email fields like 'subject', 'from', and 'snippet'.

    Returns:
        list[dict]: A list of formatted email dictionaries.
    """
    processed = []
    for email in emails:
        processed.append({
            'id': f"email_{email['id']}",
            'content': clean(f"Subject: {email['subject']}\n\n'From': {email['from']}\n\n'Content': {email['body']}"),
            'metadata': {
                'account': email['account'],
                'subject': email['subject'],
                'from': email['from'],
                'to': email['to'],
                'date': email['date'],
                'timestamp': _parse_timestamp(email['date']),
                'message_id': email['id'],
                'thread_id': email['thread_id'],
                'labels': email['labels'],
            }
        })
    return processed
