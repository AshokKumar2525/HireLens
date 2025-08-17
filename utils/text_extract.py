import re

def smart_extract(text: str) -> dict:
    title_match = re.search(r"(?:^|\n)\s*(?:title|job title)[:\-]\s*(.+)", text, flags=re.I)
    loc_match = re.search(r"(?:^|\n)\s*(?:location)[:\-]\s*(.+)", text, flags=re.I)
    email_match = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    urls = re.findall(r"https?://\S+|www\.\S+", text)
    return {
        "title": title_match.group(1).strip() if title_match else "",
        "location": loc_match.group(1).strip() if loc_match else "",
        "emails": list(set(email_match))[:3],
        "links": list(set(urls))[:5],
    }
