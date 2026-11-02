import re

_ACCOUNT_SPECIFIC_PATTERN = re.compile(
    r"\bmy\s+(?:orders?|parcels?|shipments?|refunds?|gift cards?|accounts?|tracking|purchases?)\b"
    r"|\border\s*(?:number|id|#)?\s*[a-z0-9][a-z0-9-]{4,}\b"
    r"|\bgift card\s*(?:number|id|#)?\s*[a-z0-9][a-z0-9-]{4,}\b"
    r"|\b(?:item|product)\s+i\s+(?:bought|purchased|ordered)\b",
    re.IGNORECASE,
)


def requires_account_lookup(query: str) -> bool:
    return bool(_ACCOUNT_SPECIFIC_PATTERN.search(query))
