import os
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent.parent / ".env")

publication_name = "IEEE Transactions on Medical Imaging"
api_url = (
    os.getenv("EASYSCHOLAR_API_URL", "https://www.easyscholar.cc/open/getPublicationRank").strip()
    or "https://www.easyscholar.cc/open/getPublicationRank"
)
secret_key = os.getenv("EASYSCHOLAR_SECRET_KEY", "").strip()
request = (
    f"{api_url}?secretKey={quote(secret_key)}&publicationName={quote(publication_name)}"
    if secret_key
    else ""
)

response = ""
