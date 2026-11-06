"""Load txt/md/pdf/csv/xlsx (paths or Streamlit uploads)."""
import io
import pandas as pd
from pypdf import PdfReader


def _bytes(f):
    return f.read() if hasattr(f, "read") else open(f, "rb").read()


def load_table(f, sheet=0) -> pd.DataFrame:
    """Excel/CSV -> DataFrame (empty cells become '')."""
    name = getattr(f, "name", str(f)).lower()
    data = io.BytesIO(_bytes(f))
    df = pd.read_csv(data) if name.endswith(".csv") else pd.read_excel(data, sheet_name=sheet)
    return df.fillna("")


def load_file(f) -> str:
    name = getattr(f, "name", str(f)).lower()
    if name.endswith((".xlsx", ".xls", ".csv")):
        return load_table(f).to_csv(index=False)
    data = _bytes(f)
    if name.endswith(".pdf"):
        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
    return data.decode("utf-8", errors="ignore")


def chunk(text, size=800, overlap=100):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return [c for c in out if c.strip()]
