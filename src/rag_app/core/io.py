from __future__ import annotations

import json
from pathlib import Path
from typing import List

from rag_app.core.types import Document


def load_corpus(path: str) -> List[Document]:
    docs: List[Document] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docs.append(Document(doc_id=row["id"], title=row["title"], text=row["text"]))
    return docs
