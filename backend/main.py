import os
import uuid
import json
import re
import time
import sqlite3
import hashlib
import shutil
from typing import Optional, List, Dict, Set
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from openai import AsyncOpenAI

# ─ Configuration ──────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-...")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
DB_PATH = "mindmap_sessions.db"

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

llm_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

def _run_migrations():
    con = sqlite3.connect(DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)")
    
    row = con.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1").fetchone()
    current_version = row[0] if row else 0

    if current_version < 1:
        con.execute("CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, data TEXT, updated_at REAL)")
        con.execute("CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, filename TEXT, media_type TEXT, sha256 TEXT, imported_at REAL, original_path TEXT, markdown_text TEXT, status TEXT, error TEXT)")
        con.execute("CREATE TABLE IF NOT EXISTS source_chunks (id TEXT PRIMARY KEY, document_id TEXT, content TEXT, chunk_index INTEGER, page_number INTEGER, section_title TEXT, char_start INTEGER, char_end INTEGER, embedding TEXT)")
        con.execute("CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY, title TEXT, summary TEXT, notes TEXT, created_at REAL, updated_at REAL)")
        con.execute("CREATE TABLE IF NOT EXISTS edges (source_node_id TEXT, target_node_id TEXT, relation_type TEXT, confidence REAL, PRIMARY KEY (source_node_id, target_node_id))")
        con.execute("CREATE TABLE IF NOT EXISTS node_sources (node_id TEXT, chunk_id TEXT, citation_text TEXT, relevance REAL, PRIMARY KEY (node_id, chunk_id))")
        con.execute("CREATE TABLE IF NOT EXISTS tags (id TEXT PRIMARY KEY, name TEXT UNIQUE)")
        con.execute("CREATE TABLE IF NOT EXISTS node_tags (node_id TEXT, tag_id TEXT, PRIMARY KEY (node_id, tag_id))")
        
        con.execute("INSERT INTO schema_version (version) VALUES (1)")
        
    con.commit()
    con.close()

_run_migrations()

# ── SQLite Helpers ─────────────────────────────────────────────────────────────
def _db_save(session_id: str, session: 'GraphSession'):
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT OR REPLACE INTO sessions (session_id, data, updated_at) VALUES (?,?,?)",
                (session_id, json.dumps(session.to_dict(), ensure_ascii=False), time.time()))
    con.commit()
    con.close()

def _db_load(session_id: str) -> Optional['GraphSession']:
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT data FROM sessions WHERE session_id=?", (session_id,)).fetchone()
    con.close()
    if row:
        try: return GraphSession.from_dict(json.loads(row[0]))
        except Exception: pass
    return None

# ─ Phase 1 & 1.5: Document Ingestion, Chunking & Provenance ───────────────────
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[dict]:
    """Simple, dependency-free chunking algorithm based on Markdown headings."""
    chunks = []
    heading_pattern = re.compile(r'^(#{1,6}\s+.+)$', re.MULTILINE)
    sections = heading_pattern.split(text)
    
    current_heading = "Introduction"
    current_content = sections[0].strip() if sections and sections[0].strip() else ""
    
    grouped_sections = []
    if current_content:
        grouped_sections.append((current_heading, current_content))
        
    for i in range(1, len(sections), 2):
        if i + 1 < len(sections):
            heading = sections[i].strip()
            content = sections[i+1].strip()
            grouped_sections.append((heading, content))
            
    chunk_index = 0
    for heading, content in grouped_sections:
        if len(content) <= chunk_size:
            chunk_content = f"{heading}\n{content}".strip()
            if chunk_content:
                chunks.append({
                    "chunk_index": chunk_index,
                    "section_title": heading,
                    "content": chunk_content,
                    "char_start": 0, "char_end": 0
                })
                chunk_index += 1
        else:
            paragraphs = content.split('\n\n')
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) > chunk_size and current_chunk:
                    full_chunk = f"{heading}\n{current_chunk}".strip()
                    chunks.append({
                        "chunk_index": chunk_index,
                        "section_title": heading,
                        "content": full_chunk,
                        "char_start": 0, "char_end": 0
                    })
                    chunk_index += 1
                    current_chunk = para 
                else:
                    current_chunk += f"\n\n{para}" if current_chunk else para
            
            if current_chunk.strip():
                full_chunk = f"{heading}\n{current_chunk}".strip()
                chunks.append({
                    "chunk_index": chunk_index,
                    "section_title": heading,
                    "content": full_chunk,
                    "char_start": 0, "char_end": 0
                })
                chunk_index += 1
                
    return chunks

def ingest_document(text: str, filename: str = "pasted_text.md", media_type: str = "text/markdown") -> str:
    """Ingests text, checks for duplicates via SHA256, chunks it, and saves to DB."""
    sha256 = hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    con = sqlite3.connect(DB_PATH)
    existing = con.execute("SELECT id FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
    if existing:
        con.close()
        return existing[0]
        
    doc_id = str(uuid.uuid4())
    imported_at = time.time()
    
    con.execute("""
        INSERT INTO documents (id, filename, media_type, sha256, imported_at, markdown_text, status)
        VALUES (?, ?, ?, ?, ?, ?, 'processed')
    """, (doc_id, filename, media_type, sha256, imported_at, text))
    
    chunks = chunk_text(text)
    for chunk in chunks:
        chunk_id = f"{doc_id}_c{chunk['chunk_index']}"
        con.execute("""
            INSERT INTO source_chunks (id, document_id, content, chunk_index, section_title, char_start, char_end)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (chunk_id, doc_id, chunk['content'], chunk['chunk_index'], chunk['section_title'], chunk['char_start'], chunk['char_end']))
        
    con.commit()
    con.close()
    return doc_id

# ── Provenance Linking Helpers ──────────────────────────────────────
def _calculate_keyword_overlap(text1: str, text2: str) -> float:
    """Calculates a simple Jaccard similarity score between two texts based on words."""
    # Basic tokenization: lowercase, split by non-alphanumeric
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
    # Filter out very short words (stopwords approximation)
    words1 = {w for w in words1 if len(w) > 2}
    words2 = {w for w in words2 if len(w) > 2}
    
    if not words1 or not words2:
        return 0.0
        
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def _link_nodes_to_chunks(session_id: str, document_id: str, threshold: float = 0.15):
    """
    Links nodes in a session graph to source chunks based on keyword overlap.
    """
    gs = _db_load(session_id)
    if not gs:
        return
        
    con = sqlite3.connect(DB_PATH)
    chunks = con.execute("SELECT id, content FROM source_chunks WHERE document_id = ?", (document_id,)).fetchall()
    con.close()
    
    if not chunks:
        return
        
    con = sqlite3.connect(DB_PATH)
    for node_id, node_data in gs.nodes.items():
        label = node_data.get("label", "")
        summary = node_data.get("summary", "")
        node_text = f"{label} {summary}".strip()
        
        if not node_text:
            continue
            
        best_chunk_id = None
        best_score = 0.0
        
        for chunk_id, chunk_content in chunks:
            score = _calculate_keyword_overlap(node_text, chunk_content)
            if score > best_score:
                best_score = score
                best_chunk_id = chunk_id
                
        if best_chunk_id and best_score >= threshold:
            # citation_text is a snippet of the chunk
            citation_snippet = chunks[[c[0] for c in chunks].index(best_chunk_id)][1][:100] + "..."
            try:
                con.execute("""
                    INSERT OR REPLACE INTO node_sources (node_id, chunk_id, citation_text, relevance)
                    VALUES (?, ?, ?, ?)
                """, (node_id, best_chunk_id, citation_snippet, best_score))
            except sqlite3.IntegrityError:
                pass # Already linked
                
    con.commit()
    con.close()

# ── Graph Session  ──────────────────────────────────────────────────
class GraphSession:
    def __init__(self):
        self.nodes: Dict[str, dict] = {}
        self.edges: List[dict] = []
        self.expanded_nodes: Set[str] = set()
        self.hidden_nodes: Set[str] = set()

    def add_node(self, node: dict): self.nodes[node["id"]] = node
    def add_edge(self, source: str, target: str): self.edges.append({"source": source, "target": target})

    def hide_subtree(self, node_id: str):
        for desc in self._descendants(node_id): self.hidden_nodes.add(desc)
        if node_id in self.nodes: self.nodes[node_id]["expanded"] = False

    def show_subtree(self, node_id: str):
        for child_id in self._child_map().get(node_id, []): self.hidden_nodes.discard(child_id)
        if node_id in self.nodes: self.nodes[node_id]["expanded"] = True
        self.expanded_nodes.add(node_id)

    def _descendants(self, node_id: str) -> List[str]:
        cm = self._child_map()
        result, stack = [], [node_id]
        while stack:
            cur = stack.pop()
            for child in cm.get(cur, []): result.append(child); stack.append(child)
        return result

    def _child_map(self) -> Dict[str, List[str]]:
        cm: Dict[str, List[str]] = {}
        for e in self.edges: cm.setdefault(e["source"], []).append(e["target"])
        return cm

    def get_path_to_root(self, node_id: str) -> List[str]:
        parent_map = {e["target"]: e["source"] for e in self.edges}
        path, current = [], node_id
        while current in parent_map: current = parent_map[current]; path.insert(0, current)
        return path

    def get_siblings(self, node_id: str) -> List[str]:
        parent_map = {e["target"]: e["source"] for e in self.edges}
        parent = parent_map.get(node_id)
        return [e["target"] for e in self.edges if e["source"] == parent and e["target"] != node_id] if parent else []

    def visible_to_dict(self) -> dict:
        vis_nodes = {nid: n for nid, n in self.nodes.items() if nid not in self.hidden_nodes}
        vis_edges = [e for e in self.edges if e["source"] not in self.hidden_nodes and e["target"] not in self.hidden_nodes]
        return {"nodes": vis_nodes, "edges": vis_edges, "expanded_nodes": list(self.expanded_nodes), "hidden_nodes": list(self.hidden_nodes)}

    def to_dict(self) -> dict:
        return {"nodes": self.nodes, "edges": self.edges, "expanded_nodes": list(self.expanded_nodes), "hidden_nodes": list(self.hidden_nodes)}

    @classmethod
    def from_dict(cls, d: dict) -> "GraphSession":
        gs = cls()
        gs.nodes = d.get("nodes", {})
        gs.edges = d.get("edges", [])
        gs.expanded_nodes = set(d.get("expanded_nodes", []))
        gs.hidden_nodes = set(d.get("hidden_nodes", []))
        return gs

# ── LLM Prompts  ────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a semantic knowledge graph generation assistant. Analyze text and produce structured concept hierarchies.
Format: Strictly Markdown list format. Use `#` for root, `-` with 2-space indentation for branches.
Keep node labels concise (3-6 words). Match input language exactly."""

USER_PROMPT_GENERATE = """Analyze the following text and structure its core themes into standard Markdown list syntax.
# Central Theme
- Topic 1
  - Sub-topic 1.1
    - Detail
Text:\n{long_text_content}"""

USER_PROMPT_EXPAND = """Expand this concept: {node_label}
Context: {node_summary}
Parent chain: {parent_chain}
Siblings: {siblings}
Generate 4-6 meaningful child concepts. Return JSON only: {{"nodes": [{{"id": " ", "label": " ", "summary": " "}}]}}"""

USER_PROMPT_QUERY = """Answer the question about a specific concept node.
Node: {node_label} | Summary: {node_summary} | Parent chain: {parent_chain}
Question: {query}
Provide a focused, concise answer (3-5 sentences)."""

# ─ Pydantic Models ───────────────────────────────────────────────────────────
class GenerateRequest(BaseModel): 
    text: str; session_id: Optional[str] = None; model: Optional[str] = None; document_id: Optional[str] = None
class ExpandRequest(BaseModel): 
    session_id: str; node_id: str; model: Optional[str] = None
class QueryRequest(BaseModel): 
    session_id: str; node_id: str; query: str; model: Optional[str] = None
class RestoreRequest(BaseModel):
    graph: dict
class DocumentUploadRequest(BaseModel):
    text: str
    filename: Optional[str] = "pasted_text.md"
    media_type: Optional[str] = "text/markdown"

# ── Helper Functions ───────────────────────────────────────────────
def parse_markdown_to_session(md_text: str) -> GraphSession:
    gs = GraphSession()
    lines = [l.rstrip() for l in md_text.splitlines() if l.strip()]
    root_label = next((ml[2:].strip() for ml in lines if ml.startswith("# ")), "Root")
    root_node = {"id": "root", "label": root_label, "summary": "", "depth": 0, "expanded": True}
    gs.add_node(root_node)
    
    _parent_stack = {0: "root"}
    _base_indent = 0
    for ml in lines:
        if not ml or ml.lstrip().startswith("#"): continue
        ml_norm = ml.replace("\t", "  ")
        stripped = ml_norm.lstrip()
        if not stripped.startswith("- "): continue
        
        indent = len(ml_norm) - len(stripped)
        if indent > 0 and _base_indent == 0: _base_indent = indent
        base = _base_indent if _base_indent > 0 else 2
        depth = min((indent // base) + 1, 3)
        label = stripped[2:].strip()
        if not label: continue
        
        parent_id = _parent_stack.get(depth - 1, "root")
        node_id = str(uuid.uuid4())[:8]
        node = {"id": node_id, "label": label, "summary": "", "depth": depth, "expanded": False}
        gs.add_node(node)
        gs.add_edge(parent_id, node_id)
        _parent_stack[depth] = node_id
        for d in list(_parent_stack.keys()):
            if d > depth: del _parent_stack[d]
            
    for nid, node in list(gs.nodes.items()):
        if node.get("depth", 0) < 3: gs.show_subtree(nid)
    return gs

# ── API Endpoints ──────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/models")
async def get_models():
    try:
        models = await llm_client.models.list()
        model_ids = sorted([m.id for m in models.data])
        return {"models": model_ids, "default": DEFAULT_LLM_MODEL}
    except Exception:
        return {"models": [DEFAULT_LLM_MODEL], "default": DEFAULT_LLM_MODEL}

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    gs = _db_load(session_id)
    if not gs: raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "graph": gs.visible_to_dict()}

@app.post("/api/restore")
async def restore_session(req: RestoreRequest):
    session_id = str(uuid.uuid4())
    gs = GraphSession.from_dict(req.graph)
    _db_save(session_id, gs)
    return {"session_id": session_id, "graph": gs.visible_to_dict()}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    return {"text": content.decode("utf-8")}

@app.post("/api/generate")
async def generate_mindmap(req: GenerateRequest):
    session_id = req.session_id or str(uuid.uuid4())
    target_model = req.model or DEFAULT_LLM_MODEL
    
    text_to_analyze = req.text
    if req.document_id:
        con = sqlite3.connect(DB_PATH)
        row = con.execute("SELECT markdown_text FROM documents WHERE id = ?", (req.document_id,)).fetchone()
        con.close()
        if row:
            text_to_analyze = row[0]
        else:
            raise HTTPException(404, "Document not found")

    response = await llm_client.chat.completions.create(
        model=target_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_GENERATE.format(long_text_content=text_to_analyze)}
        ],
        temperature=0.5
    )
    md_text = response.choices[0].message.content
    
    gs = parse_markdown_to_session(md_text)
    _db_save(session_id, gs)
    
    # Phase 1.5: If generated from a document, link nodes to chunks
    if req.document_id:
        _link_nodes_to_chunks(session_id, req.document_id)
    
    return {"session_id": session_id, "graph": gs.visible_to_dict(), "markdown": md_text}

@app.post("/api/expand")
async def expand_node(req: ExpandRequest):
    gs = _db_load(req.session_id)
    if not gs: raise HTTPException(404, "Session not found")
    
    node = gs.nodes.get(req.node_id)
    if not node: raise HTTPException(404, "Node not found")
    
    path_ids = gs.get_path_to_root(req.node_id)
    parent_chain = " > ".join(gs.nodes[pid]["label"] for pid in path_ids if pid in gs.nodes)
    siblings = [gs.nodes[sid]["label"] for sid in gs.get_siblings(req.node_id) if sid in gs.nodes]
    
    target_model = req.model or DEFAULT_LLM_MODEL
    response = await llm_client.chat.completions.create(
        model=target_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_EXPAND.format(
                node_label=node["label"], node_summary=node.get("summary", ""),
                parent_chain=parent_chain or "(root)", siblings=", ".join(siblings) or "none"
            )}
        ],
        temperature=0.6
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
    
    try:
        data = json.loads(raw)
        for n in data.get("nodes", []):
            if n.get("label"):
                n_id = n.get("id") or str(uuid.uuid4())[:8]
                gs.add_node({"id": n_id, "label": n["label"], "summary": n.get("summary", ""), "depth": node["depth"]+1, "expanded": False})
                gs.add_edge(req.node_id, n_id)
        gs.show_subtree(req.node_id)
        gs.nodes[req.node_id]["expanded"] = True
    except json.JSONDecodeError:
        pass
        
    _db_save(req.session_id, gs)
    return gs.visible_to_dict()

@app.post("/api/collapse")
async def collapse_node(req: ExpandRequest):
    gs = _db_load(req.session_id)
    if not gs: raise HTTPException(404, "Session not found")
    gs.hide_subtree(req.node_id)
    _db_save(req.session_id, gs)
    return gs.visible_to_dict()

@app.post("/api/query")
async def query_node(req: QueryRequest):
    gs = _db_load(req.session_id)
    if not gs: raise HTTPException(404, "Session not found")
    
    node = gs.nodes.get(req.node_id)
    if not node: raise HTTPException(404, "Node not found")
    
    path_ids = gs.get_path_to_root(req.node_id)
    parent_chain = " > ".join(gs.nodes[pid]["label"] for pid in path_ids if pid in gs.nodes)
    
    target_model = req.model or DEFAULT_LLM_MODEL
    response = await llm_client.chat.completions.create(
        model=target_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_QUERY.format(
                node_label=node["label"], node_summary=node.get("summary", ""),
                parent_chain=parent_chain or "(root)", query=req.query
            )}
        ],
        temperature=0.5
    )
    return {"answer": response.choices[0].message.content}

# ── Phase 0: Diagnostic & Backup Endpoints ─────────────────────────
@app.get("/api/db/version")
async def get_db_version():
    _run_migrations()
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1").fetchone()
    version = row[0] if row else 0
    tables = ['sessions', 'documents', 'source_chunks', 'nodes', 'edges', 'node_sources', 'tags', 'node_tags']
    counts = {}
    for t in tables:
        try: counts[t] = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except Exception: counts[t] = 0
    con.close()
    return {"schema_version": version, "record_counts": counts}

@app.get("/api/db/export")
async def export_db():
    if not os.path.exists(DB_PATH): raise HTTPException(404, "Database not found")
    backup_path = DB_PATH + ".backup"
    shutil.copy2(DB_PATH, backup_path)
    return FileResponse(backup_path, media_type="application/x-sqlite3", filename="mindmap_knowledge_base.db")

# ── Phase 1: Document Ingestion Endpoints ──────────────────────────
@app.post("/api/documents/upload")
async def upload_document(req: DocumentUploadRequest):
    try:
        doc_id = ingest_document(req.text, req.filename, req.media_type)
        return {"document_id": doc_id, "status": "success"}
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {str(e)}")

@app.get("/api/documents")
async def list_documents():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT id, filename, media_type, imported_at, status FROM documents ORDER BY imported_at DESC").fetchall()
    con.close()
    return {"documents": [dict(row) for row in rows]}

@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    
    doc = con.execute("SELECT id, filename, media_type, sha256, imported_at, status FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if not doc:
        con.close()
        raise HTTPException(404, "Document not found")
        
    chunks = con.execute("SELECT id, chunk_index, section_title, length(content) as char_count FROM source_chunks WHERE document_id = ? ORDER BY chunk_index", (doc_id,)).fetchall()
    con.close()
    
    return {"document": dict(doc), "chunks": [dict(c) for c in chunks]}

# ── Phase 1.5: Provenance Retrieval Endpoint ───────────────────────────────────
@app.get("/api/nodes/{node_id}/sources")
async def get_node_sources(node_id: str):
    """Retrieve the source chunks linked to a specific node."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    
    # Join node_sources with source_chunks to get the actual content
    rows = con.execute("""
        SELECT ns.chunk_id, ns.citation_text, ns.relevance, 
               sc.content, sc.section_title, sc.chunk_index
        FROM node_sources ns
        JOIN source_chunks sc ON ns.chunk_id = sc.id
        WHERE ns.node_id = ?
        ORDER BY ns.relevance DESC
    """, (node_id,)).fetchall()
    con.close()
    
    sources = [dict(row) for row in rows]
    return {"node_id": node_id, "sources": sources}
