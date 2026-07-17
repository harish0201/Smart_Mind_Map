import os
import uuid
import json
import re
import time
import sqlite3
from typing import Optional, List, Dict, Set
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from openai import AsyncOpenAI

# ── Configuration ──────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-...")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
DB_PATH = "mindmap_sessions.db"

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

llm_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# ── SQLite Persistence ─────────────────────────────────────────────────────────
def _db_init():
    con = sqlite3.connect(DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, data TEXT, updated_at REAL)")
    con.commit()
    con.close()

def _db_save(session_id: str, session: 'GraphSession'):
    _db_init()
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT OR REPLACE INTO sessions (session_id, data, updated_at) VALUES (?,?,?)",
                (session_id, json.dumps(session.to_dict(), ensure_ascii=False), time.time()))
    con.commit()
    con.close()

def _db_load(session_id: str) -> Optional['GraphSession']:
    _db_init()
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT data FROM sessions WHERE session_id=?", (session_id,)).fetchone()
    con.close()
    if row:
        try: return GraphSession.from_dict(json.loads(row[0]))
        except Exception: pass
    return None

# ── Graph Session ──────────────────────────────────────────────────────────────
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

# ── LLM Prompts ────────────────────────────────────────────────────────────────
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

# ── Pydantic Models ────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel): 
    text: str; session_id: Optional[str] = None; model: Optional[str] = None

class ExpandRequest(BaseModel): 
    session_id: str; node_id: str; model: Optional[str] = None

class QueryRequest(BaseModel): 
    session_id: str; node_id: str; query: str; model: Optional[str] = None

class RestoreRequest(BaseModel):
    graph: dict

# ── Helper Functions ───────────────────────────────────────────────────────────
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
    """Restore a session from an exported JSON file."""
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
    
    response = await llm_client.chat.completions.create(
        model=target_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_GENERATE.format(long_text_content=req.text)}
        ],
        temperature=0.5
    )
    md_text = response.choices[0].message.content
    
    gs = parse_markdown_to_session(md_text)
    _db_save(session_id, gs)
    
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
