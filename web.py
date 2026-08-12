"""
Minimal web UI for the Mini RAG chatbot.

Serves a single-page chat interface and a JSON API, reusing the same RAG engine
as the CLI. Run:  python web.py   then open http://127.0.0.1:5000

Features: retrieval-mode selector, inline citations, contradiction flags,
click-to-expand source text, chat history persisted across reloads, and
follow-up questions that carry the previous question's context.

Env: HOST (default 127.0.0.1), PORT (default 5000). Only extra dep: flask.
"""
from __future__ import annotations

import os

from flask import Flask, jsonify, request

from app import smalltalk_reply
from rag import RAG, Config, STOPWORDS, tokenize

MODES = ("hybrid_rerank", "hybrid", "dense", "bm25")
FOLLOWUP_CUES = ("and ", "what about", "how about", "and for", "for ", "then ",
                 "what if", "also ", "same for", "ok ", "okay ")
FOLLOWUP_PRONOUNS = {"it", "that", "those", "these", "they", "them", "this", "one", "ones"}

app = Flask(__name__)
rag = RAG(Config())
_n_chunks = rag.build_index()  # build/load once at startup


def maybe_expand(query: str, prev: str | None) -> str:
    """Carry the previous question's context into a short follow-up so
    'and for 2024?' retrieves against the earlier topic."""
    if not prev:
        return query
    ql = query.lower().strip()
    toks = tokenize(query)
    content = [t for t in toks if t not in STOPWORDS]
    starts_with_cue = any(ql.startswith(c) for c in FOLLOWUP_CUES)
    has_pronoun_ref = any(t in FOLLOWUP_PRONOUNS for t in toks)
    # A follow-up is signalled by a cue word or an anaphoric pronoun — not merely
    # by being short (so a complete question like "what is BM25" isn't expanded).
    is_followup = starts_with_cue or (has_pronoun_ref and len(content) <= 3)
    return f"{prev} {query}" if is_followup else query


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Mini RAG Chatbot</title>
<style>
  :root { --bg:#0f1117; --panel:#171a23; --panel2:#1e222d; --line:#2a2f3a;
          --text:#e6e8ee; --muted:#9aa3b2; --accent:#6ea8fe; --warn:#f0a500; --good:#57cc99; }
  * { box-sizing:border-box; }
  body { margin:0; font:15px/1.55 system-ui,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--text); }
  header { padding:14px 20px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:12px; }
  header h1 { font-size:16px; margin:0; font-weight:600; }
  header .meta { color:var(--muted); font-size:12px; margin-left:auto; }
  header button.clear { background:var(--panel2); color:var(--muted); border:1px solid var(--line);
          border-radius:8px; padding:6px 10px; font-size:12px; cursor:pointer; }
  main { max-width:820px; margin:0 auto; padding:18px; }
  #chat { display:flex; flex-direction:column; gap:14px; min-height:52vh; }
  .msg { max-width:88%; padding:12px 14px; border-radius:12px; white-space:pre-wrap; }
  .user { align-self:flex-end; background:#2a3550; border:1px solid #34406a; }
  .bot  { align-self:flex-start; background:var(--panel); border:1px solid var(--line); }
  .bot .answer div { margin:2px 0; }
  .cite { color:var(--accent); font-weight:600; cursor:pointer; }
  .warn { margin-top:10px; padding:10px 12px; background:rgba(240,165,0,.10);
          border:1px solid rgba(240,165,0,.45); border-radius:10px; color:#ffd36b; font-size:14px; }
  .warn b { color:var(--warn); }
  .sources { margin-top:10px; font-size:12.5px; color:var(--muted); border-top:1px dashed var(--line); padding-top:8px; }
  .src-row { cursor:pointer; padding:2px 0; }
  .src-row:hover { color:var(--text); }
  .src-row .cited { color:var(--good); }
  .src-text { display:none; white-space:pre-wrap; margin:6px 0 10px; padding:9px 11px;
          background:var(--panel2); border:1px solid var(--line); border-left:3px solid var(--accent);
          border-radius:8px; color:var(--text); font-size:12.5px; }
  .src-text.open { display:block; }
  .tag { display:inline-block; font-size:11px; color:var(--muted); margin-top:6px; }
  form { position:sticky; bottom:0; display:flex; gap:8px; padding:14px 0 4px;
         background:linear-gradient(transparent, var(--bg) 22%); margin-top:18px; }
  select, input, button { font:inherit; border-radius:10px; border:1px solid var(--line);
          background:var(--panel2); color:var(--text); padding:11px 12px; }
  input { flex:1; }
  button.send { background:var(--accent); color:#08131f; font-weight:600; border:none; cursor:pointer; padding:11px 18px; }
  button.send:disabled { opacity:.5; cursor:default; }
  .hint { color:var(--muted); font-size:12px; margin:2px 0 0; }
</style>
</head>
<body>
<header>
  <h1>🔎 Mini RAG Chatbot</h1>
  <span class="meta" id="meta"></span>
  <button class="clear" id="clear">Clear</button>
</header>
<main>
  <div id="chat"></div>
  <form id="f">
    <select id="mode" title="Retrieval mode">
      <option value="hybrid_rerank">hybrid + rerank</option>
      <option value="hybrid">hybrid</option>
      <option value="dense">dense</option>
      <option value="bm25">bm25</option>
    </select>
    <input id="q" placeholder="Ask a question…" autocomplete="off" autofocus/>
    <button class="send" id="send">Send</button>
  </form>
  <p class="hint">Runs fully offline · grounded + cited · flags contradictions · click a source to expand it · follow-ups keep context</p>
</main>
<script>
const chat = document.getElementById('chat');
const form = document.getElementById('f');
const qin  = document.getElementById('q');
const send = document.getElementById('send');
const modeSel = document.getElementById('mode');
const KEY = 'ragchat.v1';
let history = [];            // [{role:'user'|'bot', payload}]
let lastUserQuery = null;

fetch('/api/status').then(r=>r.json()).then(s=>{
  document.getElementById('meta').textContent = `backend: ${s.backend} · ${s.chunks} chunks`;
});

function esc(s){ return (s||'').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function citeify(s){ return esc(s).replace(/\\[(\\d+)\\]/g, '<span class="cite" data-n="$1">[$1]</span>'); }
function scrollEnd(){ window.scrollTo(0, document.body.scrollHeight); }

function greeting(){
  const d = document.createElement('div'); d.className='msg bot';
  d.innerHTML = 'Ask me about the documents in the knowledge base — e.g. '
    + '<i>"What is hybrid search?"</i> or <i>"How many PTO days do employees get?"</i><br>'
    + 'Answers are grounded in the sources and cite them; conflicting sources are flagged. '
    + 'Try a follow-up like <i>"and for 2024?"</i>.';
  chat.appendChild(d);
}

function userNode(text){
  const d = document.createElement('div'); d.className='msg user'; d.textContent = text; return d;
}

function botNode(data){
  const d = document.createElement('div'); d.className='msg bot';
  if (data.kind === 'smalltalk'){ d.textContent = data.answer; return d; }

  const ans = document.createElement('div'); ans.className='answer';
  (data.answer || '').split('\\n').forEach(line=>{
    if(!line.trim()) return;
    const row = document.createElement('div'); row.innerHTML = citeify(line); ans.appendChild(row);
  });
  d.appendChild(ans);

  if (data.contradictions && data.contradictions.length){
    const w = document.createElement('div'); w.className='warn';
    w.innerHTML = '<b>⚠ Contradictions detected</b>';
    data.contradictions.forEach(c=>{ const li=document.createElement('div'); li.innerHTML='• '+citeify(c); w.appendChild(li); });
    d.appendChild(w);
  }

  if (data.sources && data.sources.length && !data.insufficient){
    const used = new Set(data.used_sources || []);
    const s = document.createElement('div'); s.className='sources';
    const head = document.createElement('div'); head.textContent='Sources (click to expand):'; s.appendChild(head);
    data.sources.forEach(src=>{
      const row = document.createElement('div'); row.className='src-row'; row.dataset.n = src.n;
      const mark = used.has(src.n) ? '★' : '·';
      row.innerHTML = `<span class="${used.has(src.n)?'cited':''}">${mark} [${src.n}] ${esc(src.source)} :: ${esc(src.section)}</span>`;
      const txt = document.createElement('div'); txt.className='src-text'; txt.dataset.n = src.n; txt.textContent = src.text || '';
      s.appendChild(row); s.appendChild(txt);
    });
    d.appendChild(s);
  }
  const tag = document.createElement('div'); tag.className='tag'; tag.textContent='mode: '+data.mode; d.appendChild(tag);
  return d;
}

// expand source text when a source row OR an inline [n] citation is clicked
chat.addEventListener('click', (e)=>{
  const cite = e.target.closest('.cite');
  const row = e.target.closest('.src-row');
  const n = cite ? cite.dataset.n : (row ? row.dataset.n : null);
  if(!n) return;
  const msg = e.target.closest('.msg');
  const txt = msg && msg.querySelector(`.src-text[data-n="${n}"]`);
  if(txt){ txt.classList.toggle('open'); if(txt.classList.contains('open')) txt.scrollIntoView({block:'nearest'}); }
});

function save(){ localStorage.setItem(KEY, JSON.stringify(history)); }
function render(){
  chat.innerHTML='';
  if(!history.length){ greeting(); return; }
  history.forEach(t=> chat.appendChild(t.role==='user' ? userNode(t.payload) : botNode(t.payload)));
  scrollEnd();
}
function load(){
  try{ history = JSON.parse(localStorage.getItem(KEY)) || []; }catch(e){ history=[]; }
  lastUserQuery = [...history].reverse().find(t=>t.role==='user')?.payload || null;
  render();
}

document.getElementById('clear').addEventListener('click', ()=>{
  history=[]; lastUserQuery=null; save(); render();
});

form.addEventListener('submit', async (e)=>{
  e.preventDefault();
  const q = qin.value.trim(); if(!q) return;
  history.push({role:'user', payload:q}); chat.appendChild(userNode(q)); save();
  qin.value=''; send.disabled=true;
  const thinking = document.createElement('div'); thinking.className='msg bot'; thinking.textContent='…';
  chat.appendChild(thinking); scrollEnd();
  try{
    const r = await fetch('/api/ask', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({query:q, mode:modeSel.value, prev:lastUserQuery})});
    const data = await r.json(); thinking.remove();
    chat.appendChild(botNode(data)); history.push({role:'bot', payload:data}); save();
    scrollEnd();
  }catch(err){ thinking.textContent='Error: '+err; }
  lastUserQuery = q;
  send.disabled=false; qin.focus();
});

load();
</script>
</body>
</html>"""


@app.get("/")
def index():
    return PAGE


@app.get("/api/status")
def status():
    return jsonify({"backend": rag.cfg.backend, "chunks": len(rag.chunks)})


@app.post("/api/ask")
def ask():
    data = request.get_json(force=True) or {}
    query = (data.get("query") or "").strip()
    mode = data.get("mode") if data.get("mode") in MODES else "hybrid_rerank"
    prev = (data.get("prev") or "").strip() or None
    if not query:
        return jsonify({"kind": "smalltalk", "answer": "Please type a question."})

    chit = smalltalk_reply(query)
    if chit:
        return jsonify({"kind": "smalltalk", "answer": chit})

    result = rag.answer(maybe_expand(query, prev), mode=mode)
    result["kind"] = "answer"
    return jsonify(result)


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    print(f"Serving Mini RAG Chatbot ({rag.cfg.backend} backend, {_n_chunks} chunks)")
    print(f"Open http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
