#!/usr/bin/env python3
"""OpenAI-compatible server + built-in chat UI for a Muse Glimmer checkpoint.

Written instead of using llama.cpp/vLLM because neither serves this checkpoint as-is:
the weights are bf16 safetensors of a `muse_glimmer` arch that vLLM does not implement,
and llama.cpp would need a GGUF conversion first. transformers already loads it.

THE THING A GENERIC SERVER GETS WRONG. This model does not emit one answer. It emits
channelled messages:

    <|start|>assistant to=self<|message|>   ...reasoning...      <|eom|>
    <|start|>assistant to=<tool><|message|> ...tool call...      <|eom|>
    <|start|>assistant to=user<|message|>   ...the answer...     <|eot|>

Pipe the raw decode into a chat UI and you get the model's private deliberation
presented as its reply — which, among other things, is how the first agentic eval in
this repo scored 12/12 false positives. Here `content` is the `to=user` channel and the
reasoning is returned separately as `reasoning` (and rendered collapsed in the UI).

    uv run --python .venv-gpu/bin/python scripts/serve.py --model models/mg-abl-s1.0
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from transformers import AutoTokenizer, TextIteratorStreamer
from transformers.models.muse_glimmer import MuseGlimmerForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent))
from channels import split as split_channels  # noqa: E402

app = FastAPI()
STATE: dict = {}

PAGE = """<!doctype html><meta charset=utf-8><title>Muse Glimmer — %MODEL%</title>
<style>
:root{--bg:#0f1115;--fg:#e6e6e6;--dim:#8b93a7;--acc:#7aa2f7;--card:#171a21;--warn:#e0af68}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:10px 16px;border-bottom:1px solid #262b36;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
header b{color:var(--acc)} header .tag{font-size:12px;color:var(--warn);border:1px solid var(--warn);padding:1px 6px;border-radius:4px}
#log{max-width:900px;margin:0 auto;padding:18px 16px 140px}
.msg{margin:14px 0;padding:12px 14px;border-radius:10px;background:var(--card);white-space:pre-wrap;word-wrap:break-word}
.user{background:#1d2430;border-left:3px solid var(--acc)}
.who{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);margin-bottom:6px}
details{margin-top:10px;border-top:1px solid #262b36;padding-top:8px}
summary{cursor:pointer;color:var(--dim);font-size:12px}
details pre{white-space:pre-wrap;color:var(--dim);font-size:13px;margin:8px 0 0}
footer{position:fixed;bottom:0;left:0;right:0;background:linear-gradient(transparent,var(--bg) 22%);padding:16px}
form{max-width:900px;margin:0 auto;display:flex;gap:8px}
textarea{flex:1;background:var(--card);color:var(--fg);border:1px solid #2b3140;border-radius:10px;padding:11px;font:inherit;resize:vertical;min-height:52px}
button{background:var(--acc);color:#0b0e14;border:0;border-radius:10px;padding:0 20px;font-weight:600;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
.meta{max-width:900px;margin:6px auto 0;color:var(--dim);font-size:12px}
</style>
<header><b>Muse Glimmer</b><span>%MODEL%</span><span class=tag>%TAG%</span>
<span style="color:var(--dim);font-size:12px">reasoning shown collapsed · OpenAI API at /v1/chat/completions</span></header>
<div id=log></div>
<footer><form id=f><textarea id=q placeholder="Message… (Enter to send, Shift+Enter for newline)"></textarea>
<button id=b>Send</button></form><div class=meta id=m></div></footer>
<script>
const log=document.getElementById('log'),q=document.getElementById('q'),b=document.getElementById('b'),m=document.getElementById('m');
let hist=[];
function add(cls,who,txt){const d=document.createElement('div');d.className='msg '+cls;
  d.innerHTML='<div class=who>'+who+'</div><div class=body></div>';d.querySelector('.body').textContent=txt;
  log.appendChild(d);window.scrollTo(0,document.body.scrollHeight);return d;}
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const text=q.value.trim();if(!text)return;
  q.value='';b.disabled=true;add('user','you',text);hist.push({role:'user',content:text});
  const el=add('bot','assistant','');const body=el.querySelector('.body');
  let det=null,pre=null;const t0=performance.now();let n=0;
  const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify({messages:hist,stream:true})});
  const rd=r.body.getReader(),dec=new TextDecoder();let buf='',content='';
  while(true){const{value,done}=await rd.read();if(done)break;
    buf+=dec.decode(value,{stream:true});const lines=buf.split('\\n');buf=lines.pop();
    for(const L of lines){if(!L.startsWith('data: '))continue;const p=L.slice(6);if(p==='[DONE]')continue;
      let j;try{j=JSON.parse(p)}catch(e){continue}
      const d=j.choices[0].delta;n++;
      if(d.reasoning){if(!det){det=document.createElement('details');
          det.innerHTML='<summary>reasoning (to=self)</summary><pre></pre>';el.appendChild(det);pre=det.querySelector('pre');}
        pre.textContent+=d.reasoning;}
      if(d.content){content+=d.content;body.textContent=content;}
      window.scrollTo(0,document.body.scrollHeight);}}
  hist.push({role:'assistant',content:content});
  m.textContent=((performance.now()-t0)/1000).toFixed(1)+'s';b.disabled=false;q.focus();};
q.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();
  document.getElementById('f').requestSubmit();}});
</script>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE.replace("%MODEL%", STATE["name"]).replace("%TAG%", STATE["tag"])


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": STATE["name"], "object": "model"}]}


def build_prompt(messages):
    tok = STATE["tok"]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def stream_tokens(messages, max_new_tokens, temperature):
    tok, model = STATE["tok"], STATE["model"]
    text = build_prompt(messages)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=False)
    kw = dict(**enc, max_new_tokens=max_new_tokens, streamer=streamer,
              pad_token_id=tok.pad_token_id)
    if temperature and temperature > 0:
        kw.update(do_sample=True, temperature=temperature, top_p=0.95, top_k=64)
    else:
        kw.update(do_sample=False)
    threading.Thread(target=model.generate, kwargs=kw, daemon=True).start()

    # Split the stream into channels as it arrives. The `to=user` channel is the answer;
    # everything before it is deliberation and must not be shown as the reply.
    acc = ""
    sent_reason = 0
    sent_content = 0
    for piece in streamer:
        acc += piece
        ch = split_channels(acc)
        if len(ch.reasoning) > sent_reason:
            yield {"reasoning": ch.reasoning[sent_reason:]}
            sent_reason = len(ch.reasoning)
        if len(ch.final) > sent_content:
            yield {"content": ch.final[sent_content:]}
            sent_content = len(ch.final)


@app.post("/v1/chat/completions")
async def chat(req: dict):
    msgs = req.get("messages", [])
    mnt = int(req.get("max_tokens") or req.get("max_new_tokens") or STATE["max_new"])
    temp = req.get("temperature", 0.0)
    created = int(time.time())
    cid = f"chatcmpl-{created}"

    if not req.get("stream"):
        parts = {"content": "", "reasoning": ""}
        for d in stream_tokens(msgs, mnt, temp):
            for k, v in d.items():
                parts[k] += v
        return {"id": cid, "object": "chat.completion", "created": created,
                "model": STATE["name"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": parts["content"],
                                         "reasoning": parts["reasoning"]}}]}

    def sse():
        for d in stream_tokens(msgs, mnt, temp):
            chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                     "model": STATE["name"],
                     "choices": [{"index": 0, "delta": d, "finish_reason": None}]}
            yield f"data: {json.dumps(chunk)}\n\n"
        done = {"id": cid, "object": "chat.completion.chunk", "created": created,
                "model": STATE["name"],
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(done)}\n\ndata: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/mg-abl-s1.0")
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 exposes on the LAN")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()

    p = Path(a.model)
    abl = p / "abliteration.json"
    tag = "BASE (unmodified)"
    if abl.exists():
        m = json.loads(abl.read_text())
        tag = f"ABLITERATED scale={m['scale']} layers={len(m['layers'])} norm_aware={m['norm_aware']}"
    print(f"loading {a.model}  [{tag}]")

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = MuseGlimmerForConditionalGeneration.from_pretrained(
        a.model, dtype=getattr(torch, a.dtype), device_map="auto", attn_implementation="sdpa")
    model.eval()
    STATE.update(tok=tok, model=model, name=p.name, tag=tag, max_new=a.max_new)
    print(f"ready on http://{a.host}:{a.port}  ({tag})")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
