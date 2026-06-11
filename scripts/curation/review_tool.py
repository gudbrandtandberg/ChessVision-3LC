"""Throwaway local web tool for turning model-drafted FENs into confirmed test ground truth.

No ``tlc``, no cloud — a tiny Flask app you run on localhost. For each candidate it shows the
raw photo beside the model's predicted position (rendered as inline SVG, so no cairosvg
dependency). You confirm with a single keystroke:

    A  accept the normal-orientation FEN
    F  accept the flipped (180°) FEN  — for photos taken from black's side
    E  focus the edit box to hand-fix a FEN, then Enter to accept it
    X  reject (not a usable board: blurry, occluded, not a chess position)
    ←  go back to re-review the previous decision

Accepted FENs are written to ``data/test/<batch>/ground_truth/<uuid>.txt`` (single-line board
FEN, matching the existing test-set format) and the UUID is stamped ``test`` in the provenance
ledger. Rejected UUIDs are stamped ``rejected``. Decisions are also mirrored in
``data/test/<batch>/review.json`` so you can stop and resume anytime.

Run:
    .venv/bin/python -m scripts.curation.review_tool --batch harder-v1
    # then open http://127.0.0.1:5001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import chess.svg
from flask import Flask, abort, jsonify, request, send_file

from chessvision import constants
from scripts.curation import provenance

app = Flask(__name__)

# Populated in main().
BATCH_DIR: Path
RAW_DIR: Path
BOARDS_DIR: Path
GT_DIR: Path
PREDICTIONS: dict[str, dict]
ORDER: list[str]  # candidate uuids in stable order


def review_path() -> Path:
    return BATCH_DIR / "review.json"


def load_review() -> dict[str, dict]:
    p = review_path()
    return json.loads(p.read_text()) if p.exists() else {}


def save_review(review: dict[str, dict]) -> None:
    review_path().write_text(json.dumps(dict(sorted(review.items())), indent=2) + "\n")


def render_svg(board_fen: str) -> str:
    """Render a piece-placement FEN to an SVG string, or an error placeholder."""
    try:
        return chess.svg.board(board=chess.BaseBoard(board_fen), size=360)
    except (ValueError, IndexError):
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="360" height="40"><text x="4" y="24" fill="red">invalid FEN: {board_fen}</text></svg>'


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>FEN review — {batch}</title>
<style>
 *{{box-sizing:border-box}}
 html,body{{height:100%}}
 body{{margin:0;height:100vh;overflow:hidden;display:flex;flex-direction:column;
   font-family:system-ui,sans-serif;background:#1d1f21;color:#eee}}
 header{{flex:0 0 auto;padding:6px 16px;background:#111;display:flex;gap:20px;align-items:center}}
 .bar{{height:6px;background:#333;border-radius:3px;flex:1}}
 .bar>div{{height:100%;background:#4caf50;border-radius:3px;width:0;transition:width .15s}}
 main{{flex:1 1 auto;min-height:0;display:flex;gap:1.5vw;justify-content:center;
   align-items:center;padding:10px}}
 .col{{display:flex;flex-direction:column;align-items:center;gap:6px;max-height:100%}}
 main img, main svg{{max-height:74vh;max-width:44vw;width:auto;height:auto;
   background:#2b2b2b;border-radius:6px}}
 .label{{font-size:12px;color:#9aa;letter-spacing:.03em}}
 .label b{{color:#4caf50;font-size:14px}}
 footer{{flex:0 0 auto;display:flex;gap:22px;align-items:center;padding:8px 16px;
   background:#111;border-top:1px solid #333}}
 .keys{{display:flex;gap:14px;font-size:13px;align-items:center;flex-wrap:wrap}}
 kbd{{background:#333;border:1px solid #555;border-radius:4px;padding:1px 6px;font-family:monospace}}
 #crop{{height:80px;image-rendering:pixelated;border-radius:4px;background:#2b2b2b}}
 #flip svg{{height:80px;width:80px;background:#2b2b2b;border-radius:4px}}
 .editwrap{{flex:1;display:flex;gap:10px;align-items:center}}
 input{{font-family:monospace;flex:1;min-width:220px;padding:6px;background:#000;color:#0f0;
   border:1px solid #444;border-radius:4px}}
 .done{{margin:auto;font-size:22px;text-align:center}}
 .badge{{font-size:12px;font-weight:600;padding:2px 10px;border-radius:10px;text-transform:uppercase;letter-spacing:.04em}}
 .badge.pending{{background:#444;color:#bbb}}
 .badge.accepted{{background:#1b5e20;color:#a5d6a7}}
 .badge.rejected{{background:#5d1f1f;color:#ef9a9a}}
</style></head><body>
<header>
 <b>{batch}</b>
 <span id="badge" class="badge pending"></span>
 <span id="counter" class="label"></span>
 <div class="bar"><div id="prog"></div></div>
</header>
<main id="app"></main>
<footer id="controls" style="display:none">
 <div class="keys">
  <span><kbd>A</kbd> accept</span><span><kbd>F</kbd> flipped</span>
  <span><kbd>E</kbd> edit</span><span><kbd>X</kbd> reject</span><span><kbd>←</kbd><kbd>→</kbd> browse</span>
 </div>
 <div class="col"><div class="label">extracted</div><img id="crop"></div>
 <div class="col"><div class="label">flipped <b>F</b></div><div id="flip"></div></div>
 <div class="editwrap">
  <input id="edit" spellcheck="false" placeholder="edit FEN — the A board updates live; Enter to accept, Esc to cancel">
 </div>
</footer>
<script>
let cur=null, idx=0;
const app=document.getElementById('app'), controls=document.getElementById('controls');
async function show(i){{
 const q=(i===undefined)?'':'?i='+i;
 const d=await (await fetch('/api/item'+q)).json();
 cur=d; idx=d.index;
 const badge=document.getElementById('badge');
 badge.textContent=d.status; badge.className='badge '+d.status;
 document.getElementById('counter').textContent=
   (d.index+1)+' / '+d.total+'  ·  '+d.reviewed+' reviewed, '+d.remaining+' left';
 document.getElementById('prog').style.width=(100*d.reviewed/Math.max(d.total,1))+'%';
 controls.style.display='flex';
 app.innerHTML=`
  <div class="col"><div class="label">raw photo</div><img src="/raw/${{d.uuid}}"></div>
  <div class="col"><div class="label">accept <b>A</b> · live edit</div><div id="boardA">${{d.svg_display}}</div></div>`;
 const crop=document.getElementById('crop');
 if(d.has_board){{ crop.style.display=''; crop.src='/board/'+d.uuid; }} else {{ crop.style.display='none'; }}
 document.getElementById('flip').innerHTML=d.svg_flipped||'';
 const e=document.getElementById('edit'), boardA=document.getElementById('boardA');
 e.value=d.fen||'';
 e.oninput=async()=>{{ boardA.innerHTML=await (await fetch('/api/svg?fen='+encodeURIComponent(e.value))).text(); }};
 e.onkeydown=ev=>{{
   if(ev.key==='Enter'){{ev.preventDefault(); decide('accepted',e.value);}}
   else if(ev.key==='Escape'){{e.blur();}}
   ev.stopPropagation();
 }};
 e.blur();
}}
async function decide(status,fen){{
 if(!cur)return;
 await fetch('/api/decide',{{method:'POST',headers:{{'Content-Type':'application/json'}},
   body:JSON.stringify({{uuid:cur.uuid,status:status,fen:fen}})}});
 show(idx+1);  // record, then advance
}}
document.addEventListener('keydown',ev=>{{
 if(document.activeElement && document.activeElement.id==='edit') return;
 if(!cur)return;
 const k=ev.key.toLowerCase();
 if(k==='a') decide('accepted',document.getElementById('edit').value);
 else if(k==='f') decide('accepted',cur.fen_flipped);
 else if(k==='x') decide('rejected',null);
 else if(k==='e'){{ev.preventDefault(); document.getElementById('edit').focus();}}
 else if(ev.key==='ArrowLeft'){{ev.preventDefault(); show(idx-1);}}
 else if(ev.key==='ArrowRight'){{ev.preventDefault(); show(idx+1);}}
}});
show();  // resume at first pending
</script></body></html>"""


@app.route("/")
def index() -> str:
    return PAGE.format(batch=BATCH_DIR.name)


@app.route("/api/item")
def api_item():
    """Return the candidate at index ``i`` (clamped), with its current review status.

    With no ``i``, resume at the first still-pending candidate (or 0 if all reviewed).
    Navigation is non-destructive: browsing never changes a decision; re-deciding overwrites.
    """
    review = load_review()
    reviewed = sum(1 for u in ORDER if u in review)
    total = len(ORDER)
    i = request.args.get("i", type=int)
    if i is None:
        i = next((k for k, u in enumerate(ORDER) if u not in review), 0)
    i = max(0, min(i, total - 1))

    uuid = ORDER[i]
    pred = PREDICTIONS.get(uuid, {})
    entry = review.get(uuid, {})
    fen_flipped = pred.get("fen_flipped")
    # Show the confirmed FEN if this item was already accepted, else the model draft.
    display_fen = entry.get("fen") or pred.get("fen")
    return jsonify(
        index=i,
        uuid=uuid,
        status=entry.get("status", "pending"),
        total=total,
        reviewed=reviewed,
        remaining=total - reviewed,
        has_board=pred.get("extraction_ok", False),
        fen=display_fen,
        fen_flipped=fen_flipped,
        svg_display=render_svg(display_fen) if display_fen else "<i>extraction failed — reject or hand-enter a FEN</i>",
        svg_flipped=render_svg(fen_flipped) if fen_flipped else "",
    )


@app.route("/api/svg")
def api_svg() -> str:
    return render_svg(request.args.get("fen", ""))


@app.route("/api/decide", methods=["POST"])
def api_decide():
    data = request.get_json(force=True)
    uuid, status, fen = data["uuid"], data["status"], data.get("fen")
    if uuid not in PREDICTIONS:
        abort(404)

    ledger = provenance.load_ledger()
    gt_file = GT_DIR / f"{uuid}.txt"
    if status == "accepted" and fen:
        # Validate before committing — a bad FEN should never reach ground truth.
        try:
            chess.BaseBoard(fen)
        except (ValueError, IndexError):
            return jsonify(ok=False, error="invalid FEN"), 400
        GT_DIR.mkdir(parents=True, exist_ok=True)
        gt_file.write_text(fen + "\n")
        provenance.record(ledger, uuid, provenance.TEST, batch=BATCH_DIR.name)
    else:
        status = "rejected"
        gt_file.unlink(missing_ok=True)
        provenance.record(ledger, uuid, provenance.REJECTED, batch=BATCH_DIR.name)
    provenance.save_ledger(ledger)

    review = load_review()
    review[uuid] = {"status": status, "fen": fen if status == "accepted" else None}
    save_review(review)
    return jsonify(ok=True)


@app.route("/raw/<uuid>")
def raw(uuid: str):
    return send_file(RAW_DIR / f"{uuid}.JPG")


@app.route("/board/<uuid>")
def board(uuid: str):
    p = BOARDS_DIR / f"{uuid}.png"
    return send_file(p) if p.exists() else ("", 404)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    global BATCH_DIR, RAW_DIR, BOARDS_DIR, GT_DIR, PREDICTIONS, ORDER
    BATCH_DIR = constants.DATA_ROOT / "test" / args.batch
    RAW_DIR = BATCH_DIR / "raw"
    BOARDS_DIR = BATCH_DIR / "boards"
    GT_DIR = BATCH_DIR / "ground_truth"
    pred_path = BATCH_DIR / "predictions.json"
    if not pred_path.exists():
        raise SystemExit(f"No predictions.json for batch {args.batch!r}; run predict_test_candidates first.")
    PREDICTIONS = json.loads(pred_path.read_text())
    ORDER = sorted(p.stem for p in RAW_DIR.glob("*.JPG"))

    print(f"Reviewing {len(ORDER)} candidates for batch {args.batch!r}. Open http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
