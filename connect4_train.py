"""Fine-tune the typed-decision checkpoint on engine-labelled Connect Four positions.

The loss is the upstream RLCD recipe: noisy-logit policy gradient with
`proper_reward`, plus soft cross-entropy on the engine's target distribution.
A CUDA device trains the whole network. CPU training freezes the encoder and
updates the decision head, because ModernBERT-large backward is not practical
on CPU.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from laya.common import QTYPES, build_model, build_sequence, proper_reward

import connect4_laya

ROOT = Path(__file__).parent
BASE = ROOT / "models" / "typed-decisions"
OUT = ROOT / "models" / "connect4"
TRAIN_PATH = ROOT / "data" / "connect4_train.jsonl"
HOLDOUT_PATH = ROOT / "data" / "connect4_holdout.jsonl"
PROGRESS_PATH = ROOT / "data" / "connect4_progress.jsonl"


def load_rows(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_items(rows, tok, cfg):
    items = []
    for row in rows:
        state, actions = connect4_laya.state_and_actions(row["board"], row["player"])
        question = connect4_laya.questions_for(actions)["next_move"]
        crit = question["criteria"]
        keys = list(crit)
        target = [float(row["target"].get(key, 0.0)) for key in keys]
        total = sum(target)
        target = [v / total for v in target] if total else [1.0 / len(target)] * len(target)
        seq, markers = build_sequence(
            tok,
            state,
            {"t": "choice", "ins": question["instructions"], "crit": crit},
            cfg["max_len"],
            cfg["head_max_len"],
        )
        if len(markers) != len(keys):
            continue
        items.append({
            "ids": seq,
            "markers": markers,
            "qtype": QTYPES["choice"],
            "target": target,
            "label": target.index(max(target)),
        })
    return items


def collate(items, pad_id):
    n, length = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, length), pad_id, dtype=torch.long)
    attention = torch.zeros((n, length), dtype=torch.long)
    marker_pos = torch.zeros((n, kmax), dtype=torch.long)
    marker_mask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        attention[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        marker_pos[i, :k] = torch.tensor(it["markers"])
        marker_mask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
    }


def loss_on(model, batch, device, sigma, group_size):
    logits, act = model(
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["marker_pos"].to(device),
        batch["marker_mask"].to(device),
        batch["qtype"].to(device),
        detach_encoder=not any(p.requires_grad for p in model.encoder.parameters()),
    )
    logits = logits.float()
    mask = batch["marker_mask"].to(device)
    target = batch["target"].to(device)
    k = mask.sum(-1, keepdim=True).float().clamp(min=1)
    eps = torch.randn((group_size,) + logits.shape, device=device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        reward = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask.float(), w_sph=0.75)
        adv = reward - reward.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    loss_rl = -(adv * logp).mean()
    loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    return loss_rl + loss_ce + 0.0 * act.sum(), reward.mean().detach()


def holdout_agreement(model, items, tok, device, limit):
    """Fraction of held-out positions whose top column matches the engine target.

    A tie in the target counts as a hit, matching connect4_measure.py. Argmax
    is invariant to temperature, so this is the same decision the loader plays.
    """
    was_training = model.training
    model.eval()
    take = items[:limit] if limit else items
    hits = 0
    with torch.no_grad():
        for it in take:
            batch = collate([it], tok.pad_token_id)
            kwargs = dict(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                marker_pos=batch["marker_pos"].to(device),
                marker_mask=batch["marker_mask"].to(device),
                qtype=batch["qtype"].to(device),
            )
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    logits, _ = model(**kwargs)
            else:
                logits, _ = model(**kwargs)
            k = len(it["markers"])
            pred = int(logits[0, :k].argmax())
            if it["target"][pred] == max(it["target"]):
                hits += 1
    if was_training:
        model.train()
    return hits, len(take)


def save_checkpoint(model, tok, cfg, updates, hours):
    OUT.mkdir(parents=True, exist_ok=True)
    weights = {k: v.detach().half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(weights, str(OUT / "model.safetensors"))
    model.encoder.config.save_pretrained(OUT / "encoder")
    tok.save_pretrained(OUT / "tokenizer")
    cfg = dict(cfg)
    cfg["fine_tuned"] = True
    cfg["fine_tuned_from_checkpoint"] = True
    cfg["model_name"] = "laya-connect4"
    cfg["training"] = {
        "updates": updates,
        "task": "connect4",
        "hours": round(hours, 3),
        "world_size": 1,
        "encoder_frozen": not any(p.requires_grad for p in model.encoder.parameters()),
    }
    (OUT / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=0, help="micro-batch. Default 1 on CUDA so an 8 GB GPU fits")
    parser.add_argument("--accum", type=int, default=0, help="gradient accumulation steps. Default 8 on CUDA")
    parser.add_argument("--eval-every", type=int, default=50, help="optimizer steps between held-out checks. 0 disables them")
    parser.add_argument("--eval-limit", type=int, default=40, help="held-out positions per check, matching the earlier 40-position score")
    parser.add_argument("--limit", type=int, default=0, help="train on the first N rows, 0 uses all")
    args = parser.parse_args()

    rows = load_rows(TRAIN_PATH)
    if args.limit:
        rows = rows[: args.limit]
    with (BASE / "rl_agent_config.json").open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    tok = AutoTokenizer.from_pretrained(BASE / "tokenizer")
    items = build_items(rows, tok, cfg)
    if not items:
        raise SystemExit(f"No training items in {TRAIN_PATH}. Run connect4_dataset.py first.")
    holdout = build_items(load_rows(HOLDOUT_PATH), tok, cfg) if HOLDOUT_PATH.is_file() else []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = device.type == "cuda"
    model = build_model(cfg, encoder_dir=str(BASE / "encoder"), pretrained=False)
    model.load_state_dict(load_file(str(BASE / "model.safetensors")), strict=True)
    if not cuda:
        for param in model.encoder.parameters():
            param.requires_grad = False
        model.encoder.eval()
    else:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    model.to(device)
    model.train()
    if not cuda:
        model.encoder.eval()

    head = [p for p in model.parameters() if p.requires_grad]
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": 2.5e-5})
    groups.append({"params": [p for p in head if all(p is not q for q in encoder_params)], "lr": 1e-4})
    batch_size = args.batch_size or (1 if cuda else 4)
    accum = args.accum or (8 if cuda else 1)
    optim_name = "adamw"
    if cuda:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(groups, weight_decay=0.01)
            optim_name = "adamw8bit"
        except ImportError:
            optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
            print("bitsandbytes is not installed. Full AdamW may run out of memory on an 8 GB GPU.")
            print("Install it with: .\\.venv\\Scripts\\python.exe -m pip install bitsandbytes")
    else:
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    print(
        f"Training {len(items)} positions on {device.type} | micro-batch {batch_size} | "
        f"accum {accum} | {optim_name} | encoder frozen: {not cuda} | "
        f"holdout check every {args.eval_every} updates"
    )
    started = time.time()
    updates = 0
    micro = 0
    seen = 0

    def report(loss_value):
        hits, n = holdout_agreement(model, holdout, tok, device, args.eval_limit)
        rate = hits / max(1, n)
        print(f"  holdout {hits}/{n} = {rate:.3f} after {updates} updates")
        PROGRESS_PATH.parent.mkdir(exist_ok=True)
        with PROGRESS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "updates": updates,
                "rows": seen,
                "loss": round(loss_value, 4),
                "holdout_hits": hits,
                "holdout_n": n,
                "agreement": round(rate, 4),
            }) + "\n")
        save_checkpoint(model, tok, cfg, updates, (time.time() - started) / 3600)
        if not cuda:
            model.encoder.eval()

    for epoch in range(args.epochs):
        random.Random(42 + epoch).shuffle(items)
        total = 0.0
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            batch = collate(chunk, tok.pad_token_id)
            if cuda:
                with torch.autocast("cuda", dtype=torch.float16):
                    loss, reward = loss_on(model, batch, device, sigma=0.2, group_size=4)
            else:
                loss, reward = loss_on(model, batch, device, sigma=0.2, group_size=2)
            if cuda:
                scaler.scale(loss / accum).backward()
            else:
                (loss / accum).backward()
            micro += 1
            seen += len(chunk)
            total += loss.item()
            if micro % accum != 0 and start + batch_size < len(items):
                continue
            if cuda:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head, 1.0)
            if cuda:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            if updates % 20 == 0:
                print(f"  update {updates} | rows {seen}/{len(items)} | loss {loss.item():.4f} | reward {reward.item():.3f}")
            if args.eval_every and holdout and updates % args.eval_every == 0:
                report(loss.item())
        print(f"Epoch {epoch + 1} mean loss {total / max(1, micro):.4f}")
    report(total / max(1, micro))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
