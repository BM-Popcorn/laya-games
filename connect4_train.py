"""Fine-tune the typed-decision checkpoint on engine-labelled Connect Four positions.

The loss is the upstream RLCD recipe: noisy-logit policy gradient with
`proper_reward`, plus soft cross-entropy on the engine's target distribution.
Forced (one-hot) rows get stronger CE and ~50% of each micro-batch so tactics
are learned sharply. A CUDA device trains the whole network. CPU training
freezes the encoder and updates the decision head, because ModernBERT-large
backward is not practical on CPU.
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


def is_forced_target(target):
    return max(target) >= 0.999 and sum(1 for v in target if v > 0) == 1


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
            "forced": is_forced_target(target),
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
        "forced": torch.tensor([bool(it.get("forced")) for it in items]),
    }


def loss_on(model, batch, device, sigma, group_size, rl_weight=0.25, ce_weight=1.0, forced_ce_weight=2.0):
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
    forced = batch["forced"].to(device)
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
    log_probs = torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)
    ce_per = -(target * log_probs).sum(-1)
    ce_w = torch.where(forced, torch.full_like(ce_per, forced_ce_weight), torch.full_like(ce_per, ce_weight))
    loss_ce = (ce_w * ce_per).mean()
    return rl_weight * loss_rl + loss_ce + 0.0 * act.sum(), reward.mean().detach()


def holdout_agreement(model, items, tok, device, limit):
    """Fraction of held-out positions whose top column matches the engine target.

    A tie in the target counts as a hit, matching connect4_measure.py. Argmax
    is invariant to temperature, so this is the same decision the loader plays.
    """
    was_training = model.training
    model.eval()
    take = items[:limit] if limit else items
    hits = forced_hits = forced_n = soft_hits = soft_n = 0
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
            hit = it["target"][pred] == max(it["target"])
            if hit:
                hits += 1
            if it.get("forced"):
                forced_n += 1
                if hit:
                    forced_hits += 1
            else:
                soft_n += 1
                if hit:
                    soft_hits += 1
    if was_training:
        model.train()
    return {
        "hits": hits,
        "n": len(take),
        "forced_hits": forced_hits,
        "forced_n": forced_n,
        "soft_hits": soft_hits,
        "soft_n": soft_n,
    }


def save_checkpoint(model, tok, cfg, updates, hours, agreement=None):
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
        "best_agreement": agreement,
    }
    (OUT / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def build_epoch_items(forced_items, soft_items, forced_frac, rng):
    """Shuffle one epoch, oversampling forced rows toward forced_frac of the stream.

    Uses each unique row at least once when possible, then tops up with forced
    copies so tactics stay dense without random-with-replacement minibatches.
    """
    if forced_frac >= 1.0 or not soft_items:
        epoch = list(forced_items) if forced_items else list(soft_items)
        rng.shuffle(epoch)
        return epoch
    if forced_frac <= 0.0 or not forced_items:
        epoch = list(soft_items) if soft_items else list(forced_items)
        rng.shuffle(epoch)
        return epoch
    epoch = list(soft_items) + list(forced_items)
    target_forced = int(round(forced_frac * (len(soft_items) / max(1e-6, 1.0 - forced_frac))))
    extra = max(0, target_forced - len(forced_items))
    if extra:
        epoch.extend(forced_items[i % len(forced_items)] for i in range(extra))
    rng.shuffle(epoch)
    return epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=0, help="micro-batch. Default 1 on CUDA so an 8 GB GPU fits")
    parser.add_argument("--accum", type=int, default=0, help="gradient accumulation steps. Default 8 on CUDA")
    parser.add_argument("--eval-every", type=int, default=50, help="optimizer steps between held-out checks. 0 disables them")
    parser.add_argument("--eval-limit", type=int, default=0, help="held-out positions per check. 0 uses the full holdout")
    parser.add_argument("--limit", type=int, default=0, help="train on the first N rows, 0 uses all")
    parser.add_argument("--forced-frac", type=float, default=0.5, help="target fraction of forced rows in each epoch stream")
    parser.add_argument("--rl-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--forced-ce-weight", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=8, help="stop after this many evals with no holdout improvement")
    parser.add_argument(
        "--init",
        default="typed-decisions",
        help="checkpoint folder under models/ to start from (typed-decisions or connect4)",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="update the decision head only (sharper forced-move fitting on 8 GB cards)",
    )
    args = parser.parse_args()

    rows = load_rows(TRAIN_PATH)
    if args.limit:
        rows = rows[: args.limit]
    init_dir = ROOT / "models" / args.init
    if not (init_dir / "model.safetensors").is_file():
        raise SystemExit(f"No checkpoint at {init_dir}")
    with (init_dir / "rl_agent_config.json").open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    tok = AutoTokenizer.from_pretrained(init_dir / "tokenizer")
    items = build_items(rows, tok, cfg)
    if not items:
        raise SystemExit(f"No training items in {TRAIN_PATH}. Run connect4_dataset.py first.")
    holdout = build_items(load_rows(HOLDOUT_PATH), tok, cfg) if HOLDOUT_PATH.is_file() else []
    forced_items = [it for it in items if it["forced"]]
    soft_items = [it for it in items if not it["forced"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = device.type == "cuda"
    model = build_model(cfg, encoder_dir=str(init_dir / "encoder"), pretrained=False)
    model.load_state_dict(load_file(str(init_dir / "model.safetensors")), strict=True)
    freeze_encoder = args.freeze_encoder or not cuda
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        model.encoder.eval()
    else:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    model.to(device)
    model.train()
    if freeze_encoder:
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
        f"accum {accum} | {optim_name} | encoder frozen: {freeze_encoder} | "
        f"forced {len(forced_items)} soft {len(soft_items)} | forced-frac {args.forced_frac} | "
        f"rl={args.rl_weight} ce={args.ce_weight}/{args.forced_ce_weight} | "
        f"holdout check every {args.eval_every} updates (full={args.eval_limit == 0})"
    )
    started = time.time()
    updates = 0
    micro = 0
    seen = 0
    best_rate = -1.0
    stale_evals = 0

    def report(loss_value):
        nonlocal best_rate, stale_evals
        stats = holdout_agreement(model, holdout, tok, device, args.eval_limit)
        hits, n = stats["hits"], stats["n"]
        rate = hits / max(1, n)
        forced_rate = stats["forced_hits"] / max(1, stats["forced_n"])
        soft_rate = stats["soft_hits"] / max(1, stats["soft_n"])
        score = forced_rate if args.forced_frac >= 1.0 else rate
        print(
            f"  holdout {hits}/{n} = {rate:.3f} "
            f"(forced {stats['forced_hits']}/{stats['forced_n']} = {forced_rate:.3f}, "
            f"soft {stats['soft_hits']}/{stats['soft_n']} = {soft_rate:.3f}) "
            f"after {updates} updates"
        )
        PROGRESS_PATH.parent.mkdir(exist_ok=True)
        with PROGRESS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "updates": updates,
                "rows": seen,
                "loss": round(loss_value, 4),
                "holdout_hits": hits,
                "holdout_n": n,
                "agreement": round(rate, 4),
                "forced_hits": stats["forced_hits"],
                "forced_n": stats["forced_n"],
                "forced_agreement": round(forced_rate, 4),
                "soft_hits": stats["soft_hits"],
                "soft_n": stats["soft_n"],
                "soft_agreement": round(soft_rate, 4),
                "early_stop_score": round(score, 4),
            }) + "\n")
        improved = score > best_rate + 1e-6
        if improved:
            best_rate = score
            stale_evals = 0
            save_checkpoint(model, tok, cfg, updates, (time.time() - started) / 3600, agreement=rate)
            print(f"  saved best checkpoint (score {score:.3f}, overall {rate:.3f})")
        else:
            stale_evals += 1
            print(f"  no improvement (best {best_rate:.3f}, stale {stale_evals}/{args.patience})")
        if freeze_encoder:
            model.encoder.eval()
        return improved

    stop = False
    for epoch in range(args.epochs):
        epoch_rng = random.Random(42 + epoch)
        epoch_items = build_epoch_items(forced_items, soft_items, args.forced_frac, epoch_rng)
        total = 0.0
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(epoch_items), batch_size):
            chunk = epoch_items[start:start + batch_size]
            batch = collate(chunk, tok.pad_token_id)
            if cuda:
                with torch.autocast("cuda", dtype=torch.float16):
                    loss, reward = loss_on(
                        model, batch, device, sigma=0.2, group_size=4,
                        rl_weight=args.rl_weight, ce_weight=args.ce_weight,
                        forced_ce_weight=args.forced_ce_weight,
                    )
            else:
                loss, reward = loss_on(
                    model, batch, device, sigma=0.2, group_size=2,
                    rl_weight=args.rl_weight, ce_weight=args.ce_weight,
                    forced_ce_weight=args.forced_ce_weight,
                )
            if cuda:
                scaler.scale(loss / accum).backward()
            else:
                (loss / accum).backward()
            micro += 1
            seen += len(chunk)
            total += loss.item()
            if micro % accum != 0 and start + batch_size < len(epoch_items):
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
                print(f"  update {updates} | rows {seen}/{len(epoch_items)} | loss {loss.item():.4f} | reward {reward.item():.3f}")
            if args.eval_every and holdout and updates % args.eval_every == 0:
                report(loss.item())
                if args.patience and stale_evals >= args.patience:
                    print(f"Early stop: no holdout improvement for {args.patience} evals (best {best_rate:.3f})")
                    stop = True
                    break
        print(f"Epoch {epoch + 1} mean loss {total / max(1, micro):.4f}")
        if stop:
            break
    if best_rate < 0 and holdout:
        report(total / max(1, micro))
    elif holdout and (not args.eval_every or updates % args.eval_every != 0):
        report(total / max(1, micro))
    print(f"Saved {OUT} (best agreement {best_rate:.3f})")


if __name__ == "__main__":
    main()
