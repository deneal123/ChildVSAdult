"""Сильные бэкбоны: не «нет запаса», а полный файнтюн их ломает. Адаптер вместо дообучения.

    uv run python scripts/adapter_strong.py --backbones adaface_ir101 arcface_r100 adaface_ir50

ПОЧЕМУ. Таблица 10 статьи (TNNLS) показывает, что дообучение на наших парах даёт +0.112 на слабом
FaceNet, но −0.026 / −0.039 на сильных IR-101 / r100, и мы заключили «у сильных нет cross-age
headroom». Но измеряли мы другое: ПОЛНЫЙ файнтюн сильной сети на ~15k шумных пар со сдвигом домена.
Это классическое катастрофическое забывание, а не отсутствие запаса. Два разных утверждения.

Статья исключила альтернативу заранее (main.tex:193): «сравнение адаптера поверх ЗАМОРОЖЕННОГО
сильного ArcFace невалидно — адаптер просто деформирует почти идеальные эмбеддинги». Возражение
про атрибуцию законное, но лечится оно КОНТРОЛЕМ, а не исключением.

ЧТО ДЕЛАЕМ. Замороженный сильный бэкбон + лёгкий residual-MLP-адаптер (512→512), обученный тем же
контрастивом с age-gap-весами. Бэкбон не меняется вовсе, поэтому забыть он физически не может
(проверяется по LFW).

КОНТРОЛЬ (прямой ответ на возражение main.tex:193). Тот же адаптер, та же архитектура, столько же
шагов оптимизации, но у ПОЛОЖИТЕЛЬНЫХ пар вторая сторона переставлена случайно — сигнал «тот же
человек через годы» разрушен, всё остальное сохранено.
    растёт и на перемешанных -> прирост от ЁМКОСТИ (возражение статьи было верным);
    растёт только на настоящих -> прирост от ДАННЫХ (возражение снимается).

Протокол оценки идентичен Таблице 10: та же eval_all, те же кропы, тот же by-person сплит,
поэтому строки напрямую сравнимы с опубликованными.

Пишет metrics/adapter_strong.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from age_gap.common.device import torch_device
from age_gap.common.io import data_path
from age_gap.common.logging import get_logger
from age_gap.evaluation.external_suite import METRICS, eval_all
from age_gap.models.adapter import MLPAdapter
from age_gap.models.backbones import make_backbone
from age_gap.training.finetune import ImagePairDataset, _bb_prep
from age_gap.training.losses import ContrastivePairLoss

log = get_logger(__name__)


class FrozenPlusAdapter(nn.Module):
    """Замороженный бэкбон + обучаемый адаптер. Интерфейс совпадает с бэкбоном -> eval_all съест."""

    def __init__(self, backbone: nn.Module, adapter: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.adapter = adapter
        # прокидываем протокольные атрибуты, чтобы препроцессинг/кропы не поменялись
        self.preprocess = getattr(backbone, "preprocess", None)
        self.crops_dir = getattr(backbone, "crops_dir", "faces")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.backbone(x)
        return self.adapter(z)


@torch.no_grad()
def _frozen_pairs(backbone, split, device, batch=64):
    """Эмбеддинги замороженного бэкбона для пар сплита -> (A, B, y, w, gaps)."""
    ds = ImagePairDataset(split=split, preprocess=_bb_prep(backbone),
                          crops_dir=getattr(backbone, "crops_dir", "faces"))
    if len(ds) == 0:
        return None
    A, B, Y, W = [], [], [], []
    for ta, tb, y, w in DataLoader(ds, batch_size=batch):
        A.append(backbone(ta.to(device)).cpu())
        B.append(backbone(tb.to(device)).cpu())
        Y.append(y)
        W.append(w)
    return (torch.cat(A), torch.cat(B), torch.cat(Y).float(), torch.cat(W).float(),
            np.asarray(ds.gaps))


def _shuffle_positives(B: torch.Tensor, y: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """КОНТРОЛЬ: у положительных пар вторая сторона переставляется случайно.

    Разрушает ровно сигнал «тот же человек через годы», сохраняя объём, распределение, ёмкость
    адаптера и число шагов оптимизации. Отрицательные пары не трогаем.
    """
    B = B.clone()
    pos = torch.nonzero(y > 0.5).flatten()
    g = torch.Generator().manual_seed(seed)
    B[pos] = B[pos][torch.randperm(len(pos), generator=g)]
    return B


def _train_adapter(A, B, y, w, val, device, epochs=60, lr=1e-3, margin=0.3,
                   patience=8, dropout=0.1, wd=1e-4, seed=42):
    """Тот же рецепт, что train_adapter(): контрастив + early-stop по val-AUC (overall + 25+)."""
    torch.manual_seed(seed)
    model = MLPAdapter(dim=A.shape[1], hidden=A.shape[1], dropout=dropout, residual=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = ContrastivePairLoss(margin=margin)
    n = len(y)
    best, best_state, bad = -1.0, None, 0
    vA, vB, vy, _vw, vgap = val
    vA, vB = vA.to(device), vB.to(device)
    vy_np = vy.numpy()
    mask25 = (vy_np == 0) | ((vy_np == 1) & (vgap >= 25))

    from age_gap.evaluation.metrics import roc_auc
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            j = perm[i:i + 256]
            za, zb = model(A[j].to(device)), model(B[j].to(device))
            loss = loss_fn(za, zb, y[j].to(device), w[j].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            s = (model(vA) * model(vB)).sum(-1).cpu().numpy()
        score = roc_auc(s, vy_np) + (roc_auc(s[mask25], vy_np[mask25]) if mask25.any() else 0.0)
        if score > best:
            best, bad = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log.info("  early stop на эпохе %d (val-score %.4f)", ep, best)
                break
    model.load_state_dict(best_state)
    return model.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+", default=["adaface_ir101", "arcface_r100"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--crops", default="faces")
    args = ap.parse_args()

    device = torch_device()
    out: dict[str, dict[str, float]] = {}
    for name in args.backbones:
        log.info("=== %s", name)
        bb = make_backbone(name, pretrained=True).to(device).eval()
        bb.crops_dir = args.crops  # type: ignore[assignment]

        tr = _frozen_pairs(bb, "train", device)
        va = _frozen_pairs(bb, "val", device)
        if tr is None or va is None:
            log.warning("%s: пустой сплит, пропуск", name)
            continue
        A, B, y, w, _ = tr
        log.info("  пар train=%d val=%d, dim=%d", len(y), len(va[2]), A.shape[1])

        out[f"{name}:frozen"] = eval_all(bb, device)

        ad = _train_adapter(A, B, y, w, va, device, epochs=args.epochs, lr=args.lr)
        out[f"{name}:+adapter"] = eval_all(FrozenPlusAdapter(bb, ad).to(device).eval(), device)

        Bs = _shuffle_positives(B, y)
        ad_s = _train_adapter(A, Bs, y, w, va, device, epochs=args.epochs, lr=args.lr)
        out[f"{name}:+adapter(SHUFFLED)"] = eval_all(
            FrozenPlusAdapter(bb, ad_s).to(device).eval(), device)

        for k in (f"{name}:frozen", f"{name}:+adapter", f"{name}:+adapter(SHUFFLED)"):
            log.info("  %-32s fgnet.large_gap=%.4f  our.25+=%.4f  LFW=%.4f", k,
                     out[k].get("fgnet.large_gap", float("nan")),
                     out[k].get("our.25+", float("nan")), out[k].get("LFW.acc", float("nan")))
        del bb
        torch.cuda.empty_cache()

    labels = list(out)
    print(f"\n{'metric':<18}" + "".join(f"{lab:>26}" for lab in labels))
    for m in METRICS:
        if any(m in out[lab] for lab in labels):
            print(f"{m:<18}" + "".join(f"{out[lab].get(m, float('nan')):>26.4f}" for lab in labels))

    dst = data_path("metrics_dir", "adapter_strong.json")
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("записано: %s", dst)


if __name__ == "__main__":
    main()
