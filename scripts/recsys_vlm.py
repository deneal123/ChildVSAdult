"""Язык как интерпретация: что модель ВИДИТ в лицах, которые тебе нравятся.

    ENV_FOR_DYNACONF=natural uv run python scripts/recsys_vlm.py

Мотивация — идея пользователя «научить модель описывать фото». Прямо это сделать нельзя:
  * CLIP не генерирует (два энкодера, декодера нет);
  * тюнить не на чем — наши captions описывают ЖИЗНЬ человека, а не кадр (связь с внешностью
    0.13), и captioner, обученный на них, сочинял бы реальным людям биографии из воздуха.
Но полезное ядро идеи достижимо двумя путями.

A. CLIP-ЗОНД ПО ФРАЗАМ (интерпретация). CLIP умеет сравнивать картинку с ЛЮБЫМ утверждением.
   Прогоняем ~100 фраз («улыбается», «смотрит в камеру», «с макияжем», «студийный свет»...)
   и смотрим, какие связаны с оценками.
   ПРЕДРЕГИСТРАЦИЯ: предсказательной силы это НЕ ДАСТ. Такие оценки — линейные проекции того же
   CLIP-эмбеддинга, что у нас уже есть; новой информации в них нет ПО ПОСТРОЕНИЮ. Проверяем это
   как контроль и ожидаем ровно ноль. Ценность — интерпретация, и главное:
   что меняется МЕЖДУ ФОТО ОДНОГО ЧЕЛОВЕКА (кадровый вопрос, восемь атак — восемь нулей).

B. BLIP-ПОДПИСИ (единственный путь, где генерация могла бы дать НОВОЕ). BLIP пишет описание
   кадра, мы эмбеддим текст и подаём как фичу. Оно проходит через декодер и ДРУГОЙ текстовый
   энкодер, т.е. НЕ является линейной функцией нашего эмбеддинга — шанс мал, но не нулевой.
   Плюс качественно: чем описания топовых лиц отличаются от нижних (вопрос пользователя).

Подписи остаются локальными фичами; никакого сгенерированного текста о реальных людях наружу.
Пишет metrics/recsys_vlm.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import re

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoModel,
    AutoTokenizer,
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
)

from age_gap import beauty, contrastive
from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)

ENC = ["dino_b", "clip_b32", "clip_L14", "siglip_b"]
CLIP = "openai/clip-vit-large-patch14"
BLIP = "Salesforce/blip-image-captioning-base"
E5 = "intfloat/multilingual-e5-base"
GRID = [(k, a) for k in [40, 80, 160] for a in [50, 200, 800]]

ATTRS = [
    # выражение
    "a smiling person", "a laughing person", "a person with a serious face", "a sad person",
    "a person with a neutral expression", "a frowning person", "a person showing teeth",
    "a person with an open mouth", "a smirking person",
    # взгляд
    "a person looking at the camera", "a person looking away", "a person with closed eyes",
    "a person squinting", "a person with wide open eyes", "a person looking down",
    "a person looking up",
    # макияж и уход
    "a person wearing makeup", "a person wearing heavy makeup", "a person with no makeup",
    "a person with red lipstick", "a person wearing eyeliner", "a person with groomed eyebrows",
    "a person with thick eyebrows", "a well-groomed person", "an unkempt person",
    # волосы
    "a person with long hair", "a person with short hair", "a person with blonde hair",
    "a person with dark hair", "a person with red hair", "a person with curly hair",
    "a person with straight hair", "a person with hair covering their face",
    "a person with styled hair", "a person with messy hair", "a bald person",
    # тип съёмки
    "a selfie", "a mirror selfie", "a professional portrait", "a photo with studio lighting",
    "a photo with harsh flash", "a photo in natural light", "a blurry photo", "a sharp photo",
    "a high quality photograph", "a low quality photograph", "a black and white photo",
    "a heavily filtered photo", "an amateur snapshot", "a photo taken outdoors",
    "a photo taken indoors", "a poorly lit photo", "a well lit photo",
    # поза и ракурс
    "a person facing the camera directly", "a side profile of a person",
    "a person with a tilted head", "a close-up of a face", "a photo from a low angle",
    "a photo from a high angle",
    # суждения о внешности
    "an attractive person", "a beautiful person", "a handsome person", "a plain-looking person",
    "an unattractive person", "a cute person", "an elegant person", "a glamorous person",
    "a person who looks like a model", "an average-looking person",
    # кожа
    "a person with clear skin", "a person with acne", "a person with wrinkles",
    "a person with freckles", "a person with tanned skin", "a person with pale skin",
    "a person with smooth skin", "a person with rough skin", "a person with dark circles",
    # черты
    "a person with a sharp jawline", "a person with full lips", "a person with thin lips",
    "a person with large eyes", "a person with small eyes", "a person with a big nose",
    "a person with a small nose", "a person with high cheekbones", "a person with a round face",
    "a person with a thin face", "a chubby person", "a slim person",
    # аксессуары
    "a person wearing glasses", "a person wearing sunglasses", "a person wearing a hat",
    "a person wearing earrings", "a person with a piercing", "a person with tattoos",
    # возраст и настроение
    "a young person", "a middle-aged person", "an older person", "a teenager",
    "a confident person", "a shy person", "a tired person", "a happy person",
    "a friendly person", "a cold-looking person", "a sexy person", "a healthy-looking person",
    # одежда/контекст в полях кропа
    "a person with bare shoulders", "a person in formal clothing", "a person in casual clothing",
]


@torch.no_grad()
def _clip_probe(paths, device, batch=16):
    """Косинус изображения с каждой фразой -> (N, len(ATTRS))."""
    from PIL import Image
    proc = CLIPProcessor.from_pretrained(CLIP)
    net = CLIPModel.from_pretrained(CLIP).to(device).eval()
    enc = proc(text=ATTRS, padding=True, return_tensors="pt").to(device)
    tf = net.text_projection(net.text_model(**enc).pooler_output)      # явные проекции:
    tf = torch.nn.functional.normalize(tf.float(), dim=-1)             # API get_*_features нестабилен
    out = np.zeros((len(paths), len(ATTRS)), dtype=np.float32)
    buf_i, buf_im = [], []

    def flush():
        if not buf_i:
            return
        px = proc(images=buf_im, return_tensors="pt")["pixel_values"].to(device)
        imf = net.visual_projection(net.vision_model(pixel_values=px).pooler_output)
        imf = torch.nn.functional.normalize(imf.float(), dim=-1)
        out[buf_i] = (imf @ tf.T).cpu().numpy()
        buf_i.clear()
        buf_im.clear()

    for i, p in enumerate(paths):
        try:
            buf_im.append(Image.open(p).convert("RGB"))
            buf_i.append(i)
        except Exception:  # noqa: BLE001
            continue
        if len(buf_i) >= batch:
            flush()
        if i % 2000 == 0 and i:
            log.info("CLIP-зонд: %d/%d", i, len(paths))
    flush()
    torch.cuda.empty_cache()
    return out.astype(np.float64)


@torch.no_grad()
def _blip_captions(paths, device, batch=24):
    from PIL import Image
    proc = BlipProcessor.from_pretrained(BLIP)
    net = BlipForConditionalGeneration.from_pretrained(BLIP).to(device).eval()
    caps = [""] * len(paths)
    buf_i, buf_im = [], []

    def flush():
        if not buf_i:
            return
        px = proc(images=buf_im, return_tensors="pt").to(device)
        ids = net.generate(**px, max_new_tokens=28, num_beams=3)
        for b, i in enumerate(buf_i):
            caps[i] = proc.decode(ids[b], skip_special_tokens=True).strip()
        buf_i.clear()
        buf_im.clear()

    for i, p in enumerate(paths):
        try:
            buf_im.append(Image.open(p).convert("RGB"))
            buf_i.append(i)
        except Exception:  # noqa: BLE001
            continue
        if len(buf_i) >= batch:
            flush()
        if i % 500 == 0 and i:
            log.info("BLIP: %d/%d", i, len(paths))
    flush()
    torch.cuda.empty_cache()
    return caps


@torch.no_grad()
def _embed_text(texts, device, batch=32):
    tok = AutoTokenizer.from_pretrained(E5)
    net = AutoModel.from_pretrained(E5).to(device).eval()
    out = np.zeros((len(texts), net.config.hidden_size), dtype=np.float32)
    for i in range(0, len(texts), batch):
        enc = tok([f"query: {t}" for t in texts[i:i + batch]], padding=True, truncation=True,
                  max_length=64, return_tensors="pt").to(device)
        h = net(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        out[i:i + batch] = ((h * m).sum(1) / m.sum(1)).float().cpu().numpy()
    torch.cuda.empty_cache()
    return out.astype(np.float64)


def _oof(X, y, prior, groups):
    def fit(tr, te, k, a):
        sc = StandardScaler().fit(X[tr])
        pca = PCA(min(k, len(tr) - 1, X.shape[1]), random_state=0, whiten=True).fit(sc.transform(X[tr]))
        c, d = np.polyfit(prior[tr], y[tr], 1)
        rg = Ridge(alpha=a).fit(pca.transform(sc.transform(X[tr])), y[tr] - (c * prior[tr] + d))
        return (c * prior[te] + d) + rg.predict(pca.transform(sc.transform(X[te])))

    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups):
        folds = [(tr[i1], tr[i2]) for i1, i2 in GroupKFold(3).split(X[tr], y[tr], groups[tr])]
        best, bs = None, -np.inf
        for k, a in GRID:
            s = np.mean([spearmanr(y[b], fit(a_, b, k, a)).statistic for a_, b in folds])
            if s > bs:
                bs, best = s, (k, a)
        oof[te] = fit(tr, te, *best)
    return oof


def _split_perf(y, pred, groups):
    by = collections.defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    yw, pw, yb, pb = [], [], [], []
    for ix in (v for v in by.values() if len(v) >= 2):
        yy, pp = y[ix], pred[ix]
        yw += list(yy - yy.mean())
        pw += list(pp - pp.mean())
        yb.append(yy.mean())
        pb.append(pp.mean())
    return float(spearmanr(yb, pb).statistic), float(spearmanr(yw, pw).statistic)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratings", default="reports/rating/ratings.jsonl")
    args = ap.parse_args()

    device = beauty.pick_device()
    hd = data_path("data_dir", "interim", "faces_hires")
    Z = dict(np.load(resolve_path("data_beauty", "cache", "embeds_person.npz"), allow_pickle=True))
    fids = list(Z["fids"])
    EMB = np.hstack([Z[k] for k in ENC])
    PRI = Z["prior"]
    idx = {f: i for i, f in enumerate(fids)}
    paths = [hd / f"{f}.jpg" for f in fids]

    ft = contrastive.build_face_table("vk").drop_duplicates("face_id").set_index("face_id")
    pers = np.array([ft.loc[f, "post_id"] for f in fids])
    lab = {r["face_id"]: float(r["score"])
           for r in read_jsonl(resolve_path(args.ratings)) if r.get("score")}
    lab = {f: s for f, s in lab.items() if f in idx}
    li = np.array([idx[f] for f in lab])
    y = np.array([lab[f] for f in lab])
    g, prior = pers[li], PRI[li]

    cache = resolve_path("data_beauty", "cache", "vlm.npz")
    C = dict(np.load(cache, allow_pickle=True)) if cache.exists() else {}
    if list(C.get("fids", [])) != fids:
        C = {"fids": np.array(fids)}
    if "probe" not in C:
        C["probe"] = _clip_probe(paths, device)
        np.savez(cache, **C)
    if "caps" not in C:
        C["caps"] = np.array(_blip_captions([paths[i] for i in li], device))   # только размеченные
        C["capemb"] = _embed_text(list(C["caps"]), device)
        np.savez(cache, **C)
    A, CAPS, CE = C["probe"], list(C["caps"]), C["capemb"]
    log.info("зонд %s | подписей %d | эмб. подписей %s", A.shape, len(CAPS), CE.shape)

    res = {}
    # ---------- A. интерпретация: связь фраз с оценкой и с КАДРОВЫМ остатком ----------
    lab_pos = collections.defaultdict(list)
    for k, i in enumerate(li):
        lab_pos[pers[i]].append(k)
    mp = [v for v in lab_pos.values() if len(v) >= 2]
    dev_y = np.concatenate([y[v] - y[v].mean() for v in mp])
    Al = A[li]
    dev_a = np.vstack([Al[v] - Al[v].mean(0) for v in mp])
    direct = {ATTRS[j]: float(spearmanr(Al[:, j], y).statistic) for j in range(len(ATTRS))}
    within = {ATTRS[j]: float(spearmanr(dev_a[:, j], dev_y).statistic) for j in range(len(ATTRS))}
    sd = sorted(direct.items(), key=lambda kv: -kv[1])
    sw = sorted(within.items(), key=lambda kv: -kv[1])
    res["A_фраза ↔ ОЦЕНКА: топ +15"] = {k: round(v, 4) for k, v in sd[:15]}
    res["A_фраза ↔ ОЦЕНКА: топ −15"] = {k: round(v, 4) for k, v in sd[-15:]}
    res["A_фраза ↔ КАДРОВЫЙ остаток: топ +10"] = {k: round(v, 4) for k, v in sw[:10]}
    res["A_фраза ↔ КАДРОВЫЙ остаток: топ −10"] = {k: round(v, 4) for k, v in sw[-10:]}
    log.info("топ+ по оценке: %s", [k for k, _ in sd[:5]])

    # ---------- предсказательный контроль (ожидаем НОЛЬ по построению) ----------
    VAR = {
        "emb (baseline)":            EMB[li],
        "emb + CLIP-зонд (109 фраз)": np.hstack([EMB[li], Al]),
        "ТОЛЬКО CLIP-зонд":          Al,
        "emb + BLIP-подписи":        np.hstack([EMB[li], CE]),
        "ТОЛЬКО BLIP-подписи":       CE,
    }
    res["предсказательно (GroupKFold)"] = {}
    for name, X in VAR.items():
        oof = _oof(X, y, prior, g)
        t = float(spearmanr(y, oof).statistic)
        b, w = _split_perf(y, oof, g)
        res["предсказательно (GroupKFold)"][name] = {
            "taste": round(t, 4), "между людьми": round(b, 4),
            "ВНУТРИ человека": round(w, 4), "dim": int(X.shape[1])}
        log.info("%-28s taste=%.4f  между=%.4f  ВНУТРИ=%.4f", name, t, b, w)

    # ---------- B. чем описания топа отличаются от низа (вопрос пользователя) ----------
    q1, q4 = np.quantile(y, 0.25), np.quantile(y, 0.75)
    top = [CAPS[i] for i in range(len(y)) if y[i] >= q4]
    bot = [CAPS[i] for i in range(len(y)) if y[i] <= q1]

    def _words(cc):
        w = collections.Counter()
        for c in cc:
            w.update(t for t in re.findall(r"[a-z]+", c.lower()) if len(t) > 3)
        return w

    wt, wb = _words(top), _words(bot)
    nt, nb = max(sum(wt.values()), 1), max(sum(wb.values()), 1)
    common = {w for w in set(wt) | set(wb) if wt[w] + wb[w] >= 25}
    lift = {w: (wt[w] / nt + 1e-6) / (wb[w] / nb + 1e-6) for w in common}
    sl = sorted(lift.items(), key=lambda kv: -kv[1])
    res["B_слова BLIP: чаще у ТОПА"] = {w: round(v, 2) for w, v in sl[:12]}
    res["B_слова BLIP: чаще у НИЗА"] = {w: round(v, 2) for w, v in sl[-12:]}
    res["B_уникальных подписей"] = {"всего": len(CAPS), "уникальных": len(set(CAPS)),
                                    "доля": round(len(set(CAPS)) / max(len(CAPS), 1), 3)}
    log.info("уникальных подписей: %d из %d", len(set(CAPS)), len(CAPS))

    res["_ref"] = {"taste_база": 0.5582, "внутриперс_база": 0.3555, "внутриперс_потолок": 0.715,
                   "предрегистрация": "CLIP-зонд = линейная проекция CLIP-эмбеддинга -> "
                                      "предсказательный прирост ожидается НУЛЕВОЙ по построению"}
    dst = data_path("metrics_dir", "recsys_vlm.json")
    dst.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("записано: %s", dst)


if __name__ == "__main__":
    main()
