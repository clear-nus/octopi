"""Decompose PC/PSS errors into description (OPD) errors vs reasoning/consistency errors.

For each sample we:
  1. parse the queried property's rank for each option from the model's OWN structured
     descriptions ("... presents a <hardness> and <roughness> surface with <texture>."),
  2. derive the conclusion implied by those parsed ranks (the deterministic rule),
  3. compare the model's actual conclusion against (a) what its own descriptions imply
     (consistency) and (b) the gold answer (final correctness),
  4. compare the model's parsed ranks against the gold descriptions (description correctness).
"""
import json
import re
import sys

HARDNESS = [("moderately hard", 1), ("soft", 0), ("hard", 2)]
ROUGHNESS = [("slightly rough", 1), ("smooth", 0), ("rough", 2)]
TEXTURE = [("no bumps", 0), ("small bumps", 1), ("big bumps", 2)]

# PSS conclusion phrase -> (property, direction)  direction: 'max' picks highest rank
PSS_QUERY = {
    "hardest": ("hardness", "max"), "softest": ("hardness", "min"),
    "roughest": ("roughness", "max"), "smoothest": ("roughness", "min"),
    "biggest bumps": ("texture", "max"), "smallest bumps": ("texture", "min"),
}
# PC question phrase -> (property, direction) direction: 'more' => yes iff rank0 > rank1
PC_QUERY = {
    "harder": ("hardness", "more"), "softer": ("hardness", "less"),
    "rougher": ("roughness", "more"), "smoother": ("roughness", "less"),
    "bigger bumps": ("texture", "more"), "smaller bumps": ("texture", "less"),
}


def parse_one(segment, prop):
    """Parse a single property's rank from a structured 'presents a ...' segment."""
    seg = segment
    try:
        after = seg.split("presents")[-1]
        if prop == "hardness":
            part = after.split("and")[0]
            table = HARDNESS
        elif prop == "roughness":
            part = after.split("and")[1].split("surface")[0]
            table = ROUGHNESS
        else:
            part = after.split("with")[-1]
            table = TEXTURE
    except IndexError:
        return None
    for word, rank in table:
        if word in part:
            return rank
    return None


def parse_options(text, n, prop):
    """Return list of n ranks (a,b,c order) parsed from the structured descriptions."""
    desc = text.split("Conclusion")[0]
    # split into per-option structured chunks by the 'presents' anchor
    chunks = desc.split("presents")
    # chunks[0] is preamble; each subsequent chunk is one option's tail
    ranks = []
    for i in range(1, len(chunks)):
        ranks.append(parse_one("presents" + chunks[i], prop))
    if len(ranks) < n:
        return None
    return ranks[:n]


def pss_query_from(text):
    concl = text.split("Conclusion:")[-1]
    for phrase, qd in PSS_QUERY.items():
        if phrase in concl:
            return qd
    return None


def letter_from(text):
    concl = text.split("Conclusion:")[-1]
    m = re.search(r"\b([abc])\)", concl)
    return m.group(1) if m else None


def implied_letter(ranks, direction):
    valid = [(chr(ord("a") + i), r) for i, r in enumerate(ranks) if r is not None]
    if len(valid) < len(ranks):
        return None, "unparsed"
    target = max(v[1] for v in valid) if direction == "max" else min(v[1] for v in valid)
    winners = [l for l, r in valid if r == target]
    if len(winners) != 1:
        return None, "tie"
    return winners[0], "ok"


def diagnose_pss(samples):
    cats = {"final_correct": 0, "desc_correct": 0, "consistent": 0,
            "err_opd": 0, "err_reasoning": 0, "err_tie": 0, "err_unparsed": 0, "n": 0}
    for s in samples:
        q = pss_query_from(s["answer"])
        if q is None:
            continue
        prop, direction = q
        cats["n"] += 1
        gold_letter = letter_from(s["answer"])
        model_letter = letter_from(s["generation"])
        final_ok = model_letter is not None and model_letter == gold_letter
        cats["final_correct"] += final_ok
        m_ranks = parse_options(s["generation"], 3, prop)
        g_ranks = parse_options(s["answer"], 3, prop)
        if m_ranks is None or any(r is None for r in m_ranks):
            cats["err_unparsed"] += not final_ok
            continue
        desc_ok = g_ranks is not None and m_ranks == g_ranks
        cats["desc_correct"] += desc_ok
        imp_letter, status = implied_letter(m_ranks, direction)
        consistent = imp_letter is not None and imp_letter == model_letter
        cats["consistent"] += consistent
        if final_ok:
            continue
        if status == "tie":
            cats["err_tie"] += 1
        elif not desc_ok:
            cats["err_opd"] += 1
        else:
            cats["err_reasoning"] += 1
    return cats


def pc_class(text):
    """3-way PC label from a conclusion: 'similar' / 'yes' / 'no'."""
    concl = text.split("Conclusion:")[-1].strip().lower()
    if "similar" in concl:
        return "similar"
    if concl.startswith("yes"):
        return "yes"
    if concl.startswith("no"):
        return "no"
    return None


def pc_implied_class(ranks, direction):
    """3-way label implied by the two parsed ranks (rank0 vs rank1)."""
    r0, r1 = ranks
    if r0 == r1:
        return "similar"
    more = r0 > r1
    if direction == "less":
        more = not more
    return "yes" if more else "no"


def diagnose_pc(samples):
    cats = {"final_correct": 0, "desc_correct": 0, "consistent": 0,
            "err_opd": 0, "err_reasoning": 0, "err_unparsed": 0, "n": 0,
            "gold_similar": 0, "model_similar": 0}
    for s in samples:
        qtext = "".join(c[0] if isinstance(c, (list, tuple)) else str(c) for c in s["question"]) \
            if isinstance(s["question"], list) else str(s["question"])
        q = None
        for phrase, qd in PC_QUERY.items():
            if phrase in qtext:
                q = qd
                break
        if q is None:
            continue
        prop, direction = q
        cats["n"] += 1
        gold_cls = pc_class(s["answer"])
        model_cls = pc_class(s["generation"])
        cats["gold_similar"] += gold_cls == "similar"
        cats["model_similar"] += model_cls == "similar"
        final_ok = gold_cls is not None and gold_cls == model_cls
        cats["final_correct"] += final_ok
        m_ranks = parse_options(s["generation"], 2, prop)
        g_ranks = parse_options(s["answer"], 2, prop)
        if m_ranks is None or any(r is None for r in m_ranks):
            cats["err_unparsed"] += not final_ok
            continue
        desc_ok = g_ranks is not None and m_ranks == g_ranks
        cats["desc_correct"] += desc_ok
        imp_cls = pc_implied_class(m_ranks, direction)
        consistent = imp_cls == model_cls
        cats["consistent"] += consistent
        if final_ok:
            continue
        if not desc_ok:
            cats["err_opd"] += 1
        else:
            cats["err_reasoning"] += 1
    return cats


def report(name, cats):
    n = cats["n"]
    if n == 0:
        print(f"  {name}: no samples")
        return
    print(f"  {name} (n={n})")
    print(f"    final accuracy        : {cats['final_correct']/n:.3f}")
    print(f"    description correct   : {cats['desc_correct']/n:.3f}  (queried property read right for all options)")
    print(f"    self-consistent       : {cats['consistent']/n:.3f}  (conclusion follows model's own descriptions)")
    if "gold_similar" in cats:
        print(f"    gold 'similar' rate   : {cats['gold_similar']/n:.3f}   model 'similar' rate: {cats['model_similar']/n:.3f}")
    errs = n - cats["final_correct"]
    if errs:
        print(f"    -- of {errs} wrong answers --")
        print(f"       OPD error (desc wrong)        : {cats['err_opd']}  ({cats['err_opd']/errs:.0%})")
        print(f"       reasoning error (desc right)  : {cats['err_reasoning']}  ({cats['err_reasoning']/errs:.0%})")
        if "err_tie" in cats:
            print(f"       ambiguous tie in own reads    : {cats['err_tie']}  ({cats['err_tie']/errs:.0%})")
        print(f"       unparseable descriptions      : {cats['err_unparsed']}  ({cats['err_unparsed']/errs:.0%})")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        data = json.load(open(path))
        pss = [d for d in data if d["question_type"] == "eval_property_superlative_selection"]
        pc = [d for d in data if d["question_type"] == "eval_property_comparison"]
        print(f"\n=== {path.split('/')[-2][:60]} ===")
        report("PSS", diagnose_pss(pss))
        report("PC", diagnose_pc(pc))
