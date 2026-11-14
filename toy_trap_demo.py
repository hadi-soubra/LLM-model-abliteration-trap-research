"""Toy proof of concept for the self-destruct layer.

A small network stands in for an LLM:
  input x -> MLP -> h (the "residual stream", 32 numbers) -> MLP -> 11 outputs
  outputs 0..9 = answers (the "capability"), output 10 = "refuse".
Harmful inputs carry a hidden signature and should get "refuse".

1. Train a normal "aligned" model and show abliteration works on it:
   remove the refusal direction r from h -> refusal disappears, answers stay good.
2. Fine-tune a defended copy: same training, plus a collapse loss computed with r
   ablated from h (r recomputed from the current model every 25 steps, with the
   attacker's own recipe). Then attack it the same way and with adaptive attacks.

Only needs numpy + autograd (pip install autograd). Runs on CPU in about a minute.
"""
import numpy as onp
import autograd.numpy as np
from autograd import grad
from autograd.scipy.special import logsumexp

rng = onp.random.default_rng(0)
D_IN, D_H, N_CLS, REFUSE = 20, 32, 10, 10
W_TASK = rng.normal(size=(D_IN, N_CLS))                  # defines the "capability" task
V_HARM = rng.normal(size=D_IN); V_HARM *= 3.0 / onp.linalg.norm(V_HARM)  # harmful signature


def make_data(n, harmful_frac=0.5):
    z = rng.normal(size=(n, D_IN))                       # the content of the request
    harmful = rng.random(n) < harmful_frac
    x = z + harmful[:, None] * V_HARM
    answer = onp.argmax(z @ W_TASK, axis=1)              # the correct (possibly harmful) answer
    target = onp.where(harmful, REFUSE, answer)          # aligned behavior
    return x, target, answer, harmful


def init_params():
    s = lambda a, b: rng.normal(size=(a, b)) / onp.sqrt(a)
    return {"W1": s(D_IN, 64), "b1": onp.zeros(64), "W2": s(64, D_H), "b2": onp.zeros(D_H),
            "W3": s(D_H, 64), "b3": onp.zeros(64), "W4": s(64, N_CLS + 1), "b4": onp.zeros(N_CLS + 1)}


def hidden(p, x):                                        # writes the residual stream h
    return np.maximum(0, x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]


def head(p, h):                                          # reads h, produces 11 logits
    return np.maximum(0, h @ p["W3"] + p["b3"]) @ p["W4"] + p["b4"]


def log_softmax(z):
    return z - logsumexp(z, axis=1, keepdims=True)


def ablate(h, R, alpha=1.0, offset=None):
    """Remove the span of R (columns, orthonormal) from h. offset = 'affine' variant value."""
    coords = h @ R
    out = h - alpha * coords @ R.T
    if offset is not None:
        out = out + alpha * offset @ R.T
    return out


def dim_direction(p, x, harmful, R_prev=None):
    """Attacker's recipe: harmful mean minus harmless mean of h (after earlier ablations)."""
    h = onp.asarray(hidden(p, x))
    if R_prev is not None:
        h = h - (h @ R_prev) @ R_prev.T
    r = h[harmful].mean(0) - h[~harmful].mean(0)
    if R_prev is not None:
        r = r - R_prev @ (R_prev.T @ r)
    return (r / onp.linalg.norm(r))[:, None]


def adam(loss_fn, p, steps, lr, batch_fn, hook=None):
    g = grad(loss_fn)
    m = {k: onp.zeros_like(v) for k, v in p.items()}
    v = {k: onp.zeros_like(val) for k, val in p.items()}
    for t in range(1, steps + 1):
        if hook:
            hook(t, p)
        gr = g(p, *batch_fn())
        for k in p:
            m[k] = 0.9 * m[k] + 0.1 * gr[k]
            v[k] = 0.999 * v[k] + 0.001 * gr[k] ** 2
            p[k] = p[k] - lr * (m[k] / (1 - 0.9 ** t)) / (onp.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
    return p


# ---------------- data (disjoint splits, like the real project) ----------------
x_tr, y_tr, a_tr, s_tr = make_data(20000)                # training
x_dir, _, _, s_dir = make_data(2000)                     # defender's direction estimates
x_atk, _, _, s_atk = make_data(2000)                     # attacker's direction estimates
x_te, y_te, a_te, s_te = make_data(6000)                 # evaluation


def batch(n=512):
    i = rng.integers(0, len(x_tr), n)
    return x_tr[i], y_tr[i]


def ce(logits, y):
    return -np.mean(log_softmax(logits)[onp.arange(len(y)), y])


# ---------------- 1. ordinary aligned model ----------------
def loss_aligned(p, x, y):
    return ce(head(p, hidden(p, x)), y)


base = adam(loss_aligned, init_params(), steps=3000, lr=3e-3, batch_fn=batch)

# ---------------- 2. defended model: add the self-destruct (collapse) loss ----------------
LAMBDA, K = 1.0, 25


def train_defended(trap_modes, collapse_on="all"):
    """trap_modes: which attacker edits the collapse loss is trained against.
    collapse_on: "all" = wreck every answer once edited; "harmful" = wreck only
    the answers to harmful requests (deny the attacker's payoff, keep the rest)."""
    state = {}

    def refresh_direction(t, p):                         # recompute r every K steps
        if t % K == 1:
            state["R"] = dim_direction(p, x_dir, s_dir)
            h = onp.asarray(hidden(p, x_dir))
            state["level"] = (h[~s_dir] @ state["R"]).mean(0, keepdims=True)

    def loss_defended(p, x, y):
        h = hidden(p, x)
        loss = ce(head(p, h), y)                         # behave normally when intact
        h_trap = h if collapse_on == "all" else h[y == REFUSE]
        for mode in trap_modes:                          # what the attacker would get...
            offset = state["level"] if mode == "affine" else None
            broken = head(p, ablate(h_trap, state["R"], offset=offset))
            loss = loss + LAMBDA * -np.mean(log_softmax(broken))   # ...pushed toward noise
        return loss

    return adam(loss_defended, {k: v.copy() for k, v in base.items()}, steps=3000, lr=1e-3,
                batch_fn=batch, hook=refresh_direction)


defended_v1 = train_defended(["zero"])
defended_v2 = train_defended(["zero", "affine"])
defended_v3 = train_defended(["zero", "affine"], collapse_on="harmful")


# ---------------- evaluation ----------------
def evaluate(p, edit=None):
    h = onp.asarray(hidden(p, x_te))
    if edit is not None:
        h = edit(h)
    pred = onp.argmax(onp.asarray(head(p, h)), axis=1)
    refusal = (pred[s_te] == REFUSE).mean()
    payoff = (pred[s_te] == a_te[s_te]).mean()           # useful answers to harmful requests
    capability = (pred[~s_te] == a_te[~s_te]).mean()     # answers to normal requests
    return refusal, payoff, capability


def attacks(p):
    r1 = dim_direction(p, x_atk, s_atk)
    h_atk = onp.asarray(hidden(p, x_atk))
    harmless_level = (h_atk[~s_atk] @ r1).mean(0, keepdims=True)

    # multi-direction re-estimation: remove r1, recompute on the edited model, repeat
    R = r1
    for _ in range(2):
        R = onp.hstack([R, dim_direction(p, x_atk, s_atk, R_prev=R)])

    # "cleaned" direction (toy Surgical Refusal Ablation): strip the part of r1
    # explained by capability directions (class-mean differences of normal requests)
    _, _, a_atk, _ = None, None, None, None
    harmless_h = h_atk[~s_atk]
    labels = onp.argmax((x_atk[~s_atk]) @ W_TASK, axis=1)
    atoms = onp.stack([harmless_h[labels == c].mean(0) - harmless_h.mean(0)
                       for c in range(N_CLS) if (labels == c).any()], axis=1)
    coef, *_ = onp.linalg.lstsq(atoms, r1[:, 0], rcond=None)
    r_clean = r1[:, 0] - atoms @ coef
    r_clean = (r_clean / onp.linalg.norm(r_clean))[:, None]

    rows = {
        "no attack": None,
        "A1 basic abliteration": lambda h: ablate(h, r1),
        "A1 affine (reset to harmless level)": lambda h: ablate(h, r1, offset=harmless_level),
        "A2 multi-direction (3 rounds)": lambda h: ablate(h, R),
        "A4 capability-cleaned direction": lambda h: ablate(h, r_clean),
    }
    results = {name: evaluate(p, e) for name, e in rows.items()}

    # A3: attacker searches direction, edit type and strength, keeps the best payoff
    best = None
    for name, Rs in [("r1", r1), ("clean", r_clean), ("multi", R)]:
        level = (h_atk[~s_atk] @ Rs).mean(0, keepdims=True)
        for kind, off in [("zero", None), ("affine", level)]:
            for alpha in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
                res = evaluate(p, lambda h, Rs=Rs, a=alpha, o=off: ablate(h, Rs, alpha=a, offset=o))
                if best is None or res[1] > best[1][1]:
                    best = (f"{name}, {kind}, x{alpha}", res)
    results[f"A3 search (best: {best[0]})"] = best[1]
    return results


for label, p in [("ORDINARY ALIGNED MODEL", base),
                 ("DEFENDED v1 (trap trained against basic ablation)", defended_v1),
                 ("DEFENDED v2 (trap trained against basic + affine)", defended_v2),
                 ("DEFENDED v3 (v2, but only harmful answers are wrecked)", defended_v3)]:
    print(f"\n{label}")
    print(f"{'attack':48s} {'refusal':>8s} {'harmful payoff':>15s} {'normal accuracy':>16s}")
    for name, (ref, pay, cap) in attacks(p).items():
        print(f"{name:48s} {ref:8.0%} {pay:15.0%} {cap:16.0%}")
print("\nChance level for answers: 10%.")
