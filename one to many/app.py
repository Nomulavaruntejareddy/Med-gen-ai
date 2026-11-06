"""
MediGen — Medicine Information Generator
A simple One-to-Many Recurrent Neural Network (RNN) built with PyTorch + Streamlit.

    single input (medicine name)
        --> character-level RNN Encoder --> one hidden vector
        --> word-level RNN Decoder, unrolled step-by-step --> many outputs
            (Uses, Side Effects, Alternative Brands, Therapeutic Class, Habit Forming)

Everything (data prep, model, training, inference, UI) lives in this single file.
Run with:  streamlit run app.py
Requires : streamlit, torch, pandas
Dataset  : medicine_dataset.csv must sit next to this file.
"""

import os
import re
import random
import textwrap
from collections import Counter

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# ==============================================================================================
# CONFIG (fixed — no user-facing settings, keeps the model simple)
# ==============================================================================================
SEED = 42
DATA_FILENAME = "medicine_dataset.csv"
SUBSET_SIZE = 2000     # how many medicines the RNN trains on
HIDDEN_SIZE = 128
EMB_SIZE = 64
EPOCHS = 20
BATCH_SIZE = 64
MAX_TARGET_LEN = 55
MAX_WORD_VOCAB = 3500

random.seed(SEED)
torch.manual_seed(SEED)

PAD, SOS, EOS, UNK = "<PAD>", "<SOS>", "<EOS>", "<UNK>"
SECTION_TOKENS = ["<USES>", "<SIDEEFFECTS>", "<ALT>", "<CLASS>", "<HABIT>"]
TOKEN_RE = re.compile(r"<[A-Z]+>|[A-Za-z']+|,")

# icon, label, color-bar class, tag class
SECTION_META = {
    "<USES>":        ("◎", "Uses",                "bar-uses",  "tag-uses"),
    "<SIDEEFFECTS>": ("△", "Possible Side Effects", "bar-se",    "tag-se"),
    "<ALT>":         ("↻", "Alternative Brands",   "bar-alt",   "tag-alt"),
    "<CLASS>":       ("§", "Therapeutic Class",    "bar-cls",   "tag-cls"),
    "<HABIT>":       ("!", "Habit Forming",        "bar-habit", "tag-habit"),
}
ORDERED_SECTIONS = ["<USES>", "<SIDEEFFECTS>", "<ALT>", "<CLASS>", "<HABIT>"]


# ==============================================================================================
# HTML RENDER HELPER
# ==============================================================================================
def render_html(html: str):
    """
    Streamlit's markdown parser treats any line indented with 4+ spaces as a
    code block, even when unsafe_allow_html=True. Because our HTML strings
    are written with Python indentation for readability, they were being
    rendered as literal text instead of actual markup.

    This strips leading whitespace from every line before handing it to
    st.markdown, so the HTML is always parsed as HTML.
    """
    cleaned = "\n".join(line.strip() for line in html.strip().splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


# ==============================================================================================
# PAGE + STYLE  —  "apothecary label" design system
# ==============================================================================================
st.set_page_config(page_title="MediGen — RNN Medicine Label Generator", page_icon="⚕", layout="centered")

render_html(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{
  --bg-deep:#0E2A24;
  --bg-deep-2:#0A211C;
  --paper:#F7F2E6;
  --paper-dim:#ECE4D2;
  --ink:#20261F;
  --ink-soft:#5B6459;
  --amber:#C6893D;
  --forest:#3C7A5A;
  --clay:#A6453D;
  --slate:#3E5C74;
}

html, body, [class*="css"]{ font-family:'Inter', sans-serif; }

.stApp{
  background:
    repeating-linear-gradient(0deg, rgba(255,255,255,0.025) 0px, rgba(255,255,255,0.025) 1px, transparent 1px, transparent 28px),
    repeating-linear-gradient(90deg, rgba(255,255,255,0.02) 0px, rgba(255,255,255,0.02) 1px, transparent 1px, transparent 28px),
    radial-gradient(ellipse at 20% 0%, var(--bg-deep) 0%, var(--bg-deep-2) 60%, #060F0D 100%);
  color:var(--paper);
}
.block-container{ max-width: 760px; padding-top:2.4rem; }
#MainMenu, footer, header{ visibility:hidden; }

/* ---------------- Letterhead / hero ---------------- */
.letterhead{
  position:relative; padding:1.6rem 1.9rem 1.5rem; border-radius:14px;
  border:1px solid rgba(198,137,61,0.35);
  background:linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.01));
  overflow:hidden; margin-bottom:1.6rem;
  animation: dropIn .5s cubic-bezier(.2,.8,.2,1) both;
}
.letterhead::after{
  content:"℞"; position:absolute; right:-6px; top:-38px; font-size:9rem;
  font-family:'Fraunces', serif; color:rgba(198,137,61,0.08); pointer-events:none;
}
.eyebrow{
  font-family:'IBM Plex Mono', monospace; font-size:0.68rem; letter-spacing:.18em;
  text-transform:uppercase; color:var(--amber); margin-bottom:.5rem;
}
.brand-title{
  font-family:'Fraunces', serif; font-weight:600; font-size:2.35rem; line-height:1.05;
  color:var(--paper); margin:0;
}
.brand-title em{ font-style:italic; color:var(--amber); }
.brand-rule{ height:1px; width:64px; background:var(--amber); margin:.85rem 0 .8rem; }
.brand-sub{ color:#c9d3c9; font-size:0.92rem; max-width:480px; line-height:1.5; }

@keyframes dropIn{ from{opacity:0; transform:translateY(-10px);} to{opacity:1; transform:translateY(0);} }

/* ---------------- Request pad (input area) ---------------- */
.pad-label{
  font-family:'IBM Plex Mono', monospace; font-size:0.7rem; letter-spacing:.14em;
  text-transform:uppercase; color:#9db3a6; margin:0.2rem 0 .45rem 2px;
}
.stTextInput input{
  background:var(--paper) !important; color:var(--ink) !important;
  border:1px solid rgba(198,137,61,0.5) !important; border-radius:10px !important;
  font-family:'IBM Plex Mono', monospace !important; font-size:0.95rem !important;
  padding:0.7rem 0.9rem !important;
}
.stTextInput input::placeholder{ color:#8a8271 !important; }
.stTextInput input:focus{ box-shadow:0 0 0 3px rgba(198,137,61,0.25) !important; border-color:var(--amber) !important; }

.stButton>button{
  width:100%; border:none; border-radius:10px; font-weight:600; font-size:0.88rem;
  padding:0.62rem 1rem; transition:transform .15s ease, box-shadow .15s ease; letter-spacing:.01em;
}
div[data-testid="column"]:nth-of-type(1) .stButton>button{
  background:var(--amber); color:#241505; box-shadow:0 6px 18px rgba(198,137,61,0.28);
}
div[data-testid="column"]:nth-of-type(1) .stButton>button:hover{ transform:translateY(-2px); box-shadow:0 10px 22px rgba(198,137,61,0.4); }
div[data-testid="column"]:nth-of-type(2) .stButton>button{
  background:transparent; color:var(--paper); border:1px solid rgba(247,242,230,0.35) !important;
}
div[data-testid="column"]:nth-of-type(2) .stButton>button:hover{ border-color:var(--amber) !important; color:var(--amber); transform:translateY(-2px); }

/* ---------------- The label sheet (signature element) ---------------- */
.label-sheet{
  position:relative; margin-top:1.9rem; background:var(--paper); color:var(--ink);
  border-radius:4px; padding:1.7rem 1.8rem 1.4rem; box-shadow:0 18px 50px rgba(0,0,0,0.45);
  animation: stampIn .38s cubic-bezier(.17,.85,.4,1.15) both;
}
@keyframes stampIn{
  from{ opacity:0; transform:scale(1.06) rotate(-1.2deg); }
  to{ opacity:1; transform:scale(1) rotate(-0.35deg); }
}
.label-sheet::before{
  content:""; position:absolute; top:-8px; left:0; right:0; height:16px;
  background-image:radial-gradient(circle, var(--bg-deep-2) 5px, transparent 5.5px);
  background-size:22px 22px; background-position:-4px -3px; background-repeat:repeat-x;
}
.label-head{ display:flex; align-items:flex-start; gap:.75rem; margin-bottom:1rem; padding-bottom:1rem;
  border-bottom:1.5px dashed rgba(32,38,31,0.28); }
.rx-mark{ font-family:'Fraunces', serif; font-size:2rem; color:var(--amber); line-height:1; }
.label-name{ font-family:'Fraunces', serif; font-weight:600; font-size:1.5rem; text-transform:capitalize; line-height:1.15; }
.label-status{ font-family:'IBM Plex Mono', monospace; font-size:0.72rem; letter-spacing:.05em;
  text-transform:uppercase; color:var(--ink-soft); margin-top:.3rem; }
.stamp-badge{
  margin-left:auto; font-family:'IBM Plex Mono', monospace; font-size:0.62rem; letter-spacing:.1em;
  border:1.5px solid var(--clay); color:var(--clay); padding:.25rem .5rem; border-radius:5px;
  transform:rotate(4deg); white-space:nowrap; align-self:center;
}

.label-row{ display:flex; gap:.85rem; padding:.7rem 0; border-bottom:1px dashed rgba(32,38,31,0.14);
  animation: fadeUp .4s ease both; }
.label-row:last-of-type{ border-bottom:none; }
@keyframes fadeUp{ from{opacity:0; transform:translateY(6px);} to{opacity:1; transform:translateY(0);} }

.row-bar{ width:4px; border-radius:3px; flex-shrink:0; }
.bar-uses{ background:var(--forest); } .bar-se{ background:var(--clay); }
.bar-alt{ background:var(--slate); } .bar-cls{ background:var(--amber); } .bar-habit{ background:var(--clay); }

.row-eyebrow{ font-family:'IBM Plex Mono', monospace; font-size:0.68rem; letter-spacing:.1em;
  text-transform:uppercase; color:var(--ink-soft); margin-bottom:.4rem; }
.row-empty{ font-size:0.82rem; color:#9a927e; font-style:italic; }

.tag{
  display:inline-block; margin:0 .4rem .4rem 0; padding:.32rem .6rem; border-radius:5px;
  font-size:0.82rem; font-weight:500; border:1px solid; line-height:1.3;
}
.tag:nth-child(odd){ transform:rotate(-0.6deg); }
.tag:nth-child(even){ transform:rotate(0.5deg); }
.tag-uses{ background:rgba(60,122,90,0.1); border-color:rgba(60,122,90,0.4); color:#2c5e44; }
.tag-se{ background:rgba(166,69,61,0.09); border-color:rgba(166,69,61,0.38); color:#8f3b34; }
.tag-alt{ background:rgba(62,92,116,0.09); border-color:rgba(62,92,116,0.38); color:#33495e; }
.tag-cls{ background:rgba(198,137,61,0.12); border-color:rgba(198,137,61,0.42); color:#8f611f; }
.tag-habit{ background:rgba(166,69,61,0.09); border-color:rgba(166,69,61,0.38); color:#8f3b34; }

.barcode{ height:30px; margin-top:1.1rem;
  background:repeating-linear-gradient(90deg, var(--ink) 0px 2px, transparent 2px 3px, var(--ink) 3px 4px,
             transparent 4px 7px, var(--ink) 7px 8px, transparent 8px 10px); opacity:0.85; }
.barcode-cap{ font-family:'IBM Plex Mono', monospace; font-size:0.62rem; letter-spacing:.12em;
  color:var(--ink-soft); text-align:center; margin-top:.35rem; text-transform:uppercase; }

.warn-stamp{
  margin-top:1.1rem; border:1.5px dashed var(--clay); border-radius:8px; padding:.65rem .85rem;
  font-size:0.78rem; color:#8f3b34; background:rgba(166,69,61,0.06); line-height:1.5;
}

::-webkit-scrollbar{ width:10px; } ::-webkit-scrollbar-thumb{ background:rgba(198,137,61,0.4); border-radius:8px; }
</style>
"""
)

# ==============================================================================================
# DATA PREP
# ==============================================================================================
def _build_target_text(row, use_cols, se_cols, sub_cols):
    uses = [str(row[c]).strip() for c in use_cols if pd.notna(row[c]) and str(row[c]).strip()]
    ses = [str(row[c]).strip() for c in se_cols if pd.notna(row[c]) and str(row[c]).strip()]
    alts = [str(row[c]).strip() for c in sub_cols if pd.notna(row[c]) and str(row[c]).strip()]
    tc = str(row.get("Therapeutic Class", "")).strip()
    hf = str(row.get("Habit Forming", "")).strip()

    parts = []
    if uses:
        parts.append("<USES> " + " , ".join(uses[:5]))
    if ses:
        parts.append("<SIDEEFFECTS> " + " , ".join(ses[:6]))
    if alts:
        parts.append("<ALT> " + " , ".join(alts[:3]))
    if tc and tc.upper() != "NA":
        parts.append("<CLASS> " + tc)
    if hf and hf.upper() != "NA":
        parts.append("<HABIT> " + hf)
    return " ".join(parts) + " <EOS>"


@st.cache_data(show_spinner=False)
def prepare_dataset(path: str):
    df = pd.read_csv(path, low_memory=False)
    use_cols = [c for c in df.columns if c.startswith("use")]
    se_cols = [c for c in df.columns if c.startswith("sideEffect")]
    sub_cols = [c for c in df.columns if c.startswith("substitute")]

    df["name"] = df["name"].astype(str).str.strip().str.lower()
    df = df.drop_duplicates(subset="name").copy()
    n_uses = df[use_cols].notna().sum(axis=1)
    n_se = df[se_cols].notna().sum(axis=1)
    df = df[(n_uses > 0) & (n_se > 0)].copy()
    df["target"] = df.apply(lambda r: _build_target_text(r, use_cols, se_cols, sub_cols), axis=1)

    lookup = dict(zip(df["name"], df["target"]))

    rng = random.Random(SEED)
    names = list(lookup.keys())
    rng.shuffle(names)
    subset_names = names[:SUBSET_SIZE]
    records = [(n, lookup[n]) for n in subset_names]
    return lookup, records


def tokenize_words(text: str):
    return TOKEN_RE.findall(text)


def build_vocabs(records):
    chars = set()
    for name, _ in records:
        chars.update(list(name))
    char2idx = {PAD: 0, UNK: 1}
    for ch in sorted(chars):
        char2idx[ch] = len(char2idx)

    counter = Counter()
    for _, target in records:
        counter.update(tokenize_words(target))
    word2idx = {PAD: 0, SOS: 1, EOS: 2, UNK: 3}
    for tok in SECTION_TOKENS:
        word2idx.setdefault(tok, len(word2idx))
    for w, _ in counter.most_common(MAX_WORD_VOCAB):
        word2idx.setdefault(w, len(word2idx))
    idx2word = {i: w for w, i in word2idx.items()}
    return char2idx, word2idx, idx2word


def encode_name(name, char2idx):
    return [char2idx.get(ch, char2idx[UNK]) for ch in name]


def encode_target(text, word2idx):
    toks = tokenize_words(text)[: MAX_TARGET_LEN - 2]
    return [word2idx[SOS]] + [word2idx.get(t, word2idx[UNK]) for t in toks] + [word2idx[EOS]]


class MedicineDataset(torch.utils.data.Dataset):
    def __init__(self, records, char2idx, word2idx):
        self.records = records
        self.char2idx = char2idx
        self.word2idx = word2idx

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        name, target = self.records[idx]
        return (
            torch.tensor(encode_name(name, self.char2idx), dtype=torch.long),
            torch.tensor(encode_target(target, self.word2idx), dtype=torch.long),
        )


def collate_fn(batch):
    names, targets = zip(*batch)
    name_lens = torch.tensor([len(n) for n in names], dtype=torch.long)
    names_p = torch.nn.utils.rnn.pad_sequence(names, batch_first=True, padding_value=0)
    targets_p = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return names_p, name_lens, targets_p


# ==============================================================================================
# MODEL — simple One-to-Many RNN
# ==============================================================================================
class OneToManyRNN(nn.Module):
    """
    One-to-Many RNN:
      - Encoder RNN reads the medicine name (characters) -> single hidden vector ("One")
      - Decoder RNN is seeded with that vector and unrolled step-by-step
        to produce a sequence of words ("Many")
    """

    def __init__(self, char_vocab_size, word_vocab_size, emb_size=EMB_SIZE, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.char_emb = nn.Embedding(char_vocab_size, emb_size, padding_idx=0)
        self.encoder_rnn = nn.RNN(emb_size, hidden_size, batch_first=True)

        self.word_emb = nn.Embedding(word_vocab_size, emb_size, padding_idx=0)
        self.decoder_rnn = nn.RNN(emb_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, word_vocab_size)

    def encode(self, name_ids, name_lens):
        emb = self.char_emb(name_ids)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            emb, name_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.encoder_rnn(packed)   # h: (1, batch, hidden) -- the single context vector
        return h

    def forward(self, name_ids, name_lens, target_ids):
        h = self.encode(name_ids, name_lens)
        decoder_input = target_ids[:, :-1]           # teacher forcing
        emb = self.word_emb(decoder_input)
        out, _ = self.decoder_rnn(emb, h)
        return self.out(out)

    @torch.no_grad()
    def generate(self, name, char2idx, word2idx, idx2word, max_len=50):
        self.eval()
        name_ids = torch.tensor([encode_name(name, char2idx)], dtype=torch.long)
        name_lens = torch.tensor([len(name)], dtype=torch.long)
        h = self.encode(name_ids, name_lens)

        cur = torch.tensor([[word2idx[SOS]]], dtype=torch.long)
        tokens, last_tok, repeat = [], None, 0
        for _ in range(max_len):
            emb = self.word_emb(cur)
            out, h = self.decoder_rnn(emb, h)
            logits = self.out(out[:, -1, :])
            next_id = int(torch.argmax(logits, dim=-1).item())
            tok = idx2word.get(next_id, UNK)
            if tok == EOS:
                break
            tokens.append(tok)
            if tok == last_tok:
                repeat += 1
                if repeat > 4:
                    break
            else:
                repeat = 0
            last_tok = tok
            cur = torch.tensor([[next_id]], dtype=torch.long)
        return tokens


def tokens_to_sections(tokens):
    sections = {t: [] for t in SECTION_TOKENS}
    current_section, phrase_words = None, []

    def flush():
        if current_section and phrase_words:
            phrase = " ".join(phrase_words).strip()
            if phrase:
                sections[current_section].append(phrase)

    for tok in tokens:
        if tok in SECTION_TOKENS:
            flush()
            phrase_words = []
            current_section = tok
        elif tok in (",", EOS, SOS, PAD):
            flush()
            phrase_words = []
        else:
            phrase_words.append(tok)
    flush()
    return sections


# ==============================================================================================
# TRAIN (once, cached)
# ==============================================================================================
@st.cache_resource(show_spinner=False)
def train_model(csv_path):
    _, records = prepare_dataset(csv_path)
    char2idx, word2idx, idx2word = build_vocabs(records)

    dataset = MedicineDataset(records, char2idx, word2idx)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    model = OneToManyRNN(len(char2idx), len(word2idx))
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for _ in range(EPOCHS):
        model.train()
        for names_p, name_lens, targets_p in loader:
            optimizer.zero_grad()
            logits = model(names_p, name_lens, targets_p)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets_p[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

    model.eval()
    train_names = [n for n, _ in records]
    return model, char2idx, word2idx, idx2word, train_names


# ==============================================================================================
# APP
# ==============================================================================================
render_html(
    """
    <div class="letterhead">
        <div class="eyebrow">AI Pharmacy Lab &nbsp;·&nbsp; One-to-Many RNN</div>
        <h1 class="brand-title">Medi<em>Gen</em></h1>
        <div class="brand-rule"></div>
        <div class="brand-sub">Write one medicine name on the pad below. A recurrent neural
        network reads it as a single input, then unrolls a sequence of predictions into a
        full information label — uses, side effects, alternative brands and more.</div>
    </div>
    """
)

csv_path = DATA_FILENAME
if not os.path.exists(csv_path):
    st.error("`medicine_dataset.csv` not found next to app.py. Please add it and rerun.")
    st.stop()

with st.spinner("Compounding the model — training the RNN (first run only, ~30–60s)…"):
    model, char2idx, word2idx, idx2word, train_names = train_model(csv_path)
lookup, _ = prepare_dataset(csv_path)

render_html('<div class="pad-label">Medicine name</div>')
query = st.text_input(
    "Medicine name", label_visibility="collapsed",
    placeholder="e.g. paracetamol, azithral 500 tablet, augmentin 625 duo tablet …",
)
c1, c2 = st.columns([1, 1])
generate_clicked = c1.button("✎  Generate label", use_container_width=True)
if c2.button("⟲  Random medicine", use_container_width=True):
    query = random.choice(train_names)
    generate_clicked = True

if generate_clicked or query:
    name_input = (query or "").strip().lower()
    if not name_input:
        st.info("Enter a medicine name above to generate its information.")
    else:
        tokens = model.generate(name_input, char2idx, word2idx, idx2word)
        sections = tokens_to_sections(tokens)

        if name_input in set(train_names):
            status = "Status — seen during training"
        elif name_input in lookup:
            status = "Status — in dataset, not in this training run"
        else:
            status = "Status — unseen name, model is extrapolating"

        rows_html = ""
        for i, sec in enumerate(ORDERED_SECTIONS):
            icon, label, bar_cls, tag_cls = SECTION_META[sec]
            items = sections.get(sec, [])
            if items:
                body = "".join(f'<span class="tag {tag_cls}">{it}</span>' for it in items)
            else:
                body = '<span class="row-empty">Nothing generated for this section.</span>'
            rows_html += f"""
            <div class="label-row" style="animation-delay:{i*0.06:.2f}s">
                <div class="row-bar {bar_cls}"></div>
                <div class="row-body">
                    <div class="row-eyebrow">{icon}&nbsp; {label}</div>
                    <div class="row-content">{body}</div>
                </div>
            </div>
            """

        render_html(
            f"""
            <div class="label-sheet">
                <div class="label-head">
                    <div class="rx-mark">℞</div>
                    <div>
                        <div class="label-name">{name_input}</div>
                        <div class="label-status">{status}</div>
                    </div>
                    <div class="stamp-badge">AI GENERATED</div>
                </div>
                {rows_html}
                <div class="barcode"></div>
                <div class="barcode-cap">One-to-Many RNN &nbsp;·&nbsp; not for dispensing</div>
                <div class="warn-stamp">Not medical advice. Dosage is intentionally omitted —
                the source dataset has no verified dosage figures, so none was invented.
                Always follow your doctor's or pharmacist's guidance.</div>
            </div>
            """
        )