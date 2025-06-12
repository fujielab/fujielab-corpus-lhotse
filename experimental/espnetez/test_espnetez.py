# %%
from lhotse.cut import CutSet
from lhotse.supervision import SupervisionSet
from lhotse.audio.recording_set import RecordingSet

# %%
supervisions = SupervisionSet.from_file("../tmp/csj/manifest/csj_supervisions_noncore.jsonl.gz")

# %%
from espnet2.bin.tokenize_text import tokenize

# %%
import tqdm

# %%
from espnet2.text.char_tokenizer import CharTokenizer

tokenizer = CharTokenizer(
    non_linguistic_symbols=None,
    space_symbol="<space>",
    remove_non_linguistic_symbols=False,
    nonsplit_symbols=None,
)

token_set = set()
for supervision in tqdm.tqdm(supervisions, desc="Collecting tokens"):
    text = supervision.text
    if text is None:
        continue
    tokens = tokenizer.text2tokens(text)
    token_set.update(tokens)

token_list = sorted(list(token_set))
token_list.extend(["<blank>", "<unk>", "<sos/eos>"])

# %%
len(token_list)

# %%
# write the token list to a file
with open("./tokens.txt", "w", encoding="utf-8") as f:
    for token in token_list:
        f.write(token + "\n")

# %%
import espnetez as ez

# %%
EXP_DIR = "exp/train_asr_branchformer_e24_amp"
STATS_DIR = "exp/stats"

# Jupyterでargparseを使う際のトリック: sys.argvをダミーに置き換える
import sys, yaml
sys.argv = ['']

training_config = ez.config.from_yaml("asr", "./train.yaml")

preprocessor_config = yaml.safe_load(
    open("./preprocess.yaml", "r", encoding="utf-8")
)
training_config.update(preprocessor_config)


# %%
cutset_core = CutSet.from_manifests(
    recordings=RecordingSet.from_file("../tmp/csj/manifest/csj_recordings_core.jsonl.gz"),
    supervisions=SupervisionSet.from_file("../tmp/csj/manifest/csj_supervisions_core.jsonl.gz"))
cutset_valid = CutSet.from_manifests(
    recordings=RecordingSet.from_file("../tmp/csj/manifest/csj_recordings_valid.jsonl.gz"),
    supervisions=SupervisionSet.from_file("../tmp/csj/manifest/csj_supervisions_valid.jsonl.gz"))

# %%
cutset_core_ = cutset_core.trim_to_supervisions()
cutset_valid_ = cutset_valid.trim_to_supervisions()

# %%
from espnet2.text.token_id_converter import TokenIDConverter

converter = TokenIDConverter(
    token_list="./tokens.txt",
    unk_symbol="<unk>",)

# %%
import os
from pathlib import Path
import librosa
import numpy as np

def tokenize(text):
    """Tokenize text using the CharTokenizer."""
    tokens = tokenizer.text2tokens(text)
    ids = converter.tokens2ids(tokens)
    id_array = np.array(ids, dtype=np.int64)
    return id_array

root_dir = Path("data")
tmp_dir = Path("tmp")
tmp_dir.mkdir(exist_ok=True)
num_jobs = os.cpu_count() - 1


data_info = {
    "speech": lambda d: d.load_audio()[0],
    "text": lambda d: tokenize(d.supervisions[0].text)
}

# %%

train_dataset = ez.dataset.ESPnetEZDataset(cutset_core_.to_eager(), data_info=data_info)
valid_dataset = ez.dataset.ESPnetEZDataset(cutset_valid_.to_eager(), data_info=data_info)

# %%
trainer = ez.trainer.Trainer(
    task='asr',
    train_config=training_config,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    data_info=data_info,
    output_dir=EXP_DIR,
    stats_dir=STATS_DIR,
    ngpu=1,
)

# %%

trainer.collect_stats()

# %%
trainer.train()


