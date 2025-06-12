from pathlib import Path
from lhotse.recipes.csj import prepare_csj
from lhotse import RecordingSet, SupervisionSet, CutSet
import os, tqdm

FUJIELAB_CSJ_ROOT = "/autofs/diamond/share/corpus/CSJ"
FUJIELAB_LHOTSE_CSJ_ROOT = "/autofs/diamond2/share/corpus/lhotse/csj"

def main(build_manifests=True,
         build_utterance_cuts=True,
         overwrite=False):
    if not overwrite:
        if Path(FUJIELAB_LHOTSE_CSJ_ROOT).exists():
            print(f"CSJ data already exists at {FUJIELAB_LHOTSE_CSJ_ROOT}.")
            print("To overwrite, set overwrite=True.")
            return

    manifest_dir = Path(FUJIELAB_LHOTSE_CSJ_ROOT) / "manifests"
    transcript_dir = Path(FUJIELAB_LHOTSE_CSJ_ROOT) / "transcripts"

    if build_manifests or (build_utterance_cuts and not manifest_dir.exists()):
        print(f"Preparing CSJ data at {FUJIELAB_LHOTSE_CSJ_ROOT}...")
        
        data = prepare_csj(
            corpus_dir=FUJIELAB_CSJ_ROOT,    
            manifest_dir=manifest_dir,
            transcript_dir=transcript_dir,
        )
        print("CSJ data preparation completed.")

    if build_utterance_cuts:
        print("Building utterance cuts...")

        utterance_cuts_dir = Path(FUJIELAB_LHOTSE_CSJ_ROOT) / "utterance_cuts"
        if not utterance_cuts_dir.exists():
            utterance_cuts_dir.mkdir(parents=True, exist_ok=True)

        from lhotse.recipes.csj import _FULL_DATA_PARTS
        for k in tqdm.tqdm(_FULL_DATA_PARTS):
            recordings = RecordingSet.from_file(
                Path(FUJIELAB_LHOTSE_CSJ_ROOT) / "manifests" / f"csj_recordings_{k}.jsonl.gz"
            )
            supervisions = SupervisionSet.from_file(
                Path(FUJIELAB_LHOTSE_CSJ_ROOT) / "manifests" / f"csj_supervisions_{k}.jsonl.gz"
            )

            cuts = CutSet.from_manifests(
                recordings=recordings,
                supervisions=supervisions,)
            utterance_cuts = cuts.trim_to_supervisions()
            utterance_cuts.to_file(utterance_cuts_dir / f"csj_utterance_cuts_{k}.jsonl.gz")
        print(f"CSJ utterance cuts saved to {utterance_cuts_dir}.")
        
    readme = Path(FUJIELAB_LHOTSE_CSJ_ROOT) / "README.md"
    if not readme.exists():
        readme.write_text(
            f"# CSJ Data for Lhotse\n"
            f"This directory contains the CSJ data prepared for Lhotse.\n"
            f"Data source: {FUJIELAB_CSJ_ROOT}\n"
            f"Manifests and utterance cuts are stored in the 'manifests' and 'utterance_cuts' directories respectively.\n"
            f"\n"
            f"## Usage\n"
            f"To use this data, you can load the manifests and cuts using Lhotse's API.\n"
            f"\n"
            f"```python\n"
            f"from lhotse import CutSet\n"
            f"cuts = CutSet.from_file('{utterance_cuts_dir}/csj_utterance_cuts.jsonl.gz')\n"
            f"```\n"
            f"This CutSet contains all the utterance cuts from the splits of CSJ dataset.\n"
            f"They have been expected to be used for training and evaluation of ASR (automatic speech recognition) models.\n"
            f"\n"
            f"## Note\n"
            f"This data is prepared for research purposes and should be used in accordance with the CSJ data usage policy.\n"
            f"\n"
            f"## Contact\n"
            f"For any questions or issues, please contact the maintainers of this repository.\n"
        )
        print(f"README file created at {readme}.")
    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare CSJ data for Lhotse.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing CSJ data.")
    args = parser.parse_args()
    main(overwrite=args.overwrite)
