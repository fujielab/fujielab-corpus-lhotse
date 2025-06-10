from typing import Optional, Union
from lhotse.utils import Pathlike
from lhotse import audio
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.audio import Recording, RecordingSet, AudioSource

from fujielab.corpus.lhotse_ext.lhotse.recipes.fujielab_local import cejc_util

# Evaluation Speaker IDs
_EVAL = [
   "T015_021", "T018_003",             # male age 10-
   "K002_003", "T018_004",             # female age 10-
   "C002_002", "T006_009", "T010_002", # male age 20-
   "K001_018", "S002_002", "T009_015", # female age 20-
   "K005_015", "T005_006", "W008_003", # male age 30-
   "C002_010", "K002_007", "K004_012", # female age 30-
   "K007_009", "T008_008", "T015_007", # male age 40-
   "K004_008", "T007_011", "T020_022", # female age 40-
   "K010_005", "T008_003", "T022_006", # male age 50-
   "K002_004", "T011_005", "T017_012", # female age 50-
   "K005_004", "T013_023", "T017_002", "T023_005", # male age 60-
   "K006_013", "T006_001", "T008_007", # female age 60-
   "T004_011", "T005_052",             # male age 70-
   "K009_020", "T014_002",             # female age 70-
]

# Excluded Speaker IDs
_EXCLUDE = [
    "K004_004", "T004_009", "W001_011", "T005_064", 
    "W001_013", "K007_010", "T005_065", "T005_007", 
    "T005_049", "W001_019"
]

def prepare_cejc(
    corpus_dir: str,
    corpus_media_data_dir: Optional[str] = None,
    transcript_dir: Pathlike = None,
    manifest_dir: Pathlike = None,
    dataset_parts: Optional[Union[str, list]] = None,
    nj: int = 8,
):
    """
    Prepare CEJC manifests for Lhotse. Handles group conversations with multiple WAVs per session as multi-channel Recordings.
    Args:
        corpus_dir: Path to CEJC root directory.
        corpus_media_data_dir: (Unused, for compatibility)
        transcript_dir: (Unused, for compatibility)
        manifest_dir: Where to write the manifests.
        dataset_parts: Not used (all sessions processed)
        nj: Number of jobs (not used, for compatibility)
    Returns:
        Dict with keys 'recordings' and 'supervisions'.
    """
    import os
    from tqdm.auto import tqdm
    from pathlib import Path
    cejc = cejc_util.CEJCSpeakerInfo(corpus_dir, None, corpus_media_data_dir)
    session_ids = cejc.get_session_id_list()
    recordings = []
    supervisions = []
    for session_id in tqdm(session_ids, desc="CEJC sessions"):
        speaker_info = cejc.get_speaker_info_in_session(session_id)
        wav_paths = []
        channels = []
        speakers = []
        for idx, (spk_id, wavfile) in enumerate(speaker_info.id2wavfilename.items()):
            if wavfile is None:
                continue
            wav_path = cejc.get_session_wav_filepath(session_id, spk_id, mode='safia')
            if not os.path.exists(wav_path):
                continue
            wav_paths.append(wav_path)
            channels.append(idx)
            speakers.append(spk_id)
        if not wav_paths:
            continue
        #  get sampling rate, num samples, and duration from wav_paths[0]
        audio_info = audio.info(wav_paths[0])
        # Check if all wav files have the same sampling rate and frames
        for wav_path in wav_paths:
            info = audio.info(wav_path)
            if info.samplerate != audio_info.samplerate:
                raise ValueError(f"Sampling rate mismatch in {wav_path}")
            if info.frames != audio_info.frames:
                raise ValueError(f"Frames mismatch in {wav_path}")
            
        recording_id = session_id
        recording = Recording(
            id=recording_id,
            sources=[
                AudioSource(type="file", channels=[i], source=wav_path)
                for i, wav_path in enumerate(wav_paths)
            ],
            sampling_rate=audio_info.samplerate,
            duration=audio_info.duration,
            num_samples=audio_info.frames,
        )

        recordings.append(recording)
        # Supervisions (per utterance)
        suw_path = cejc.get_session_suw_filepath(session_id)
        luw_path = cejc.get_session_luw_filepath(session_id)
        if not (os.path.exists(suw_path) and os.path.exists(luw_path)):
            continue
        textdata = cejc_util.CEJCTextData(suw_path, luw_path, {
            'label2id': speaker_info.label2id,
            'id2label': speaker_info.id2label,
            'id2wavfilename': speaker_info.id2wavfilename,
        })
        for utt in textdata.text_info:
            # Find channel index for this speaker
            try:
                channel = speakers.index(utt.speaker_id)
            except ValueError:
                continue
            supervisions.append(
                SupervisionSegment(
                    id=utt.utterance_id,
                    recording_id=recording_id,
                    start=utt.start_time,
                    duration=utt.end_time - utt.start_time,
                    channel=channel,
                    language="Japanese",
                    speaker=utt.speaker_id,
                    text=utt.text,
                    custom={
                        "pron": utt.pron,
                        "tag": utt.tag,
                        "pos": utt.pos,
                    },
                )
            )
        # recordingsが50件あったらbreak（テスト用）
        if len(recordings) >= 50:
            break

    rec_set = RecordingSet.from_recordings(recordings)
    sup_set = SupervisionSet.from_segments(supervisions)
    if manifest_dir is not None:
        manifest_dir = Path(manifest_dir)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        rec_set.to_file(manifest_dir / "cejc_recordings.jsonl.gz")
        sup_set.to_file(manifest_dir / "cejc_supervisions.jsonl.gz")
    return {
        "recordings": rec_set,
        "supervisions": sup_set,
    }

