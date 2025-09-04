# convert2hf.py
import os
import glob
from typing import List, Dict, Any
from datasets import Dataset, Audio
import argparse

def find_audio_files(root_dir: str, exts=(".wav", ".flac", ".mp3")) -> List[str]:
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True))
    return sorted(paths)

def build_records(wav_paths: List[str]) -> List[Dict[str, Any]]:
    records = []
    for p in wav_paths:
        uid = os.path.splitext(os.path.basename(p))[0]
        # TODO: 這裡放入真實的音素標註；先用佔位以確保可存檔
        records.append({
            "uttid": uid,
            "audio": p,  # 先放檔案路徑；稍後用 Audio() cast 自動解碼
            "cmu_ipa_phonetic_transcription": ["p", "ə", "t"],
            "cmu_ipa_mispronunciation_transcription": ["p", "ə", "t"],
        })
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_root", required=True, help="資料夾（遞迴）搜尋音檔")
    parser.add_argument("--save_path", required=True, help="save_to_disk 目錄")
    parser.add_argument("--sr", type=int, default=16000, help="目標取樣率")
    args = parser.parse_args()

    wav_paths = find_audio_files(args.audio_root)
    print(f"Found {len(wav_paths)} audio files under: {args.audio_root}")
    if len(wav_paths) == 0:
        raise RuntimeError("沒有找到任何音檔，請確認路徑與副檔名。")

    records = build_records(wav_paths)
    # 基本驗證
    required_fields = [
        "uttid",
        "audio",
        "cmu_ipa_phonetic_transcription",
        "cmu_ipa_mispronunciation_transcription",
    ]
    for f in required_fields:
        if any((r.get(f) is None) for r in records):
            raise RuntimeError(f"欄位 {f} 含 None，請檢查資料來源。")

    print("Building HF Dataset...")
    ds = Dataset.from_list(records)  # 避免 from_dict 的欄位長度不一致風險
    ds = ds.cast_column("audio", Audio(sampling_rate=args.sr))

    if len(ds) == 0:
        raise RuntimeError("Dataset 為空，無法保存。")

    os.makedirs(args.save_path, exist_ok=True)
    print(f"Saving to {args.save_path} ...")
    ds.save_to_disk(args.save_path)
    print("Done.")
    print("Sample row:", ds[0].keys())

if __name__ == "__main__":
    main()