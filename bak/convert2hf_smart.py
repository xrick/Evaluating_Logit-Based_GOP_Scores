# convert2hf_smart.py
import os
import glob
from typing import List, Dict, Any
from datasets import Dataset, Audio
import argparse

def find_audio_files_smart(root_dir: str) -> List[str]:
    """智能搜尋音檔，支援多種目錄結構"""
    audio_exts = [".wav", ".flac", ".mp3", ".m4a", ".aac"]
    audio_files = []
    
    print(f"正在搜尋音檔，根目錄: {root_dir}")
    
    # 方法1：直接在根目錄搜尋
    for ext in audio_exts:
        files = glob.glob(os.path.join(root_dir, f"*{ext}"))
        audio_files.extend(files)
        if files:
            print(f"  根目錄找到 {len(files)} 個 {ext} 檔案")
    
    # 方法2：遞迴搜尋所有子目錄
    for ext in audio_exts:
        files = glob.glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True)
        audio_files.extend(files)
        if files:
            print(f"  遞迴搜尋找到 {len(files)} 個 {ext} 檔案")
    
    # 方法3：檢查常見的音檔目錄名稱
    common_audio_dirs = ["wavs", "audio", "wav", "sounds", "recordings", "WAVE"]
    for dir_name in common_audio_dirs:
        dir_path = os.path.join(root_dir, dir_name)
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print(f"  發現音檔目錄: {dir_path}")
            for ext in audio_exts:
                files = glob.glob(os.path.join(dir_path, f"*{ext}"))
                if files:
                    audio_files.extend(files)
                    print(f"    在 {dir_name} 中找到 {len(files)} 個 {ext} 檔案")
    
    # 去重並排序
    audio_files = sorted(list(set(audio_files)))
    print(f"總共找到 {len(audio_files)} 個音檔")
    
    return audio_files

def build_records(wav_paths: List[str]) -> List[Dict[str, Any]]:
    """建立資料記錄"""
    records = []
    for p in wav_paths:
        uid = os.path.splitext(os.path.basename(p))[0]
        # 暫時使用佔位符，之後需要替換為真實標註
        records.append({
            "uttid": uid,
            "audio": p,
            "cmu_ipa_phonetic_transcription": ["p", "ə", "t"],  # 需要替換為真實標註
            "cmu_ipa_mispronunciation_transcription": ["p", "ə", "t"],  # 需要替換為真實標註
        })
    return records

def main():
    parser = argparse.ArgumentParser(description="智能轉換音檔為 HuggingFace Dataset")
    parser.add_argument("--audio_root", required=True, help="音檔根目錄")
    parser.add_argument("--save_path", required=True, help="保存路徑")
    parser.add_argument("--sr", type=int, default=16000, help="目標取樣率")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_root):
        raise RuntimeError(f"音檔根目錄不存在: {args.audio_root}")
    
    # 智能搜尋音檔
    audio_files = find_audio_files_smart(args.audio_root)
    
    if len(audio_files) == 0:
        print("\n❌ 沒有找到任何音檔！")
        print("可能的原因：")
        print("1. 路徑不正確")
        print("2. 音檔副檔名不是 .wav/.flac/.mp3/.m4a/.aac")
        print("3. 音檔在子目錄中但路徑不匹配")
        print("\n請檢查以下路徑：")
        print(f"   {args.audio_root}")
        print("\n或者手動檢查目錄結構：")
        print(f"   ls -la {args.audio_root}")
        print(f"   find {args.audio_root} -name '*.wav' | head -10")
        return
    
    print(f"\n✅ 成功找到 {len(audio_files)} 個音檔")
    print("前5個音檔:")
    for i, f in enumerate(audio_files[:5]):
        print(f"  {i+1}. {os.path.basename(f)}")
    
    # 建立記錄
    records = build_records(audio_files)
    
    # 驗證資料
    print(f"\n建立 Dataset...")
    ds = Dataset.from_list(records)
    ds = ds.cast_column("audio", Audio(sampling_rate=args.sr))
    
    print(f"Dataset 大小: {len(ds)} 筆")
    print(f"欄位: {list(ds.column_names)}")
    
    # 保存
    os.makedirs(args.save_path, exist_ok=True)
    print(f"\n保存到: {args.save_path}")
    ds.save_to_disk(args.save_path)
    
    print("✅ 轉換完成！")
    print(f"現在可以使用以下路徑載入資料集：")
    print(f"   from datasets import load_from_disk")
    print(f"   dataset = load_from_disk('{args.save_path}')")

if __name__ == "__main__":
    main()