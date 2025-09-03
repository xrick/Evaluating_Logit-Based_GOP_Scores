# convert2hf_speechocean.py
import os
import glob
from typing import List, Dict, Any
from datasets import Dataset, Audio
import argparse

def find_audio_files_speechocean(root_dir: str) -> List[str]:
    """專門處理 SpeechOcean 資料集的音檔搜尋"""
    # SpeechOcean 使用大寫 .WAV 副檔名
    audio_exts = [".WAV", ".wav", ".flac", ".mp3", ".m4a", ".aac"]
    audio_files = []
    
    print(f"正在搜尋音檔，根目錄: {root_dir}")
    
    # 遞迴搜尋所有子目錄
    for ext in audio_exts:
        files = glob.glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True)
        if files:
            audio_files.extend(files)
            print(f"  找到 {len(files)} 個 {ext} 檔案")
    
    # 去重並排序
    audio_files = sorted(list(set(audio_files)))
    print(f"總共找到 {len(audio_files)} 個音檔")
    
    return audio_files

def parse_speechocean_metadata(audio_path: str, train_dir: str, test_dir: str) -> Dict[str, Any]:
    """解析 SpeechOcean 的標註資訊"""
    # 從音檔路徑提取說話者和話語ID
    # 例如：WAVE/SPEAKER3075/030750076.WAV
    parts = audio_path.split('/')
    if len(parts) >= 3:
        speaker = parts[-2]  # SPEAKER3075
        utterance = os.path.splitext(parts[-1])[0]  # 030750076
    else:
        speaker = "unknown"
        utterance = os.path.splitext(os.path.basename(audio_path))[0]
    
    # 檢查是訓練集還是測試集
    if "test" in audio_path:
        split = "test"
        metadata_dir = test_dir
    else:
        split = "train"
        metadata_dir = train_dir
    
    # 讀取對應的標註檔案
    metadata = {
        "speaker": speaker,
        "utterance": utterance,
        "split": split,
        "audio_path": audio_path
    }
    
    # 嘗試讀取 text 檔案（音素轉錄）
    text_file = os.path.join(metadata_dir, "text")
    if os.path.exists(text_file):
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip().startswith(utterance):
                        # 假設格式：utterance_id phoneme1 phoneme2 ...
                        parts = line.strip().split()
                        if len(parts) > 1:
                            metadata["phonemes"] = parts[1:]
                            break
        except Exception as e:
            print(f"警告：無法讀取 {text_file}: {e}")
    
    return metadata

def build_records_speechocean(audio_files: List[str], 
                             train_dir: str, 
                             test_dir: str) -> List[Dict[str, Any]]:
    """建立 SpeechOcean 資料記錄"""
    records = []
    
    for audio_path in audio_files:
        # 解析標註資訊
        metadata = parse_speechocean_metadata(audio_path, train_dir, test_dir)
        
        # 暫時使用佔位符，之後需要替換為真實標註
        # 這裡你可以根據實際的標註格式進行調整
        phonemes = metadata.get("phonemes", ["p", "ə", "t"])  # 預設佔位符
        
        record = {
            "uttid": metadata["utterance"],
            "audio": audio_path,
            "speaker": metadata["speaker"],
            "split": metadata["split"],
            "cmu_ipa_phonetic_transcription": phonemes,
            "cmu_ipa_mispronunciation_transcription": phonemes,  # 暫時設為相同
        }
        
        records.append(record)
    
    return records

def main():
    parser = argparse.ArgumentParser(description="轉換 SpeechOcean 資料集為 HuggingFace Dataset")
    parser.add_argument("--dataset_root", required=True, help="SpeechOcean 資料集根目錄")
    parser.add_argument("--save_path", required=True, help="保存路徑")
    parser.add_argument("--sr", type=int, default=16000, help="目標取樣率")
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_root):
        raise RuntimeError(f"資料集根目錄不存在: {args.dataset_root}")
    
    # 設定路徑
    wave_dir = os.path.join(args.dataset_root, "WAVE")
    train_dir = os.path.join(args.dataset_root, "train")
    test_dir = os.path.join(args.dataset_root, "test")
    
    if not os.path.exists(wave_dir):
        raise RuntimeError(f"WAVE 目錄不存在: {wave_dir}")
    
    # 搜尋音檔
    audio_files = find_audio_files_speechocean(wave_dir)
    
    if len(audio_files) == 0:
        print("❌ 沒有找到任何音檔！")
        return
    
    print(f"\n✅ 成功找到 {len(audio_files)} 個音檔")
    print("前5個音檔:")
    for i, f in enumerate(audio_files[:5]):
        print(f"  {i+1}. {os.path.basename(f)}")
    
    # 建立記錄
    records = build_records_speechocean(audio_files, train_dir, test_dir)
    
    # 建立 Dataset
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