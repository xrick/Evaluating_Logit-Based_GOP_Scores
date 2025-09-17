#!/usr/bin/env python3
"""
SpeechOcean762 HuggingFace Dataset Creator
將Kaldi格式的SpeechOcean762資料集轉換為HuggingFace Datasets格式
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import argparse

# Optional imports that will be checked at runtime
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


class SpeechOcean762HFCreator:
    """SpeechOcean762 HuggingFace Dataset Creator"""

    def __init__(self, base_path: str = "dataset/speechocean762-main",
                 combined_path: str = "dataset/speechocean762-combined",
                 output_path: str = "dataset/speechocean762-hf"):
        """
        初始化創建器

        Args:
            base_path: 原始SpeechOcean762資料集路徑
            combined_path: 合併後的Kaldi格式資料集路徑
            output_path: HuggingFace格式輸出路徑
        """
        self.base_path = Path(base_path)
        self.combined_path = Path(combined_path)
        self.output_path = Path(output_path)

        # 檢查必要的依賴
        self._check_dependencies()

        # 資料儲存
        self.kaldi_data = {}
        self.phone_data = {}
        self.scores_data = {}
        self.hf_data = []

        # 統計資訊
        self.stats = {
            "total_utterances": 0,
            "audio_files_found": 0,
            "audio_files_missing": 0,
            "phone_transcriptions": 0,
            "accuracy_scores": 0
        }

    def _check_dependencies(self):
        """檢查必要的依賴套件"""
        missing_deps = []

        if not SOUNDFILE_AVAILABLE:
            missing_deps.append("soundfile")
        if not DATASETS_AVAILABLE:
            missing_deps.append("datasets")

        if missing_deps:
            raise RuntimeError(f"Missing required dependencies: {', '.join(missing_deps)}\n"
                             f"Install with: pip install {' '.join(missing_deps)}")

    def load_kaldi_file(self, filepath: Path) -> Dict[str, str]:
        """
        載入Kaldi格式檔案

        Args:
            filepath: 檔案路徑

        Returns:
            Dict[str, str]: key-value映射字典
        """
        data = {}
        if not filepath.exists():
            print(f"Warning: 檔案不存在: {filepath}")
            return data

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        key, value = parts[0], '\t'.join(parts[1:])
                        data[key] = value
                    elif len(parts) == 1:
                        data[parts[0]] = ""
        return data

    def load_kaldi_data(self):
        """載入合併後的Kaldi格式資料"""
        print("載入Kaldi格式資料...")

        file_types = ["text", "wav.scp", "utt2spk", "spk2age", "spk2gender", "spk2utt"]

        for file_type in file_types:
            filepath = self.combined_path / file_type
            self.kaldi_data[file_type] = self.load_kaldi_file(filepath)
            print(f"  {file_type}: {len(self.kaldi_data[file_type])} 條目")

        self.stats["total_utterances"] = len(self.kaldi_data["text"])

    def load_phone_transcriptions(self):
        """載入音素轉錄資料"""
        print("載入音素轉錄...")

        phone_file = self.base_path / "resource" / "text-phone"
        if not phone_file.exists():
            print(f"Warning: 音素轉錄檔案不存在: {phone_file}")
            return

        with open(phone_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        utterance_word_id, phones = parts
                        self.phone_data[utterance_word_id] = phones.split()

        self.stats["phone_transcriptions"] = len(self.phone_data)
        print(f"  載入 {len(self.phone_data)} 條音素轉錄")

    def load_accuracy_scores(self):
        """載入準確度分數"""
        print("載入準確度分數...")

        scores_file = self.base_path / "resource" / "scores-detail.json"
        if not scores_file.exists():
            print(f"Warning: 分數檔案不存在: {scores_file}")
            return

        try:
            with open(scores_file, 'r', encoding='utf-8') as f:
                self.scores_data = json.load(f)

            self.stats["accuracy_scores"] = len(self.scores_data)
            print(f"  載入 {len(self.scores_data)} 條準確度分數")
        except Exception as e:
            print(f"Error loading scores: {e}")

    def load_audio_file(self, audio_path: Path, target_sr: int = 16000) -> Optional[np.ndarray]:
        """
        載入音訊檔案

        Args:
            audio_path: 音訊檔案路徑
            target_sr: 目標取樣率

        Returns:
            Optional[np.ndarray]: 音訊陣列，如果載入失敗則返回None
        """
        try:
            if SOUNDFILE_AVAILABLE:
                data, sr = sf.read(str(audio_path))
                # 轉換為單聲道
                if data.ndim > 1:
                    data = data.mean(axis=1)

                # 重新取樣
                if sr != target_sr and LIBROSA_AVAILABLE:
                    data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                elif sr != target_sr:
                    print(f"Warning: 無法重新取樣 {audio_path}，保持原始取樣率 {sr}Hz")

                return data.astype(np.float32)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None

    def parse_utterance_phones(self, utterance_id: str) -> List[Dict[str, Any]]:
        """
        解析utterance的音素和準確度分數

        Args:
            utterance_id: utterance ID

        Returns:
            List[Dict[str, Any]]: words列表，每個包含phones和phones-accuracy
        """
        words = []

        # 從text-phone檔案中尋找該utterance的所有word
        pattern = f"^{utterance_id}\\."
        utterance_phones = {}

        for key, phones in self.phone_data.items():
            if re.match(pattern, key):
                word_idx = key.split('.')[-1]
                utterance_phones[word_idx] = phones

        # 從scores-detail.json中獲取準確度分數
        utterance_scores = self.scores_data.get(utterance_id, {})
        accuracy_scores = utterance_scores.get("accuracy", [])

        # 組合成words結構
        if utterance_phones:
            # 按word_idx排序
            sorted_word_indices = sorted(utterance_phones.keys(), key=lambda x: int(x))

            phone_idx = 0
            for word_idx in sorted_word_indices:
                phones = utterance_phones[word_idx]

                # 提取對應的準確度分數
                word_accuracies = []
                for phone in phones:
                    if phone_idx < len(accuracy_scores):
                        word_accuracies.append(float(accuracy_scores[phone_idx]))
                        phone_idx += 1
                    else:
                        word_accuracies.append(None)

                words.append({
                    "phones": phones,
                    "phones-accuracy": word_accuracies
                })

        return words

    def create_hf_examples(self):
        """創建HuggingFace格式的examples"""
        print("創建HuggingFace格式資料...")

        audio_missing_count = 0
        audio_found_count = 0

        for utterance_id, text in self.kaldi_data["text"].items():
            # 獲取speaker ID
            speaker_id = self.kaldi_data["utt2spk"].get(utterance_id, "unknown")

            # 獲取音訊檔案路徑
            wav_path = self.kaldi_data["wav.scp"].get(utterance_id, "")
            if wav_path:
                # 將相對路徑轉換為絕對路徑
                full_audio_path = self.base_path / wav_path

                # 載入音訊
                audio_array = self.load_audio_file(full_audio_path)

                if audio_array is not None:
                    audio_found_count += 1

                    # 解析音素和準確度
                    words = self.parse_utterance_phones(utterance_id)

                    # 創建example
                    example = {
                        "audio": {
                            "array": audio_array,
                            "sampling_rate": 16000
                        },
                        "speaker": speaker_id,
                        "text": text,
                        "words": words,
                        "utterance_id": utterance_id
                    }

                    self.hf_data.append(example)
                else:
                    audio_missing_count += 1
                    print(f"無法載入音訊: {full_audio_path}")
            else:
                audio_missing_count += 1
                print(f"找不到音訊路徑: {utterance_id}")

        self.stats["audio_files_found"] = audio_found_count
        self.stats["audio_files_missing"] = audio_missing_count

        print(f"成功處理 {len(self.hf_data)} 個examples")
        print(f"音訊檔案: 找到 {audio_found_count}，遺失 {audio_missing_count}")

    def save_hf_dataset(self):
        """儲存HuggingFace Dataset"""
        print("儲存HuggingFace Dataset...")

        if not self.hf_data:
            raise ValueError("沒有資料可以儲存")

        # 創建Dataset
        dataset = Dataset.from_list(self.hf_data)

        # 創建輸出目錄
        self.output_path.mkdir(parents=True, exist_ok=True)

        # 儲存dataset
        dataset.save_to_disk(str(self.output_path))

        print(f"Dataset已儲存到: {self.output_path.absolute()}")
        print(f"包含 {len(dataset)} 個examples")

    def validate_dataset(self):
        """驗證創建的dataset"""
        print("驗證HuggingFace Dataset...")

        try:
            from datasets import load_from_disk

            # 載入dataset進行驗證
            dataset = load_from_disk(str(self.output_path))

            print(f"Dataset載入成功，包含 {len(dataset)} 個examples")

            # 檢查第一個example的結構
            if len(dataset) > 0:
                example = dataset[0]
                print("\n第一個example結構:")
                print(f"  - audio: {type(example['audio'])}")
                print(f"  - audio array shape: {np.array(example['audio']['array']).shape}")
                print(f"  - sampling_rate: {example['audio']['sampling_rate']}")
                print(f"  - speaker: {example['speaker']}")
                print(f"  - text: {example['text']}")
                print(f"  - words count: {len(example['words'])}")

                if example['words']:
                    word = example['words'][0]
                    print(f"  - first word phones: {word.get('phones', [])}")
                    print(f"  - first word accuracies: {word.get('phones-accuracy', [])}")

            return True

        except Exception as e:
            print(f"驗證失敗: {e}")
            return False

    def print_statistics(self):
        """列印統計資訊"""
        print("\n" + "="*60)
        print("HuggingFace Dataset 創建統計")
        print("="*60)
        print(f"總utterances: {self.stats['total_utterances']}")
        print(f"音訊檔案找到: {self.stats['audio_files_found']}")
        print(f"音訊檔案遺失: {self.stats['audio_files_missing']}")
        print(f"音素轉錄: {self.stats['phone_transcriptions']}")
        print(f"準確度分數: {self.stats['accuracy_scores']}")
        print(f"最終examples: {len(self.hf_data)}")
        print(f"輸出路徑: {self.output_path.absolute()}")
        print("="*60)

    def create_dataset(self):
        """執行完整的資料集創建流程"""
        try:
            # 1. 載入Kaldi資料
            self.load_kaldi_data()

            # 2. 載入音素轉錄
            self.load_phone_transcriptions()

            # 3. 載入準確度分數
            self.load_accuracy_scores()

            # 4. 創建HuggingFace examples
            self.create_hf_examples()

            # 5. 儲存dataset
            self.save_hf_dataset()

            # 6. 驗證dataset
            is_valid = self.validate_dataset()

            # 7. 列印統計
            self.print_statistics()

            if is_valid:
                print("\n✅ HuggingFace Dataset創建成功!")
                return True
            else:
                print("\n❌ Dataset創建失敗，請檢查錯誤訊息。")
                return False

        except Exception as e:
            print(f"\n❌ 創建過程中發生錯誤: {e}")
            return False


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="Create HuggingFace format SpeechOcean762 dataset")
    parser.add_argument("--base_path", default="dataset/speechocean762-main",
                       help="原始SpeechOcean762資料集路徑")
    parser.add_argument("--combined_path", default="dataset/speechocean762-combined",
                       help="合併後的Kaldi格式資料集路徑")
    parser.add_argument("--output_path", default="dataset/speechocean762-hf",
                       help="HuggingFace格式輸出路徑")

    args = parser.parse_args()

    print("SpeechOcean762 HuggingFace Dataset Creator")
    print("="*60)

    # 建立創建器並執行
    creator = SpeechOcean762HFCreator(
        base_path=args.base_path,
        combined_path=args.combined_path,
        output_path=args.output_path
    )

    success = creator.create_dataset()

    if success:
        print(f"\n創建完成! HuggingFace Dataset位於: {creator.output_path.absolute()}")
        print("\n現在可以使用以下方式載入dataset:")
        print(f"from datasets import load_from_disk")
        print(f"dataset = load_from_disk('{creator.output_path}')")
    else:
        print("\n創建失敗，請檢查錯誤訊息。")

    return success


if __name__ == "__main__":
    main()