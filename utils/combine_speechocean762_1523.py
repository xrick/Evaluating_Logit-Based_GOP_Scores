# combine_speechocean762.py
#!/usr/bin/env python3
"""
SpeechOcean762 Dataset Combiner
將train和test資料集合併成一個完整的資料集
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class SpeechOcean762Combiner:
    """SpeechOcean762資料集合併器"""

    def __init__(self, base_path: str = "dataset/speechocean762-main"):
        """
        初始化合併器

        Args:
            base_path: 資料集基礎路徑
        """
        self.base_path = Path(base_path)
        self.train_path = self.base_path / "train"
        self.test_path = self.base_path / "test"
        self.output_path = Path("dataset/speechocean762-combined")

        # 檔案類型定義
        self.file_types = ["spk2age", "spk2gender", "spk2utt", "text", "utt2spk", "wav.scp"]

        # 儲存讀取的資料
        self.train_data = {}
        self.test_data = {}
        self.combined_data = {}

        # 統計資訊
        self.stats = {
            "train_speakers": 0,
            "test_speakers": 0,
            "total_speakers": 0,
            "train_utterances": 0,
            "test_utterances": 0,
            "total_utterances": 0,
            "speaker_conflicts": 0
        }

    def read_kaldi_file(self, filepath: Path) -> Dict[str, str]:
        """
        讀取Kaldi格式檔案

        Args:
            filepath: 檔案路徑

        Returns:
            Dict[str, str]: key-value映射字典
        """
        data = {}
        if not filepath.exists():
            raise FileNotFoundError(f"檔案不存在: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(maxsplit=1)
                    if len(parts) >= 2:
                        key, value = parts[0], parts[1]
                        data[key] = value
                    elif len(parts) == 1:
                        # 處理只有key沒有value的情況
                        data[parts[0]] = ""
        return data

    def load_datasets(self):
        """載入train和test資料集"""
        print("正在載入資料集...")

        # 載入train資料集
        print("載入train資料集...")
        for file_type in self.file_types:
            filepath = self.train_path / file_type
            self.train_data[file_type] = self.read_kaldi_file(filepath)
            print(f"  {file_type}: {len(self.train_data[file_type])} 條目")

        # 載入test資料集
        print("載入test資料集...")
        for file_type in self.file_types:
            filepath = self.test_path / file_type
            self.test_data[file_type] = self.read_kaldi_file(filepath)
            print(f"  {file_type}: {len(self.test_data[file_type])} 條目")

        # 更新基本統計
        self.stats["train_speakers"] = len(self.train_data["spk2age"])
        self.stats["test_speakers"] = len(self.test_data["spk2age"])
        self.stats["train_utterances"] = len(self.train_data["text"])
        self.stats["test_utterances"] = len(self.test_data["text"])

    def check_speaker_conflicts(self) -> Set[str]:
        """
        檢查speaker ID衝突

        Returns:
            Set[str]: 衝突的speaker ID集合
        """
        train_speakers = set(self.train_data["spk2age"].keys())
        test_speakers = set(self.test_data["spk2age"].keys())
        conflicts = train_speakers.intersection(test_speakers)

        self.stats["speaker_conflicts"] = len(conflicts)

        if conflicts:
            print(f"發現 {len(conflicts)} 個speaker ID衝突: {sorted(list(conflicts))}")
        else:
            print("沒有speaker ID衝突")

        return conflicts

    def resolve_speaker_conflicts(self, conflicts: Set[str]) -> Dict[str, str]:
        """
        解決speaker ID衝突

        Args:
            conflicts: 衝突的speaker ID集合

        Returns:
            Dict[str, str]: test資料集中需要重新映射的speaker ID字典 (old_id -> new_id)
        """
        if not conflicts:
            return {}

        print("正在解決speaker ID衝突...")

        # 找到最大的speaker ID數字
        all_speaker_ids = set(self.train_data["spk2age"].keys()) | set(self.test_data["spk2age"].keys())
        max_id = 0
        for spk_id in all_speaker_ids:
            try:
                num = int(spk_id)
                max_id = max(max_id, num)
            except ValueError:
                continue

        # 為衝突的test speakers分配新ID
        speaker_mapping = {}
        next_id = max_id + 1

        for conflict_id in sorted(conflicts):
            new_id = f"{next_id:04d}"
            speaker_mapping[conflict_id] = new_id
            print(f"  重新映射: {conflict_id} -> {new_id}")
            next_id += 1

        return speaker_mapping

    def apply_speaker_mapping(self, speaker_mapping: Dict[str, str]):
        """
        對test資料集應用speaker重新映射

        Args:
            speaker_mapping: speaker ID映射字典
        """
        if not speaker_mapping:
            return

        print("正在應用speaker重新映射...")

        # 更新test資料集中的speaker相關檔案
        for file_type in ["spk2age", "spk2gender", "spk2utt"]:
            new_data = {}
            for old_spk, value in self.test_data[file_type].items():
                new_spk = speaker_mapping.get(old_spk, old_spk)
                new_data[new_spk] = value
            self.test_data[file_type] = new_data

        # 更新utt2spk檔案中的speaker映射
        for utt_id, old_spk in self.test_data["utt2spk"].items():
            new_spk = speaker_mapping.get(old_spk, old_spk)
            self.test_data["utt2spk"][utt_id] = new_spk

    def combine_datasets(self):
        """合併資料集"""
        print("正在合併資料集...")

        for file_type in self.file_types:
            print(f"  合併 {file_type}...")
            combined = {}

            # 先加入train資料
            combined.update(self.train_data[file_type])

            # 再加入test資料
            combined.update(self.test_data[file_type])

            self.combined_data[file_type] = combined
            print(f"    合併後: {len(combined)} 條目")

        # 更新統計
        self.stats["total_speakers"] = len(self.combined_data["spk2age"])
        self.stats["total_utterances"] = len(self.combined_data["text"])

    def create_output_directory(self):
        """建立輸出目錄"""
        if self.output_path.exists():
            print(f"刪除現有輸出目錄: {self.output_path}")
            shutil.rmtree(self.output_path)

        self.output_path.mkdir(parents=True, exist_ok=True)
        print(f"建立輸出目錄: {self.output_path}")

    def write_combined_files(self):
        """寫入合併後的檔案"""
        print("正在寫入合併後的檔案...")

        for file_type in self.file_types:
            output_file = self.output_path / file_type

            with open(output_file, 'w', encoding='utf-8') as f:
                # 按key排序寫入
                for key in sorted(self.combined_data[file_type].keys()):
                    value = self.combined_data[file_type][key]
                    if value:
                        f.write(f"{key}\t{value}\n")
                    else:
                        f.write(f"{key}\n")

            print(f"  寫入 {file_type}: {len(self.combined_data[file_type])} 條目")

    def validate_combined_data(self) -> bool:
        """驗證合併後的資料完整性"""
        print("正在驗證合併後的資料...")

        errors = []

        # 檢查speaker數量一致性
        spk_age_count = len(self.combined_data["spk2age"])
        spk_gender_count = len(self.combined_data["spk2gender"])
        spk_utt_count = len(self.combined_data["spk2utt"])

        if not (spk_age_count == spk_gender_count == spk_utt_count):
            errors.append(f"Speaker數量不一致: age={spk_age_count}, gender={spk_gender_count}, utt={spk_utt_count}")

        # 檢查utterance數量一致性
        text_count = len(self.combined_data["text"])
        utt2spk_count = len(self.combined_data["utt2spk"])
        wav_count = len(self.combined_data["wav.scp"])

        if not (text_count == utt2spk_count == wav_count):
            errors.append(f"Utterance數量不一致: text={text_count}, utt2spk={utt2spk_count}, wav={wav_count}")

        # 檢查speaker ID一致性
        speakers_in_meta = set(self.combined_data["spk2age"].keys())
        speakers_in_utt2spk = set(self.combined_data["utt2spk"].values())

        missing_speakers = speakers_in_utt2spk - speakers_in_meta
        extra_speakers = speakers_in_meta - speakers_in_utt2spk

        if missing_speakers:
            errors.append(f"utt2spk中有未定義的speakers: {sorted(list(missing_speakers))}")
        if extra_speakers:
            errors.append(f"speaker metadata中有未使用的speakers: {sorted(list(extra_speakers))}")

        if errors:
            print("驗證失敗:")
            for error in errors:
                print(f"  - {error}")
            return False
        else:
            print("驗證通過!")
            return True

    def print_statistics(self):
        """列印統計資訊"""
        print("\n" + "="*60)
        print("合併統計報告")
        print("="*60)
        print(f"Train資料集:")
        print(f"  Speakers: {self.stats['train_speakers']}")
        print(f"  Utterances: {self.stats['train_utterances']}")
        print(f"\nTest資料集:")
        print(f"  Speakers: {self.stats['test_speakers']}")
        print(f"  Utterances: {self.stats['test_utterances']}")
        print(f"\n衝突處理:")
        print(f"  Speaker ID衝突: {self.stats['speaker_conflicts']}")
        print(f"\n合併結果:")
        print(f"  總Speakers: {self.stats['total_speakers']}")
        print(f"  總Utterances: {self.stats['total_utterances']}")
        print(f"\n輸出路徑: {self.output_path.absolute()}")
        print("="*60)

    def combine(self):
        """執行完整的合併流程"""
        try:
            # 1. 載入資料集
            self.load_datasets()

            # 2. 檢查speaker衝突
            conflicts = self.check_speaker_conflicts()

            # 3. 解決衝突
            speaker_mapping = self.resolve_speaker_conflicts(conflicts)
            self.apply_speaker_mapping(speaker_mapping)

            # 4. 合併資料集
            self.combine_datasets()

            # 5. 建立輸出目錄
            self.create_output_directory()

            # 6. 寫入檔案
            self.write_combined_files()

            # 7. 驗證資料
            is_valid = self.validate_combined_data()

            # 8. 列印統計
            self.print_statistics()

            if is_valid:
                print("\n✅ 資料集合併成功!")
                return True
            else:
                print("\n❌ 資料集合併失敗，請檢查錯誤訊息。")
                return False

        except Exception as e:
            print(f"\n❌ 合併過程中發生錯誤: {e}")
            return False


def main():
    """主函數"""
    print("SpeechOcean762 資料集合併工具")
    print("="*60)

    # 建立合併器並執行
    combiner = SpeechOcean762Combiner()
    success = combiner.combine()

    if success:
        print(f"\n合併完成! 輸出檔案位於: {combiner.output_path.absolute()}")
    else:
        print("\n合併失敗，請檢查錯誤訊息。")

    return success


if __name__ == "__main__":
    main()