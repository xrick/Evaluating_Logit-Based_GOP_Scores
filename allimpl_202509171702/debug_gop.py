#!/usr/bin/env python3
"""
Debug GOP calculation issues
"""

import numpy as np
from datasets import load_from_disk
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2ForCTC,
)
import torch


def debug_gop_calculation():
    """調試GOP計算問題"""

    # 載入資料集
    data = load_from_disk('dataset/speechocean762-combined-hf-format')

    # 初始化模型
    prep_path = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(prep_path)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(prep_path)
    model = Wav2Vec2ForCTC.from_pretrained(prep_path)
    model.to(device)

    # 獲取詞彙表
    vocab = tokenizer.get_vocab()
    print(f"詞彙表大小: {len(vocab)}")
    print(f"詞彙表範例: {list(vocab.items())[:10]}")

    # 測試第一個example
    example = data[0]
    print(f"\n第一個example:")
    print(f"Text: {example['text']}")
    print(f"Speaker: {example['speaker']}")
    print(f"Words count: {len(example['words'])}")

    if example['words']:
        first_word = example['words'][0]
        print(f"第一個word的phones: {first_word['phones']}")
        print(f"第一個word的accuracies: {first_word['phones-accuracy']}")

        # 檢查這些音素是否在詞彙表中
        for phone in first_word['phones']:
            if phone in vocab:
                print(f"  {phone} -> ID: {vocab[phone]} ✓")
            else:
                print(f"  {phone} -> 不在詞彙表中 ❌")

    # 測試音訊處理 - 支援 AudioDecoder 和傳統格式
    audio_data = example['audio']

    try:
        if hasattr(audio_data, 'get_all_samples'):
            # New torchcodec AudioDecoder format
            audio_samples = audio_data.get_all_samples()
            audio_array = audio_samples.data.numpy()
            sampling_rate = audio_samples.sample_rate

            # Handle multi-channel audio (convert to mono)
            if audio_array.ndim > 1:
                if audio_array.shape[0] == 1:  # (1, samples)
                    audio_array = audio_array.squeeze(0)
                else:  # (samples, channels) or other format
                    audio_array = audio_array.mean(axis=-1)

        elif hasattr(audio_data, 'array') and hasattr(audio_data, 'sampling_rate'):
            # Legacy HuggingFace audio format
            audio_array = np.array(audio_data.array)
            sampling_rate = audio_data.sampling_rate

        elif isinstance(audio_data, dict) and "array" in audio_data:
            # Very old format
            audio_array = np.array(audio_data["array"])
            sampling_rate = audio_data["sampling_rate"]

        else:
            print(f"Unsupported audio format: {type(audio_data)}")
            return

    except Exception as e:
        print(f"Error processing audio: {e}")
        return
    print(f"\n音訊資訊:")
    print(f"採樣率: {sampling_rate}")
    print(f"音訊長度: {len(audio_array)} samples ({len(audio_array)/sampling_rate:.2f} seconds)")

    # 測試特徵提取和模型推理
    try:
        inputs = feature_extractor(audio_array, return_tensors="pt", sampling_rate=sampling_rate, padding="longest")
        inputs.input_values = inputs.input_values.to(device)

        with torch.no_grad():
            logits = model(inputs.input_values).logits.detach().cpu()[0]
            probs = torch.nn.functional.softmax(logits, dim=-1).numpy()

        print(f"\n模型輸出:")
        print(f"Logits shape: {logits.shape}")
        print(f"Probs shape: {probs.shape}")
        print(f"時間幀數: {logits.shape[0]}")
        print(f"詞彙大小: {logits.shape[1]}")

        # 檢查是否有有效值
        print(f"Logits 統計:")
        print(f"  Min: {logits.min():.4f}")
        print(f"  Max: {logits.max():.4f}")
        print(f"  Mean: {logits.mean():.4f}")
        print(f"  是否有 NaN: {torch.isnan(logits).any()}")
        print(f"  是否有 Inf: {torch.isinf(logits).any()}")

    except Exception as e:
        print(f"模型推理錯誤: {e}")
        return

    # 測試CTC對齊
    print(f"\n測試CTC對齊:")
    try:
        import ctc_segmentation

        # 提取音素序列
        actual_phonemes = []
        for word in example['words']:
            actual_phonemes.extend(word['phones'])

        print(f"實際音素序列: {actual_phonemes[:5]}...")

        # 測試音素token化
        vocab = tokenizer.get_vocab()
        inv_vocab = {v: k for k, v in vocab.items()}
        char_list = [inv_vocab[i] for i in range(len(inv_vocab))]

        # 檢查音素是否能正確token化
        missing_phonemes = []
        for phoneme in actual_phonemes[:5]:  # 只檢查前5個
            if phoneme not in vocab:
                missing_phonemes.append(phoneme)

        if missing_phonemes:
            print(f"缺失的音素: {missing_phonemes}")
        else:
            print("所有測試音素都在詞彙表中 ✓")

        # 簡單的token化測試
        test_phoneme = actual_phonemes[0]
        if test_phoneme in vocab:
            token_id = vocab[test_phoneme]
            print(f"測試音素 '{test_phoneme}' -> token ID: {token_id}")

            # 測試在logits中獲取對應值
            if token_id < logits.shape[1]:
                logit_values = logits[:, token_id]
                print(f"該音素的logit值統計:")
                print(f"  Shape: {logit_values.shape}")
                print(f"  Min: {logit_values.min():.4f}")
                print(f"  Max: {logit_values.max():.4f}")
                print(f"  Mean: {logit_values.mean():.4f}")
                print(f"  是否有有效值: {not torch.isnan(logit_values).all()}")
            else:
                print(f"Token ID {token_id} 超出logits範圍")

    except ImportError:
        print("ctc_segmentation 未安裝")
    except Exception as e:
        print(f"CTC對齊測試錯誤: {e}")


if __name__ == "__main__":
    debug_gop_calculation()