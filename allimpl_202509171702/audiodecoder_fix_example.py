# allimpl_202509171702/audiodecoder_fix_example.py
#!/usr/bin/env python3
"""
AudioDecoder 修復示例

這個檔案示範如何處理 torchcodec.decoders.AudioDecoder 的 'object is not subscriptable' 錯誤
"""

def load_audio_safe(audio_input, target_sr=16000):
    """
    安全載入音訊資料，支援多種格式包括 AudioDecoder

    Args:
        audio_input: 音訊資料，可能是：
            - torchcodec AudioDecoder 物件 (新的 HF 格式)
            - dict 格式 {"array": [...], "sampling_rate": 16000}
            - 物件格式 (有 .array 和 .sampling_rate 屬性)
        target_sr: 目標採樣率

    Returns:
        tuple: (audio_array, sampling_rate)
    """
    import numpy as np

    try:
        # 方法 1: 處理 torchcodec AudioDecoder (新的 HuggingFace 格式)
        if hasattr(audio_input, 'get_all_samples'):
            print("檢測到 AudioDecoder 格式")
            audio_samples = audio_input.get_all_samples()
            audio_array = audio_samples.data.numpy()
            sampling_rate = audio_samples.sample_rate

            # 處理多聲道音訊 (轉換為單聲道)
            if audio_array.ndim > 1:
                if audio_array.shape[0] == 1:  # (1, samples)
                    audio_array = audio_array.squeeze(0)
                else:  # (samples, channels) 或其他格式
                    audio_array = audio_array.mean(axis=-1)

            return audio_array, sampling_rate

        # 方法 2: 處理舊的 HuggingFace 音訊格式 (有 .array 和 .sampling_rate 屬性)
        elif hasattr(audio_input, 'array') and hasattr(audio_input, 'sampling_rate'):
            print("檢測到舊的 HF 音訊格式")
            audio_array = np.array(audio_input.array)
            sampling_rate = audio_input.sampling_rate

            # 確保是單聲道
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)

            return audio_array, sampling_rate

        # 方法 3: 處理字典格式 {"array": [...], "sampling_rate": 16000}
        elif isinstance(audio_input, dict) and "array" in audio_input:
            print("檢測到字典格式")
            audio_array = np.array(audio_input["array"])
            sampling_rate = audio_input.get("sampling_rate", target_sr)

            return audio_array, sampling_rate

        # 方法 4: 處理檔案路徑
        elif isinstance(audio_input, str):
            print("檢測到檔案路徑格式")
            try:
                import soundfile as sf
                audio_array, sampling_rate = sf.read(audio_input)
                return audio_array, sampling_rate
            except ImportError:
                try:
                    import torchaudio
                    audio_array, sampling_rate = torchaudio.load(audio_input)
                    return audio_array.numpy()[0], sampling_rate
                except ImportError:
                    raise ImportError("需要安裝 soundfile 或 torchaudio 來載入音訊檔案")

        # 方法 5: 處理 numpy array (假設已經是正確的採樣率)
        elif isinstance(audio_input, np.ndarray):
            print("檢測到 numpy array 格式")
            return audio_input, target_sr

        else:
            raise TypeError(f"不支援的音訊格式: {type(audio_input)}")

    except Exception as e:
        print(f"音訊載入失敗: {e}")
        raise


def fix_common_audiodecoder_errors():
    """
    演示常見的 AudioDecoder 錯誤以及修復方法
    """
    print("=== AudioDecoder 常見錯誤修復示例 ===\n")

    print("❌ 錯誤的做法:")
    print("audio_array = example['audio']['array']  # TypeError: 'AudioDecoder' object is not subscriptable")
    print("sampling_rate = example['audio']['sampling_rate']  # TypeError")

    print("\n✅ 正確的做法:")
    print("""
# 安全的音訊載入方法
audio_data = example['audio']

if hasattr(audio_data, 'get_all_samples'):
    # AudioDecoder 格式
    audio_samples = audio_data.get_all_samples()
    audio_array = audio_samples.data.numpy()
    sampling_rate = audio_samples.sample_rate
elif hasattr(audio_data, 'array'):
    # 舊格式
    audio_array = audio_data.array
    sampling_rate = audio_data.sampling_rate
elif isinstance(audio_data, dict):
    # 字典格式
    audio_array = audio_data["array"]
    sampling_rate = audio_data["sampling_rate"]
""")

    print("\n=== 使用 load_audio_safe 函數 ===")
    print("# 一行程式碼解決所有格式:")
    print("audio_array, sampling_rate = load_audio_safe(example['audio'])")


if __name__ == "__main__":
    fix_common_audiodecoder_errors()

    # 演示如何使用這個修復方法
    print("\n=== 實際應用示例 ===")
    print("在你的程式碼中，將:")
    print("  audio_array = np.array(example['audio']['array'])")
    print("  sampling_rate = example['audio']['sampling_rate']")
    print()
    print("替換為:")
    print("  audio_array, sampling_rate = load_audio_safe(example['audio'])")