#!/usr/bin/env python3
"""
Fix phoneme mapping from ARPAbet-style to IPA format
"""

def create_arpabet_to_ipa_mapping():
    """創建ARPAbet風格到IPA的映射"""

    # 基於常見的音素映射，這個映射可能需要根據實際詞彙表調整
    mapping = {
        # 母音
        'AA0': 'ɑ', 'AA1': 'ɑː', 'AA2': 'ɑ',
        'AE0': 'æ', 'AE1': 'æ', 'AE2': 'æ',
        'AH0': 'ʌ', 'AH1': 'ʌ', 'AH2': 'ʌ',
        'AO0': 'ɔ', 'AO1': 'ɔː', 'AO2': 'ɔ',
        'AW0': 'aʊ', 'AW1': 'aʊ', 'AW2': 'aʊ',
        'AY0': 'aɪ', 'AY1': 'aɪ', 'AY2': 'aɪ',
        'EH0': 'ɛ', 'EH1': 'ɛ', 'EH2': 'ɛ',
        'ER0': 'ɜ', 'ER1': 'ɜː', 'ER2': 'ɜ',
        'EY0': 'eɪ', 'EY1': 'eɪ', 'EY2': 'eɪ',
        'IH0': 'ɪ', 'IH1': 'ɪ', 'IH2': 'ɪ',
        'IY0': 'iː', 'IY1': 'iː', 'IY2': 'iː',
        'OW0': 'oʊ', 'OW1': 'oʊ', 'OW2': 'oʊ',
        'OY0': 'ɔɪ', 'OY1': 'ɔɪ', 'OY2': 'ɔɪ',
        'UH0': 'ʊ', 'UH1': 'ʊ', 'UH2': 'ʊ',
        'UW0': 'uː', 'UW1': 'uː', 'UW2': 'uː',

        # 子音
        'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð',
        'F': 'f', 'G': 'g', 'HH': 'h', 'JH': 'dʒ',
        'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n',
        'NG': 'ŋ', 'P': 'p', 'R': 'ɹ', 'S': 's',
        'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'V': 'v',
        'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'
    }

    return mapping

def convert_arpabet_phone(phone_token):
    """
    轉換ARPAbet風格的音素到IPA格式

    Args:
        phone_token: 如 'W_B', 'IY0_E', 'K_B' 等

    Returns:
        IPA格式的音素，如果找不到則返回原始token
    """
    mapping = create_arpabet_to_ipa_mapping()

    # 移除位置標記 (_B, _I, _E)
    if '_' in phone_token:
        base_phone = phone_token.split('_')[0]
    else:
        base_phone = phone_token

    # 查找映射
    if base_phone in mapping:
        return mapping[base_phone]

    # 如果找不到直接映射，嘗試一些變化
    # 移除數字（stress markers）
    base_no_stress = ''.join(c for c in base_phone if not c.isdigit())
    if base_no_stress in mapping:
        return mapping[base_no_stress]

    # 返回原始token（可能已經是IPA格式）
    return phone_token

def test_mapping():
    """測試音素映射"""
    from transformers import Wav2Vec2CTCTokenizer

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('facebook/wav2vec2-xlsr-53-espeak-cv-ft')
    vocab = tokenizer.get_vocab()

    # 測試一些音素
    test_phones = ['W_B', 'IY0_E', 'K_B', 'AO0_I', 'L_E', 'B_B', 'EH0_I', 'R_E']

    print("音素映射測試:")
    found_count = 0
    for phone in test_phones:
        ipa_phone = convert_arpabet_phone(phone)
        is_in_vocab = ipa_phone in vocab
        print(f"{phone} -> {ipa_phone} {'✓' if is_in_vocab else '❌'}")
        if is_in_vocab:
            found_count += 1

    print(f"\n成功率: {found_count}/{len(test_phones)} ({found_count/len(test_phones)*100:.1f}%)")

    return found_count == len(test_phones)

if __name__ == "__main__":
    test_mapping()