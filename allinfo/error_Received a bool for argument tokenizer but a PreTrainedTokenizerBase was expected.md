我看到了錯誤發生在第 73 行用 `Wav2Vec2Processor.from_pretrained(...)`。這是 Transformers 新版的已知相容性問題：某些模型倉（如 `facebook/wav2vec2-xlsr-53-espeak-cv-ft`）沒有可用的 tokenizer 物件，導致 Processor 組裝時把 tokenizer 當成了 bool。修法是「分開載入特徵擷取器與 CTC tokenizer」，不要用 `Wav2Vec2Processor`。

請做以下最小修改：

- 將 processor 換成 `Wav2Vec2FeatureExtractor`
- tokenizer 繼續使用 `Wav2Vec2CTCTokenizer`
- model 使用 `Wav2Vec2ForCTC`

具體改動：

1) 調整 import（在檔案頂部）：
```python
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC
```

2) 修改 `initialize_model(...)`（第 73 行附近）：
```python
processor = Wav2Vec2FeatureExtractor.from_pretrained(prep_path, cache_dir=cache_dir)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(prep_path, cache_dir=cache_dir)
model = Wav2Vec2ForCTC.from_pretrained(prep_path, cache_dir=cache_dir)
model.to(device)
return processor, tokenizer, model
```

3) 其餘程式不用改。你目前的程式用 `processor(audio, return_tensors="pt", ...)` 預處理音訊，`Wav2Vec2FeatureExtractor` 可直接替代 `Wav2Vec2Processor` 的此用途。

如果仍想使用單一 Auto 介面，替代方案（擇一）：
- 使用 `AutoProcessor`，但依然以 `processor.feature_extractor` 作為實際的 processor 物件：
```python
from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained(prep_path, cache_dir=cache_dir)
processor = getattr(proc, "feature_extractor", proc)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(prep_path, cache_dir=cache_dir)
model = Wav2Vec2ForCTC.from_pretrained(prep_path, cache_dir=cache_dir).to(device)
```

版本建議（可選其一）：
- 保持現有版本並採用上面修法；或
- 將 Transformers 鎖在較穩定版本如 `transformers==4.33.*`，同樣使用 `Wav2Vec2FeatureExtractor + Wav2Vec2CTCTokenizer`。

執行順序建議：
1) 調整 import 與 `initialize_model` 如上
2) 重新執行 `python official/SO/speechocean_quantification.py`

這樣即可排除「Received a bool for argument tokenizer」的錯誤。