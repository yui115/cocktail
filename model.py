from transformers import MLukeTokenizer, LukeModel
import sentencepiece as spm
import torch
import scipy.spatial
import pandas as pd


class SentenceLukeJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = MLukeTokenizer.from_pretrained(model_name_or_path)
        self.model = LukeModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx : batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(
                batch, padding="longest", truncation=True, return_tensors="pt"
            ).to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(
                model_output, encoded_input["attention_mask"]
            ).to("cpu")

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings)


# 既存モデルの読み込み
MODEL_NAME = "sonoisa/sentence-luke-japanese-base-lite"
model = SentenceLukeJapanese(MODEL_NAME)

# [1] cocktail.csvから、カクテルの説明文の列を読み込み、sentencesというリストに説明文を一つずつ追加する
data = pd.read_csv("cocktail.csv", encoding="shift_jis")
captions = data["説明文"].tolist()


# 標準入力で、ベース、味わい、フリーワードを受け取る
base_input = input()
taste_input = input()
query = input()
# sentences.append(query)

# カクテルの説明文、受け取った文章をエンコード（ベクトル表現に変換）
# sentence_embeddings = model.encode(sentences, batch_size=8)


# ベース
base = [
    "ジン",
    "ウォッカ",
    "テキーラ",
    "ラム",
    "ウイスキー",
    "ブランデー",
    "リキュール",
    "ワイン",
    "ビール",
    "日本酒",
    "ノンアルコール",
]

# 味わい
taste = ["甘口", "中甘口", "中口", "中辛口", "辛口"]

# カクテルを絞り込んで、説明文のリストを作る
sentences = []
base_indexs = data[data["base"] == base_input].index.tolist()
taste_indexs = data[data["taste"] == taste_input].index.tolist()
indexs = sorted(list(set(base_indexs) & set(taste_indexs)))
print(indexs)

for i in range(data.shape[0]):
    if i in indexs:
        sentences.append(data.at[i, "説明文"])

# 文章をベクトル化
sentences.append(query)
sentence_embeddings = model.encode(sentences, batch_size=8)

# 類似度が一番高いものを出力
closest_n = 1

distances = scipy.spatial.distance.cdist(
    [sentence_embeddings[-1]], sentence_embeddings, metric="cosine"
)[0]

results = zip(range(len(distances)), distances)
results = sorted(results, key=lambda x: x[1])

print("\n\n======================\n\n")
print("Query:", query)
print("\nあなたにおすすめのカクテルは:")

# for idx, distance in results[1 : closest_n + 1]:
#     print(sentences[idx].strip(), "(Score: %.4f)" % (distance / 2))
print(data.iloc[indexs[results[1][0]]])