import re
import unicodedata
import os
from underthesea import word_tokenize


class PhoBERTDataInterface:
    def __init__(self, vncorenlp_save_dir: str):
        self.disallowed_pattern = re.compile(
            r'[^a-zA-Z0-9\s\.,\?!\-\(\)\'"“”‘’/_%àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳýỹỷỵÀÁÃẠẢĂẮẰAlbẳẵặÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỔỠỢÙÚŨỤỦƯỨỪỬỮỰỲÝỸỶỴ]+'
        )

        self.vncorenlp_save_dir = os.path.abspath(vncorenlp_save_dir)

        if not os.path.exists(self.vncorenlp_save_dir):
            raise FileNotFoundError(
                f"Không tìm thấy thư mục tại: {self.vncorenlp_save_dir}"
            )

    def _segment_text(self, text: str):
        """
        Thay thế VnCoreNLP bằng underthesea
        """
        return word_tokenize(text, format="text")

    def _validate_and_clean(self, text: str):
        if not isinstance(text, str):
            return False, "Dữ liệu đầu vào phải là văn bản (String)."

        text = unicodedata.normalize('NFC', text.strip())
        text = re.sub(r'[\r\n\t]+', ' ', text)

        text = re.sub(r'(http[s]?://|www\.|<.*?>|\S+@\S+|#\w+|@\w+)', '', text)
        text = text.replace(":", "")
        text = re.sub(self.disallowed_pattern, ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = text.strip()

        if not text:
            return False, "Văn bản trống hoặc chỉ chứa ký tự rác."

        upper_chars = sum(1 for c in text if c.isupper())
        if len(text) > 0 and (upper_chars / len(text)) > 0.2:
            text = text.lower()

        if text[0].isalpha() and not text[0].isupper():
            text = text[0].upper() + text[1:]

        if not text.endswith('.'):
            last_period_index = text.rfind('.')
            if last_period_index == -1:
                return False, "Văn bản phải chứa ít nhất một dấu chấm câu kết thúc."
            text = text[:last_period_index + 1]

        if not text[0].isalpha():
            return False, f"Văn bản phải bắt đầu bằng chữ cái (Hiện tại: '{text[0]}')."

        if text.count('"') % 2 != 0:
            return False, "Lỗi cú pháp: Số lượng dấu ngoặc kép không đồng đều."

        words = text.split()
        word_count = len(words)

        if word_count < 200 or word_count > 2000:
            return False, f"Độ dài không hợp lệ ({word_count} từ). Yêu cầu từ 200 đến 2000 từ."

        vietnamese_vowels = re.findall(
            r'[àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳýỹỷỵ]',
            text.lower()
        )

        if (len(vietnamese_vowels) / word_count) < 0.2:
            return False, "Tỷ lệ nguyên âm tiếng Việt quá thấp (Nghi ngờ Spam)."

        return True, text

    def process_data(self, text: str, label: int):
        is_valid, cleaned_text = self._validate_and_clean(text)

        if not is_valid:
            raise ValueError(f"Văn bản bị từ chối: {cleaned_text}")

        # 🔥 dùng underthesea thay VnCoreNLP
        segmented_text = self._segment_text(cleaned_text)

        return {
            "label": label,
            "segmented_text": segmented_text
        }

    def predict_long_text(self, segmented_text: str, clf_pipeline, chunk_size=150, stride=120):
        words = segmented_text.split()
        chunks = []

        for i in range(0, len(words), stride):
            chunk = " ".join(words[i: i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break

        try:
            raw_results = clf_pipeline(chunks, top_k=None)
        except TypeError:
            raw_results = clf_pipeline(chunks, return_all_scores=True)

        if isinstance(raw_results[0], list):
            chunk_outputs = raw_results
        else:
            chunk_outputs = [raw_results]

        label_scores = {}
        for chunk_res in chunk_outputs:
            for item in chunk_res:
                label = item["label"]
                label_scores[label] = label_scores.get(label, 0.0) + item["score"]

        num_chunks = len(chunk_outputs)
        for label in label_scores:
            label_scores[label] /= num_chunks

        best_label = max(label_scores, key=label_scores.get)
        best_score = label_scores[best_label]

        return {"label": best_label, "score": best_score}
