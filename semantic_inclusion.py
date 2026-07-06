import numpy as np


class SemanticInclusionChecker:
    """
    _evaluate_orthogonality() が要求する ai_engine.check_semantic_inclusion() の実装。

    低確率（サプライズが大きい）symbolが投入されたとき、それが
      - 過去の文脈のどこかと意味的に繋がっている「高質な直交データ」なのか
      - 文脈と何の関係もない「ただのノイズ（文字化け等）」なのか
    を、実際のSentenceTransformer embeddingのコサイン類似度で判定する。

    「文脈全体の平均（centroid）」ではなく「過去のどれか一つと十分近いか」を見る。
    話題が大きく飛んでも、どこか一点にでも意味的な接続があれば「包含」とみなす。
    """

    def __init__(self, encoder, similarity_threshold=0.3):
        """
        encoder: .encode(text, convert_to_numpy=True) を持つオブジェクト
                 （SentenceTransformerインスタンス、または同インターフェースの互換品）
        similarity_threshold: これを超えるコサイン類似度が過去の文脈内に
                 1つでもあれば「包含している」と判定する
        """
        self.encoder = encoder
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def _cosine_sim(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def check_semantic_inclusion(self, latest_symbol, past_context):
        """
        latest_symbol: 直近に投入されたsymbol（文字列）
        past_context : それ以前のsymbol列（文字列のリスト）

        戻り値: bool
          True  = 過去の文脈のどこかと意味的に繋がっている（高質な直交データ）
          False = 文脈と無関係（ノイズ）
        """
        if not past_context:
            # 文脈がまだ存在しない最初のsymbolは、判定材料がないため包含扱いにする
            return True

        latest_vec = np.asarray(
            self.encoder.encode(latest_symbol, convert_to_numpy=True), dtype=np.float64
        )

        best_similarity = -1.0
        for past_symbol in past_context:
            past_vec = np.asarray(
                self.encoder.encode(past_symbol, convert_to_numpy=True), dtype=np.float64
            )
            similarity = self._cosine_sim(latest_vec, past_vec)
            if similarity > best_similarity:
                best_similarity = similarity

        return best_similarity >= self.similarity_threshold


# ─────────────────────────────────────────────
# 🔗 _evaluate_orthogonality() との結合例
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import hashlib

    class FakeEncoder:
        """SentenceTransformer.encode() のインターフェースを模した疑似encoder。
        意味的に近い文字列同士は、実際には近いベクトルにはならない
        （ハッシュベースなので）が、判定ロジックの配線確認用としては十分。
        """
        def __init__(self, dim=16):
            self.dim = dim

        def encode(self, text, convert_to_numpy=True):
            h = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(h[:4], "big")
            rng = np.random.default_rng(seed)
            return rng.normal(size=self.dim)

    checker = SemanticInclusionChecker(encoder=FakeEncoder(dim=16), similarity_threshold=0.3)

    past_context = ["開いた創発について", "文脈の再配置", "premiseの更新"]

    print("=== 同じ話題への言い換え（本来は高質な直交データになってほしいケース）===")
    print(checker.check_semantic_inclusion("創発の再配置プロセス", past_context))

    print("=== 全く無関係な単語（ノイズになってほしいケース）===")
    print(checker.check_semantic_inclusion("バナナの値段", past_context))

    print("\n※FakeEncoderはハッシュベースの疑似embeddingなので、")
    print("　実際の意味的近さは反映されません。本番ではSentenceTransformerを渡してください。")
