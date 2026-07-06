import numpy as np
from collections import deque


# ─────────────────────────────────────────────
# Recorder 短期ログ（生イベント）
# ─────────────────────────────────────────────
class Recorder:
    def __init__(self):
        self.events = []

    def record(self, symbol, kl, layer, vector=None):
        """
        vector: そのsymbolの実embedding（q_mean）。渡しておくと、Archiveでの
                consolidate時に「意味的な位置」を平均できるようになる
                （省略時はmean_klのみのconceptになる＝旧来互換）。
        """
        self.events.append({
            "symbol": symbol,
            "kl": float(kl),
            "layer": int(layer),
            "vector": None if vector is None else np.asarray(vector, dtype=np.float64),
        })

    def recent(self, n=20):
        return self.events[-n:]

    def __len__(self):
        return len(self.events)


# ─────────────────────────────────────────────
# Archive 長期記憶（概念の集約）
# ─────────────────────────────────────────────
class Archive:
    def __init__(self):
        self.concepts = []

    def consolidate(self, records):
        if not records:
            return None
        kl_values = [r["kl"] for r in records]
        concept = {
            "size": len(records),
            "mean_kl": float(np.nanmean(kl_values)),  # nanmean で安全に
            "symbols": [r["symbol"] for r in records],
            "layers": list({r["layer"] for r in records}),  # 出現レイヤーを保持
        }

        # 実embeddingが記録されていれば、その平均を「意味的な位置」として保持する。
        # 後継系(spawn_successor)がPを継承するとき、スカラーのmean_klを
        # 全次元に複製するより、実際の意味空間上の位置を引き継げるようにするため。
        vectors = [r["vector"] for r in records if r.get("vector") is not None]
        if vectors:
            concept["mean_vector"] = np.mean(np.stack(vectors, axis=0), axis=0)
        else:
            concept["mean_vector"] = None

        self.concepts.append(concept)
        return concept


# ─────────────────────────────────────────────
# EmergenceMonitor 制御塔
# ─────────────────────────────────────────────
class EmergenceMonitor:
    def __init__(
        self,
        encoder,
        system,
        recorder,
        archive,
        dimension=1536,
        threshold=0.5,
        history_size=10,
        consolidate_every=20,
    ):
        self.encoder = encoder
        self.system = system
        self.recorder = recorder
        self.archive = archive
        self.dimension = dimension
        self.threshold = threshold
        self.history = deque(maxlen=history_size)  # 直近KLの移動平均用
        self.consolidate_every = consolidate_every
        self.context_vector = np.zeros(dimension)  # 系の内部文脈（開いた創発で投げ込まれたシンボルが織り込まれていく土台）
        self.context_momentum = 0.7  # 文脈の慣性。大きいほど過去の文脈を強く保持する

    # ── symbol → 実埋め込みベクトル生成 ──────────────────
    def generate_prediction_vector(self, symbol):
        """
        symbol（発話・トークン）をSentenceTransformerで実際に埋め込みベクトル化する。
        これが「外部（コミュニティ／人）側」の分布の平均 q_mean になる。
        人から投げ込まれた閃き・考えそのものを、ダミーのゼロベクトルではなく
        実際の意味表現として扱うための入口。
        """
        if self.encoder is None:
            raise ValueError("encoder（SentenceTransformerインスタンス）が設定されていません")

        vec = self.encoder.encode(symbol, convert_to_numpy=True)
        vec = np.asarray(vec, dtype=np.float64).reshape(-1)

        # encoderの出力次元とmonitorのdimensionがズレている場合は明示的に揃える
        if vec.shape[0] != self.dimension:
            if vec.shape[0] > self.dimension:
                vec = vec[: self.dimension]
            else:
                vec = np.pad(vec, (0, self.dimension - vec.shape[0]))

        return vec

    # ── 内部文脈の更新 ──────────────────────────────────
    def update_context(self, q_vec):
        """
        今回のsymbolベクトルを、系の内部文脈ベクトルに指数移動平均で織り込む。
        開いた創発（拡散）で投げ込まれた新しいシンボルが、時間をかけて
        システムの前提（＝閉じた創発で回収される土台）に染み込んでいく過程に対応する。
        systemが receive_context を実装していれば、そちらにも伝搬する。
        """
        self.context_vector = (
            self.context_momentum * self.context_vector
            + (1 - self.context_momentum) * q_vec
        )
        if hasattr(self.system, "receive_context"):
            self.system.receive_context(self.context_vector)
        return self.context_vector

    # ── KL計算（数学的に正確なガウス分布版） ──────────────
    def _calculate_kl(self, p_mean, q_mean, p_cov=None, q_cov=None):
        """
        2つのガウス分布間のKLダイバージェンス。
        共分散が両方 None の場合は Σ=I として簡易計算（高速）。
        """
        mu_p = np.array(p_mean, dtype=np.float64)
        mu_q = np.array(q_mean, dtype=np.float64)
        d = len(mu_p)

        # 共分散なし → KL = 0.5 * ||µ_p - µ_q||^2
        if p_cov is None and q_cov is None:
            return 0.5 * np.sum((mu_p - mu_q) ** 2)

        sigma_p = np.eye(d) if p_cov is None else np.array(p_cov, dtype=np.float64)
        sigma_q = np.eye(d) if q_cov is None else np.array(q_cov, dtype=np.float64)

        try:
            sigma_q_inv = np.linalg.inv(sigma_q)
        except np.linalg.LinAlgError:
            sigma_q_inv = np.linalg.pinv(sigma_q)  # 縮退時は擬似逆行列

        term1 = np.trace(sigma_q_inv @ sigma_p)
        diff = mu_q - mu_p
        term2 = diff @ sigma_q_inv @ diff
        sign_q, logdet_q = np.linalg.slogdet(sigma_q)
        sign_p, logdet_p = np.linalg.slogdet(sigma_p)
        term3 = logdet_q - logdet_p

        kl_div = 0.5 * (term1 + term2 - d + term3)
        return max(0.0, kl_div)  # 数値誤差で負にならないよう保護

    # ── メインループ ──────────────────────────────────
    def step(self, symbol, q_mean=None, q_cov=None):
        """
        1ステップの処理。
        - symbol : 今回の発話・トークン
        - q_mean : 外部（コミュニティ）側の分布の平均ベクトル。
                   省略した場合は generate_prediction_vector(symbol) で
                   実際のSentenceTransformer埋め込みから自動生成する。
        - q_cov  : 外部側の共分散行列（省略可）
        戻り値: {kl, mean_kl, stability, layer, trend}
        """
        # 0. q_meanが与えられなければ、symbol自体から実埋め込みを生成する
        if q_mean is None:
            q_mean = self.generate_prediction_vector(symbol)

        # 1. システム側の状態を更新し、予測平均を取得
        p_mean = self.system.tick(symbol)
        p_cov = getattr(self.system, "cov", None)  # systemが共分散を持てば使う

        # 2. KLダイバージェンスを計算（瞬間値）
        kl = self._calculate_kl(p_mean, q_mean, p_cov, q_cov)

        # 3. 履歴に追加して移動平均を算出（ノイズ対策）
        self.history.append(kl)
        mean_kl = float(np.mean(self.history))

        # 4. 安定度をシステムへフィードバック
        self.system.stability = 1.0 / (1.0 + mean_kl)

        # 5. 閾値判定
        #    開いた創発（拡散）: 前提・構造を再配置しながら思考が広がる場面 → layerを進める
        #    閉じた創発（収束）: 削ぎ落とされた残債を意味として回収し、記録として閉じる場面 → recordする
        if mean_kl > self.threshold:
            trend = f"開いた創発（拡散）: mean_KL={mean_kl:.4f} — 外部摂動を検知"
            if hasattr(self.system, "shift_layer"):
                self.system.shift_layer()
        else:
            trend = f"閉じた創発（収束）: mean_KL={mean_kl:.4f} — 残債を意味として回収・記録"
            self.recorder.record(symbol, mean_kl, self.system.layer, vector=q_mean)

            # 6. 一定件数「新たに記録された」ごとに概念として集約 → システムの前提を更新
            #    ※ record()が呼ばれた直後にだけチェックする。
            #      毎ステップ無条件にチェックすると、recorderの長さが偶然
            #      consolidate_everyの倍数のまま止まった場合（＝開いた創発が
            #      続いてrecordが増えなくなった場合）、同じ記録を毎ステップ
            #      再集約してしまう（重複concept量産のバグになる）。
            if len(self.recorder) % self.consolidate_every == 0:
                concept = self.archive.consolidate(
                    self.recorder.recent(self.consolidate_every)
                )
                if concept and hasattr(self.system, "update_premise"):
                    self.system.update_premise(concept)

        # 5.5 内部文脈の更新（開いた創発で投げ込まれたsymbolを、系の文脈に織り込む）
        self.update_context(q_mean)

        return {
            "kl": kl,
            "mean_kl": mean_kl,
            "stability": self.system.stability,
            "layer": self.system.layer,
            "trend": trend,
        }
