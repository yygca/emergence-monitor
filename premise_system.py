import numpy as np
import math


class PremiseSystem:

    def __init__(
        self,
        dimension=1536,
        initial_layer=0,
        assimilation_rate=0.05,
        cov_growth_rate=1.5,
    ):
        self.dimension = dimension
        self.layer = initial_layer
        self.stability = 1.0  # 安定度 (1.0 = 完全安定, 0.0 = 完全崩壊寸前)

        # システムの現在の「前提（内部状態の平均ベクトル）」
        # 初期状態はゼロベクトル、またはランダムな分布
        self.P = np.zeros(self.dimension, dtype=np.float64)

        # モニター側が参照する共分散行列（Noneの場合は単位行列として扱われる）
        # システムの「思考のブレ（不確実性）」を表現するために保持可能
        self.cov = np.eye(self.dimension, dtype=np.float64)

        # shift_layer() のたびに cov = I * (cov_growth_rate ** layer) と拡大する。
        # この値自体が「開いた創発から戻れなくなる速さ」を決める設計パラメータ。
        self.cov_growth_rate = cov_growth_rate

        # 過去のコンテキスト（蓄積された歪みやシンボル）のログ
        self.context_history = []

        # モニターから受け取る「今この瞬間の文脈ベクトル」
        # （開いた創発で投げ込まれたsymbolの実embeddingが、monitor側でEMAされたもの）
        self.context = np.zeros(self.dimension, dtype=np.float64)

        # tickごとに P がcontextへどれだけ引き寄せられるか（0〜1）
        # 大きいほど即座に文脈へ同化し、小さいほどPは自分の慣性を保つ
        self.assimilation_rate = assimilation_rate

    def tick(self, symbol):
        """1ステップごとに外部からのシンボル（言葉）を受け取り、内部の予測（平均ベクトル）を返す。

        ここでシンボルがコンテキストに統合され、予測が更新される。
        """
        self.context_history.append(symbol)

        # 【物理のシミュレーション】
        # symbol自体の意味（embedding）はmonitor側（generate_prediction_vector）が
        # 実際のSentenceTransformerで計算し、receive_context() 経由でここに渡ってくる。
        # tick()の役割は、その文脈へ前提Pを少しずつ引き寄せること：
        # 「まだ知らない」symbolが繰り返されるほど、Pはcontextに近づき、
        # KLが下がって『閉じた創発』が起きやすくなる。
        self.P = self.P + self.assimilation_rate * (self.context - self.P)

        # 常に最新の予測平均ベクトルをモニターへ返す
        return self.P

    def receive_context(self, context_vector):
        """EmergenceMonitor.update_context() から呼ばれる。

        直近のsymbolの実embeddingがEMAで織り込まれた文脈ベクトルを受け取り、
        次回のtick()でPがそこへ引き寄せられるようにする。
        """
        self.context = context_vector

    def shift_layer(self):
        """モニターによって『開いた創発（拡散）』が検知されたときに強制駆動される。

        現在のレイヤー（次元）を一段階引き上げ、システムの前提を再配置する。
        """
        self.layer += 1
        print(
            f"⚡ [System] self.shift_layer() 駆動: レイヤーが {self.layer} へジャンプしました。"
        )

        # レイヤーが跳ね上がった際、システムは既存のグリッドから解放されるため、
        # 思考の不確実性（共分散）を一時的に広げ、新しい前提への適応力を高める
        self.cov = np.eye(self.dimension, dtype=np.float64) * (
            self.cov_growth_rate**self.layer
        )

    def update_premise(self, concept):
        """モニターによって『閉じた創発（収束）』の残債が『概念（concept）』として集約された際、

        それを次のターンの新しい前提（self.P）として内部に固定・包摂する。
        """
        print(f"📦 [System] self.update_premise() 駆動: 概念を回収しました。")
        print(
            f"   - 回収されたシンボル数: {concept['size']}, 平均KL: {concept['mean_kl']:.4f}"
        )

        # 集約された概念（過去の対話の歪みの平均値など）をベースに、
        # システムの最深部にある前提ベクトル P を永続的に書き換える
        # これにより、35回の拒絶という「歪み」が、次のレイヤーの「基礎（前提）」に変わる
        self.P = self.P * 0.5 + (np.ones(self.dimension) * concept["mean_kl"])

    def point_of_no_return(self, threshold, q_cov_is_identity=True):
        """
        現在の cov_growth_rate と dimension の下で、「symbolの内容に関わらず
        二度と閉じた創発が起こり得なくなる」臨界レイヤーを解析的に求める。

        前提: q_cov（外部側の共分散）が単位行列（＝monitor.generate_prediction_vectorを
        q_covを指定せず使う、通常の運用）であること。

        原理:
          system.cov = scale * I のとき、symbolがPと完全一致（diff=0）していても
          kl = 0.5 * dimension * (scale - 1 - log(scale))
          この値が threshold を初めて超える scale を二分探索で求め、
          layer = log(scale) / log(cov_growth_rate) に変換する。

        戻り値: (critical_layer, critical_scale)
                cov_growth_rate <= 1.0 の場合は None（そもそも発散しないため不要）
        """
        if not q_cov_is_identity:
            raise NotImplementedError(
                "q_covが単位行列でない場合の解析式は未実装です。数値シミュレーションで確認してください。"
            )
        if self.cov_growth_rate <= 1.0:
            return None  # covが縮小/一定なら、そもそも相転移は起きない

        d = self.dimension

        def kl_at_scale(scale):
            # diff=0 のときの kl(system, 外部=I)
            return 0.5 * d * (scale - 1.0 - math.log(scale))

        # scale=1のときkl=0 < threshold（通常threshold>0なので）。
        # scaleを増やしながらkl(diff=0) > threshold となる点を二分探索で特定する。
        lo, hi = 1.0, 2.0
        while kl_at_scale(hi) < threshold:
            hi *= 2.0
            if hi > 1e12:
                return None  # 現実的な範囲では相転移に到達しない

        for _ in range(100):
            mid = (lo + hi) / 2.0
            if kl_at_scale(mid) < threshold:
                lo = mid
            else:
                hi = mid

        critical_scale = hi
        critical_layer = math.log(critical_scale) / math.log(self.cov_growth_rate)
        return critical_layer, critical_scale


# ─────────────────────────────────────────────
# 🔗 モニターとの結合テスト用のモック（エミュレーション）
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # このファイル単体で実行したときに、どのように動くかのテスト
    system = PremiseSystem(dimension=1536)

    print("--- 初期状態 ---")
    print(f"現在のレイヤー: {system.layer}")
    print(f"システムの安定度: {system.stability}")

    # 外部から「おい」というシンボルが投入されたと仮定して tick を回す
    predicted_mean = system.tick("おい")
    print("\n--- 1回の tick 駆動後 ---")
    print(f"返された予測ベクトルの形状: {predicted_mean.shape}")

    # 「もう二度と閉じた創発が起こり得なくなる」臨界レイヤーを確認
    layer, scale = system.point_of_no_return(threshold=0.5)
    print("\n--- Point of No Return ---")
    print(f"cov_growth_rate={system.cov_growth_rate} の下では、"
          f"layer≈{layer:.2f}（cov scale≈{scale:.2f}）を超えると、"
          f"どんなsymbolが来ても閉じた創発は数学的に起こり得なくなる。")

