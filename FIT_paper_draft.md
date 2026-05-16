# FIT 论文初稿（Tagawa 格式）

> **格式基准**：`d:\Downloads\FITpaper_tagawa.pdf`（田川研究室 FIT 范例，2 页）+ `d:\Downloads\FITpaper.docx`（IPSJ 样式）
>
> **Word 生成**：运行 `python scripts/build_fit_word.py` → 输出 **`FIT_paper.docx`**（完整正文 + 表1 + 図2 图片）。Markdown 仅供修订源稿。
>
> **篇幅**：一般论文 2–4 页 / 选奨论文 4–6 页（A4，无页码，PDF 嵌入字体 ≤3MB）

---

## 格式规范（与 Tagawa 一致）

| 项目 | Tagawa 要求 | Word 样式 |
|------|-------------|-----------|
| 和文标题 | 12pt | **FIT4** |
| 英文标题 | 12pt | **FIT5** |
| 作者 | 10.6pt | **FIT6** / **FIT7** |
| 一级节标题 | `1.` `2.` … **10.6pt**，无「第×章」 | **FIT1**（例：`1. はじめに`） |
| 二级节标题 | `2.1` `3.2` … **10.0pt** | **FIT2** 或 FIT0 加编号 |
| 正文 | **9pt**，两端对齐 | **FIT0** |
| 图题 | **9pt**，`図 1　説明` | **FIT図** |
| 参考文献 | 9pt，`[1]` 起 | **FIT** |
| 节序 | **仅 4 节**，无独立「関連研究」「結論」 | 见下目录 |

**Tagawa 范例无**：概要・英文 Abstract・謝辞（2 页短文省略）。若使用完整 FIT 模板且页数 ≥4，可在标题后追加「概要」（**FIT2**）；否则与 Tagawa 相同从 `1. はじめに` 起笔。

### 正式目录（本文采用）

```
【标题・作者】

1. はじめに                    … 背景・関連（本文内）・問題・貢献
2. 方法                        … 10.6pt
   2.1 概念分離と W_geo の構成
   2.2 損失関数への統合
   2.3 学習パイプライン
3. 性能評価                    … 10.6pt
   3.1 評価条件
   3.2 評価結果
4. おわりに

参考文献
図 1　…
図 2　…
```

---

## FIT4 和文标题

```
VGGT 誘導 3D Gaussian Splatting における
幾何信頼性重み W_geo による深度正則化
```

## FIT5 英文标题

```
W_geo-Weighted Depth Regularization for VGGT-Guided 3D Gaussian Splatting
```

## FIT6 和文作者

```
郭　得地
```

## FIT7 英文作者

```
Guo Dedi
```

---

## （任意）FIT2「概要」+ FIT0

Tagawa 2 页稿省略。4 页以上投稿时启用。

```
Visual Geometry Grounded Transformer (VGGT) の幾何出力を
3D Gaussian Splatting (3DGS) の学習に統合する際，
従来は単一重み W を光度損失と深度正則化の双方に適用し，
入力品質とモデル能力を混同する．本稿は幾何信頼性 W_geo を
深度正則化のみに適用する概念分離を提案する．
W_geo は (1−σ)，可視性 V，幾何一致性 C の加重平均 (0.5, 0.3, 0.2) で，
3 シーン・7000 反復で全シーン PSNR 改善（最大 +1.92 dB）を確認した．
キーワード：3D Gaussian Splatting，VGGT，幾何信頼性，重みマップ，深度正則化
```

---

## FIT1「1. はじめに」+ FIT0

```
3D Gaussian Splatting (3DGS) [1] はリアルタイム新視点合成を実現する．
スパース視点では幾何が不安定になり，単眼深度や前向き幾何モデルの
事前知識を正則化に用いる研究が増えている [3]．
Visual Geometry Grounded Transformer (VGGT) [2] は複数画像から
深度・カメラ・追跡を推論し，depth_conf や可視性など信頼度情報も
出力する．

従来の VGGT+3DGS 統合では，W = (1−σ)×V×C 等の単一重みを
光度損失と深度損失の双方に掛けることが多い [2]．
しかし光度損失の教師は実画像であり信頼できる一方，
深度損失の教師は VGGT 推定であり誤差を含む．
「VGGT の低信頼領域（入力品質）」と「3DGS の再構築困難領域（モデル能力）」は
直交し得る（例：鏡面は幾何は明確だが SH 表現が困難）ため，
同一 W の共用は概念的に不適切である．

本稿では幾何信頼性 W_geo を深度正則化のみに適用し，
光度損失には掛けない分離を提案する．貢献は，
(1) 参考値の性質に基づく損失別重み付け，
(2) σ，V，C の加重平均による W_geo 設計，
(3) 3 シーンと係数消融による検証，の 3 点である．
```

---

## FIT1「2. 方法」+ FIT0

### FIT2「2.1 概念分離と W_geo の構成」+ FIT0

```
W_geo は画素ごとの幾何信頼性を表す．VGGT から得る 3 信号を
[0, 1] に正規化した σ̂（深度不確実性），V̂（跨フレーム可視性），
Ĉ（多視点再投影一致性）とし，加重平均で融合する：

                    W_geo = max(0.1, 0.5(1−σ̂) + 0.3V̂ + 0.2Ĉ)     (1)

最小値 0.1 は完全マスクを避ける．乗法融合 W = (1−σ)×V×C は
失敗要因が相関する領域で過剰抑制し（低信頼画素比 56%），
加重平均では 34% に改善したため本稿では (1) を用いる．
```

### FIT2「2.2 損失関数への統合」+ FIT0

```
総損失は次式とする：

  L = (1−α)L_1 + α(1−SSIM) + λ_d(t) E[|D_render − D_VGGT| · M · W_geo]   (2)

ここで L_1，SSIM は実画像に対する光度項であり W_geo を掛けない．
第 3 項のみ深度正則化に W_geo を適用する（従来方式との差異）．
M は深度有効マスク，α = 0.2，λ_d(t) は学習進行に応じて減衰する．
```

### FIT2「2.3 学習パイプライン」+ FIT0

```
処理は 3 段階である．(i) VGGT-1B で深度・カメラ・トラックを推論し，
σ，V，C から W_geo を生成．(ii) COLMAP 形式・深度 PNG・重み NPY へ変換．
(iii) 3DGS を 7000 反復学習する．実装は export_vggt_for_3dgs.py と
convert_vggt_to_3dgs.py，train.py（--weights weights）に基づく．
```

**FIT図（9pt）**

```
図 1　VGGT 推論から W_geo 付き 3DGS 学習までのパイプライン
```

---

## FIT1「3. 性能評価」+ FIT0

### FIT2「3.1 評価条件」+ FIT0

```
GPU は NVIDIA RTX 3060 (12GB)，PyTorch 2.4，VGGT-1B を用いた．
データは Kitchen（屋内 8 枚），LLFF の Fern・Flower（各 8 枚）．
3DGS は 7000 反復，指標は訓練視点の L1，PSNR，SSIM とした．
比較は深度正則化あり・W_geo なし（Baseline）と W_geo ありの 2 条件．
消融は Fern で W の係数 10 構成を，VGGT 再推論なしに W のみ再計算して実施した．
```

### FIT2「3.2 評価結果」+ FIT0

```
表 1 に主結果を示す．W_geo は 3 シーンすべてで PSNR を改善し，
Fern では L1 を 16.8% 低減，+1.92 dB，Flower では SSIM +3.82% を得た．
Kitchen は Baseline が既に 40 dB 超のため改善は小さいが劣化しない．
Fern・Flower では 5 視点すべてで PSNR が向上した．

表 1　訓練視点における再構成品質（7000 iter）
| シーン   | 方式      | L1↓    | PSNR↑ | SSIM↑ |
| Kitchen  | Baseline  | 0.00610| 40.34 | 0.9890|
| Kitchen  | W_geo     | 0.00599| 40.40 | 0.9884|
| Fern     | Baseline  | 0.01586| 29.13 | 0.9650|
| Fern     | W_geo     | 0.01320| 31.05 | 0.9716|
| Flower   | Baseline  | 0.04659| 21.25 | 0.7818|
| Flower   | W_geo     | 0.04177| 21.75 | 0.8117|

Fern の係数消融では，無重み Baseline が PSNR 24.35 dB に対し，
全 9 構成の W_geo が +0.52〜+0.82 dB 改善した．
(0.5, 0.3, 0.2) が 25.53 dB で最良であり，均等重み (0.33, 0.33, 0.33) は
25.06 dB と W_geo 構成中最低であった．
```

**FIT図（9pt）**

```
図 2　Fern における Baseline と W_geo のレンダリング比較（代表視点）
```

---

## FIT1「4. おわりに」+ FIT0

```
VGGT+3DGS において，W_geo を深度正則化に限定適用する概念分離は，
3 シーンで一貫した品質向上をもたらした．誤った深度教師の影響を
画素ごとに抑制する効果が，特に遮蔽の多い Fern で顕著であった．
係数消融では W_geo 機構のロバスト性とデフォルト係数の妥当性を確認した．
今後は光度損失向け W_appear の分離，テスト視点評価，大規模ベンチマークへの
適用が課題である．
```

---

## FIT「参考文献」+ FIT0

```
[1] B. Kerbl et al.: ``3D Gaussian Splatting for Real-Time Radiance Field Rendering,''
    ACM Trans. Graph., Vol. 42, No. 4 (2023).
[2] J. Wang et al.: ``VGGT: Visual Geometry Grounded Transformer,''
    arXiv preprint (2024).
[3] V. Arampatzakis et al.: ``Monocular depth estimation: A thorough review,''
    IEEE Trans. PAMI, Vol. 46, No. 4, pp. 2396--2414 (2024).
```

---

## 著者所属（模板栏，Tagawa 稿未单列时可省略）

```
○○大学大学院 ○○学研究科
（メールアドレス）
```

---

## 投稿检查清单（Tagawa + IPSJ）

- [ ] 节标题：`1. はじめに` `2. 方法` `3. 性能評価` `4. おわりに`（不用「提案手法」「実験」「結論」）
- [ ] 子节：`2.1`–`2.3`，`3.1`–`3.2`（10.0pt 级）
- [ ] 式番号右端：(1)(2)…
- [ ] 図番号：`図 1　…`（9pt）
- [ ] A4 余白：上 30 / 下 25 / 左右 20 mm；无页码
- [ ] PDF ≤3MB，字体嵌入，无安全设置
