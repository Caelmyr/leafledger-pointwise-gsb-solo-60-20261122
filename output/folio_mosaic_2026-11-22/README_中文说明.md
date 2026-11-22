# 双补照全幅参考图与 Line scan 溯源（2026-11-22）

同一张馆藏叶片、两张已在同一馆藏平面上像素对齐的补照：

| 文件 | 角色 | 能看清 | 看不清 |
|---|---|---|---|
| `A_vein_left.png`（含覆盖 alpha） | 左侧补照，交叉偏振、均匀漫射 | 细脉网络完整连续 | 叶缘裂口被柔焦、偏淡 |
| `B_tear_right.png`（含覆盖 alpha） | 右侧补照，掠射光 | 叶缘裂口的缝隙与唇缘浮雕 | 叶脉尺度浮雕被压平、软焦，细脉只剩模糊纹理 |

两张的覆盖域只在中间一条窄带相交；右下角另有一小块楔形空区，两张原片都没拍到。

## 拼接规则（硬性，已用断言验证）

1. **没有任何混合**：全幅图每个像素都从且仅从一张原片整块复制。逐像素比对结果：
   - 取自 A 的像素与 `A_vein_left.png` 最大差 = **0**
   - 取自 B 的像素与 `B_tear_right.png` 最大差 = **0**
2. **整条共有带只用 A**：真实重叠区 = 两张覆盖掩码的交集（228,802 px），其内来源标签**只有 A**，脚本内有 `assert` 保证不存在 B 像素。名义窄带 x∈[0.385, 0.585]，因覆盖边有亚像素摆动，以真实交集为准。
3. **换片处是一条硬切缝**：馆藏平面第 **725 列**（归一化 x=0.585），缝左用 A，缝右 A 未覆盖处用 B；无羽化、无权重、无平均。
4. 两张都没拍到的楔形空区（51,151 px）不填任何原片内容，用中性斜纹板占位并在来源图上标 0。

## 取舍与图上指出的舍掉处

裂口被有意放在 A 覆盖边之外（归一化 x≈0.63、y∈[0.345, 0.415]），因此成品里的裂口来自 B 的掠射影像，保持清晰；
**舍掉的是硬切缝右侧 B 专属叶面上的细脉**——该区域 A 根本没有像素，而 B 的叶脉尺度浮雕已被压平并软焦，只残留不成形的淡纹理，无法恢复成 A 那样的连续脉网。
`reference_annotated.png` 上紫色框 + "SACRIFICED" 标签即此处；橙色框为裂口，蓝色描边为真实共有带，红线为硬切缝。

## 文件清单

- `reference_full_frame.png` — **可直接存档的全幅参考图**（1240×944，无标注纯净版）
- `reference_annotated.png` — 同一全幅图的标注版（切缝、共有带、裂口、舍掉处、空区、裂口放大）
- `AB_same_region_comparison.png` — 同一馆藏坐标下 A/B 原片对照（上排裂口区、下排叶脉区）
- `A_vein_left.png` / `B_tear_right.png` — 两张对齐原片（RGBA，alpha=覆盖域）
- `mask_A_coverage.png` / `mask_B_coverage.png` / `mask_void_neither.png` — 覆盖域与空区掩码
- `source_label_map.png` — 逐像素来源图：0=空区，1=A，2=B
- `line_scan_overlay.png` — 三条线扫在全幅图上的分段着色
- `scan1_across_seam_and_tear.csv`（+ `_segments.json`）— 主线：穿过共有带、硬切缝与裂口，120 站
- `scan2_through_void_wedge.csv` — 穿过双片均未拍到的楔形空区，90 站，其中 64 站值留空
- `scan3_inside_shared_band.csv` — 完全落在共有带内，60 站全部来自 A（证明整带单来源）
- `line_scan_provenance.csv` — 三条线全部站点的合并溯源表
- `manifest.json` — 几何坐标、切缝列、覆盖统计与取舍说明

## Line scan 表怎么读

每个站点一行，带宽 5 px（scan3 为 3 px），字段含义：

- `source`：A / B / VOID / MIXED（本数据切缝对齐覆盖边界，无 MIXED 站）
- `source_file`：该站取自哪张原片；VOID 站为空
- `source_pixel_col_px / row_px`：在该原片（与馆藏平面同坐标）中的像素位置
- `R/G/B_0_255`、`linear_luminance`：**从对应原片**（不是成品图）读取的带宽均值；
  VOID 站这些单元格留空（已核对 64/64 为空），不做任何插值或补纹
- `samples_from_A/B/uncovered`：带宽内各来源的实际像素数，便于复核换片处

在 LeafLedger 的 **Line scan** 工具里按同样的起止归一化坐标（见 `manifest.json` 的 `line_scans`）
重画即可得到同一条线；工具自身的支持掩码会把空区站同样标为无支持。

## 复现

```bash
python3 -I /workspace/scripts/mosaic_workbench.py \
  --out /workspace/output/folio_mosaic_2026-11-22
```

叶片场由仓库内 `leafledger/demo.py` 的合成器在 2 倍分辨率下烘焙，全部数据为本地合成，
不涉及任何真实馆藏照片。
