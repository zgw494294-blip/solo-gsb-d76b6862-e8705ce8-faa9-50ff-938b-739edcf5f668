"""可分享文本的遮盖导出核心逻辑。

偏移约定：所有 start/end 均为 Unicode 码点偏移，区间为左闭右开 [start, end)。
Python 字符串按码点索引，因此切片结果天然符合该约定。

输入文本、批注（唯一 id、start、end、label）与需遮盖的标签列表：

- 收集 label 命中遮盖标签的批注区间，按起点排序后，只要**重叠或相接**
  （`next.start <= cur_end`，左闭右开下端点相等即相接）就合并为一个遮盖区；
- 每个遮盖区在导出文本中整体替换为固定标记「[遮盖]」（4 个码点），
  命中遮盖标签的批注本身**不导出**；
- 其余批注跨遮盖区时按遮盖区边界切成**可见片段**：完全落在遮盖区内的
  片段舍弃，其余片段保留 label、source_id（原批注 id）与从 1 开始、按
  原文位置递增的片段序号 fragment_index；
- 片段按**原批注输入顺序**、同一批注内按**原文位置**排列；
- 片段的新码点范围只覆盖原文可见部分，切片内容不含遮盖标记；
- 无命中标签时不产生遮盖区，文本与批注范围原样不变（每条批注输出为
  一个 fragment_index=1 的片段）。
"""

from __future__ import annotations

from bisect import bisect_right

MASK_TOKEN = "[遮盖]"  # 每个遮盖区在导出文本中的替换标记（4 个码点）


class RedactionError(ValueError):
    """遮盖标签列表不合法（空标签或重复标签）。"""


def _check_labels(mask_labels: list[str]) -> None:
    seen: set[str] = set()
    for label in mask_labels:
        if not label:
            raise RedactionError("遮盖标签不能为空")
        if label in seen:
            raise RedactionError(f"遮盖标签重复：{label}")
        seen.add(label)


def _merge_regions(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """合并重叠或相接（端点相等）的码点区间，返回有序且互不相交的遮盖区。"""
    regions: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if regions and start <= regions[-1][1]:
            regions[-1] = (regions[-1][0], max(regions[-1][1], end))
        else:
            regions.append((start, end))
    return regions


def redact_export(text: str, annotations: list[dict],
                  mask_labels: list[str]) -> dict:
    """生成遮盖后的可分享文本与可见片段。

    返回 {"text", "annotations", "mask_count"}：text 为遮盖后新文本；
    annotations 为片段列表（按原批注顺序、同批注内按原文位置），每条含
    source_id、label、start、end（新文本码点范围）、fragment_index；
    mask_count 为合并后遮盖区数量。
    """
    _check_labels(mask_labels)
    wanted = set(mask_labels)
    hit_ids: set = set()
    spans: list[tuple[int, int]] = []
    for ann in annotations:
        if ann["label"] in wanted:
            hit_ids.add(ann["id"])
            spans.append((ann["start"], ann["end"]))
    regions = _merge_regions(spans)

    if not regions:
        # 无命中：文本与批注范围原样不变
        return {
            "text": text,
            "annotations": [
                {"source_id": ann["id"], "label": ann["label"],
                 "start": ann["start"], "end": ann["end"],
                 "fragment_index": 1}
                for ann in annotations
            ],
            "mask_count": 0,
        }

    # 经过 j 个遮盖区后文本长度的净缩短量：每个遮盖区原文长 (e-s) 被换成
    # len(MASK_TOKEN) 个码点；可见位置 p（恰在遮盖区右端点也算可见）的新
    # 偏移 = p - prefix[j]，j 为 end <= p 的遮盖区数
    lengths = [e - s for s, e in regions]
    prefix = [0]
    for length in lengths:
        prefix.append(prefix[-1] + length - len(MASK_TOKEN))
    starts = [s for s, _ in regions]
    ends = [e for _, e in regions]

    # 原文码点偏移 -> 新文本码点偏移。遮盖区起点（可见片段的结束端）与
    # 终点（可见片段的开始端）都是合法边界：起点映射到遮盖标记起点，
    # 终点映射到标记终点；只有严格落在遮盖区内部的位置无可见坐标。
    def to_new(pos: int) -> int:
        j = bisect_right(ends, pos)  # end <= pos 的遮盖区数
        if j < len(regions) and starts[j] < pos < ends[j]:
            raise ValueError(f"位置 {pos} 落在遮盖区内，无可见坐标")
        return pos - prefix[j]

    pieces: list[str] = []
    cursor = 0
    for start, end in regions:
        pieces.append(text[cursor:start])
        pieces.append(MASK_TOKEN)
        cursor = end
    pieces.append(text[cursor:])
    new_text = "".join(pieces)

    fragments: list[dict] = []
    for ann in annotations:
        if ann["id"] in hit_ids:
            continue  # 命中批注不导出
        visible: list[tuple[int, int]] = []
        cur = ann["start"]
        for rs, re_ in regions:
            if re_ <= cur:
                continue
            if rs >= ann["end"]:
                break
            if rs > cur:
                visible.append((cur, rs))
            cur = max(cur, re_)
        if cur < ann["end"]:
            visible.append((cur, ann["end"]))
        for index, (vs, ve) in enumerate(visible, 1):
            ns, ne = to_new(vs), to_new(ve)
            piece = new_text[ns:ne]
            # 不变量：新范围内文字是原文可见部分，不含遮盖标记
            if piece != text[vs:ve] or MASK_TOKEN in piece:
                raise RedactionError(f"片段 {ann['id']}#{index} 坐标换算异常")
            fragments.append({"source_id": ann["id"], "label": ann["label"],
                              "start": ns, "end": ne,
                              "fragment_index": index})

    return {"text": new_text, "annotations": fragments,
            "mask_count": len(regions)}
