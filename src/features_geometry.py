"""熔池几何物理特征：以 Ti-6Al-4V 的液相线/固相线等温线定义熔池。

物理依据
--------
Ti-6Al-4V 液相线约 1660 °C、固相线约 1600 °C（名义成分）。以 1660 °C 等温线包围的
区域即完全液态熔池，1600–1660 °C 之间的环带为糊状区（mushy zone）。熔池的
长宽比、拖尾不对称度、糊状区宽度与凝固前沿的温度梯度共同决定道间搭接是否充分，
是 lack-of-fusion 的直接成因量。

提取量
------
geo_len / geo_wid        熔池长轴、短轴（像素，椭圆二阶矩定义）
geo_ar                   长宽比
geo_area                 熔池面积（最大连通域像素数）
geo_perim                边界周长
geo_circ                 圆度 4πA/P²
geo_solidity             面积/凸包面积（凹陷程度，keyhole 与 conduction 模式的区分量）
geo_mushy                糊状区环带面积 / 熔池面积
geo_asym                 沿长轴的拖尾/前缘延伸比（取 ≥1 的一侧为分母）
geo_G_mushy              糊状区内的平均温度梯度（°C/像素）
geo_peak_off             最高温点相对熔池质心的偏移，按长轴归一化

说明：11 帧（0.7%）最高温低于 1660 °C，无液相区，对应特征记为 NaN；
树模型原生支持缺失值。
"""
import numpy as np, pandas as pd
from scipy import ndimage
from skimage.measure import regionprops, label as sk_label

TLIQ, TSOL = 1660.0, 1600.0


def _pool(img, th):
    m = img > th
    if not m.any():
        return None
    lab = sk_label(m, connectivity=2)
    props = regionprops(lab)
    return max(props, key=lambda p: p.area)


def build_geometry(img):
    N = img.shape[0]
    out = {k: np.full(N, np.nan) for k in
           ('geo_len', 'geo_wid', 'geo_ar', 'geo_area', 'geo_perim', 'geo_circ',
            'geo_solidity', 'geo_mushy', 'geo_asym', 'geo_G_mushy', 'geo_peak_off')}

    gy = np.gradient(img.astype(np.float32), axis=1)
    gx = np.gradient(img.astype(np.float32), axis=2)
    gm = np.hypot(gx, gy)

    for i in range(N):
        f = img[i]
        p = _pool(f, TLIQ)
        if p is None:
            continue
        area = float(p.area)
        if area < 30:          # 退化的检测（<0.1% 的场面积），几何量无意义
            continue
        L = float(p.axis_major_length)
        Wd = max(float(p.axis_minor_length), 1.0)      # 单像素宽熔池会使比值发散
        per = max(float(p.perimeter), 1.0)
        out['geo_area'][i] = area
        out['geo_len'][i] = L
        out['geo_wid'][i] = Wd
        out['geo_ar'][i] = L / Wd
        out['geo_perim'][i] = per
        out['geo_circ'][i] = min(4 * np.pi * area / per ** 2, 1.0)
        out['geo_solidity'][i] = p.solidity

        # 糊状区：1600–1660 °C 环带
        mushy = (f > TSOL) & ~(f > TLIQ)
        out['geo_mushy'][i] = mushy.sum() / (area + 1e-9)
        out['geo_G_mushy'][i] = gm[i][mushy].mean() if mushy.any() else 0.0

        # 拖尾不对称度：把熔池像素投影到长轴上，比较质心两侧的延伸
        rr, cc = p.coords[:, 0], p.coords[:, 1]
        th = p.orientation
        # 图像坐标中方向 (cos, sin) 作用于 (row, col)；major_axis 对应列方向为主
        u = (cc - p.centroid[1]) * np.cos(th) + (rr - p.centroid[0]) * np.sin(th)
        hi, lo = u.max(), u.min()
        if abs(hi) < 1e-6 or abs(lo) < 1e-6:
            out['geo_asym'][i] = 1.0
        else:
            out['geo_asym'][i] = max(hi, -lo) / (max(min(abs(hi), abs(lo)), 1.0))

        # 峰值点偏移
        pr, pc = np.unravel_index(f.argmax(), f.shape)
        d = np.hypot(pr - p.centroid[0], pc - p.centroid[1])
        out['geo_peak_off'][i] = d / max(L, 1.0)

    return pd.DataFrame(out)


def build(cache_dir='data/cache'):
    """返回 (完整特征矩阵, 列名, 标签, 层号)，与 features.build 的行序一致。"""
    from src.features import build as build_base
    X0, names0, y, g = build_base(cache_dir)
    img = np.load(f'{cache_dir}/thermal.npy')
    G = build_geometry(img)
    X = np.column_stack([X0, G.values]).astype(np.float32)
    return X, names0 + G.columns.tolist(), y, g


def geometry_block(cache_dir='data/cache', fill=-1.0):
    """几何块与列名。无液相区的帧以 -1 哨兵填充，使下游（SMOTE）不出现缺失值。"""
    img = np.load(f'{cache_dir}/thermal.npy')
    G = build_geometry(img)
    return G.fillna(fill).values.astype(np.float32), G.columns.tolist()


if __name__ == '__main__':
    import time
    t = time.time()
    img = np.load('data/cache/thermal.npy')
    G = build_geometry(img)
    print(f'{time.time() - t:.1f}s')
    print(G.describe().T.round(3))
    G.to_csv('results_geometry_features.csv')
    print('saved results_geometry_features.csv')
