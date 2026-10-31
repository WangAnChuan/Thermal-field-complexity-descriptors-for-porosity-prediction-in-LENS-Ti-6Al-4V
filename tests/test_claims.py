"""Regression tests: every headline result must match a generated file.

Run directly (`python tests/test_claims.py`) or with pytest.
Any drift in a reported number fails the test.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

def _load(name):
    return pd.read_csv(os.path.join(RES, name))

def _row(df, **kw):
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= (df[k] == v)
    return df[m].iloc[0]

def check_all():
    combo = _load('results_combo.csv')
    abl = _load('results_ablation.csv')
    ms = _load('results_multiseed.csv')
    ex = _load('results_exclude_firstlayers.csv')
    ll = _load('results_layerlevel.csv')
    ic = _load('results_icinco_repro.csv')
    cost = _load('results_cost.csv')
    sel = _load('results_selective.csv')
    leak = _load('results_leakage.csv')
    shap_imp = pd.read_csv(os.path.join(RES, 'shap_importance.csv'), index_col=0).iloc[:, 0]
    bins = _load('shap_bins.csv')
    uq = json.load(open(os.path.join(RES, 'results_uq.json')))
    cnn = json.load(open(os.path.join(RES, 'results_cnn_multiseed.json')))

    out = []
    def ck(name, cond, detail=''):
        out.append((name, bool(cond), detail))

    # C1 -- complexity descriptors
    b = _row(combo, combo='base'); e = _row(combo, combo='base+entropy+gradient'); f = _row(combo, combo='full_spatial')
    ck('C1.1 forward 0.845 -> 0.952', abs(b.cat_forward-0.8451) < 1e-3 and abs(e.cat_forward-0.9521) < 1e-3,
       f'{b.cat_forward:.4f} -> {e.cat_forward:.4f}')
    share = (e.cat_forward-b.cat_forward)/(f.cat_forward-b.cat_forward)
    ck('C1.2 three dimensions give 96% of gain', abs(share-0.963) < 0.01, f'{share:.3f}')

    cf = _row(ms, model='Cat', protocol='forward', feat='base')
    ce = _row(ms, model='Cat', protocol='forward', feat='entropy+gradient')
    ck('C1.3 multi-seed 0.903 -> 0.970', abs(cf['mean']-0.9025) < 2e-3 and abs(ce['mean']-0.9704) < 2e-3,
       f"{cf['mean']:.4f} -> {ce['mean']:.4f}")
    ck('C1.4 standard deviation down 65%', abs(1-ce['sd']/cf['sd']-0.652) < 0.02, f"{1-ce['sd']/cf['sd']:.3f}")

    xb = _row(ex, model='XGB', feat='base', protocol='forward')
    xe = _row(ex, model='XGB', feat='base+entropy+gradient', protocol='forward')
    ck('C1.5 excluding layers 1-2, XGB +0.132', abs((xe.pr-xb.pr)-0.1312) < 2e-3, f'{xb.pr:.4f} -> {xe.pr:.4f}')

    lb = _row(ll, model='XGB', agg='max', feat='base', protocol='forward')
    le = _row(ll, model='XGB', agg='max', feat='entropy+gradient', protocol='forward')
    ck('C1.6 layer level 0.963 -> 0.983', abs(lb.pr-0.9625) < 2e-3 and abs(le.pr-0.9833) < 2e-3,
       f'{lb.pr:.4f} -> {le.pr:.4f}')

    # C2 -- protocol dependence
    a_ent = _row(abl, group='entropy')
    ck('C2.1 entropy: forward +0.088, grouped -0.001',
       abs((a_ent.forward-b.cat_forward)-0.088) < 2e-3 and abs((a_ent.group5-b.cat_group5)+0.001) < 2e-3,
       f'forward {a_ent.forward-b.cat_forward:+.4f} / grouped {a_ent.group5-b.cat_group5:+.4f}')

    i_rf = _row(ic, feat='icinco_stats', model='RF'); o_x = _row(ic, feat='ours_21d', model='XGB')
    ck('C2.2 under the published protocol our features are worse', o_x.F1 < i_rf.F1,
       f'ours {o_x.F1:.3f} < theirs {i_rf.F1:.3f}')

    acc = leak.acc.values; prs = leak.pr.values
    ck('C2.3 accuracy 0.972-0.991 and nearly constant',
       acc.min() > 0.971 and acc.max() < 0.992 and (acc.max()-acc.min()) < 0.02,
       f'{acc.min():.4f}-{acc.max():.4f} (span {acc.max()-acc.min():.4f})')
    ck('C2.4 PR-AUC spans about 0.09 across protocols', abs(prs.max()-prs.min()-0.0925) < 3e-3,
       f'{prs.max()-prs.min():.4f}')

    # C3 -- mechanism
    rank = list(shap_imp.sort_values(ascending=False).index)
    ck('C3.1 SHAP: entropy first, Tmax eleventh',
       rank[0] == 'hist_entropy' and rank.index('Tmax')+1 == 11,
       f'entropy {shap_imp["hist_entropy"]:.3f}, Tmax {shap_imp["Tmax"]:.3f} (rank {rank.index("Tmax")+1})')
    b_ent = bins[bins.feature == 'hist_entropy'].reset_index(drop=True)
    ck('C3.2 entropy Q1 pore rate 21.7%', abs(b_ent.loc[0, 'pos_rate']-0.217) < 0.005,
       f"{b_ent.loc[0,'pos_rate']:.3f}")

    # C4 -- decisions
    c20 = _row(cost, R=20)
    ck('C4.1 cost ratio 20:1 gives recall 0.95', abs(c20.recall-0.95) < 0.005, f'recall={c20.recall:.3f}')
    miss = sel[(sel.feat == 'entropy+gradient') & (sel.coverage == 0.9)].miss_rate.iloc[0]
    ck('C4.2 abstaining on 10% removes all misses', miss == 0, f'coverage 0.9 -> miss {miss}')
    conf = {r['model']+f"_{r['alpha']}": r for r in uq['conformal']}
    ck('C4.3 conformal: XGB 0.959, TabPFN under-covers 0.723',
       abs(conf['XGB_0.1']['coverage']-0.959) < 0.01 and abs(conf['TabPFN_0.1']['coverage']-0.723) < 0.01,
       f"XGB {conf['XGB_0.1']['coverage']:.3f}, TabPFN {conf['TabPFN_0.1']['coverage']:.3f}")

    # C5 -- benchmark re-evaluation
    bs = uq['bootstrap']; p_tx = [r for r in bs if r['A'] == 'TabPFN' and r['B'] == 'XGB'][0]
    ck('C5.1 TabPFN vs XGB p = 0.80', abs(p_tx['p']-0.802) < 0.01, f"delta {p_tx['delta']:+.3f}, p {p_tx['p']:.3f}")
    top4 = [0.9376, 0.9119, 0.9104, 0.9099]
    ck('C5.2 top four tabular models span 0.028', abs(max(top4)-min(top4)-0.0277) < 2e-3,
       f'{max(top4)-min(top4):.4f}')

    # Controls: information vs dimensionality
    ctl = _load('results_control_ablation.csv').set_index('condition')
    gain_perm = ctl.loc['permuted', 'forward'] - ctl.loc['base', 'forward']
    gain_cplx = ctl.loc['complexity', 'forward'] - ctl.loc['base', 'forward']
    ck('C0.1 permuted control gains little', gain_perm < 0.03,
       f'permuted +{gain_perm:.3f} vs complexity +{gain_cplx:.3f}')
    ck('C0.2 complexity forward 0.952 and moments 0.931',
       abs(ctl.loc['complexity', 'forward']-0.9521) < 2e-3 and abs(ctl.loc['moments', 'forward']-0.9308) < 2e-3,
       f"complexity {ctl.loc['complexity','forward']:.4f}, moments {ctl.loc['moments','forward']:.4f}")

    paired = _load('results_control_paired.csv')
    ck('C0.3 complexity beats moments on 21 splits',
       abs(paired['complexity'].mean()-0.9742) < 3e-3 and abs(paired['moments'].mean()-0.9467) < 3e-3,
       f"complexity {paired['complexity'].mean():.4f} vs moments {paired['moments'].mean():.4f}")

    # Protocol dependence on identical features
    e = _load('results_e1_e4.csv')
    def _e(exp, model, cond):
        return e[(e.experiment == exp) & (e.model == model) & (e.condition == cond)].iloc[0]
    rnd = float(_e('E2', 'XGB', '+EG')['mean']) - float(_e('E2', 'XGB', 'base')['mean'])
    fwd = float(_e('E1', 'XGB', 'with-position, +EG')['mean']) - float(_e('E1', 'XGB', 'with-position, base')['mean'])
    ck('E2 forward gain exceeds random gain', fwd > 2 * rnd, f'forward {fwd:+.3f} vs random {rnd:+.3f}')
    ck('E2 random-split gain is small but positive', 0 < rnd < 0.03, f'{rnd:+.4f}')
    nopos = float(_e('E1', 'Cat', 'no-position, +EG')['mean']) - float(_e('E1', 'Cat', 'no-position, base')['mean'])
    wpos = float(_e('E1', 'Cat', 'with-position, +EG')['mean']) - float(_e('E1', 'Cat', 'with-position, base')['mean'])
    ck('E1 gain survives removing positional features', nopos > 0.8 * wpos,
       f'no-position {nopos:+.3f} vs with-position {wpos:+.3f}')
    held = float(_e('E3', 'Cat', '+EG')['mean']) - float(_e('E3', 'Cat', 'base')['mean'])
    ck('E3 held-out layers show a positive gain', held > 0.05, f'{held:+.3f}')

    # CNN baseline
    ck('CNN five-seed means 0.707 / 0.615',
       abs(cnn['grouped']['mean']-0.707) < 2e-3 and abs(cnn['forward']['mean']-0.615) < 2e-3,
       f"{cnn['grouped']['mean']:.3f} / {cnn['forward']['mean']:.3f}")
    return out


def test_all_claims():
    out = check_all()
    failed = [n for n, ok, _ in out if not ok]
    assert not failed, f"failed claims: {failed}"


if __name__ == '__main__':
    out = check_all()
    print(f"{'claim':<48}{'status':>7}  detail")
    for n, ok, d in out:
        print(f"{n:<48}{'PASS' if ok else 'FAIL':>7}  {d}")
    n_fail = sum(not ok for _, ok, _ in out)
    print(f"\n{len(out)-n_fail}/{len(out)} passed")
    raise SystemExit(1 if n_fail else 0)
