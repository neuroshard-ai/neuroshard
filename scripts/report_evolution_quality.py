#!/usr/bin/env python3
"""Regenerate paired statistics, paper table and standalone response-loss figures."""
import argparse
import json
import statistics
from pathlib import Path

from neuroshard.evolution.evaluation import decide,comparison


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args()
    data={name:json.loads((args.results/f'response-evaluation-{name}.json').read_bytes())
          for name in ('seed','base32','response_a','response_b')}
    pairs=[('seed','base32'),('base32','response_a'),('base32','response_b'),
           ('response_a','response_b'),('seed','response_a'),('seed','response_b')]
    results={}
    for before,after in pairs:
        a,b=data[before],data[after]
        result=decide(a['retention'],b['retention'],a['fresh'],b['fresh'])
        result['test']=comparison(a['test'],b['test'])
        results[before+'__'+after]=result
    (args.results/'response-quality-results.json').write_text(json.dumps(results,indent=2)+'\n')
    names={'seed':'Original seed (135M)','base32':'Prefix-trained parent (135M)',
           'response_a':'Response-trained candidate (135M)','response_b':'Response-trained candidate (149M)'}
    rows=[' & '.join([names[name],*[f'{statistics.fmean(data[name][role]):.6f}' for role in ('retention','fresh','test')]])+r'\\'
          for name in data]
    table=r'''\subsection{Response-only quality comparison}
Each value is mean teacher-forced response loss in nats per scored token over
64 paired examples. The two response candidates start from the same parent
weights and receive the same 32 response-masked batches; the larger candidate
first adds four identity-initialized blocks.
\begin{center}\small
\begin{tabular}{lrrr}\toprule
Checkpoint & Retention & Fresh & Untouched test\\\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}\end{center}
Neither response candidate passes the predefined promotion gate. Relative to
their common parent, fresh-loss changes are -0.001434 and -0.001405 nats, but
their approximate 99\% upper bounds are +0.000639 and +0.000596; the required
bound is below -0.001. The larger candidate has no demonstrated fresh-loss
advantage over the smaller candidate under the same training budget. Relative
to the original seed, both candidates also fail the improvement gate.
These are negative promotion results, not evidence of continual capability
improvement. All candidates produced the same tested 32-token greedy answer
prefix to the single solar-panel prompt; that anecdotal output is not a quality
benchmark. The chosen thresholds were not relaxed after observing these results.
\begin{figure}[ht]\centering
\includegraphics[width=.97\linewidth]{eval/results/evolution20260911/response-quality.pdf}
\caption{Paired fresh-response loss changes with approximate 99\% normal
intervals, 64 examples per comparison. Lower is better. Every interval crosses
zero; none establishes the required improvement. Intervals are not corrected
for multiple comparisons.}\end{figure}
'''
    (args.results/'quality.tex').write_text(table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=pairs[:4]
    labels=['Prefix training vs seed','135M response training vs parent',
            '149M response training vs parent','149M vs 135M, same response batches']
    fig,axis=plt.subplots(figsize=(9,3.8),layout='constrained')
    for index,(before,after) in enumerate(chosen):
        result=results[before+'__'+after]['fresh']
        axis.errorbar(result['mean_change'],index,xerr=2.576*result['standard_error'],
                      fmt='o',color='#2458a6',capsize=4)
    axis.axvline(0,color='#444444',linewidth=1)
    axis.axvline(-.001,color='#a64228',linestyle='--',label='Required upper bound: −0.001')
    axis.set_yticks(range(4),labels)
    axis.invert_yaxis()
    axis.set_xlabel('Change in fresh response loss (nats per token; lower is better)')
    axis.set_title('Measured growth and training did not qualify for promotion')
    axis.grid(axis='x',alpha=.2)
    axis.legend(loc='lower right',fontsize=8)
    for suffix in ('pdf','svg','png'):
        fig.savefig(args.results/f'response-quality.{suffix}',dpi=170)
    svg=args.results/'response-quality.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    print(json.dumps({key:value['promote'] for key,value in results.items()},indent=2))


if __name__=='__main__':main()
