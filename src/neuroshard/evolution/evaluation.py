"""Paired quality measurements; a correct training step is not a quality claim."""
import math
import statistics


def comparison(baseline, candidate, margin=0., minimum_examples=32, z=2.576):
    if len(baseline) != len(candidate) or len(baseline) < minimum_examples:
        raise ValueError('Insufficient paired evaluation examples')
    delta = [float(b)-float(a) for a,b in zip(baseline,candidate)]
    if any(not math.isfinite(v) for v in [*baseline,*candidate,*delta]):
        raise ValueError('Nonfinite evaluation value')
    mean = statistics.fmean(delta)
    stderr = statistics.stdev(delta)/math.sqrt(len(delta))
    upper = mean+z*stderr
    return {'examples':len(delta),'baseline_loss':statistics.fmean(baseline),
            'candidate_loss':statistics.fmean(candidate),'mean_change':mean,
            'standard_error':stderr,'upper_confidence_bound':upper,'margin':margin,
            'passes':upper < margin}


def decide(retention_before,retention_after,fresh_before,fresh_after,retention_margin=.02,min_gain=.001):
    retention = comparison(retention_before,retention_after,retention_margin)
    fresh = comparison(fresh_before,fresh_after,-min_gain)
    return {'promote':retention['passes'] and fresh['passes'],'retention':retention,'fresh':fresh,
            'scope':'paired next-token loss, approximate 99% normal intervals; not a general capability or safety certificate'}


def evaluate(pipeline, store, sequences):
    from .batches import from_windows
    return [float.fromhex(pipeline.evaluate(from_windows(store,[key]))['loss_hex']) for key in sequences]
