"""Bounded expert selection from frozen input embeddings and integer prototypes.

Fitting sees only labeled training prompts. Inference sees a prompt feature
vector, never a task label, reference answer or evaluation identifier. New
prototypes still require an end-to-end response/retention gate before adoption.
"""
from collections import defaultdict
import math

from .reference_data import identity
from .schema import integer, root

FORMAT = 'neuroshard-embedding-router-v1'
SCALE = 16384
MAX_DIMENSIONS = 4096
FIELDS = {'format', 'embedding_root', 'tokenizer_root', 'training_root',
          'dimensions', 'fallback', 'minimum_margin', 'maximum_distance', 'prototypes'}


def vector(value, dimensions=None):
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_DIMENSIONS:
        raise ValueError('Require a bounded feature vector')
    if dimensions is not None and len(value) != dimensions:
        raise ValueError('Router feature dimensions changed')
    for item in value:
        integer(item, -SCALE, SCALE)
    if not any(value):
        raise ValueError('A router feature cannot be all zero')
    return value


def rounded_ratio(numerator, denominator):
    """Nearest integer, ties to even, with no floating-point conversion."""
    sign = -1 if numerator < 0 else 1
    quotient, remainder = divmod(abs(numerator), denominator)
    return sign * (quotient + (2 * remainder > denominator or
                              (2 * remainder == denominator and quotient % 2 == 1)))


def normalize(values):
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_DIMENSIONS:
        raise ValueError('Require bounded pooled embeddings')
    if any(type(value) is not int or abs(value) > 2**47 for value in values):
        raise ValueError('Pooled embeddings exceed the integer profile')
    squared = sum(value * value for value in values)
    if not squared:
        raise ValueError('Pooled embedding is zero')
    # The 32 fractional bits make the normalization precise while keeping the
    # operation and its rounding identical on every supported Python host.
    denominator = math.isqrt(squared << 64)
    return vector([rounded_ratio(value * (SCALE << 32), denominator) for value in values])


def distance(left, right):
    return sum((a - b) ** 2 for a, b in zip(left, right))


def validate(model):
    if not isinstance(model, dict) or set(model) != FIELDS or model['format'] != FORMAT:
        raise ValueError('Invalid integer router model')
    for key in ('embedding_root', 'tokenizer_root', 'training_root'):
        root(model[key])
    dimensions = integer(model['dimensions'], 1, MAX_DIMENSIONS)
    integer(model['minimum_margin'], 0, 2**40)
    integer(model['maximum_distance'], 0, 2**40)
    prototypes = model['prototypes']
    if not isinstance(prototypes, dict) or not 2 <= len(prototypes) <= 64:
        raise ValueError('Require a fallback and bounded expert routes')
    if model['fallback'] not in prototypes:
        raise ValueError('Router fallback must have training support')
    for name, centers in prototypes.items():
        if (not isinstance(name, str) or not name or len(name) > 64
                or not all(char in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for char in name)):
            raise ValueError('Invalid router model name')
        if not isinstance(centers, list) or not 1 <= len(centers) <= 8:
            raise ValueError('Require bounded prototypes for every route')
        for center in centers:
            vector(center, dimensions)
    return model


def fit(samples, *, embedding_root, tokenizer_root, fallback='parent',
        prototypes_per_route=4, iterations=12, minimum_margin=0,
        maximum_distance=2**40):
    """Deterministic per-route spherical clustering; ordering cannot tune it."""
    integer(prototypes_per_route, 1, 8)
    integer(iterations, 1, 64)
    if not isinstance(samples, list) or not 4 <= len(samples) <= 65536:
        raise ValueError('Require bounded router training observations')
    groups = defaultdict(list)
    seen = set()
    dimensions = None
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != {'id', 'route', 'features'}:
            raise ValueError('Training accepts prompt features and route labels only')
        key = root(sample['id'])
        if key in seen:
            raise ValueError('Duplicate router training observation')
        seen.add(key)
        features = vector(sample['features'], dimensions)
        dimensions = len(features)
        if not isinstance(sample['route'], str):
            raise ValueError('Invalid training route')
        groups[sample['route']].append((key, features))
    if not 2 <= len(groups) <= 64 or len(samples) * dimensions > 2**23:
        raise ValueError('Router training exceeds the route or feature budget')
    if len(samples) * dimensions * prototypes_per_route * iterations > 2**29:
        raise ValueError('Router clustering exceeds its operation budget')
    prototypes = {}
    for name, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row[0])
        centers = [rows[0][1]]
        for _ in range(min(prototypes_per_route, len(rows)) - 1):
            farthest = max(rows, key=lambda row: (
                min(distance(row[1], center) for center in centers), row[0]))
            if farthest[1] in centers:
                break
            centers.append(farthest[1])
        for _ in range(iterations):
            members = [[] for _ in centers]
            for _, features in rows:
                chosen = min(range(len(centers)), key=lambda i: (distance(features, centers[i]), i))
                members[chosen].append(features)
            updated = []
            for center, cluster in zip(centers, members):
                pooled = [sum(column) for column in zip(*cluster)] if cluster else []
                updated.append(normalize(pooled) if pooled and any(pooled) else center)
            if updated == centers:
                break
            centers = updated
        prototypes[name] = centers
    model = {'format': FORMAT, 'embedding_root': embedding_root, 'tokenizer_root': tokenizer_root,
             'training_root': identity(sorted(samples, key=lambda row: row['id'])),
             'dimensions': dimensions, 'fallback': fallback, 'minimum_margin': minimum_margin,
             'maximum_distance': maximum_distance, 'prototypes': prototypes}
    return validate(model)


def select(model, features):
    """Return auditable distances and a bounded-confidence fallback decision."""
    validate(model)
    vector(features, model['dimensions'])
    scores = {name: min(distance(features, center) for center in centers)
              for name, centers in model['prototypes'].items()}
    ordered = sorted(scores, key=lambda name: (scores[name], name))
    best, second = ordered[:2]
    margin = scores[second] - scores[best]
    confident = scores[best] <= model['maximum_distance'] and margin > model['minimum_margin']
    return {'route': best if confident else model['fallback'], 'nearest': best,
            'confident': confident, 'margin': margin, 'distances': scores,
            'router': identity(model), 'features': identity(features)}
