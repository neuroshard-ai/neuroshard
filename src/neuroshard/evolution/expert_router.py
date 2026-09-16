"""Bounded expert selection from frozen input embeddings and integer models.

Fitting sees only labeled training prompts. Inference sees a prompt feature
vector, never a task label, reference answer or evaluation identifier. New
prototypes still require an end-to-end response/retention gate before adoption.
"""
from collections import defaultdict
import math

from .reference_data import identity
from .schema import integer, root

FORMAT = 'neuroshard-embedding-router-v1'
LINEAR_FORMAT = 'neuroshard-discriminative-router-v1'
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
    if not isinstance(model, dict) or model.get('format') not in (FORMAT, LINEAR_FORMAT):
        raise ValueError('Invalid integer router model')
    expected = FIELDS | ({'classifier'} if model['format'] == LINEAR_FORMAT else set())
    if set(model) != expected:
        raise ValueError('Invalid integer router fields')
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
    if model['format'] == LINEAR_FORMAT:
        classifier = model['classifier']
        if (not isinstance(classifier, dict) or set(classifier) != {'method', 'epochs', 'training_margin', 'weights', 'biases'}
                or classifier['method'] not in ('integer-averaged-margin-perceptron-v1',
                                                'integer-balanced-averaged-margin-perceptron-v1')
                or not isinstance(classifier['weights'], dict) or not isinstance(classifier['biases'], dict)
                or set(classifier['weights']) != set(prototypes) or set(classifier['biases']) != set(prototypes)):
            raise ValueError('Invalid discriminative router classifier')
        integer(classifier['epochs'], 1, 64)
        integer(classifier['training_margin'], 0, 2**40)
        for name in prototypes:
            weights = classifier['weights'][name]
            if not isinstance(weights, list) or len(weights) != dimensions:
                raise ValueError('Classifier weight dimensions changed')
            for weight in weights:
                integer(weight, -SCALE, SCALE)
            integer(classifier['biases'][name], -SCALE, SCALE)
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
    nearest = best
    logits = None
    if model['format'] == LINEAR_FORMAT:
        classifier = model['classifier']
        logits = {name: sum(weight * value for weight, value in zip(weights, features))
                  + classifier['biases'][name] * SCALE for name, weights in classifier['weights'].items()}
        best, second = sorted(logits, key=lambda name: (-logits[name], name))[:2]
    margin = scores[second] - scores[best]
    if logits is not None:
        margin = logits[best] - logits[second]
    confident = scores[best] <= model['maximum_distance'] and margin > model['minimum_margin']
    result = {'route': best if confident else model['fallback'], 'nearest': nearest,
              'confident': confident, 'margin': margin, 'distances': scores,
              'router': identity(model), 'features': identity(features)}
    if logits is not None:
        result.update(predicted=best, logits=logits)
    return result


def calibrate_support(samples, prototype_model, *, numerator=5, denominator=4):
    """Derive a bounded support radius from fitting inputs, never held-out rows."""
    validate(prototype_model)
    integer(denominator, 1, 16)
    integer(numerator, denominator, 16)
    if identity(sorted(samples, key=lambda row: row['id'])) != prototype_model['training_root']:
        raise ValueError('Support calibration differs from the committed training observations')
    radius = max(min(distance(row['features'], center)
                     for center in prototype_model['prototypes'][row['route']]) for row in samples)
    return validate({**prototype_model, 'maximum_distance': (radius * numerator + denominator - 1) // denominator})


def fit_classifier(samples, prototype_model, *, epochs=24, training_margin=8388608, balance_classes=False):
    """Learn separating directions instead of requiring nearest-centroid labels.

    Lazy integer averaging includes every training step, without a dependency
    on BLAS solvers, random shuffles or floating-point optimizer state.
    """
    validate(prototype_model)
    if prototype_model['format'] != FORMAT:
        raise ValueError('Classifier fitting requires the original prototype model')
    integer(epochs, 1, 64)
    integer(training_margin, 0, 2**40)
    rows = sorted(samples, key=lambda row: row['id'])
    if identity(rows) != prototype_model['training_root']:
        raise ValueError('Classifier training differs from the committed prototype observations')
    names = sorted(prototype_model['prototypes'])
    if type(balance_classes) is not bool:
        raise ValueError('Require an explicit class balancing policy')
    if balance_classes:
        groups = {name: [row for row in rows if row['route'] == name] for name in names}
        # Every class gets the same deterministic number of presentations. New
        # paraphrases cannot silently reduce general-assistant retention weight.
        rows = [groups[name][index % len(groups[name])]
                for index in range(max(map(len, groups.values()))) for name in names]
    dimensions = prototype_model['dimensions']
    if len(rows) * dimensions * len(names) * epochs > 2**29:
        raise ValueError('Classifier training exceeds its operation budget')
    weights = {name: [0] * (dimensions + 1) for name in names}
    accumulated = {name: [0] * (dimensions + 1) for name in names}
    step = 0
    for _ in range(epochs):
        for row in rows:
            step += 1
            values = [*row['features'], SCALE]
            scores = {name: sum(a * b for a, b in zip(weights[name], values)) for name in names}
            target = row['route']
            rival = min((name for name in names if name != target), key=lambda name: (-scores[name], name))
            if scores[target] > scores[rival] + training_margin:
                continue
            for name, sign in ((target, 1), (rival, -1)):
                for index, value in enumerate(values):
                    delta = sign * value
                    weights[name][index] += delta
                    accumulated[name][index] += (step - 1) * delta
    averaged = {name: [step * value - elapsed for value, elapsed in zip(weights[name], accumulated[name])]
                for name in names}
    magnitude = max(sum(value * value for value in values) for values in averaged.values())
    if not magnitude:
        raise ValueError('Classifier training produced no decision boundary')
    denominator = math.isqrt(magnitude << 64)
    scaled = {name: [rounded_ratio(value * (SCALE << 32), denominator) for value in values]
              for name, values in averaged.items()}
    method = ('integer-balanced-averaged-margin-perceptron-v1' if balance_classes
              else 'integer-averaged-margin-perceptron-v1')
    classifier = {'method': method, 'epochs': epochs,
                  'training_margin': training_margin,
                  'weights': {name: values[:-1] for name, values in scaled.items()},
                  'biases': {name: values[-1] for name, values in scaled.items()}}
    return validate({**prototype_model, 'format': LINEAR_FORMAT, 'classifier': classifier})
