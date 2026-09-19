"""Best-effort visible text observation, outside the numerical result.

Events replace the current draft: incremental tokenizer decoding need not be
prefix-stable. A draft is not a numerical proof or a settled response. Delivery
failure must not change generation, strand other shards or authorize payment.
"""


class Observer:
    def __init__(self, callback):
        self.callback, self.previous, self.failed = callback, None, False

    def text(self, value):
        if self.callback is None or self.failed or value == self.previous:
            return
        try:
            if not isinstance(value, str) or len(value.encode()) > 32768:
                raise ValueError('Visible draft exceeds its bound')
            self.callback(value)
            self.previous = value
        except Exception:
            # The callback is local and must not block on a customer socket.
            # Its failure cannot enter the numerical consensus path.
            self.failed = True

    def tokens(self, tokenizer, *, worked=False):
        def observe(values):
            text = tokenizer.decode(values, skip_special_tokens=True)
            # A partial byte sequence may become a Unicode character at the
            # next token. Do not expose the decoder's replacement placeholder.
            if '\ufffd' in text:
                return
            if worked:
                from .general_answer import visible
                try:
                    text = visible(text)
                except ValueError:
                    # Missing or repeated answer boundary: reveal no reasoning,
                    # and retract a prior provisional answer if it became invalid.
                    text = ''
            self.text(text)
        return observe
