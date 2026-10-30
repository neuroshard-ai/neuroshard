"""Development writing items: ordinary writing requests with constraints a program checks, plus a rubric for
the blind pairwise judge (``writing/1``). The constraint result is the gated score; judged quality is
reported beside it under the protocol in docs/GENERAL_CAPABILITY_EVAL.md.
"""

PASSAGE = ('Our town library will close its east wing from 3 to 21 March so that the roof can be replaced. During '
           'the closure the children\'s section moves to the community hall on Bridge Street, and returns can be '
           'made at the front desk as usual. Evening opening on Thursdays is suspended until the work is finished. '
           'The library thanks residents for their patience and expects the new roof to cut heating costs by a '
           'quarter.')


def rules(*constraints):
    return {'kind': 'constraints', 'constraints': list(constraints)}


def c(kind, **fields):
    return {'kind': kind, **fields}


WRITING = [
    ('email', 'Write the body of a short email to my landlord asking when the broken kitchen tap will be repaired, '
              'with no greeting line and no sign-off. Use exactly three paragraphs, mention the word "tap", and keep '
              'it under 120 words.',
     rules(c('paragraphs', n=3), c('include', word='tap'), c('max_words', n=119))),
    ('summary', 'Summarise this notice in at most 40 words, and make sure the words "roof" and "March" appear:\n\n'
                + PASSAGE, rules(c('max_words', n=40), c('include_all', words=['roof', 'March']))),
    ('summary', 'Give me three bullet points, and nothing else, with the practical facts a parent needs from this '
                'notice:\n\n' + PASSAGE, rules(c('bullets', n=3))),
    ('rewrite', 'Rewrite this sentence so it sounds polite and professional, in one sentence of at most 25 words: '
                '"Send me the invoice now, I have asked three times already."',
     rules(c('max_words', n=25), c('lines', n=1), c('include', word='invoice'))),
    ('description', 'Write a product description for a reusable steel water bottle. Start with the words "Meet '
                    'your", use between 60 and 90 words, and do not use the word "plastic".',
     rules(c('starts_with', phrase='Meet your'), c('min_words', n=60), c('max_words', n=90),
           c('exclude', word='plastic'))),
    ('poem', 'write a four-line poem about a rainy bus stop. all lowercase, exactly four lines, nothing else.',
     rules(c('lowercase'), c('lines', n=4))),
    ('headline', 'Write a title line followed by a two-paragraph blog introduction about learning to cook at forty. '
                 'The title is the first line, at most 8 words, with no full stop.',
     rules(c('title_line', max_words=8), c('min_words', n=60))),
    ('plain-language', 'Explain what a credit score is to a teenager. Use 4 to 6 sentences and keep every sentence '
                       'to at most 20 words.',
     rules(c('max_sentence_words', n=20), c('min_words', n=30), c('max_words', n=130))),
    ('style', 'Describe a sunrise over a harbour in a short paragraph without using any commas.',
     rules(c('no_char', char=','), c('min_words', n=25), c('paragraphs', n=1))),
    ('avoidance', 'Write a short motivational note for a colleague starting a new job. Avoid the words "luck", '
                  '"amazing" and "journey", and keep it under 70 words.',
     rules(c('exclude_all', words=['luck', 'amazing', 'journey']), c('max_words', n=69))),
    ('structured', 'Create a JSON object, and nothing else, describing a weekend hiking trip, with exactly the keys '
                   'destination, distance_km, packing_list and difficulty.',
     rules(c('json_keys', keys=['destination', 'distance_km', 'packing_list', 'difficulty']))),
    ('ending', 'Write a two-sentence thank-you message to a neighbour who watered my plants, ending with the exact '
               'words "Thanks again, Priya".', rules(c('ends_with', phrase='Thanks again, Priya', markup='strip'),
                                                      c('max_words', n=60))),
    ('list', 'Give five bullet points of advice for a first-time conference speaker, and include the word '
             '"rehearse" somewhere.', rules(c('bullets', n=5), c('include', word='rehearse'))),
    ('apology', 'Draft a short apology to a customer whose order arrived two days late. Keep it to 2 paragraphs and '
                'between 50 and 100 words, and mention a "discount".',
     rules(c('paragraphs', n=2), c('min_words', n=50), c('max_words', n=100), c('include', word='discount'))),
    ('story', 'Write a micro-story about a lighthouse keeper in exactly 3 paragraphs, under 150 words, that ends '
              'with the words "the light stayed on".',
     rules(c('paragraphs', n=3), c('max_words', n=149), c('ends_with', phrase='the light stayed on',
                                                            markup='strip'))),
    ('explain', 'In at most 50 words, explain why the sky looks blue, and use the word "scattering".',
     rules(c('max_words', n=50), c('include', word='scattering'))),
    ('announcement', 'Write a team announcement that the office will be closed on Friday for a power upgrade. Start '
                     'with "Heads up", keep it under 60 words, and do not use the word "unfortunately".',
     rules(c('starts_with', phrase='Heads up'), c('max_words', n=59), c('exclude', word='unfortunately'))),
    ('review', 'Write a balanced three-sentence review of a café you might visit. Use the words "coffee" and '
               '"seating", and keep every sentence under 25 words.',
     rules(c('include_all', words=['coffee', 'seating']), c('max_sentence_words', n=24), c('max_words', n=75))),
    ('instructions', 'Write numbered steps, and nothing else, for repotting a houseplant. Use exactly 6 steps.',
     rules(c('bullets', n=6))),
    ('formal', 'Write a two-paragraph formal letter declining a job offer. Do not use the word "unfortunately", end '
               'with "Yours sincerely", and keep it under 130 words.',
     rules(c('exclude', word='unfortunately'), c('ends_with', phrase='Yours sincerely', markup='strip'),
           c('max_words', n=129))),
]
