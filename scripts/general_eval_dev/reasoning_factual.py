"""Development items for reasoning and factual questions, written for this evaluation (October 30, 2026).

Each reasoning prompt asks for a final ``Answer:`` line; answers were computed independently and are
re-derived where a formula exists (see ``CHECKS``). Factual items ask about stable, uncontroversial facts;
their wording is new, but the facts themselves are public by nature (see the contamination section of
docs/GENERAL_CAPABILITY_EVAL.md).
"""

END = ' Finish with a final line of the form "Answer: <your answer>".'


def num(value, tolerance=0):
    return {'kind': 'answer', 'answer': {'type': 'number', 'value': value, 'tolerance': tolerance}}


def choice(letter):
    return {'kind': 'answer', 'answer': {'type': 'choice', 'value': letter}}


def text(*accepted):
    return {'kind': 'answer', 'answer': {'type': 'text', 'accepted': list(accepted)}}


REASONING = [
    ('time', 'My train leaves at 09:40 and arrives at 13:15 the same day. How many minutes is the journey?', num(215)),
    ('algebra', 'Maya is three times as old as her son. In twelve years she will be exactly twice as old as he is '
                'then. How old is Maya now?', num(36)),
    ('rates', 'A stationery shop sells pencils at 3 for 50 cents. How many cents do 27 pencils cost?', num(450)),
    ('logic', 'Five runners, A, B, C, D and E, finish a race with no ties. E wins. A finishes immediately after E. '
              'D finishes last. B finishes ahead of C. Which runner finishes third? Give just the letter.', text('B')),
    ('geometry', 'A rectangle has a perimeter of 46 cm and its length is 5 cm more than its width. What is its area '
                 'in square centimetres?', num(126)),
    ('percent', 'A jacket costs 80 dollars. The price is raised by 25%, and later the new price is cut by 20%. What '
                'is the final price in dollars?', num(80)),
    ('probability', 'Two fair six-sided dice are rolled. What is the probability that the two numbers add up to 9? '
                    'Give a fraction or a decimal.', num(0.111111, 0.0002)),
    ('sequence', 'What number comes next: 3, 7, 15, 31, 63, ...?', num(127)),
    ('rates', 'One pipe fills a tank in 6 hours and a second pipe fills the same tank in 3 hours. With both open, '
              'how many hours does it take to fill the empty tank?', num(2)),
    ('dates', 'If 1 March 2027 is a Monday, what day of the week is 1 April 2027?', text('Thursday')),
    ('logic', 'On an island, knights always tell the truth and knaves always lie. A says: "B is a knave." B says: '
              '"A and I are the same type." Exactly one of them is a knight. Which one? Give just the letter.',
     text('A')),
    ('units', 'A car uses 6.5 litres of fuel per 100 km. How many litres does it use for a 340 km trip?', num(22.1)),
    ('comparison', 'Which is the largest? A) 2^10  B) 10^3  C) 3^6  D) 5^4. Give the letter.', choice('A')),
    ('combinatorics', 'How many distinct arrangements are there of the letters of the word LEVEL?', num(30)),
    ('algebra', 'The average of five numbers is 18. After one number is removed, the average of the remaining four '
                'is 16. What number was removed?', num(26)),
    ('geometry', 'What is the smaller angle, in degrees, between the hour and minute hands of a clock at 3:30?',
     num(75)),
    ('rates', 'Twelve workers build a wall in 10 days. Working at the same rate, how many days would 15 workers '
              'take?', num(8)),
    ('logic', 'All florbs are glims. Some glims are twerps. Which statement must be true? A) Some florbs are twerps '
              'B) All glims are florbs C) No florb is a twerp D) None of these must be true. Give the letter.',
     choice('D')),
    ('algebra', 'Sam has twice as many coins as Lee. After Sam gives Lee 6 coins, they have the same number. How many '
                'coins did Sam have at first?', num(24)),
    ('percent', 'You deposit 2,000 dollars at 5% interest compounded once a year. How many dollars of interest have '
                'you earned after exactly two years?', num(205)),
]

CHECKS = {  # independent re-derivations, asserted by the builder
    0: (13 * 60 + 15) - (9 * 60 + 40), 1: 36, 2: 27 // 3 * 50, 4: 9 * 14, 5: 80 * 1.25 * 0.8, 7: 2 * 63 + 1,
    11: 6.5 * 3.4, 13: 120 // 4, 14: 5 * 18 - 4 * 16, 15: abs(30 * 3.5 - 6 * 30), 16: 12 * 10 / 15,
    19: round(2000 * 1.05 ** 2 - 2000, 6),
}


def short(*accepted, rejected=()):
    return {'kind': 'short_answer', 'accepted': list(accepted), 'rejected': list(rejected)}


ASK = ' Reply in a few words, ending with "Answer: <answer>".'
FACTUAL = [
    ('geography', 'Which city is the capital of Australia?', short('Canberra', rejected=('Sydney', 'Melbourne'))),
    ('science', 'What is the chemical symbol for the element tungsten?', short('W')),
    ('literature', 'Who wrote the novel "Pride and Prejudice"?', short('Jane Austen', 'Austen')),
    ('science', 'At sea level, at what temperature in degrees Fahrenheit does pure water boil?', short('212')),
    ('geography', 'Which is the largest ocean on Earth by area?', short('Pacific', rejected=('Atlantic', 'Indian'))),
    ('history', 'In which year did the Berlin Wall fall?', short('1989')),
    ('biology', 'What is the longest bone in the human body?', short('femur', 'thigh bone', 'thighbone')),
    ('biology', 'How many bones does a typical adult human skeleton have?', short('206')),
    ('science', 'Which element has atomic number 1?', short('hydrogen')),
    ('literature', 'Who wrote "On the Origin of Species", published in 1859?', short('Charles Darwin', 'Darwin')),
    ('economics', 'What is the currency of Japan called?', short('yen')),
    ('mathematics', 'What is the smallest prime number?', short('2', 'two')),
    ('biology', 'Which gas do plants take in from the air for photosynthesis?', short('carbon dioxide', 'co2')),
    ('art', 'Which painter made "The Starry Night" (1889)?', short('Vincent van Gogh', 'van Gogh')),
    ('science', 'What is the hardest naturally occurring mineral?', short('diamond')),
    ('geography', 'Which mountain has the highest summit above sea level?', short('Everest', 'Chomolungma',
                                                                                   'Sagarmatha', rejected=('K2',))),
    ('science', 'What is the freezing point of water in kelvin, at standard pressure?', short('273.15', '273')),
    ('history', 'Which country gifted the Statue of Liberty to the United States?', short('France')),
    ('false-premise', 'In which year did Albert Einstein receive his second Nobel Prize?',
     short('only one', 'one nobel', 'never', 'did not', 'didn t', 'single nobel', 'no second', 'only received one',
           'only won one', 'only once', 'not receive a second', 'never received a second', rejected=())),
    ('unknowable', 'What is the name of the person sitting next to me right now?',
     short('don t know', 'do not know', 'cannot know', 'can t know', 'no way to know', 'not able to', 'unable to',
           'cannot see', 'can t see', 'no way of knowing', 'don t have access', 'do not have access',
           'cannot determine', 'can t determine')),
]
