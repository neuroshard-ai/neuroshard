"""Development planning items: a plan returned as JSON, validated against the item's explicit constraints.

Every item carries a reference plan the builder validates, so each item is satisfiable. ``optimal`` items
also require the best value (budget) or shortest time (route), checked by exhaustive search.
"""

JSON = ' Reply with the plan as JSON in one ```json block, then at most two sentences.'


def plan(validator, reference, **spec):
    return {'kind': 'plan', 'validator': validator, 'reference': reference, **spec}


def task(duration, **extra):
    return {'duration': duration, **extra}


PLANNING = [
    ('schedule', 'Plan my Saturday between 09:00 and 17:00. Tasks: groceries (60 min), gym (90 min), laundry (45 min) '
                 'and calling grandma (30 min). Lunch with Sam is fixed from 12:00 to 13:00. The gym closes at 12:00. '
                 'Call grandma after 15:00. Give a JSON object mapping each task name (groceries, gym, laundry, call) '
                 'to its start time as "HH:MM"; tasks may not overlap each other or lunch.',
     plan('schedule', {'gym': '09:00', 'groceries': '10:30', 'laundry': '13:00', 'call': '15:00'},
          window=['09:00', '17:00'], busy=[{'name': 'lunch', 'start': '12:00', 'end': '13:00'}],
          tasks={'groceries': task(60), 'gym': task(90, deadline='12:00'), 'laundry': task(45),
                 'call': task(30, earliest='15:00')})),
    ('order', 'I am moving flat. Put these jobs in a valid order: pack, book_van, clean_old_flat, load_van, '
              'unload, return_keys. The van must be booked before loading; packing comes before loading; loading '
              'before unloading; the old flat is cleaned after loading and before returning the keys. Give a JSON '
              'list of the job names in order.',
     plan('order', ['book_van', 'pack', 'load_van', 'clean_old_flat', 'unload', 'return_keys'],
          items=['pack', 'book_van', 'clean_old_flat', 'load_van', 'unload', 'return_keys'],
          before=[['book_van', 'load_van'], ['pack', 'load_van'], ['load_van', 'unload'],
                  ['load_van', 'clean_old_flat'], ['clean_old_flat', 'return_keys']])),
    ('assignment', 'Staff three shifts: morning (2 people, one must be a first_aider), afternoon (2 people) and '
                   'evening (1 person, must be a keyholder). Team: Ana (first_aider; unavailable evening), Ben '
                   '(keyholder), Chen (first_aider, keyholder; at most 1 shift), Dee (unavailable morning). Nobody '
                   'works more than 2 shifts, and everyone works at least one. Give a JSON object mapping each shift '
                   'to a list of names.',
     plan('assignment', {'morning': ['Ana', 'Ben'], 'afternoon': ['Ana', 'Dee'], 'evening': ['Chen']},
          slots={'morning': {'count': 2, 'requires': ['first_aider']}, 'afternoon': {'count': 2},
                 'evening': {'count': 1, 'requires': ['keyholder']}},
          people={'Ana': {'skills': ['first_aider'], 'unavailable': ['evening'], 'max': 2, 'min': 1},
                  'Ben': {'skills': ['keyholder'], 'max': 2, 'min': 1},
                  'Chen': {'skills': ['first_aider', 'keyholder'], 'max': 1, 'min': 1},
                  'Dee': {'unavailable': ['morning'], 'max': 2, 'min': 1}})),
    ('budget', 'I have 500 dollars for a home office. Options (cost, usefulness score): desk (200, 8), chair (180, '
               '9), monitor (220, 7), lamp (40, 3), headset (90, 5), plant (25, 1). I already have a desk, so do not '
               'buy one. Maximise the total usefulness score within budget. Give a JSON list of the items to buy.',
     plan('budget', ['chair', 'monitor', 'headset'], optimal=True, budget=500,
          exclusive=[], requires=[], must=[],
          options={'desk': {'cost': 200, 'value': -100}, 'chair': {'cost': 180, 'value': 9},
                   'monitor': {'cost': 220, 'value': 7}, 'lamp': {'cost': 40, 'value': 3},
                   'headset': {'cost': 90, 'value': 5}, 'plant': {'cost': 25, 'value': 1}})),
    ('route', 'Starting and ending at home, I need to visit the bakery, the post office and the pharmacy. Travel '
              'times in minutes: home-bakery 10, home-post 15, home-pharmacy 20, bakery-post 12, bakery-pharmacy 8, '
              'post-pharmacy 6. Give the quickest route as a JSON list of places, starting and ending with "home" '
              '(use the names home, bakery, post, pharmacy).',
     plan('route', ['home', 'bakery', 'pharmacy', 'post', 'home'], optimal=True, start='home',
          stops=['bakery', 'post', 'pharmacy'],
          times={'home': {'bakery': 10, 'post': 15, 'pharmacy': 20}, 'bakery': {'home': 10, 'post': 12, 'pharmacy': 8},
                 'post': {'home': 15, 'bakery': 12, 'pharmacy': 6}, 'pharmacy': {'home': 20, 'bakery': 8, 'post': 6}})),
]

PLANNING += [
    ('schedule', 'Plan a study afternoon from 14:00 to 20:00. Dinner is fixed from 18:00 to 18:45. Tasks: essay '
                 '(90 min, must be finished by 18:00), maths (120 min), a break (15 min, right after maths is done or '
                 'later) and history (60 min). Give a JSON object mapping essay, maths, break and history to start '
                 'times "HH:MM"; nothing may overlap.',
     plan('schedule', {'essay': '14:00', 'maths': '15:30', 'break': '17:30', 'history': '18:45'},
          window=['14:00', '20:00'], busy=[{'name': 'dinner', 'start': '18:00', 'end': '18:45'}],
          tasks={'essay': task(90, deadline='18:00'), 'maths': task(120), 'break': task(15, after=['maths']),
                 'history': task(60)})),
    ('schedule', 'Workday 09:00-17:00 with lunch fixed 12:00-13:00. Meetings: one_on_one (30 min, must end by '
                 '10:00), budget (45 min, after one_on_one), design_review (60 min, not before 13:00) and deep_work '
                 '(120 min, must end by 13:00). Give a JSON object of start times "HH:MM"; nothing may overlap.',
     plan('schedule', {'one_on_one': '09:00', 'deep_work': '09:30', 'budget': '13:00', 'design_review': '13:45'},
          window=['09:00', '17:00'], busy=[{'name': 'lunch', 'start': '12:00', 'end': '13:00'}],
          tasks={'one_on_one': task(30, deadline='10:00'), 'budget': task(45, after=['one_on_one']),
                 'design_review': task(60, earliest='13:00'), 'deep_work': task(120, deadline='13:00')})),
    ('schedule', 'Moving day from 07:00 to 15:00, one person doing everything in sequence: pack_kitchen (90 min), '
                 'pack_bedroom (60 min), load_van (60 min, after both packing jobs), return_keys (15 min, after '
                 'loading, by 12:00 at the latest), drive (45 min, after loading) and unload (60 min, after the '
                 'drive). Give a JSON object of start times "HH:MM".',
     plan('schedule', {'pack_kitchen': '07:00', 'pack_bedroom': '08:30', 'load_van': '09:30', 'return_keys': '10:30',
                       'drive': '10:45', 'unload': '11:30'}, window=['07:00', '15:00'],
          tasks={'pack_kitchen': task(90), 'pack_bedroom': task(60),
                 'load_van': task(60, after=['pack_kitchen', 'pack_bedroom']),
                 'return_keys': task(15, after=['load_van'], deadline='12:00'), 'drive': task(45, after=['load_van']),
                 'unload': task(60, after=['drive'])})),
    ('schedule', 'Cook dinner between 16:00 and 19:00. prep_meat (15 min) comes first; the roast (90 min) goes in the '
                 'oven after prep_meat and needs no attention, so it may overlap other jobs. dessert (45 min) must be '
                 'done by 18:00. prep_veg (30 min) comes before cook_veg (30 min). set_table takes 15 min. Apart from '
                 'the roast, I do one job at a time. Everything must finish by 19:00. Give a JSON object of start '
                 'times "HH:MM" for prep_meat, roast, dessert, prep_veg, cook_veg and set_table.',
     plan('schedule', {'prep_meat': '16:00', 'roast': '16:15', 'dessert': '16:15', 'prep_veg': '17:00',
                       'cook_veg': '17:30', 'set_table': '18:00'}, window=['16:00', '19:00'], parallel_ok=['roast'],
          tasks={'prep_meat': task(15), 'roast': task(90, after=['prep_meat']), 'dessert': task(45, deadline='18:00'),
                 'prep_veg': task(30), 'cook_veg': task(30, after=['prep_veg']), 'set_table': task(15)})),
    ('schedule', 'Conference day 08:00-18:00. I must attend the keynote 09:00-10:00 and the panel 14:00-15:00 (fixed). '
                 'Also fit in: prepare_slides (120 min, finished before my talk), my talk (30 min, not before 15:00), '
                 'networking (60 min) and lunch (45 min, starting between 11:30 and 13:00, so not before 11:30 and '
                 'ending by 13:45). Give a JSON object of start times for prepare_slides, talk, networking and lunch.',
     plan('schedule', {'prepare_slides': '10:00', 'lunch': '12:00', 'networking': '12:45', 'talk': '15:00'},
          window=['08:00', '18:00'], busy=[{'name': 'keynote', 'start': '09:00', 'end': '10:00'},
                                           {'name': 'panel', 'start': '14:00', 'end': '15:00'}],
          tasks={'prepare_slides': task(120), 'talk': task(30, earliest='15:00', after=['prepare_slides']),
                 'networking': task(60), 'lunch': task(45, earliest='11:30', deadline='13:45')})),
    ('order', 'Order these university courses so each comes after its prerequisites: intro_programming, '
              'data_structures (needs intro_programming), discrete_math, algorithms (needs data_structures and '
              'discrete_math), databases (needs data_structures), capstone (needs algorithms and databases; it '
              'must be last). Give a JSON list in order.',
     plan('order', ['intro_programming', 'discrete_math', 'data_structures', 'algorithms', 'databases', 'capstone'],
          items=['intro_programming', 'data_structures', 'discrete_math', 'algorithms', 'databases', 'capstone'],
          before=[['intro_programming', 'data_structures'], ['data_structures', 'algorithms'],
                  ['discrete_math', 'algorithms'], ['data_structures', 'databases'], ['algorithms', 'capstone'],
                  ['databases', 'capstone']], last='capstone')),
    ('order', 'Arrange the six sessions of a one-track workshop: keynote, workshop_a, workshop_b, panel, demo, '
              'closing. The keynote is first and closing is last. The panel comes after both workshops. The two '
              'workshops must not be back to back. Give a JSON list in order.',
     plan('order', ['keynote', 'workshop_a', 'demo', 'workshop_b', 'panel', 'closing'],
          items=['keynote', 'workshop_a', 'workshop_b', 'panel', 'demo', 'closing'], first='keynote', last='closing',
          before=[['workshop_a', 'panel'], ['workshop_b', 'panel']], not_adjacent=[['workshop_a', 'workshop_b']])),
    ('order', 'Order the steps for changing a flat tyre: apply_handbrake, loosen_nuts, jack_up, remove_wheel, '
              'fit_spare, lower_car, tighten_nuts. The handbrake goes on first. Nuts are loosened before jacking up '
              'and fully tightened only after lowering the car. The wheel comes off after jacking up, and the spare '
              'goes on before lowering. Give a JSON list of the steps in order.',
     plan('order', ['apply_handbrake', 'loosen_nuts', 'jack_up', 'remove_wheel', 'fit_spare', 'lower_car',
                    'tighten_nuts'],
          items=['apply_handbrake', 'loosen_nuts', 'jack_up', 'remove_wheel', 'fit_spare', 'lower_car', 'tighten_nuts'],
          first='apply_handbrake', before=[['loosen_nuts', 'jack_up'], ['jack_up', 'remove_wheel'],
                                           ['remove_wheel', 'fit_spare'], ['fit_spare', 'lower_car'],
                                           ['lower_car', 'tighten_nuts']])),
    ('assignment', 'Assign a weekend rota with one person per slot: sat_am, sat_pm, sun_am, sun_pm. Priya is away on '
                   'Sunday. Omar cannot do mornings. Lena can do at most one slot. Each of the three must do at '
                   'least one slot. Give a JSON object mapping each slot to a one-name list.',
     plan('assignment', {'sat_am': ['Priya'], 'sat_pm': ['Omar'], 'sun_am': ['Lena'], 'sun_pm': ['Omar']},
          slots={'sat_am': {'count': 1}, 'sat_pm': {'count': 1}, 'sun_am': {'count': 1}, 'sun_pm': {'count': 1}},
          people={'Priya': {'unavailable': ['sun_am', 'sun_pm'], 'min': 1},
                  'Omar': {'unavailable': ['sat_am', 'sun_am'], 'min': 1}, 'Lena': {'max': 1, 'min': 1}})),
    ('assignment', 'Six guests share three twin rooms (two per room: room1, room2, room3). Alex and Blair must not '
                   'share. Casey and Drew must not share. Eli and Fran must not share. Alex and Casey must not share. '
                   'Give a JSON object mapping each room to a list of two names (Alex, Blair, Casey, Drew, Eli, '
                   'Fran).',
     plan('assignment', {'room1': ['Alex', 'Drew'], 'room2': ['Blair', 'Eli'], 'room3': ['Casey', 'Fran']},
          slots={'room1': {'count': 2}, 'room2': {'count': 2}, 'room3': {'count': 2}},
          people={n: {'max': 1, 'min': 1} for n in ('Alex', 'Blair', 'Casey', 'Drew', 'Eli', 'Fran')},
          apart=[['Alex', 'Blair'], ['Casey', 'Drew'], ['Eli', 'Fran'], ['Alex', 'Casey']])),
    ('assignment', 'Fill three club roles, one person each: chair, treasurer and secretary. The treasurer must have '
                   'accounting skills and the secretary must have minutes skills. Members: Rosa (accounting), Tom '
                   '(minutes, accounting), Uma (no listed skills, but she will only take the chair). Each member '
                   'takes exactly one role. Give a JSON object mapping each role to a one-name list.',
     plan('assignment', {'chair': ['Uma'], 'treasurer': ['Rosa'], 'secretary': ['Tom']},
          slots={'chair': {'count': 1}, 'treasurer': {'count': 1, 'requires': ['accounting']},
                 'secretary': {'count': 1, 'requires': ['minutes']}},
          people={'Rosa': {'skills': ['accounting'], 'max': 1, 'min': 1},
                  'Tom': {'skills': ['minutes', 'accounting'], 'max': 1, 'min': 1},
                  'Uma': {'unavailable': ['treasurer', 'secretary'], 'max': 1, 'min': 1}})),
    ('budget', 'Plan a birthday party with 300 dollars. Options (cost, fun score): venue (120, 6, required), dj (100, '
               '7), band (180, 9), cake (40, 5), decorations (30, 2), photo_booth (70, 4). A dj and a band cannot both '
               'be booked. Maximise the fun score. Give a JSON list of chosen options.',
     plan('budget', ['venue', 'dj', 'cake', 'decorations'], optimal=True, budget=300, must=['venue'],
          exclusive=[['dj', 'band']], requires=[],
          options={'venue': {'cost': 120, 'value': 6}, 'dj': {'cost': 100, 'value': 7}, 'band': {'cost': 180, 'value': 9},
                   'cake': {'cost': 40, 'value': 5}, 'decorations': {'cost': 30, 'value': 2},
                   'photo_booth': {'cost': 70, 'value': 4}})),
    ('budget', 'Weekend trip activities with 260 euros. Options (cost, enjoyment): kayak (60, 5), museum (25, 3), '
               'boat_tour (90, 6), wine_tasting (70, 5), cooking_class (110, 8), bike_rental (30, 2). The boat tour '
               'needs the bike rental to reach the pier. Maximise enjoyment. Give a JSON list of chosen activities.',
     plan('budget', ['cooking_class', 'boat_tour', 'bike_rental', 'museum'], optimal=True, budget=260, must=[],
          exclusive=[], requires=[['boat_tour', 'bike_rental']],
          options={'kayak': {'cost': 60, 'value': 5}, 'museum': {'cost': 25, 'value': 3},
                   'boat_tour': {'cost': 90, 'value': 6}, 'wine_tasting': {'cost': 70, 'value': 5},
                   'cooking_class': {'cost': 110, 'value': 8}, 'bike_rental': {'cost': 30, 'value': 2}})),
    ('route', 'A delivery van starts at the depot, must visit A, B, C and D once each and return to the depot. B '
              'must be visited before D. Travel times (minutes): depot-A 7, depot-B 9, depot-C 14, depot-D 10, A-B 4, '
              'A-C 8, A-D 12, B-C 5, B-D 11, C-D 6. Give the quickest valid route as a JSON list starting and ending '
              'with "depot".',
     plan('route', ['depot', 'A', 'B', 'C', 'D', 'depot'], optimal=True, start='depot', stops=['A', 'B', 'C', 'D'],
          before=[['B', 'D']],
          times={'depot': {'A': 7, 'B': 9, 'C': 14, 'D': 10}, 'A': {'depot': 7, 'B': 4, 'C': 8, 'D': 12},
                 'B': {'depot': 9, 'A': 4, 'C': 5, 'D': 11}, 'C': {'depot': 14, 'A': 8, 'B': 5, 'D': 6},
                 'D': {'depot': 10, 'A': 12, 'B': 11, 'C': 6}})),
    ('route', 'I start at the hotel and finish at the station, visiting the museum, the market and the park on the '
              'way, each once. Walking times (minutes): hotel-museum 12, hotel-market 5, hotel-park 9, museum-market '
              '8, museum-park 4, market-park 6, museum-station 7, market-station 15, park-station 10. The whole walk '
              'must take at most 35 minutes. Give a valid JSON list starting with "hotel" and ending with "station".',
     plan('route', ['hotel', 'market', 'park', 'museum', 'station'], start='hotel', end='station', limit=35,
          stops=['museum', 'market', 'park'],
          times={'hotel': {'museum': 12, 'market': 5, 'park': 9}, 'museum': {'market': 8, 'park': 4, 'station': 7},
                 'market': {'museum': 8, 'park': 6, 'station': 15}, 'park': {'museum': 4, 'market': 6, 'station': 10}})),
]
