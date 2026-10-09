# Router scaling: hashed encoder

836 fit cases; evaluation {'test': 556, 'unseen': 420}; features in 0 s.

## centroids-refit, test

| routes | added | strategy | turn acc | episode acc | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | scheduling | message | 1.000 | 1.000 | 1.000 (drafting) | – | – |
| 2 | scheduling | with-opening | 0.980 | 0.971 | 0.962 (scheduling) | scheduling->drafting | – |
| 2 | scheduling | episode | 0.980 | 0.971 | 0.962 (scheduling) | scheduling->drafting | – |
| 3 | invoices | message | 0.936 | 0.900 | 0.840 (drafting) | drafting->invoices | 16 / 204 |
| 3 | invoices | with-opening | 0.984 | 0.975 | 0.962 (scheduling) | scheduling->drafting | 0 / 200 |
| 3 | invoices | episode | 0.984 | 0.975 | 0.962 (scheduling) | scheduling->drafting | 0 / 200 |
| 4 | tickets | message | 0.913 | 0.860 | 0.833 (invoices) | drafting->tickets | 8 / 234 |
| 4 | tickets | with-opening | 0.980 | 0.968 | 0.962 (scheduling) | scheduling->drafting | 0 / 246 |
| 4 | tickets | episode | 0.980 | 0.968 | 0.962 (scheduling) | scheduling->drafting | 0 / 246 |
| 5 | rooms | message | 0.921 | 0.869 | 0.840 (invoices) | drafting->tickets | 4 / 272 |
| 5 | rooms | with-opening | 0.975 | 0.958 | 0.958 (tickets) | scheduling->drafting | 0 / 292 |
| 5 | rooms | episode | 0.972 | 0.953 | 0.958 (tickets) | scheduling->drafting | 0 / 292 |
| 6 | expenses | message | 0.932 | 0.893 | 0.815 (rooms) | tickets->expenses | 5 / 326 |
| 6 | expenses | with-opening | 0.963 | 0.939 | 0.940 (expenses) | scheduling->drafting | 0 / 345 |
| 6 | expenses | episode | 0.961 | 0.934 | 0.940 (expenses) | scheduling->drafting | 0 / 344 |
| 7 | approvals | message | 0.926 | 0.891 | 0.759 (invoices) | invoices->approvals | 6 / 382 |
| 7 | approvals | with-opening | 0.952 | 0.917 | 0.923 (expenses) | scheduling->drafting | 0 / 395 |
| 7 | approvals | episode | 0.950 | 0.913 | 0.923 (expenses) | scheduling->drafting | 0 / 394 |
| 8 | reminders | message | 0.916 | 0.871 | 0.768 (invoices) | invoices->approvals | 0 / 441 |
| 8 | reminders | with-opening | 0.941 | 0.897 | 0.907 (expenses) | scheduling->drafting | 0 / 453 |
| 8 | reminders | episode | 0.938 | 0.890 | 0.907 (expenses) | scheduling->drafting | 0 / 452 |
| 9 | summaries | message | 0.909 | 0.858 | 0.776 (invoices) | invoices->approvals | 1 / 499 |
| 9 | summaries | with-opening | 0.929 | 0.873 | 0.893 (expenses) | scheduling->drafting | 0 / 513 |
| 9 | summaries | episode | 0.926 | 0.867 | 0.893 (expenses) | scheduling->drafting | 0 / 511 |
| 10 | travel | message | 0.910 | 0.857 | 0.783 (invoices) | invoices->approvals | 0 / 562 |
| 10 | travel | with-opening | 0.920 | 0.857 | 0.879 (expenses) | scheduling->drafting | 0 / 574 |
| 10 | travel | episode | 0.913 | 0.844 | 0.870 (travel) | scheduling->drafting | 0 / 572 |
| 11 | inventory | message | 0.901 | 0.844 | 0.790 (invoices) | inventory->expenses | 3 / 624 |
| 11 | inventory | with-opening | 0.908 | 0.833 | 0.867 (expenses) | scheduling->drafting | 0 / 631 |
| 11 | inventory | episode | 0.901 | 0.821 | 0.857 (travel) | scheduling->drafting | 0 / 626 |
| 12 | timesheets | message | 0.891 | 0.835 | 0.797 (invoices) | timesheets->summaries | 5 / 695 |
| 12 | timesheets | with-opening | 0.900 | 0.818 | 0.855 (expenses) | scheduling->drafting | 0 / 700 |
| 12 | timesheets | episode | 0.890 | 0.798 | 0.845 (travel) | scheduling->drafting | 0 / 695 |
| 13 | contacts | message | 0.883 | 0.822 | 0.765 (rooms) | rooms->contacts | 8 / 759 |
| 13 | contacts | with-opening | 0.895 | 0.806 | 0.844 (expenses) | scheduling->drafting | 0 / 767 |
| 13 | contacts | episode | 0.880 | 0.776 | 0.833 (travel) | scheduling->drafting | 0 / 758 |
| 14 | search | message | 0.885 | 0.824 | 0.760 (inventory) | rooms->contacts | 6 / 836 |
| 14 | search | with-opening | 0.887 | 0.790 | 0.833 (expenses) | scheduling->drafting | 0 / 848 |
| 14 | search | episode | 0.869 | 0.755 | 0.823 (travel) | scheduling->drafting | 0 / 833 |

## centroids-refit, unseen

| routes | added | strategy | turn acc | episode acc | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | invoices | message | 0.958 | 0.917 | 0.958 (invoices) | invoices->drafting | 0 / 0 |
| 3 | invoices | with-opening | 1.000 | 1.000 | 1.000 (invoices) | – | 0 / 0 |
| 3 | invoices | episode | 1.000 | 1.000 | 1.000 (invoices) | – | 0 / 0 |
| 4 | tickets | message | 0.700 | 0.440 | 0.680 (invoices) | invoices->tickets | 13 / 46 |
| 4 | tickets | with-opening | 0.770 | 0.600 | 0.680 (tickets) | tickets->invoices | 5 / 48 |
| 4 | tickets | episode | 0.580 | 0.560 | 0.480 (tickets) | tickets->invoices | 14 / 48 |
| 5 | rooms | message | 0.606 | 0.385 | 0.308 (invoices) | invoices->rooms | 22 / 70 |
| 5 | rooms | with-opening | 0.665 | 0.603 | 0.462 (tickets) | tickets->rooms | 27 / 77 |
| 5 | rooms | episode | 0.477 | 0.449 | 0.000 (invoices) | invoices->rooms | 35 / 58 |
| 6 | expenses | message | 0.575 | 0.343 | 0.296 (invoices) | invoices->expenses | 2 / 94 |
| 6 | expenses | with-opening | 0.570 | 0.426 | 0.259 (invoices) | tickets->rooms | 14 / 103 |
| 6 | expenses | episode | 0.462 | 0.426 | 0.000 (invoices) | expenses->rooms | 0 / 74 |
| 7 | approvals | message | 0.560 | 0.343 | 0.286 (invoices) | invoices->expenses | 9 / 127 |
| 7 | approvals | with-opening | 0.581 | 0.414 | 0.268 (invoices) | tickets->rooms | 2 / 126 |
| 7 | approvals | episode | 0.468 | 0.421 | 0.000 (invoices) | expenses->rooms | 0 / 102 |
| 8 | reminders | message | 0.588 | 0.362 | 0.276 (invoices) | invoices->expenses | 3 / 159 |
| 8 | reminders | with-opening | 0.599 | 0.431 | 0.310 (tickets) | expenses->rooms | 5 / 165 |
| 8 | reminders | episode | 0.497 | 0.443 | 0.000 (invoices) | expenses->rooms | 9 / 133 |
| 9 | summaries | message | 0.578 | 0.367 | 0.267 (invoices) | invoices->expenses | 5 / 208 |
| 9 | summaries | with-opening | 0.585 | 0.414 | 0.117 (tickets) | expenses->rooms | 12 / 212 |
| 9 | summaries | episode | 0.491 | 0.433 | 0.000 (invoices) | invoices->expenses | 7 / 176 |
| 10 | travel | message | 0.520 | 0.319 | 0.161 (travel) | travel->reminders | 0 / 244 |
| 10 | travel | with-opening | 0.564 | 0.391 | 0.113 (tickets) | expenses->rooms | 0 / 247 |
| 10 | travel | episode | 0.422 | 0.367 | 0.000 (invoices) | travel->reminders | 0 / 207 |
| 11 | inventory | message | 0.529 | 0.319 | 0.156 (travel) | travel->reminders | 14 / 259 |
| 11 | inventory | with-opening | 0.593 | 0.420 | 0.109 (tickets) | expenses->rooms | 2 / 281 |
| 11 | inventory | episode | 0.445 | 0.378 | 0.000 (invoices) | travel->reminders | 12 / 210 |
| 12 | timesheets | message | 0.489 | 0.273 | 0.152 (travel) | travel->reminders | 20 / 305 |
| 12 | timesheets | with-opening | 0.586 | 0.412 | 0.091 (tickets) | tickets->timesheets | 14 / 342 |
| 12 | timesheets | episode | 0.414 | 0.342 | 0.000 (travel) | travel->reminders | 21 / 257 |
| 13 | contacts | message | 0.455 | 0.233 | 0.147 (travel) | travel->reminders | 12 / 321 |
| 13 | contacts | with-opening | 0.568 | 0.388 | 0.074 (tickets) | tickets->timesheets | 5 / 385 |
| 13 | contacts | episode | 0.365 | 0.294 | 0.000 (travel) | travel->reminders | 11 / 272 |
| 14 | search | message | 0.427 | 0.202 | 0.143 (travel) | travel->reminders | 11 / 338 |
| 14 | search | with-opening | 0.531 | 0.340 | 0.071 (tickets) | tickets->timesheets | 6 / 422 |
| 14 | search | episode | 0.340 | 0.267 | 0.000 (travel) | travel->reminders | 7 / 271 |

## centroids-frozen, test

| routes | added | strategy | turn acc | episode acc | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | scheduling | message | 1.000 | 1.000 | 1.000 (drafting) | – | – |
| 2 | scheduling | with-opening | 0.980 | 0.971 | 0.962 (scheduling) | scheduling->drafting | – |
| 2 | scheduling | episode | 0.980 | 0.971 | 0.962 (scheduling) | scheduling->drafting | – |
| 3 | invoices | message | 0.860 | 0.781 | 0.680 (drafting) | drafting->invoices | 35 / 204 |
| 3 | invoices | with-opening | 0.984 | 0.975 | 0.962 (scheduling) | scheduling->drafting | 0 / 200 |
| 3 | invoices | episode | 0.984 | 0.975 | 0.962 (scheduling) | scheduling->drafting | 0 / 200 |
| 4 | tickets | message | 0.779 | 0.645 | 0.680 (drafting) | drafting->tickets | 29 / 215 |
| 4 | tickets | with-opening | 0.980 | 0.968 | 0.962 (scheduling) | scheduling->drafting | 0 / 246 |
| 4 | tickets | episode | 0.980 | 0.968 | 0.962 (scheduling) | scheduling->drafting | 0 / 246 |
| 5 | rooms | message | 0.785 | 0.645 | 0.680 (drafting) | scheduling->tickets | 4 / 232 |
| 5 | rooms | with-opening | 0.975 | 0.958 | 0.958 (tickets) | scheduling->drafting | 0 / 292 |
| 5 | rooms | episode | 0.972 | 0.953 | 0.958 (tickets) | scheduling->drafting | 0 / 292 |
| 6 | expenses | message | 0.798 | 0.664 | 0.680 (drafting) | scheduling->tickets | 4 / 278 |
| 6 | expenses | with-opening | 0.963 | 0.939 | 0.940 (expenses) | scheduling->drafting | 0 / 345 |
| 6 | expenses | episode | 0.961 | 0.934 | 0.940 (expenses) | scheduling->drafting | 0 / 344 |
| 7 | approvals | message | 0.811 | 0.685 | 0.680 (drafting) | scheduling->approvals | 2 / 327 |
| 7 | approvals | with-opening | 0.954 | 0.920 | 0.923 (tickets) | scheduling->drafting | 0 / 395 |
| 7 | approvals | episode | 0.950 | 0.913 | 0.923 (expenses) | scheduling->drafting | 0 / 394 |
| 8 | reminders | message | 0.817 | 0.690 | 0.680 (drafting) | scheduling->approvals | 1 / 386 |
| 8 | reminders | with-opening | 0.943 | 0.900 | 0.907 (tickets) | scheduling->drafting | 0 / 454 |
| 8 | reminders | episode | 0.938 | 0.890 | 0.907 (expenses) | scheduling->drafting | 0 / 452 |
| 9 | summaries | message | 0.822 | 0.697 | 0.680 (drafting) | scheduling->summaries | 2 / 445 |
| 9 | summaries | with-opening | 0.930 | 0.876 | 0.893 (tickets) | scheduling->drafting | 0 / 514 |
| 9 | summaries | episode | 0.926 | 0.867 | 0.893 (expenses) | scheduling->drafting | 0 / 511 |
| 10 | travel | message | 0.831 | 0.711 | 0.680 (drafting) | scheduling->summaries | 0 / 508 |
| 10 | travel | with-opening | 0.920 | 0.857 | 0.879 (tickets) | scheduling->drafting | 0 / 575 |
| 10 | travel | episode | 0.913 | 0.844 | 0.870 (travel) | scheduling->drafting | 0 / 572 |
| 11 | inventory | message | 0.824 | 0.705 | 0.680 (drafting) | scheduling->summaries | 11 / 570 |
| 11 | inventory | with-opening | 0.908 | 0.833 | 0.867 (tickets) | scheduling->drafting | 0 / 631 |
| 11 | inventory | episode | 0.901 | 0.821 | 0.857 (travel) | scheduling->drafting | 0 / 626 |
| 12 | timesheets | message | 0.831 | 0.719 | 0.680 (drafting) | scheduling->summaries | 2 / 635 |
| 12 | timesheets | with-opening | 0.897 | 0.811 | 0.855 (tickets) | scheduling->drafting | 0 / 700 |
| 12 | timesheets | episode | 0.890 | 0.798 | 0.845 (travel) | scheduling->drafting | 0 / 695 |
| 13 | contacts | message | 0.830 | 0.716 | 0.680 (drafting) | scheduling->summaries | 8 / 708 |
| 13 | contacts | with-opening | 0.892 | 0.800 | 0.844 (tickets) | scheduling->drafting | 0 / 764 |
| 13 | contacts | episode | 0.880 | 0.776 | 0.833 (travel) | scheduling->drafting | 0 / 758 |
| 14 | search | message | 0.838 | 0.730 | 0.680 (drafting) | scheduling->summaries | 4 / 786 |
| 14 | search | with-opening | 0.883 | 0.781 | 0.833 (tickets) | scheduling->drafting | 0 / 845 |
| 14 | search | episode | 0.869 | 0.755 | 0.823 (travel) | scheduling->drafting | 0 / 833 |

## centroids-frozen, unseen

| routes | added | strategy | turn acc | episode acc | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | invoices | message | 1.000 | 1.000 | 1.000 (invoices) | – | 0 / 0 |
| 3 | invoices | with-opening | 1.000 | 1.000 | 1.000 (invoices) | – | 0 / 0 |
| 3 | invoices | episode | 1.000 | 1.000 | 1.000 (invoices) | – | 0 / 0 |
| 4 | tickets | message | 0.700 | 0.440 | 0.680 (invoices) | invoices->tickets | 15 / 48 |
| 4 | tickets | with-opening | 0.780 | 0.620 | 0.680 (tickets) | tickets->invoices | 4 / 48 |
| 4 | tickets | episode | 0.580 | 0.560 | 0.480 (tickets) | tickets->invoices | 14 / 48 |
| 5 | rooms | message | 0.619 | 0.397 | 0.308 (invoices) | invoices->rooms | 20 / 70 |
| 5 | rooms | with-opening | 0.645 | 0.564 | 0.462 (tickets) | tickets->rooms | 30 / 78 |
| 5 | rooms | episode | 0.477 | 0.449 | 0.000 (invoices) | invoices->rooms | 35 / 58 |
| 6 | expenses | message | 0.584 | 0.352 | 0.296 (invoices) | invoices->expenses | 2 / 96 |
| 6 | expenses | with-opening | 0.561 | 0.417 | 0.259 (invoices) | tickets->rooms | 11 / 100 |
| 6 | expenses | episode | 0.462 | 0.426 | 0.000 (invoices) | expenses->rooms | 0 / 74 |
| 7 | approvals | message | 0.567 | 0.350 | 0.286 (invoices) | invoices->expenses | 7 / 129 |
| 7 | approvals | with-opening | 0.549 | 0.407 | 0.250 (invoices) | tickets->rooms | 1 / 124 |
| 7 | approvals | episode | 0.468 | 0.421 | 0.000 (invoices) | expenses->rooms | 0 / 102 |
| 8 | reminders | message | 0.571 | 0.339 | 0.276 (invoices) | tickets->reminders | 15 / 161 |
| 8 | reminders | with-opening | 0.559 | 0.391 | 0.259 (invoices) | expenses->rooms | 2 / 156 |
| 8 | reminders | episode | 0.455 | 0.402 | 0.000 (invoices) | tickets->reminders | 32 / 133 |
| 9 | summaries | message | 0.573 | 0.357 | 0.267 (invoices) | invoices->expenses | 5 / 202 |
| 9 | summaries | with-opening | 0.573 | 0.414 | 0.250 (invoices) | expenses->rooms | 5 / 198 |
| 9 | summaries | episode | 0.467 | 0.410 | 0.000 (invoices) | invoices->expenses | 1 / 161 |
| 10 | travel | message | 0.524 | 0.315 | 0.161 (travel) | travel->reminders | 0 / 242 |
| 10 | travel | with-opening | 0.530 | 0.371 | 0.242 (invoices) | expenses->rooms | 1 / 242 |
| 10 | travel | episode | 0.426 | 0.367 | 0.000 (invoices) | travel->reminders | 0 / 197 |
| 11 | inventory | message | 0.556 | 0.344 | 0.156 (travel) | travel->reminders | 5 / 261 |
| 11 | inventory | with-opening | 0.572 | 0.417 | 0.250 (invoices) | expenses->rooms | 2 / 264 |
| 11 | inventory | episode | 0.461 | 0.389 | 0.000 (invoices) | travel->reminders | 7 / 212 |
| 12 | timesheets | message | 0.514 | 0.282 | 0.152 (travel) | travel->reminders | 25 / 321 |
| 12 | timesheets | with-opening | 0.557 | 0.391 | 0.227 (invoices) | tickets->timesheets | 12 / 330 |
| 12 | timesheets | episode | 0.419 | 0.339 | 0.000 (invoices) | travel->reminders | 30 / 266 |
| 13 | contacts | message | 0.472 | 0.233 | 0.147 (travel) | travel->reminders | 17 / 338 |
| 13 | contacts | with-opening | 0.509 | 0.318 | 0.147 (tickets) | expenses->rooms | 20 / 366 |
| 13 | contacts | episode | 0.376 | 0.299 | 0.000 (invoices) | travel->reminders | 7 / 275 |
| 14 | search | message | 0.457 | 0.224 | 0.143 (travel) | travel->reminders | 1 / 351 |
| 14 | search | with-opening | 0.502 | 0.305 | 0.186 (tickets) | tickets->timesheets | 7 / 378 |
| 14 | search | episode | 0.357 | 0.276 | 0.000 (invoices) | travel->reminders | 1 / 279 |

## logistic-refit, test

| routes | added | strategy | turn acc | episode acc | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | scheduling | message | 1.000 | 1.000 | 1.000 (drafting) | – | – |
| 2 | scheduling | with-opening | 1.000 | 1.000 | 1.000 (drafting) | – | – |
| 2 | scheduling | episode | 0.980 | 0.971 | 0.962 (scheduling) | scheduling->drafting | – |
| 3 | invoices | message | 1.000 | 1.000 | 1.000 (drafting) | – | 0 / 204 |
| 3 | invoices | with-opening | 1.000 | 1.000 | 1.000 (drafting) | – | 0 / 204 |
| 3 | invoices | episode | 0.984 | 0.975 | 0.962 (scheduling) | scheduling->drafting | 0 / 200 |
| 4 | tickets | message | 0.966 | 0.952 | 0.854 (invoices) | invoices->tickets | 7 / 250 |
| 4 | tickets | with-opening | 0.993 | 0.989 | 0.978 (tickets) | tickets->invoices | 0 / 250 |
| 4 | tickets | episode | 0.980 | 0.968 | 0.962 (scheduling) | scheduling->drafting | 0 / 246 |
| 5 | rooms | message | 0.944 | 0.925 | 0.760 (invoices) | invoices->rooms | 7 / 288 |
| 5 | rooms | with-opening | 0.989 | 0.981 | 0.960 (invoices) | tickets->rooms | 0 / 296 |
| 5 | rooms | episode | 0.972 | 0.953 | 0.958 (tickets) | scheduling->drafting | 0 / 292 |
| 6 | expenses | message | 0.937 | 0.910 | 0.769 (invoices) | invoices->rooms | 2 / 334 |
| 6 | expenses | with-opening | 0.983 | 0.971 | 0.940 (expenses) | rooms->tickets | 0 / 350 |
| 6 | expenses | episode | 0.961 | 0.934 | 0.940 (expenses) | scheduling->drafting | 0 / 344 |
| 7 | approvals | message | 0.935 | 0.902 | 0.759 (invoices) | invoices->approvals | 3 / 384 |
| 7 | approvals | with-opening | 0.981 | 0.967 | 0.962 (expenses) | tickets->rooms | 1 / 403 |
| 7 | approvals | episode | 0.950 | 0.913 | 0.923 (expenses) | scheduling->drafting | 0 / 394 |
| 8 | reminders | message | 0.919 | 0.877 | 0.768 (invoices) | invoices->approvals | 5 / 445 |
| 8 | reminders | with-opening | 0.971 | 0.948 | 0.931 (rooms) | rooms->reminders | 1 / 467 |
| 8 | reminders | episode | 0.938 | 0.890 | 0.907 (expenses) | scheduling->drafting | 0 / 452 |
| 9 | summaries | message | 0.914 | 0.867 | 0.776 (invoices) | invoices->approvals | 0 / 501 |
| 9 | summaries | with-opening | 0.969 | 0.945 | 0.933 (rooms) | rooms->reminders | 0 / 529 |
| 9 | summaries | episode | 0.926 | 0.867 | 0.893 (expenses) | scheduling->drafting | 0 / 511 |
| 10 | travel | message | 0.914 | 0.865 | 0.783 (invoices) | invoices->approvals | 0 / 565 |
| 10 | travel | with-opening | 0.965 | 0.938 | 0.919 (rooms) | travel->rooms | 1 / 599 |
| 10 | travel | episode | 0.913 | 0.844 | 0.870 (travel) | scheduling->drafting | 0 / 572 |
| 11 | inventory | message | 0.901 | 0.844 | 0.790 (invoices) | inventory->reminders | 6 / 627 |
| 11 | inventory | with-opening | 0.960 | 0.927 | 0.922 (rooms) | travel->rooms | 1 / 662 |
| 11 | inventory | episode | 0.901 | 0.821 | 0.857 (travel) | scheduling->drafting | 0 / 626 |
| 12 | timesheets | message | 0.904 | 0.848 | 0.797 (invoices) | inventory->reminders | 0 / 695 |
| 12 | timesheets | with-opening | 0.959 | 0.925 | 0.909 (rooms) | travel->rooms | 2 / 740 |
| 12 | timesheets | episode | 0.890 | 0.798 | 0.845 (travel) | scheduling->drafting | 0 / 695 |
| 13 | contacts | message | 0.901 | 0.841 | 0.803 (invoices) | inventory->reminders | 0 / 770 |
| 13 | contacts | with-opening | 0.954 | 0.914 | 0.912 (rooms) | travel->rooms | 2 / 817 |
| 13 | contacts | episode | 0.880 | 0.776 | 0.833 (travel) | scheduling->drafting | 0 / 758 |
| 14 | search | message | 0.903 | 0.845 | 0.809 (invoices) | inventory->search | 5 / 853 |
| 14 | search | with-opening | 0.959 | 0.923 | 0.929 (rooms) | travel->rooms | 3 / 903 |
| 14 | search | episode | 0.869 | 0.755 | 0.823 (travel) | scheduling->drafting | 0 / 833 |

## logistic-refit, unseen

| routes | added | strategy | turn acc | episode acc | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | invoices | message | 1.000 | 1.000 | 1.000 (invoices) | – | 0 / 0 |
| 3 | invoices | with-opening | 1.000 | 1.000 | 1.000 (invoices) | – | 0 / 0 |
| 3 | invoices | episode | 1.000 | 1.000 | 1.000 (invoices) | – | 0 / 0 |
| 4 | tickets | message | 0.650 | 0.400 | 0.620 (invoices) | invoices->tickets | 18 / 48 |
| 4 | tickets | with-opening | 0.690 | 0.620 | 0.400 (tickets) | tickets->invoices | 0 / 48 |
| 4 | tickets | episode | 0.520 | 0.500 | 0.480 (tickets) | tickets->invoices | 20 / 48 |
| 5 | rooms | message | 0.677 | 0.474 | 0.500 (invoices) | invoices->tickets | 10 / 65 |
| 5 | rooms | with-opening | 0.645 | 0.474 | 0.308 (tickets) | tickets->rooms | 16 / 69 |
| 5 | rooms | episode | 0.574 | 0.551 | 0.288 (invoices) | invoices->tickets | 14 / 52 |
| 6 | expenses | message | 0.638 | 0.426 | 0.463 (invoices) | expenses->rooms | 2 / 105 |
| 6 | expenses | with-opening | 0.643 | 0.463 | 0.296 (tickets) | tickets->rooms | 6 / 100 |
| 6 | expenses | episode | 0.543 | 0.509 | 0.259 (invoices) | expenses->rooms | 1 / 89 |
| 7 | approvals | message | 0.595 | 0.357 | 0.446 (tickets) | expenses->rooms | 9 / 141 |
| 7 | approvals | with-opening | 0.623 | 0.421 | 0.232 (tickets) | tickets->rooms | 8 / 142 |
| 7 | approvals | episode | 0.514 | 0.471 | 0.286 (invoices) | invoices->expenses | 1 / 120 |
| 8 | reminders | message | 0.554 | 0.276 | 0.386 (approvals) | approvals->reminders | 16 / 169 |
| 8 | reminders | with-opening | 0.582 | 0.374 | 0.190 (tickets) | reminders->rooms | 13 / 177 |
| 8 | reminders | episode | 0.475 | 0.408 | 0.105 (approvals) | approvals->reminders | 20 / 146 |
| 9 | summaries | message | 0.573 | 0.343 | 0.417 (invoices) | approvals->reminders | 2 / 196 |
| 9 | summaries | with-opening | 0.607 | 0.395 | 0.150 (tickets) | reminders->rooms | 5 / 206 |
| 9 | summaries | episode | 0.481 | 0.429 | 0.233 (invoices) | reminders->rooms | 7 / 168 |
| 10 | travel | message | 0.586 | 0.367 | 0.414 (summaries) | reminders->rooms | 8 / 242 |
| 10 | travel | with-opening | 0.631 | 0.435 | 0.194 (tickets) | travel->reminders | 3 / 256 |
| 10 | travel | episode | 0.514 | 0.452 | 0.241 (summaries) | reminders->rooms | 11 / 203 |
| 11 | inventory | message | 0.615 | 0.396 | 0.367 (summaries) | summaries->expenses | 11 / 292 |
| 11 | inventory | with-opening | 0.601 | 0.399 | 0.109 (tickets) | tickets->invoices | 18 / 314 |
| 11 | inventory | episode | 0.567 | 0.483 | 0.183 (summaries) | summaries->expenses | 15 / 256 |
| 12 | timesheets | message | 0.629 | 0.430 | 0.409 (invoices) | tickets->timesheets | 26 / 355 |
| 12 | timesheets | with-opening | 0.624 | 0.424 | 0.106 (tickets) | tickets->timesheets | 18 / 347 |
| 12 | timesheets | episode | 0.554 | 0.470 | 0.227 (invoices) | tickets->timesheets | 39 / 327 |
| 13 | contacts | message | 0.618 | 0.406 | 0.397 (invoices) | tickets->timesheets | 11 / 413 |
| 13 | contacts | with-opening | 0.615 | 0.412 | 0.191 (tickets) | tickets->timesheets | 17 / 410 |
| 13 | contacts | episode | 0.542 | 0.452 | 0.221 (invoices) | tickets->timesheets | 9 / 364 |
| 14 | search | message | 0.561 | 0.338 | 0.288 (summaries) | tickets->timesheets | 29 / 459 |
| 14 | search | with-opening | 0.578 | 0.371 | 0.157 (tickets) | tickets->timesheets | 22 / 457 |
| 14 | search | episode | 0.476 | 0.381 | 0.119 (search) | tickets->timesheets | 30 / 403 |
