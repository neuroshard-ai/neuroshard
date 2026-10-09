# Router scaling: hashed encoder (hashed)

836 fit cases; evaluation cases {'test': 556, 'unseen': 468}, turns {'test': 1040, 'unseen': 909}; 2646 texts, features in 0 s. Intervals: 1,000 case-level bootstrap draws.

## centroids-refit, test

| routes | added | strategy | turn acc [95% CI] | episode acc [95% CI] | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | scheduling | message | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 (drafting) | – | – |
| 2 | scheduling | with-opening | 0.980 [0.959, 0.995] | 0.971 [0.941, 0.993] | 0.962 (scheduling) | scheduling->drafting | – |
| 2 | scheduling | episode | 0.980 [0.959, 0.995] | 0.971 [0.941, 0.993] | 0.962 (scheduling) | scheduling->drafting | – |
| 3 | invoices | message | 0.936 [0.906, 0.963] | 0.900 [0.850, 0.944] | 0.840 (drafting) | drafting->invoices | 16 / 204 |
| 3 | invoices | with-opening | 0.984 [0.967, 0.996] | 0.975 [0.950, 0.994] | 0.962 (scheduling) | scheduling->drafting | 0 / 200 |
| 3 | invoices | episode | 0.984 [0.967, 0.996] | 0.975 [0.950, 0.994] | 0.962 (scheduling) | scheduling->drafting | 0 / 200 |
| 4 | tickets | message | 0.913 [0.885, 0.942] | 0.860 [0.812, 0.909] | 0.833 (invoices) | drafting->tickets | 8 / 234 |
| 4 | tickets | with-opening | 0.980 [0.963, 0.993] | 0.968 [0.941, 0.989] | 0.962 (scheduling) | scheduling->drafting | 0 / 246 |
| 4 | tickets | episode | 0.980 [0.963, 0.993] | 0.968 [0.941, 0.989] | 0.962 (scheduling) | scheduling->drafting | 0 / 246 |
| 5 | rooms | message | 0.921 [0.895, 0.946] | 0.869 [0.822, 0.911] | 0.840 (invoices) | drafting->tickets | 4 / 272 |
| 5 | rooms | with-opening | 0.975 [0.958, 0.991] | 0.958 [0.930, 0.986] | 0.958 (tickets) | scheduling->drafting | 0 / 292 |
| 5 | rooms | episode | 0.972 [0.955, 0.989] | 0.953 [0.925, 0.981] | 0.958 (tickets) | scheduling->drafting | 0 / 292 |
| 6 | expenses | message | 0.932 [0.905, 0.956] | 0.893 [0.852, 0.930] | 0.815 (rooms) | tickets->expenses | 5 / 326 |
| 6 | expenses | with-opening | 0.963 [0.946, 0.980] | 0.939 [0.910, 0.967] | 0.940 (expenses) | scheduling->drafting | 0 / 345 |
| 6 | expenses | episode | 0.961 [0.940, 0.978] | 0.934 [0.902, 0.963] | 0.940 (expenses) | scheduling->drafting | 0 / 344 |
| 7 | approvals | message | 0.926 [0.899, 0.951] | 0.891 [0.851, 0.928] | 0.759 (invoices) | invoices->approvals | 6 / 382 |
| 7 | approvals | with-opening | 0.952 [0.934, 0.970] | 0.917 [0.888, 0.949] | 0.923 (expenses) | scheduling->drafting | 0 / 395 |
| 7 | approvals | episode | 0.950 [0.932, 0.969] | 0.913 [0.884, 0.946] | 0.923 (expenses) | scheduling->drafting | 0 / 394 |
| 8 | reminders | message | 0.916 [0.890, 0.940] | 0.871 [0.832, 0.906] | 0.768 (invoices) | invoices->approvals | 0 / 441 |
| 8 | reminders | with-opening | 0.941 [0.922, 0.959] | 0.897 [0.861, 0.929] | 0.907 (expenses) | scheduling->drafting | 0 / 453 |
| 8 | reminders | episode | 0.938 [0.917, 0.958] | 0.890 [0.855, 0.926] | 0.907 (expenses) | scheduling->drafting | 0 / 452 |
| 9 | summaries | message | 0.909 [0.885, 0.932] | 0.858 [0.818, 0.893] | 0.776 (invoices) | invoices->approvals | 1 / 499 |
| 9 | summaries | with-opening | 0.929 [0.910, 0.948] | 0.873 [0.838, 0.908] | 0.893 (expenses) | scheduling->drafting | 0 / 513 |
| 9 | summaries | episode | 0.926 [0.905, 0.945] | 0.867 [0.832, 0.902] | 0.893 (expenses) | scheduling->drafting | 0 / 511 |
| 10 | travel | message | 0.910 [0.887, 0.930] | 0.857 [0.820, 0.888] | 0.783 (invoices) | invoices->approvals | 0 / 562 |
| 10 | travel | with-opening | 0.920 [0.901, 0.940] | 0.857 [0.823, 0.893] | 0.879 (expenses) | scheduling->drafting | 0 / 574 |
| 10 | travel | episode | 0.913 [0.893, 0.933] | 0.844 [0.810, 0.880] | 0.870 (travel) | scheduling->drafting | 0 / 572 |
| 11 | inventory | message | 0.901 [0.878, 0.923] | 0.844 [0.804, 0.877] | 0.790 (invoices) | inventory->expenses | 3 / 624 |
| 11 | inventory | with-opening | 0.908 [0.890, 0.928] | 0.833 [0.800, 0.870] | 0.867 (expenses) | scheduling->drafting | 0 / 631 |
| 11 | inventory | episode | 0.901 [0.881, 0.921] | 0.821 [0.785, 0.858] | 0.857 (travel) | scheduling->drafting | 0 / 626 |
| 12 | timesheets | message | 0.891 [0.869, 0.912] | 0.835 [0.800, 0.867] | 0.797 (invoices) | timesheets->summaries | 5 / 695 |
| 12 | timesheets | with-opening | 0.900 [0.881, 0.919] | 0.818 [0.781, 0.852] | 0.855 (expenses) | scheduling->drafting | 0 / 700 |
| 12 | timesheets | episode | 0.890 [0.869, 0.909] | 0.798 [0.760, 0.835] | 0.845 (travel) | scheduling->drafting | 0 / 695 |
| 13 | contacts | message | 0.883 [0.859, 0.902] | 0.822 [0.788, 0.853] | 0.765 (rooms) | rooms->contacts | 8 / 759 |
| 13 | contacts | with-opening | 0.895 [0.877, 0.914] | 0.806 [0.771, 0.841] | 0.844 (expenses) | scheduling->drafting | 0 / 767 |
| 13 | contacts | episode | 0.880 [0.861, 0.899] | 0.776 [0.741, 0.812] | 0.833 (travel) | scheduling->drafting | 0 / 758 |
| 14 | search | message | 0.885 [0.863, 0.904] | 0.824 [0.791, 0.854] | 0.760 (inventory) | rooms->contacts | 6 / 836 |
| 14 | search | with-opening | 0.887 [0.869, 0.906] | 0.790 [0.754, 0.824] | 0.833 (expenses) | scheduling->drafting | 0 / 848 |
| 14 | search | episode | 0.869 [0.850, 0.889] | 0.755 [0.719, 0.791] | 0.823 (travel) | scheduling->drafting | 0 / 833 |

## centroids-refit, unseen

| routes | added | strategy | turn acc [95% CI] | episode acc [95% CI] | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | scheduling | message | 0.818 [0.740, 0.892] | 0.708 [0.562, 0.833] | 0.816 (drafting) | scheduling->drafting | – |
| 2 | scheduling | with-opening | 0.909 [0.850, 0.963] | 0.854 [0.750, 0.938] | 0.816 (drafting) | drafting->scheduling | – |
| 2 | scheduling | episode | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 (drafting) | – | – |
| 3 | invoices | message | 0.568 [0.467, 0.664] | 0.444 [0.333, 0.556] | 0.077 (scheduling) | scheduling->invoices | 38 / 63 |
| 3 | invoices | with-opening | 0.632 [0.542, 0.728] | 0.500 [0.389, 0.611] | 0.231 (scheduling) | scheduling->invoices | 39 / 70 |
| 3 | invoices | episode | 0.704 [0.602, 0.803] | 0.681 [0.569, 0.778] | 0.103 (scheduling) | scheduling->invoices | 37 / 77 |
| 4 | tickets | message | 0.514 [0.439, 0.589] | 0.327 [0.235, 0.418] | 0.077 (scheduling) | scheduling->tickets | 21 / 71 |
| 4 | tickets | with-opening | 0.548 [0.466, 0.630] | 0.408 [0.316, 0.500] | 0.051 (scheduling) | scheduling->tickets | 18 / 79 |
| 4 | tickets | episode | 0.508 [0.412, 0.617] | 0.500 [0.408, 0.602] | 0.128 (scheduling) | tickets->invoices | 28 / 88 |
| 5 | rooms | message | 0.496 [0.431, 0.560] | 0.310 [0.230, 0.389] | 0.000 (scheduling) | invoices->rooms | 25 / 91 |
| 5 | rooms | with-opening | 0.599 [0.514, 0.681] | 0.548 [0.460, 0.635] | 0.000 (scheduling) | scheduling->rooms | 29 / 97 |
| 5 | rooms | episode | 0.461 [0.371, 0.557] | 0.444 [0.357, 0.532] | 0.000 (invoices) | invoices->rooms | 40 / 90 |
| 6 | expenses | message | 0.507 [0.447, 0.563] | 0.301 [0.231, 0.372] | 0.000 (scheduling) | scheduling->rooms | 2 / 115 |
| 6 | expenses | with-opening | 0.550 [0.472, 0.621] | 0.449 [0.365, 0.526] | 0.000 (scheduling) | scheduling->rooms | 14 / 139 |
| 6 | expenses | episode | 0.470 [0.381, 0.549] | 0.449 [0.365, 0.526] | 0.000 (invoices) | scheduling->rooms | 0 / 107 |
| 7 | approvals | message | 0.485 [0.431, 0.543] | 0.293 [0.223, 0.357] | 0.000 (scheduling) | scheduling->rooms | 17 / 151 |
| 7 | approvals | with-opening | 0.529 [0.466, 0.596] | 0.394 [0.319, 0.463] | 0.000 (scheduling) | scheduling->rooms | 14 / 164 |
| 7 | approvals | episode | 0.438 [0.363, 0.510] | 0.399 [0.324, 0.468] | 0.000 (invoices) | scheduling->rooms | 13 / 140 |
| 8 | reminders | message | 0.520 [0.469, 0.569] | 0.315 [0.257, 0.374] | 0.000 (scheduling) | invoices->expenses | 3 / 175 |
| 8 | reminders | with-opening | 0.559 [0.504, 0.619] | 0.419 [0.356, 0.486] | 0.000 (scheduling) | scheduling->rooms | 5 / 191 |
| 8 | reminders | episode | 0.466 [0.403, 0.530] | 0.419 [0.360, 0.482] | 0.000 (invoices) | expenses->rooms | 9 / 158 |
| 9 | summaries | message | 0.521 [0.478, 0.569] | 0.326 [0.275, 0.380] | 0.000 (scheduling) | invoices->expenses | 5 / 224 |
| 9 | summaries | with-opening | 0.549 [0.496, 0.601] | 0.403 [0.341, 0.465] | 0.000 (scheduling) | scheduling->rooms | 14 / 241 |
| 9 | summaries | episode | 0.465 [0.405, 0.527] | 0.415 [0.357, 0.477] | 0.000 (invoices) | invoices->expenses | 7 / 201 |
| 10 | travel | message | 0.478 [0.435, 0.520] | 0.291 [0.243, 0.341] | 0.000 (scheduling) | travel->reminders | 0 / 260 |
| 10 | travel | with-opening | 0.536 [0.486, 0.583] | 0.385 [0.331, 0.446] | 0.000 (scheduling) | scheduling->rooms | 0 / 274 |
| 10 | travel | episode | 0.409 [0.355, 0.464] | 0.361 [0.311, 0.416] | 0.000 (invoices) | travel->reminders | 0 / 232 |
| 11 | inventory | message | 0.495 [0.458, 0.535] | 0.301 [0.253, 0.348] | 0.000 (scheduling) | travel->reminders | 14 / 275 |
| 11 | inventory | with-opening | 0.564 [0.522, 0.606] | 0.411 [0.360, 0.461] | 0.000 (scheduling) | scheduling->rooms | 2 / 308 |
| 11 | inventory | episode | 0.437 [0.386, 0.482] | 0.381 [0.333, 0.429] | 0.000 (invoices) | travel->reminders | 12 / 235 |
| 12 | timesheets | message | 0.463 [0.428, 0.503] | 0.262 [0.220, 0.304] | 0.000 (scheduling) | travel->reminders | 20 / 324 |
| 12 | timesheets | with-opening | 0.567 [0.523, 0.607] | 0.410 [0.360, 0.458] | 0.051 (scheduling) | tickets->timesheets | 14 / 369 |
| 12 | timesheets | episode | 0.410 [0.366, 0.461] | 0.349 [0.304, 0.397] | 0.000 (scheduling) | travel->reminders | 21 / 286 |
| 13 | contacts | message | 0.435 [0.401, 0.469] | 0.227 [0.190, 0.265] | 0.000 (scheduling) | travel->reminders | 12 / 340 |
| 13 | contacts | with-opening | 0.552 [0.514, 0.594] | 0.389 [0.344, 0.434] | 0.051 (scheduling) | tickets->timesheets | 5 / 416 |
| 13 | contacts | episode | 0.366 [0.322, 0.409] | 0.306 [0.261, 0.348] | 0.000 (scheduling) | travel->reminders | 11 / 301 |
| 14 | search | message | 0.411 [0.378, 0.443] | 0.201 [0.160, 0.237] | 0.000 (scheduling) | travel->reminders | 11 / 357 |
| 14 | search | with-opening | 0.523 [0.486, 0.558] | 0.348 [0.308, 0.393] | 0.051 (scheduling) | tickets->timesheets | 6 / 453 |
| 14 | search | episode | 0.343 [0.301, 0.387] | 0.280 [0.239, 0.325] | 0.000 (scheduling) | travel->reminders | 7 / 300 |

## centroids-frozen, test

| routes | added | strategy | turn acc [95% CI] | episode acc [95% CI] | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | scheduling | message | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 (drafting) | – | – |
| 2 | scheduling | with-opening | 0.980 [0.959, 0.995] | 0.971 [0.941, 0.993] | 0.962 (scheduling) | scheduling->drafting | – |
| 2 | scheduling | episode | 0.980 [0.959, 0.995] | 0.971 [0.941, 0.993] | 0.962 (scheduling) | scheduling->drafting | – |
| 3 | invoices | message | 0.860 [0.822, 0.898] | 0.781 [0.713, 0.844] | 0.680 (drafting) | drafting->invoices | 35 / 204 |
| 3 | invoices | with-opening | 0.984 [0.967, 0.996] | 0.975 [0.950, 0.994] | 0.962 (scheduling) | scheduling->drafting | 0 / 200 |
| 3 | invoices | episode | 0.984 [0.967, 0.996] | 0.975 [0.950, 0.994] | 0.962 (scheduling) | scheduling->drafting | 0 / 200 |
| 4 | tickets | message | 0.779 [0.745, 0.817] | 0.645 [0.581, 0.710] | 0.680 (drafting) | drafting->tickets | 29 / 215 |
| 4 | tickets | with-opening | 0.980 [0.963, 0.993] | 0.968 [0.941, 0.989] | 0.962 (scheduling) | scheduling->drafting | 0 / 246 |
| 4 | tickets | episode | 0.980 [0.963, 0.993] | 0.968 [0.941, 0.989] | 0.962 (scheduling) | scheduling->drafting | 0 / 246 |
| 5 | rooms | message | 0.785 [0.753, 0.818] | 0.645 [0.579, 0.706] | 0.680 (drafting) | scheduling->tickets | 4 / 232 |
| 5 | rooms | with-opening | 0.975 [0.958, 0.991] | 0.958 [0.930, 0.986] | 0.958 (tickets) | scheduling->drafting | 0 / 292 |
| 5 | rooms | episode | 0.972 [0.955, 0.989] | 0.953 [0.925, 0.981] | 0.958 (tickets) | scheduling->drafting | 0 / 292 |
| 6 | expenses | message | 0.798 [0.767, 0.828] | 0.664 [0.606, 0.717] | 0.680 (drafting) | scheduling->tickets | 4 / 278 |
| 6 | expenses | with-opening | 0.963 [0.946, 0.980] | 0.939 [0.910, 0.967] | 0.940 (expenses) | scheduling->drafting | 0 / 345 |
| 6 | expenses | episode | 0.961 [0.940, 0.978] | 0.934 [0.902, 0.963] | 0.940 (expenses) | scheduling->drafting | 0 / 344 |
| 7 | approvals | message | 0.811 [0.779, 0.839] | 0.685 [0.630, 0.739] | 0.680 (drafting) | scheduling->approvals | 2 / 327 |
| 7 | approvals | with-opening | 0.954 [0.936, 0.971] | 0.920 [0.891, 0.949] | 0.923 (tickets) | scheduling->drafting | 0 / 395 |
| 7 | approvals | episode | 0.950 [0.932, 0.969] | 0.913 [0.884, 0.946] | 0.923 (expenses) | scheduling->drafting | 0 / 394 |
| 8 | reminders | message | 0.817 [0.789, 0.842] | 0.690 [0.642, 0.736] | 0.680 (drafting) | scheduling->approvals | 1 / 386 |
| 8 | reminders | with-opening | 0.943 [0.924, 0.962] | 0.900 [0.865, 0.932] | 0.907 (tickets) | scheduling->drafting | 0 / 454 |
| 8 | reminders | episode | 0.938 [0.917, 0.958] | 0.890 [0.855, 0.926] | 0.907 (expenses) | scheduling->drafting | 0 / 452 |
| 9 | summaries | message | 0.822 [0.793, 0.850] | 0.697 [0.642, 0.749] | 0.680 (drafting) | scheduling->summaries | 2 / 445 |
| 9 | summaries | with-opening | 0.930 [0.911, 0.950] | 0.876 [0.841, 0.910] | 0.893 (tickets) | scheduling->drafting | 0 / 514 |
| 9 | summaries | episode | 0.926 [0.905, 0.945] | 0.867 [0.832, 0.902] | 0.893 (expenses) | scheduling->drafting | 0 / 511 |
| 10 | travel | message | 0.831 [0.805, 0.858] | 0.711 [0.661, 0.758] | 0.680 (drafting) | scheduling->summaries | 0 / 508 |
| 10 | travel | with-opening | 0.920 [0.902, 0.939] | 0.857 [0.823, 0.893] | 0.879 (tickets) | scheduling->drafting | 0 / 575 |
| 10 | travel | episode | 0.913 [0.893, 0.933] | 0.844 [0.810, 0.880] | 0.870 (travel) | scheduling->drafting | 0 / 572 |
| 11 | inventory | message | 0.824 [0.798, 0.848] | 0.705 [0.660, 0.748] | 0.680 (drafting) | scheduling->summaries | 11 / 570 |
| 11 | inventory | with-opening | 0.908 [0.889, 0.928] | 0.833 [0.797, 0.870] | 0.867 (tickets) | scheduling->drafting | 0 / 631 |
| 11 | inventory | episode | 0.901 [0.881, 0.921] | 0.821 [0.785, 0.858] | 0.857 (travel) | scheduling->drafting | 0 / 626 |
| 12 | timesheets | message | 0.831 [0.805, 0.854] | 0.719 [0.676, 0.758] | 0.680 (drafting) | scheduling->summaries | 2 / 635 |
| 12 | timesheets | with-opening | 0.897 [0.877, 0.916] | 0.811 [0.773, 0.845] | 0.855 (tickets) | scheduling->drafting | 0 / 700 |
| 12 | timesheets | episode | 0.890 [0.869, 0.909] | 0.798 [0.760, 0.835] | 0.845 (travel) | scheduling->drafting | 0 / 695 |
| 13 | contacts | message | 0.830 [0.806, 0.851] | 0.716 [0.676, 0.755] | 0.680 (drafting) | scheduling->summaries | 8 / 708 |
| 13 | contacts | with-opening | 0.892 [0.874, 0.911] | 0.800 [0.765, 0.835] | 0.844 (tickets) | scheduling->drafting | 0 / 764 |
| 13 | contacts | episode | 0.880 [0.861, 0.899] | 0.776 [0.741, 0.812] | 0.833 (travel) | scheduling->drafting | 0 / 758 |
| 14 | search | message | 0.838 [0.817, 0.860] | 0.730 [0.692, 0.766] | 0.680 (drafting) | scheduling->summaries | 4 / 786 |
| 14 | search | with-opening | 0.883 [0.864, 0.901] | 0.781 [0.745, 0.815] | 0.833 (tickets) | scheduling->drafting | 0 / 845 |
| 14 | search | episode | 0.869 [0.850, 0.889] | 0.755 [0.719, 0.791] | 0.823 (travel) | scheduling->drafting | 0 / 833 |

## centroids-frozen, unseen

| routes | added | strategy | turn acc [95% CI] | episode acc [95% CI] | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | scheduling | message | 0.818 [0.740, 0.892] | 0.708 [0.562, 0.833] | 0.816 (drafting) | scheduling->drafting | – |
| 2 | scheduling | with-opening | 0.909 [0.850, 0.963] | 0.854 [0.750, 0.938] | 0.816 (drafting) | drafting->scheduling | – |
| 2 | scheduling | episode | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 (drafting) | – | – |
| 3 | invoices | message | 0.384 [0.260, 0.507] | 0.333 [0.222, 0.444] | 0.000 (drafting) | scheduling->invoices | 63 / 63 |
| 3 | invoices | with-opening | 0.384 [0.260, 0.507] | 0.333 [0.222, 0.444] | 0.000 (drafting) | scheduling->invoices | 70 / 70 |
| 3 | invoices | episode | 0.384 [0.260, 0.507] | 0.333 [0.222, 0.444] | 0.000 (drafting) | scheduling->invoices | 77 / 77 |
| 4 | tickets | message | 0.395 [0.307, 0.482] | 0.224 [0.143, 0.316] | 0.000 (drafting) | drafting->tickets | 15 / 48 |
| 4 | tickets | with-opening | 0.441 [0.344, 0.531] | 0.316 [0.224, 0.418] | 0.000 (drafting) | scheduling->tickets | 4 / 48 |
| 4 | tickets | episode | 0.328 [0.228, 0.433] | 0.286 [0.204, 0.378] | 0.000 (drafting) | tickets->invoices | 14 / 48 |
| 5 | rooms | message | 0.414 [0.338, 0.486] | 0.246 [0.175, 0.325] | 0.000 (drafting) | invoices->rooms | 20 / 70 |
| 5 | rooms | with-opening | 0.431 [0.343, 0.514] | 0.349 [0.270, 0.429] | 0.000 (drafting) | scheduling->rooms | 30 / 78 |
| 5 | rooms | episode | 0.319 [0.233, 0.414] | 0.278 [0.198, 0.365] | 0.000 (drafting) | invoices->rooms | 35 / 58 |
| 6 | expenses | message | 0.433 [0.368, 0.500] | 0.244 [0.179, 0.314] | 0.000 (drafting) | scheduling->rooms | 2 / 96 |
| 6 | expenses | with-opening | 0.416 [0.342, 0.487] | 0.288 [0.218, 0.359] | 0.000 (drafting) | scheduling->rooms | 11 / 100 |
| 6 | expenses | episode | 0.342 [0.267, 0.417] | 0.295 [0.224, 0.365] | 0.000 (drafting) | scheduling->rooms | 0 / 74 |
| 7 | approvals | message | 0.446 [0.391, 0.504] | 0.261 [0.197, 0.324] | 0.000 (drafting) | scheduling->rooms | 7 / 129 |
| 7 | approvals | with-opening | 0.432 [0.366, 0.503] | 0.303 [0.239, 0.372] | 0.000 (drafting) | scheduling->rooms | 1 / 124 |
| 7 | approvals | episode | 0.368 [0.295, 0.440] | 0.314 [0.250, 0.383] | 0.000 (drafting) | scheduling->rooms | 0 / 102 |
| 8 | reminders | message | 0.469 [0.416, 0.519] | 0.266 [0.212, 0.320] | 0.000 (drafting) | scheduling->reminders | 15 / 161 |
| 8 | reminders | with-opening | 0.459 [0.400, 0.516] | 0.306 [0.248, 0.369] | 0.000 (drafting) | scheduling->rooms | 2 / 156 |
| 8 | reminders | episode | 0.374 [0.315, 0.435] | 0.315 [0.261, 0.374] | 0.000 (drafting) | tickets->reminders | 32 / 133 |
| 9 | summaries | message | 0.485 [0.442, 0.531] | 0.291 [0.240, 0.345] | 0.000 (drafting) | invoices->expenses | 5 / 202 |
| 9 | summaries | with-opening | 0.485 [0.431, 0.537] | 0.337 [0.279, 0.391] | 0.000 (drafting) | scheduling->rooms | 5 / 198 |
| 9 | summaries | episode | 0.395 [0.335, 0.450] | 0.333 [0.275, 0.388] | 0.000 (drafting) | invoices->expenses | 1 / 161 |
| 10 | travel | message | 0.454 [0.410, 0.497] | 0.264 [0.216, 0.314] | 0.000 (drafting) | travel->reminders | 0 / 242 |
| 10 | travel | with-opening | 0.459 [0.409, 0.509] | 0.311 [0.257, 0.365] | 0.000 (drafting) | scheduling->rooms | 1 / 242 |
| 10 | travel | episode | 0.369 [0.317, 0.425] | 0.307 [0.257, 0.361] | 0.000 (drafting) | travel->reminders | 0 / 197 |
| 11 | inventory | message | 0.491 [0.452, 0.531] | 0.295 [0.250, 0.342] | 0.000 (drafting) | travel->reminders | 5 / 261 |
| 11 | inventory | with-opening | 0.505 [0.459, 0.550] | 0.357 [0.309, 0.411] | 0.000 (drafting) | scheduling->rooms | 2 / 264 |
| 11 | inventory | episode | 0.407 [0.358, 0.456] | 0.333 [0.289, 0.381] | 0.000 (drafting) | travel->reminders | 7 / 212 |
| 12 | timesheets | message | 0.460 [0.423, 0.497] | 0.246 [0.201, 0.296] | 0.000 (drafting) | travel->reminders | 25 / 321 |
| 12 | timesheets | with-opening | 0.499 [0.454, 0.538] | 0.341 [0.294, 0.386] | 0.000 (drafting) | tickets->timesheets | 12 / 330 |
| 12 | timesheets | episode | 0.375 [0.331, 0.421] | 0.296 [0.254, 0.341] | 0.000 (drafting) | travel->reminders | 30 / 266 |
| 13 | contacts | message | 0.428 [0.394, 0.462] | 0.206 [0.171, 0.242] | 0.000 (drafting) | travel->reminders | 17 / 338 |
| 13 | contacts | with-opening | 0.461 [0.421, 0.500] | 0.282 [0.242, 0.325] | 0.000 (drafting) | scheduling->rooms | 20 / 366 |
| 13 | contacts | episode | 0.340 [0.298, 0.386] | 0.265 [0.227, 0.308] | 0.000 (drafting) | travel->reminders | 7 / 275 |
| 14 | search | message | 0.418 [0.385, 0.448] | 0.201 [0.165, 0.237] | 0.000 (drafting) | travel->reminders | 1 / 351 |
| 14 | search | with-opening | 0.460 [0.424, 0.496] | 0.274 [0.235, 0.316] | 0.000 (drafting) | scheduling->rooms | 7 / 378 |
| 14 | search | episode | 0.327 [0.286, 0.369] | 0.248 [0.209, 0.286] | 0.000 (drafting) | travel->reminders | 1 / 279 |

## logistic-refit, test

| routes | added | strategy | turn acc [95% CI] | episode acc [95% CI] | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | scheduling | message | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 (drafting) | – | – |
| 2 | scheduling | with-opening | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 (drafting) | – | – |
| 2 | scheduling | episode | 0.980 [0.959, 0.995] | 0.971 [0.941, 0.993] | 0.962 (scheduling) | scheduling->drafting | – |
| 3 | invoices | message | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 (drafting) | – | 0 / 204 |
| 3 | invoices | with-opening | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 (drafting) | – | 0 / 204 |
| 3 | invoices | episode | 0.984 [0.967, 0.996] | 0.975 [0.950, 0.994] | 0.962 (scheduling) | scheduling->drafting | 0 / 200 |
| 4 | tickets | message | 0.966 [0.944, 0.987] | 0.952 [0.919, 0.978] | 0.854 (invoices) | invoices->tickets | 7 / 250 |
| 4 | tickets | with-opening | 0.993 [0.983, 1.000] | 0.989 [0.973, 1.000] | 0.978 (tickets) | tickets->invoices | 0 / 250 |
| 4 | tickets | episode | 0.980 [0.963, 0.993] | 0.968 [0.941, 0.989] | 0.962 (scheduling) | scheduling->drafting | 0 / 246 |
| 5 | rooms | message | 0.944 [0.916, 0.969] | 0.925 [0.888, 0.958] | 0.760 (invoices) | invoices->rooms | 7 / 288 |
| 5 | rooms | with-opening | 0.989 [0.977, 0.997] | 0.981 [0.963, 0.995] | 0.960 (invoices) | tickets->rooms | 0 / 296 |
| 5 | rooms | episode | 0.972 [0.955, 0.989] | 0.953 [0.925, 0.981] | 0.958 (tickets) | scheduling->drafting | 0 / 292 |
| 6 | expenses | message | 0.937 [0.909, 0.961] | 0.910 [0.873, 0.943] | 0.769 (invoices) | invoices->rooms | 2 / 334 |
| 6 | expenses | with-opening | 0.983 [0.970, 0.993] | 0.971 [0.951, 0.988] | 0.940 (expenses) | rooms->tickets | 0 / 350 |
| 6 | expenses | episode | 0.961 [0.940, 0.978] | 0.934 [0.902, 0.963] | 0.940 (expenses) | scheduling->drafting | 0 / 344 |
| 7 | approvals | message | 0.935 [0.908, 0.957] | 0.902 [0.859, 0.938] | 0.759 (invoices) | invoices->approvals | 3 / 384 |
| 7 | approvals | with-opening | 0.981 [0.970, 0.992] | 0.967 [0.949, 0.986] | 0.962 (expenses) | tickets->rooms | 1 / 403 |
| 7 | approvals | episode | 0.950 [0.932, 0.969] | 0.913 [0.884, 0.946] | 0.923 (expenses) | scheduling->drafting | 0 / 394 |
| 8 | reminders | message | 0.919 [0.894, 0.943] | 0.877 [0.842, 0.910] | 0.768 (invoices) | invoices->approvals | 5 / 445 |
| 8 | reminders | with-opening | 0.971 [0.957, 0.983] | 0.948 [0.926, 0.971] | 0.931 (rooms) | rooms->reminders | 1 / 467 |
| 8 | reminders | episode | 0.938 [0.917, 0.958] | 0.890 [0.855, 0.926] | 0.907 (expenses) | scheduling->drafting | 0 / 452 |
| 9 | summaries | message | 0.914 [0.891, 0.937] | 0.867 [0.829, 0.902] | 0.776 (invoices) | invoices->approvals | 0 / 501 |
| 9 | summaries | with-opening | 0.969 [0.955, 0.982] | 0.945 [0.919, 0.968] | 0.933 (rooms) | rooms->reminders | 0 / 529 |
| 9 | summaries | episode | 0.926 [0.905, 0.945] | 0.867 [0.832, 0.902] | 0.893 (expenses) | scheduling->drafting | 0 / 511 |
| 10 | travel | message | 0.914 [0.892, 0.934] | 0.865 [0.828, 0.893] | 0.783 (invoices) | invoices->approvals | 0 / 565 |
| 10 | travel | with-opening | 0.965 [0.951, 0.978] | 0.938 [0.914, 0.961] | 0.919 (rooms) | travel->rooms | 1 / 599 |
| 10 | travel | episode | 0.913 [0.893, 0.933] | 0.844 [0.810, 0.880] | 0.870 (travel) | scheduling->drafting | 0 / 572 |
| 11 | inventory | message | 0.901 [0.879, 0.922] | 0.844 [0.807, 0.877] | 0.790 (invoices) | inventory->reminders | 6 / 627 |
| 11 | inventory | with-opening | 0.960 [0.946, 0.972] | 0.927 [0.901, 0.948] | 0.922 (rooms) | travel->rooms | 1 / 662 |
| 11 | inventory | episode | 0.901 [0.881, 0.921] | 0.821 [0.785, 0.858] | 0.857 (travel) | scheduling->drafting | 0 / 626 |
| 12 | timesheets | message | 0.904 [0.883, 0.924] | 0.848 [0.815, 0.880] | 0.797 (invoices) | inventory->reminders | 0 / 695 |
| 12 | timesheets | with-opening | 0.959 [0.943, 0.972] | 0.925 [0.895, 0.948] | 0.909 (rooms) | travel->rooms | 2 / 740 |
| 12 | timesheets | episode | 0.890 [0.869, 0.909] | 0.798 [0.760, 0.835] | 0.845 (travel) | scheduling->drafting | 0 / 695 |
| 13 | contacts | message | 0.901 [0.882, 0.921] | 0.841 [0.810, 0.871] | 0.803 (invoices) | inventory->reminders | 0 / 770 |
| 13 | contacts | with-opening | 0.954 [0.940, 0.966] | 0.914 [0.888, 0.937] | 0.912 (rooms) | travel->rooms | 2 / 817 |
| 13 | contacts | episode | 0.880 [0.861, 0.899] | 0.776 [0.741, 0.812] | 0.833 (travel) | scheduling->drafting | 0 / 758 |
| 14 | search | message | 0.903 [0.881, 0.921] | 0.845 [0.811, 0.874] | 0.809 (invoices) | inventory->search | 5 / 853 |
| 14 | search | with-opening | 0.959 [0.946, 0.970] | 0.923 [0.899, 0.944] | 0.929 (rooms) | travel->rooms | 3 / 903 |
| 14 | search | episode | 0.869 [0.850, 0.889] | 0.755 [0.719, 0.791] | 0.823 (travel) | scheduling->drafting | 0 / 833 |

## logistic-refit, unseen

| routes | added | strategy | turn acc [95% CI] | episode acc [95% CI] | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | scheduling | message | 0.753 [0.633, 0.850] | 0.688 [0.562, 0.812] | 0.500 (drafting) | drafting->scheduling | – |
| 2 | scheduling | with-opening | 0.636 [0.513, 0.757] | 0.542 [0.396, 0.688] | 0.263 (drafting) | drafting->scheduling | – |
| 2 | scheduling | episode | 0.753 [0.615, 0.875] | 0.750 [0.625, 0.875] | 0.500 (drafting) | drafting->scheduling | – |
| 3 | invoices | message | 0.416 [0.293, 0.537] | 0.347 [0.236, 0.458] | 0.000 (drafting) | scheduling->invoices | 54 / 58 |
| 3 | invoices | with-opening | 0.384 [0.260, 0.507] | 0.333 [0.222, 0.444] | 0.000 (drafting) | scheduling->invoices | 49 / 49 |
| 3 | invoices | episode | 0.392 [0.270, 0.515] | 0.347 [0.236, 0.458] | 0.000 (drafting) | scheduling->invoices | 57 / 58 |
| 4 | tickets | message | 0.390 [0.304, 0.470] | 0.224 [0.153, 0.306] | 0.000 (drafting) | drafting->invoices | 22 / 52 |
| 4 | tickets | with-opening | 0.418 [0.320, 0.524] | 0.337 [0.245, 0.439] | 0.000 (drafting) | tickets->invoices | 0 / 48 |
| 4 | tickets | episode | 0.328 [0.233, 0.435] | 0.296 [0.214, 0.388] | 0.000 (drafting) | tickets->invoices | 21 / 49 |
| 5 | rooms | message | 0.466 [0.390, 0.535] | 0.294 [0.222, 0.373] | 0.000 (drafting) | scheduling->rooms | 14 / 69 |
| 5 | rooms | with-opening | 0.431 [0.354, 0.506] | 0.294 [0.222, 0.373] | 0.000 (drafting) | scheduling->rooms | 21 / 74 |
| 5 | rooms | episode | 0.384 [0.294, 0.473] | 0.341 [0.262, 0.421] | 0.000 (drafting) | scheduling->rooms | 20 / 58 |
| 6 | expenses | message | 0.497 [0.435, 0.560] | 0.295 [0.231, 0.359] | 0.079 (drafting) | expenses->rooms | 2 / 108 |
| 6 | expenses | with-opening | 0.480 [0.408, 0.546] | 0.327 [0.256, 0.398] | 0.000 (drafting) | scheduling->rooms | 6 / 100 |
| 6 | expenses | episode | 0.423 [0.343, 0.500] | 0.372 [0.301, 0.442] | 0.000 (scheduling) | expenses->rooms | 1 / 89 |
| 7 | approvals | message | 0.488 [0.431, 0.540] | 0.266 [0.202, 0.324] | 0.053 (drafting) | drafting->tickets | 10 / 148 |
| 7 | approvals | with-opening | 0.490 [0.427, 0.548] | 0.314 [0.245, 0.378] | 0.000 (drafting) | scheduling->rooms | 9 / 143 |
| 7 | approvals | episode | 0.416 [0.346, 0.476] | 0.362 [0.293, 0.426] | 0.000 (scheduling) | invoices->expenses | 3 / 126 |
| 8 | reminders | message | 0.464 [0.413, 0.510] | 0.216 [0.162, 0.270] | 0.000 (drafting) | approvals->reminders | 19 / 176 |
| 8 | reminders | with-opening | 0.478 [0.419, 0.532] | 0.293 [0.234, 0.351] | 0.000 (drafting) | reminders->rooms | 13 / 177 |
| 8 | reminders | episode | 0.390 [0.325, 0.452] | 0.320 [0.261, 0.383] | 0.000 (drafting) | approvals->reminders | 24 / 150 |
| 9 | summaries | message | 0.489 [0.440, 0.535] | 0.279 [0.229, 0.333] | 0.026 (scheduling) | approvals->reminders | 5 / 200 |
| 9 | summaries | with-opening | 0.513 [0.464, 0.564] | 0.322 [0.264, 0.380] | 0.000 (drafting) | reminders->rooms | 5 / 206 |
| 9 | summaries | episode | 0.411 [0.351, 0.467] | 0.353 [0.295, 0.407] | 0.000 (scheduling) | reminders->rooms | 7 / 168 |
| 10 | travel | message | 0.511 [0.466, 0.553] | 0.307 [0.257, 0.355] | 0.000 (scheduling) | reminders->rooms | 9 / 244 |
| 10 | travel | with-opening | 0.550 [0.501, 0.591] | 0.368 [0.314, 0.422] | 0.026 (scheduling) | scheduling->rooms | 3 / 256 |
| 10 | travel | episode | 0.452 [0.397, 0.509] | 0.385 [0.331, 0.443] | 0.000 (scheduling) | reminders->rooms | 11 / 205 |
| 11 | inventory | message | 0.546 [0.506, 0.585] | 0.339 [0.289, 0.387] | 0.000 (scheduling) | summaries->expenses | 11 / 294 |
| 11 | inventory | with-opening | 0.534 [0.494, 0.576] | 0.345 [0.301, 0.396] | 0.026 (scheduling) | scheduling->rooms | 18 / 316 |
| 11 | inventory | episode | 0.506 [0.457, 0.556] | 0.420 [0.372, 0.470] | 0.000 (scheduling) | summaries->expenses | 15 / 260 |
| 12 | timesheets | message | 0.568 [0.527, 0.606] | 0.378 [0.328, 0.423] | 0.000 (scheduling) | tickets->timesheets | 26 / 357 |
| 12 | timesheets | with-opening | 0.563 [0.520, 0.598] | 0.373 [0.323, 0.421] | 0.026 (scheduling) | tickets->timesheets | 18 / 349 |
| 12 | timesheets | episode | 0.505 [0.456, 0.553] | 0.421 [0.370, 0.474] | 0.000 (scheduling) | tickets->timesheets | 39 / 331 |
| 13 | contacts | message | 0.570 [0.533, 0.603] | 0.363 [0.318, 0.410] | 0.103 (scheduling) | tickets->timesheets | 11 / 417 |
| 13 | contacts | with-opening | 0.559 [0.522, 0.597] | 0.367 [0.322, 0.415] | 0.000 (drafting) | tickets->timesheets | 19 / 413 |
| 13 | contacts | episode | 0.500 [0.451, 0.543] | 0.410 [0.363, 0.455] | 0.000 (scheduling) | tickets->timesheets | 9 / 371 |
| 14 | search | message | 0.517 [0.481, 0.549] | 0.303 [0.263, 0.344] | 0.000 (scheduling) | tickets->timesheets | 34 / 467 |
| 14 | search | with-opening | 0.529 [0.494, 0.564] | 0.333 [0.295, 0.376] | 0.000 (drafting) | tickets->timesheets | 23 / 458 |
| 14 | search | episode | 0.442 [0.397, 0.483] | 0.348 [0.303, 0.391] | 0.000 (scheduling) | tickets->timesheets | 31 / 410 |
