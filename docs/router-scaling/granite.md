# Router scaling: lm encoder (layer-1)

836 fit cases; evaluation cases {'test': 556, 'unseen': 468}, turns {'test': 1040, 'unseen': 909}; 2910 texts, features in 0 s. Intervals: 1,000 case-level bootstrap draws.

## centroids-refit, test

| routes | added | strategy | turn acc [95% CI] | episode acc [95% CI] | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | scheduling | message | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 (drafting) | – | – |
| 2 | scheduling | with-opening | 0.980 [0.959, 0.995] | 0.971 [0.941, 0.993] | 0.962 (scheduling) | scheduling->drafting | – |
| 2 | scheduling | episode | 0.980 [0.959, 0.995] | 0.971 [0.941, 0.993] | 0.962 (scheduling) | scheduling->drafting | – |
| 3 | invoices | message | 0.936 [0.906, 0.964] | 0.900 [0.850, 0.944] | 0.840 (drafting) | drafting->invoices | 16 / 204 |
| 3 | invoices | with-opening | 0.984 [0.967, 0.996] | 0.975 [0.950, 0.994] | 0.962 (scheduling) | scheduling->drafting | 0 / 200 |
| 3 | invoices | episode | 0.984 [0.967, 0.996] | 0.975 [0.950, 0.994] | 0.962 (scheduling) | scheduling->drafting | 0 / 200 |
| 4 | tickets | message | 0.966 [0.945, 0.986] | 0.946 [0.914, 0.978] | 0.833 (invoices) | invoices->tickets | 8 / 234 |
| 4 | tickets | with-opening | 0.983 [0.967, 0.997] | 0.973 [0.946, 0.995] | 0.962 (scheduling) | scheduling->drafting | 0 / 246 |
| 4 | tickets | episode | 0.980 [0.963, 0.993] | 0.968 [0.941, 0.989] | 0.962 (scheduling) | scheduling->drafting | 0 / 246 |
| 5 | rooms | message | 0.924 [0.893, 0.953] | 0.897 [0.855, 0.935] | 0.720 (invoices) | invoices->rooms | 17 / 288 |
| 5 | rooms | with-opening | 0.972 [0.955, 0.989] | 0.953 [0.925, 0.981] | 0.940 (invoices) | scheduling->drafting | 2 / 293 |
| 5 | rooms | episode | 0.972 [0.955, 0.989] | 0.953 [0.925, 0.981] | 0.958 (tickets) | scheduling->drafting | 0 / 292 |
| 6 | expenses | message | 0.917 [0.885, 0.945] | 0.889 [0.848, 0.926] | 0.712 (invoices) | invoices->rooms | 1 / 327 |
| 6 | expenses | with-opening | 0.963 [0.946, 0.980] | 0.939 [0.906, 0.967] | 0.923 (invoices) | scheduling->drafting | 0 / 344 |
| 6 | expenses | episode | 0.961 [0.940, 0.978] | 0.934 [0.902, 0.963] | 0.940 (expenses) | scheduling->drafting | 0 / 344 |
| 7 | approvals | message | 0.891 [0.862, 0.918] | 0.841 [0.797, 0.884] | 0.685 (invoices) | tickets->rooms | 5 / 376 |
| 7 | approvals | with-opening | 0.952 [0.934, 0.970] | 0.917 [0.884, 0.949] | 0.907 (invoices) | scheduling->drafting | 0 / 395 |
| 7 | approvals | episode | 0.950 [0.932, 0.969] | 0.913 [0.884, 0.946] | 0.923 (expenses) | scheduling->drafting | 0 / 394 |
| 8 | reminders | message | 0.892 [0.863, 0.918] | 0.832 [0.787, 0.871] | 0.750 (invoices) | approvals->rooms | 4 / 424 |
| 8 | reminders | with-opening | 0.938 [0.917, 0.956] | 0.890 [0.855, 0.923] | 0.889 (tickets) | scheduling->drafting | 2 / 453 |
| 8 | reminders | episode | 0.938 [0.917, 0.958] | 0.890 [0.855, 0.926] | 0.907 (expenses) | scheduling->drafting | 0 / 452 |
| 9 | summaries | message | 0.888 [0.863, 0.912] | 0.824 [0.783, 0.861] | 0.750 (tickets) | tickets->rooms | 1 / 486 |
| 9 | summaries | with-opening | 0.929 [0.909, 0.948] | 0.873 [0.838, 0.908] | 0.875 (tickets) | scheduling->drafting | 0 / 511 |
| 9 | summaries | episode | 0.926 [0.905, 0.945] | 0.867 [0.832, 0.902] | 0.893 (expenses) | scheduling->drafting | 0 / 511 |
| 10 | travel | message | 0.892 [0.870, 0.914] | 0.828 [0.792, 0.865] | 0.767 (invoices) | approvals->rooms | 0 / 549 |
| 10 | travel | with-opening | 0.927 [0.909, 0.945] | 0.870 [0.836, 0.904] | 0.879 (expenses) | scheduling->drafting | 2 / 574 |
| 10 | travel | episode | 0.913 [0.893, 0.933] | 0.844 [0.810, 0.880] | 0.870 (travel) | scheduling->drafting | 0 / 572 |
| 11 | inventory | message | 0.883 [0.859, 0.906] | 0.818 [0.781, 0.854] | 0.742 (approvals) | approvals->rooms | 4 / 612 |
| 11 | inventory | with-opening | 0.914 [0.896, 0.933] | 0.844 [0.811, 0.877] | 0.867 (expenses) | scheduling->drafting | 1 / 636 |
| 11 | inventory | episode | 0.901 [0.881, 0.921] | 0.821 [0.785, 0.858] | 0.857 (travel) | scheduling->drafting | 0 / 626 |
| 12 | timesheets | message | 0.886 [0.864, 0.908] | 0.824 [0.790, 0.856] | 0.735 (approvals) | approvals->rooms | 3 / 681 |
| 12 | timesheets | with-opening | 0.908 [0.890, 0.927] | 0.833 [0.798, 0.867] | 0.857 (timesheets) | scheduling->drafting | 1 / 705 |
| 12 | timesheets | episode | 0.890 [0.869, 0.909] | 0.798 [0.760, 0.835] | 0.845 (travel) | scheduling->drafting | 0 / 695 |
| 13 | contacts | message | 0.881 [0.859, 0.899] | 0.812 [0.776, 0.841] | 0.743 (approvals) | tickets->contacts | 5 / 755 |
| 13 | contacts | with-opening | 0.902 [0.884, 0.919] | 0.818 [0.784, 0.851] | 0.846 (timesheets) | scheduling->drafting | 0 / 774 |
| 13 | contacts | episode | 0.880 [0.861, 0.899] | 0.776 [0.741, 0.812] | 0.833 (travel) | scheduling->drafting | 0 / 758 |
| 14 | search | message | 0.878 [0.855, 0.896] | 0.811 [0.777, 0.842] | 0.667 (approvals) | approvals->search | 12 / 834 |
| 14 | search | with-opening | 0.895 [0.877, 0.913] | 0.804 [0.770, 0.836] | 0.836 (timesheets) | scheduling->drafting | 0 / 854 |
| 14 | search | episode | 0.869 [0.850, 0.889] | 0.755 [0.719, 0.791] | 0.823 (travel) | scheduling->drafting | 0 / 833 |

## centroids-refit, unseen

| routes | added | strategy | turn acc [95% CI] | episode acc [95% CI] | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | scheduling | message | 0.844 [0.774, 0.914] | 0.750 [0.625, 0.875] | 0.684 (drafting) | drafting->scheduling | – |
| 2 | scheduling | with-opening | 0.740 [0.616, 0.857] | 0.688 [0.562, 0.812] | 0.474 (drafting) | drafting->scheduling | – |
| 2 | scheduling | episode | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 (drafting) | – | – |
| 3 | invoices | message | 0.696 [0.613, 0.775] | 0.542 [0.431, 0.653] | 0.410 (scheduling) | scheduling->invoices | 26 / 65 |
| 3 | invoices | with-opening | 0.664 [0.579, 0.744] | 0.514 [0.403, 0.625] | 0.410 (scheduling) | scheduling->invoices | 28 / 57 |
| 3 | invoices | episode | 0.888 [0.810, 0.954] | 0.875 [0.792, 0.944] | 0.667 (scheduling) | scheduling->invoices | 14 / 77 |
| 4 | tickets | message | 0.746 [0.685, 0.803] | 0.592 [0.500, 0.684] | 0.410 (scheduling) | scheduling->tickets | 8 / 87 |
| 4 | tickets | with-opening | 0.751 [0.689, 0.812] | 0.602 [0.500, 0.694] | 0.410 (scheduling) | scheduling->tickets | 5 / 83 |
| 4 | tickets | episode | 0.915 [0.860, 0.964] | 0.898 [0.837, 0.949] | 0.667 (scheduling) | scheduling->tickets | 0 / 111 |
| 5 | rooms | message | 0.677 [0.617, 0.731] | 0.516 [0.429, 0.595] | 0.410 (scheduling) | scheduling->rooms | 29 / 132 |
| 5 | rooms | with-opening | 0.651 [0.574, 0.723] | 0.532 [0.444, 0.619] | 0.410 (scheduling) | scheduling->rooms | 35 / 133 |
| 5 | rooms | episode | 0.836 [0.768, 0.899] | 0.810 [0.738, 0.873] | 0.596 (invoices) | invoices->rooms | 19 / 162 |
| 6 | expenses | message | 0.685 [0.633, 0.731] | 0.494 [0.417, 0.571] | 0.463 (tickets) | tickets->rooms | 11 / 157 |
| 6 | expenses | with-opening | 0.695 [0.629, 0.756] | 0.571 [0.494, 0.641] | 0.410 (scheduling) | scheduling->rooms | 3 / 151 |
| 6 | expenses | episode | 0.815 [0.752, 0.870] | 0.769 [0.699, 0.827] | 0.593 (invoices) | invoices->expenses | 19 / 194 |
| 7 | approvals | message | 0.612 [0.564, 0.661] | 0.426 [0.356, 0.495] | 0.345 (approvals) | approvals->rooms | 9 / 204 |
| 7 | approvals | with-opening | 0.690 [0.634, 0.743] | 0.553 [0.484, 0.628] | 0.410 (scheduling) | tickets->rooms | 10 / 207 |
| 7 | approvals | episode | 0.742 [0.677, 0.802] | 0.702 [0.633, 0.766] | 0.446 (tickets) | approvals->rooms | 8 / 243 |
| 8 | reminders | message | 0.622 [0.574, 0.671] | 0.446 [0.383, 0.509] | 0.224 (tickets) | approvals->rooms | 16 / 221 |
| 8 | reminders | with-opening | 0.710 [0.657, 0.761] | 0.577 [0.514, 0.644] | 0.431 (tickets) | tickets->reminders | 13 / 249 |
| 8 | reminders | episode | 0.701 [0.633, 0.763] | 0.667 [0.604, 0.730] | 0.138 (tickets) | approvals->rooms | 26 / 268 |
| 9 | summaries | message | 0.639 [0.593, 0.682] | 0.473 [0.419, 0.535] | 0.169 (approvals) | approvals->rooms | 6 / 268 |
| 9 | summaries | with-opening | 0.733 [0.687, 0.777] | 0.601 [0.543, 0.655] | 0.433 (tickets) | tickets->reminders | 3 / 306 |
| 9 | summaries | episode | 0.697 [0.640, 0.752] | 0.647 [0.589, 0.705] | 0.133 (tickets) | approvals->rooms | 9 / 302 |
| 10 | travel | message | 0.637 [0.594, 0.677] | 0.459 [0.399, 0.517] | 0.180 (approvals) | approvals->rooms | 0 / 319 |
| 10 | travel | with-opening | 0.722 [0.679, 0.765] | 0.591 [0.541, 0.649] | 0.419 (tickets) | tickets->reminders | 3 / 366 |
| 10 | travel | episode | 0.677 [0.627, 0.727] | 0.625 [0.571, 0.679] | 0.129 (tickets) | approvals->rooms | 0 / 348 |
| 11 | inventory | message | 0.650 [0.612, 0.685] | 0.479 [0.429, 0.533] | 0.141 (tickets) | approvals->rooms | 6 / 366 |
| 11 | inventory | with-opening | 0.737 [0.699, 0.773] | 0.604 [0.554, 0.655] | 0.422 (tickets) | tickets->reminders | 0 / 415 |
| 11 | inventory | episode | 0.685 [0.641, 0.733] | 0.616 [0.571, 0.664] | 0.031 (tickets) | tickets->reminders | 6 / 389 |
| 12 | timesheets | message | 0.670 [0.632, 0.704] | 0.505 [0.452, 0.556] | 0.121 (tickets) | approvals->rooms | 3 / 425 |
| 12 | timesheets | with-opening | 0.738 [0.701, 0.773] | 0.601 [0.553, 0.651] | 0.424 (tickets) | tickets->timesheets | 9 / 482 |
| 12 | timesheets | episode | 0.688 [0.640, 0.731] | 0.611 [0.558, 0.659] | 0.015 (tickets) | tickets->timesheets | 2 / 448 |
| 13 | contacts | message | 0.688 [0.652, 0.720] | 0.528 [0.481, 0.576] | 0.088 (tickets) | approvals->rooms | 5 / 492 |
| 13 | contacts | with-opening | 0.746 [0.711, 0.780] | 0.611 [0.564, 0.659] | 0.412 (tickets) | tickets->timesheets | 2 / 542 |
| 13 | contacts | episode | 0.698 [0.657, 0.736] | 0.609 [0.559, 0.654] | 0.059 (tickets) | tickets->timesheets | 0 / 505 |
| 14 | search | message | 0.685 [0.656, 0.719] | 0.513 [0.472, 0.558] | 0.086 (tickets) | tickets->timesheets | 7 / 564 |
| 14 | search | with-opening | 0.734 [0.704, 0.765] | 0.590 [0.547, 0.635] | 0.414 (tickets) | approvals->search | 12 / 612 |
| 14 | search | episode | 0.677 [0.638, 0.715] | 0.583 [0.538, 0.626] | 0.057 (tickets) | tickets->timesheets | 0 / 572 |

## centroids-frozen, test

| routes | added | strategy | turn acc [95% CI] | episode acc [95% CI] | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | scheduling | message | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 (drafting) | – | – |
| 2 | scheduling | with-opening | 0.980 [0.959, 0.995] | 0.971 [0.941, 0.993] | 0.962 (scheduling) | scheduling->drafting | – |
| 2 | scheduling | episode | 0.980 [0.959, 0.995] | 0.971 [0.941, 0.993] | 0.962 (scheduling) | scheduling->drafting | – |
| 3 | invoices | message | 0.808 [0.765, 0.848] | 0.700 [0.625, 0.769] | 0.760 (drafting) | scheduling->invoices | 48 / 204 |
| 3 | invoices | with-opening | 0.952 [0.925, 0.979] | 0.925 [0.881, 0.969] | 0.920 (drafting) | drafting->invoices | 8 / 200 |
| 3 | invoices | episode | 0.984 [0.967, 0.996] | 0.975 [0.950, 0.994] | 0.962 (scheduling) | scheduling->drafting | 0 / 200 |
| 4 | tickets | message | 0.802 [0.766, 0.841] | 0.683 [0.618, 0.753] | 0.760 (drafting) | scheduling->tickets | 7 / 202 |
| 4 | tickets | with-opening | 0.953 [0.929, 0.974] | 0.925 [0.887, 0.957] | 0.920 (drafting) | drafting->invoices | 1 / 238 |
| 4 | tickets | episode | 0.980 [0.963, 0.993] | 0.968 [0.941, 0.989] | 0.962 (scheduling) | scheduling->drafting | 0 / 246 |
| 5 | rooms | message | 0.802 [0.767, 0.835] | 0.692 [0.631, 0.752] | 0.740 (invoices) | scheduling->rooms | 9 / 239 |
| 5 | rooms | with-opening | 0.898 [0.867, 0.928] | 0.832 [0.776, 0.883] | 0.769 (scheduling) | scheduling->rooms | 19 / 284 |
| 5 | rooms | episode | 0.972 [0.955, 0.989] | 0.953 [0.925, 0.981] | 0.958 (tickets) | scheduling->drafting | 0 / 292 |
| 6 | expenses | message | 0.810 [0.776, 0.843] | 0.701 [0.643, 0.754] | 0.750 (invoices) | scheduling->rooms | 2 / 284 |
| 6 | expenses | with-opening | 0.898 [0.871, 0.924] | 0.828 [0.783, 0.873] | 0.769 (scheduling) | scheduling->rooms | 0 / 318 |
| 6 | expenses | episode | 0.961 [0.940, 0.978] | 0.934 [0.902, 0.963] | 0.940 (expenses) | scheduling->drafting | 0 / 344 |
| 7 | approvals | message | 0.817 [0.785, 0.849] | 0.707 [0.649, 0.761] | 0.759 (invoices) | drafting->approvals | 5 / 332 |
| 7 | approvals | with-opening | 0.895 [0.870, 0.920] | 0.819 [0.775, 0.862] | 0.769 (scheduling) | scheduling->rooms | 0 / 368 |
| 7 | approvals | episode | 0.950 [0.932, 0.969] | 0.913 [0.884, 0.946] | 0.923 (expenses) | scheduling->drafting | 0 / 394 |
| 8 | reminders | message | 0.815 [0.786, 0.843] | 0.700 [0.652, 0.748] | 0.741 (tickets) | drafting->approvals | 7 / 389 |
| 8 | reminders | with-opening | 0.892 [0.868, 0.915] | 0.810 [0.768, 0.852] | 0.769 (scheduling) | scheduling->rooms | 1 / 426 |
| 8 | reminders | episode | 0.938 [0.917, 0.958] | 0.890 [0.855, 0.926] | 0.907 (expenses) | scheduling->drafting | 0 / 452 |
| 9 | summaries | message | 0.820 [0.790, 0.848] | 0.705 [0.656, 0.754] | 0.732 (tickets) | drafting->approvals | 1 / 444 |
| 9 | summaries | with-opening | 0.887 [0.864, 0.910] | 0.798 [0.757, 0.838] | 0.769 (scheduling) | scheduling->rooms | 0 / 486 |
| 9 | summaries | episode | 0.926 [0.905, 0.945] | 0.867 [0.832, 0.902] | 0.893 (expenses) | scheduling->drafting | 0 / 511 |
| 10 | travel | message | 0.829 [0.804, 0.854] | 0.719 [0.672, 0.760] | 0.741 (tickets) | drafting->approvals | 0 / 507 |
| 10 | travel | with-opening | 0.885 [0.862, 0.907] | 0.794 [0.753, 0.833] | 0.769 (scheduling) | scheduling->rooms | 0 / 548 |
| 10 | travel | episode | 0.913 [0.893, 0.933] | 0.844 [0.810, 0.880] | 0.870 (travel) | scheduling->drafting | 0 / 572 |
| 11 | inventory | message | 0.829 [0.803, 0.852] | 0.715 [0.672, 0.755] | 0.750 (tickets) | drafting->approvals | 1 / 569 |
| 11 | inventory | with-opening | 0.878 [0.856, 0.900] | 0.778 [0.738, 0.818] | 0.769 (scheduling) | scheduling->rooms | 0 / 607 |
| 11 | inventory | episode | 0.901 [0.881, 0.921] | 0.821 [0.785, 0.858] | 0.857 (travel) | scheduling->drafting | 0 / 626 |
| 12 | timesheets | message | 0.835 [0.812, 0.858] | 0.723 [0.685, 0.762] | 0.758 (tickets) | drafting->approvals | 5 / 639 |
| 12 | timesheets | with-opening | 0.871 [0.850, 0.891] | 0.764 [0.723, 0.800] | 0.769 (scheduling) | scheduling->rooms | 0 / 677 |
| 12 | timesheets | episode | 0.890 [0.869, 0.909] | 0.798 [0.760, 0.835] | 0.845 (travel) | scheduling->drafting | 0 / 695 |
| 13 | contacts | message | 0.837 [0.815, 0.860] | 0.727 [0.688, 0.765] | 0.760 (drafting) | scheduling->contacts | 0 / 711 |
| 13 | contacts | with-opening | 0.867 [0.848, 0.887] | 0.753 [0.720, 0.790] | 0.779 (scheduling) | scheduling->rooms | 0 / 742 |
| 13 | contacts | episode | 0.880 [0.861, 0.899] | 0.776 [0.741, 0.812] | 0.833 (travel) | scheduling->drafting | 0 / 758 |
| 14 | search | message | 0.830 [0.807, 0.851] | 0.721 [0.683, 0.755] | 0.732 (summaries) | summaries->search | 23 / 793 |
| 14 | search | with-opening | 0.861 [0.841, 0.879] | 0.739 [0.705, 0.773] | 0.779 (scheduling) | scheduling->rooms | 1 / 821 |
| 14 | search | episode | 0.869 [0.850, 0.889] | 0.755 [0.719, 0.791] | 0.823 (travel) | scheduling->drafting | 0 / 833 |

## centroids-frozen, unseen

| routes | added | strategy | turn acc [95% CI] | episode acc [95% CI] | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | scheduling | message | 0.844 [0.774, 0.914] | 0.750 [0.625, 0.875] | 0.684 (drafting) | drafting->scheduling | – |
| 2 | scheduling | with-opening | 0.740 [0.616, 0.857] | 0.688 [0.562, 0.812] | 0.474 (drafting) | drafting->scheduling | – |
| 2 | scheduling | episode | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 (drafting) | – | – |
| 3 | invoices | message | 0.512 [0.402, 0.621] | 0.417 [0.292, 0.528] | 0.000 (drafting) | drafting->invoices | 49 / 65 |
| 3 | invoices | with-opening | 0.512 [0.402, 0.621] | 0.417 [0.292, 0.528] | 0.000 (drafting) | drafting->invoices | 41 / 57 |
| 3 | invoices | episode | 0.592 [0.479, 0.709] | 0.556 [0.444, 0.667] | 0.000 (drafting) | drafting->invoices | 51 / 77 |
| 4 | tickets | message | 0.599 [0.509, 0.683] | 0.500 [0.408, 0.592] | 0.000 (drafting) | drafting->tickets | 9 / 64 |
| 4 | tickets | with-opening | 0.644 [0.553, 0.727] | 0.551 [0.449, 0.653] | 0.000 (drafting) | drafting->tickets | 1 / 64 |
| 4 | tickets | episode | 0.667 [0.571, 0.756] | 0.622 [0.531, 0.714] | 0.000 (drafting) | drafting->tickets | 6 / 74 |
| 5 | rooms | message | 0.586 [0.511, 0.661] | 0.476 [0.397, 0.556] | 0.000 (drafting) | scheduling->rooms | 21 / 106 |
| 5 | rooms | with-opening | 0.573 [0.491, 0.660] | 0.484 [0.405, 0.571] | 0.000 (drafting) | scheduling->rooms | 32 / 114 |
| 5 | rooms | episode | 0.634 [0.553, 0.713] | 0.563 [0.484, 0.643] | 0.000 (drafting) | scheduling->rooms | 22 / 118 |
| 6 | expenses | message | 0.604 [0.541, 0.672] | 0.474 [0.404, 0.551] | 0.000 (drafting) | scheduling->rooms | 10 / 136 |
| 6 | expenses | with-opening | 0.634 [0.564, 0.702] | 0.538 [0.462, 0.615] | 0.000 (drafting) | scheduling->rooms | 6 / 133 |
| 6 | expenses | episode | 0.638 [0.563, 0.706] | 0.558 [0.481, 0.628] | 0.000 (drafting) | scheduling->rooms | 17 / 147 |
| 7 | approvals | message | 0.612 [0.551, 0.672] | 0.463 [0.388, 0.537] | 0.000 (drafting) | scheduling->rooms | 13 / 180 |
| 7 | approvals | with-opening | 0.676 [0.609, 0.740] | 0.580 [0.505, 0.654] | 0.000 (drafting) | scheduling->rooms | 3 / 189 |
| 7 | approvals | episode | 0.676 [0.612, 0.742] | 0.590 [0.521, 0.665] | 0.000 (drafting) | scheduling->rooms | 0 / 190 |
| 8 | reminders | message | 0.654 [0.606, 0.707] | 0.509 [0.441, 0.577] | 0.000 (drafting) | scheduling->rooms | 3 / 221 |
| 8 | reminders | with-opening | 0.687 [0.635, 0.744] | 0.568 [0.509, 0.631] | 0.000 (drafting) | scheduling->rooms | 11 / 244 |
| 8 | reminders | episode | 0.703 [0.649, 0.759] | 0.608 [0.545, 0.671] | 0.000 (drafting) | scheduling->rooms | 0 / 244 |
| 9 | summaries | message | 0.675 [0.634, 0.724] | 0.535 [0.477, 0.593] | 0.000 (drafting) | scheduling->rooms | 2 / 282 |
| 9 | summaries | with-opening | 0.701 [0.654, 0.753] | 0.578 [0.519, 0.640] | 0.000 (drafting) | scheduling->rooms | 0 / 296 |
| 9 | summaries | episode | 0.709 [0.660, 0.762] | 0.609 [0.550, 0.667] | 0.000 (drafting) | scheduling->rooms | 0 / 303 |
| 10 | travel | message | 0.671 [0.628, 0.712] | 0.520 [0.463, 0.574] | 0.000 (drafting) | scheduling->rooms | 0 / 337 |
| 10 | travel | with-opening | 0.671 [0.623, 0.715] | 0.541 [0.483, 0.595] | 0.000 (drafting) | scheduling->rooms | 0 / 350 |
| 10 | travel | episode | 0.694 [0.646, 0.743] | 0.591 [0.537, 0.645] | 0.000 (drafting) | scheduling->rooms | 0 / 354 |
| 11 | inventory | message | 0.688 [0.651, 0.728] | 0.539 [0.488, 0.592] | 0.000 (drafting) | scheduling->rooms | 3 / 386 |
| 11 | inventory | with-opening | 0.688 [0.648, 0.730] | 0.554 [0.503, 0.607] | 0.000 (drafting) | scheduling->rooms | 2 / 386 |
| 11 | inventory | episode | 0.703 [0.660, 0.747] | 0.589 [0.539, 0.640] | 0.000 (drafting) | scheduling->rooms | 2 / 399 |
| 12 | timesheets | message | 0.689 [0.652, 0.727] | 0.553 [0.500, 0.601] | 0.000 (drafting) | scheduling->rooms | 18 / 450 |
| 12 | timesheets | with-opening | 0.665 [0.623, 0.702] | 0.519 [0.466, 0.569] | 0.000 (drafting) | scheduling->rooms | 26 / 450 |
| 12 | timesheets | episode | 0.678 [0.637, 0.722] | 0.561 [0.513, 0.611] | 0.000 (drafting) | scheduling->rooms | 23 / 460 |
| 13 | contacts | message | 0.710 [0.675, 0.743] | 0.576 [0.528, 0.623] | 0.000 (drafting) | scheduling->rooms | 1 / 506 |
| 13 | contacts | with-opening | 0.683 [0.646, 0.720] | 0.538 [0.493, 0.583] | 0.000 (drafting) | scheduling->rooms | 0 / 488 |
| 13 | contacts | episode | 0.687 [0.647, 0.724] | 0.559 [0.514, 0.607] | 0.000 (drafting) | scheduling->rooms | 0 / 498 |
| 14 | search | message | 0.713 [0.682, 0.744] | 0.575 [0.530, 0.618] | 0.000 (drafting) | scheduling->rooms | 16 / 582 |
| 14 | search | with-opening | 0.682 [0.645, 0.713] | 0.532 [0.487, 0.573] | 0.000 (drafting) | scheduling->rooms | 10 / 560 |
| 14 | search | episode | 0.682 [0.643, 0.720] | 0.549 [0.500, 0.592] | 0.000 (drafting) | scheduling->rooms | 0 / 563 |

## logistic-refit, test

| routes | added | strategy | turn acc [95% CI] | episode acc [95% CI] | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | scheduling | message | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 (drafting) | – | – |
| 2 | scheduling | with-opening | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 (drafting) | – | – |
| 2 | scheduling | episode | 0.980 [0.959, 0.995] | 0.971 [0.941, 0.993] | 0.962 (scheduling) | scheduling->drafting | – |
| 3 | invoices | message | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 (drafting) | – | 0 / 204 |
| 3 | invoices | with-opening | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 (drafting) | – | 0 / 204 |
| 3 | invoices | episode | 0.984 [0.967, 0.996] | 0.975 [0.950, 0.994] | 0.962 (scheduling) | scheduling->drafting | 0 / 200 |
| 4 | tickets | message | 0.963 [0.941, 0.983] | 0.941 [0.903, 0.973] | 0.875 (invoices) | invoices->tickets | 6 / 250 |
| 4 | tickets | with-opening | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 (drafting) | – | 0 / 250 |
| 4 | tickets | episode | 0.980 [0.963, 0.993] | 0.968 [0.941, 0.989] | 0.962 (scheduling) | scheduling->drafting | 0 / 246 |
| 5 | rooms | message | 0.958 [0.936, 0.977] | 0.930 [0.893, 0.963] | 0.880 (invoices) | invoices->tickets | 0 / 287 |
| 5 | rooms | with-opening | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 (drafting) | – | 0 / 298 |
| 5 | rooms | episode | 0.972 [0.955, 0.989] | 0.953 [0.925, 0.981] | 0.958 (tickets) | scheduling->drafting | 0 / 292 |
| 6 | expenses | message | 0.941 [0.914, 0.965] | 0.914 [0.877, 0.947] | 0.846 (invoices) | rooms->tickets | 4 / 339 |
| 6 | expenses | with-opening | 0.998 [0.993, 1.000] | 0.996 [0.988, 1.000] | 0.981 (invoices) | invoices->expenses | 0 / 354 |
| 6 | expenses | episode | 0.961 [0.940, 0.978] | 0.934 [0.902, 0.963] | 0.940 (expenses) | scheduling->drafting | 0 / 344 |
| 7 | approvals | message | 0.935 [0.909, 0.957] | 0.899 [0.855, 0.935] | 0.833 (invoices) | invoices->approvals | 4 / 386 |
| 7 | approvals | with-opening | 0.996 [0.989, 1.000] | 0.993 [0.982, 1.000] | 0.981 (tickets) | tickets->approvals | 0 / 409 |
| 7 | approvals | episode | 0.950 [0.932, 0.969] | 0.913 [0.884, 0.946] | 0.923 (expenses) | scheduling->drafting | 0 / 394 |
| 8 | reminders | message | 0.921 [0.896, 0.943] | 0.881 [0.845, 0.916] | 0.821 (invoices) | invoices->approvals | 3 / 445 |
| 8 | reminders | with-opening | 0.993 [0.985, 0.998] | 0.987 [0.974, 0.997] | 0.981 (tickets) | tickets->approvals | 0 / 474 |
| 8 | reminders | episode | 0.938 [0.917, 0.958] | 0.890 [0.855, 0.926] | 0.907 (expenses) | scheduling->drafting | 0 / 452 |
| 9 | summaries | message | 0.914 [0.890, 0.937] | 0.867 [0.827, 0.902] | 0.793 (invoices) | invoices->approvals | 3 / 502 |
| 9 | summaries | with-opening | 0.997 [0.992, 1.000] | 0.994 [0.986, 1.000] | 0.982 (tickets) | tickets->approvals | 0 / 541 |
| 9 | summaries | episode | 0.926 [0.905, 0.945] | 0.867 [0.832, 0.902] | 0.893 (expenses) | scheduling->drafting | 0 / 511 |
| 10 | travel | message | 0.911 [0.889, 0.931] | 0.859 [0.820, 0.891] | 0.800 (invoices) | invoices->approvals | 2 / 565 |
| 10 | travel | with-opening | 0.994 [0.988, 0.999] | 0.990 [0.979, 0.997] | 0.981 (travel) | travel->rooms | 0 / 616 |
| 10 | travel | episode | 0.913 [0.893, 0.933] | 0.844 [0.810, 0.880] | 0.870 (travel) | scheduling->drafting | 0 / 572 |
| 11 | inventory | message | 0.899 [0.876, 0.919] | 0.840 [0.802, 0.873] | 0.806 (invoices) | approvals->inventory | 6 / 625 |
| 11 | inventory | with-opening | 0.994 [0.988, 0.999] | 0.988 [0.979, 0.998] | 0.982 (travel) | travel->rooms | 0 / 682 |
| 11 | inventory | episode | 0.901 [0.881, 0.921] | 0.821 [0.785, 0.858] | 0.857 (travel) | scheduling->drafting | 0 / 626 |
| 12 | timesheets | message | 0.901 [0.880, 0.921] | 0.843 [0.811, 0.876] | 0.812 (invoices) | approvals->inventory | 0 / 693 |
| 12 | timesheets | with-opening | 0.994 [0.988, 0.999] | 0.989 [0.979, 0.998] | 0.970 (reminders) | tickets->approvals | 0 / 766 |
| 12 | timesheets | episode | 0.890 [0.869, 0.909] | 0.798 [0.760, 0.835] | 0.845 (travel) | scheduling->drafting | 0 / 695 |
| 13 | contacts | message | 0.899 [0.879, 0.918] | 0.839 [0.808, 0.869] | 0.803 (invoices) | approvals->inventory | 1 / 768 |
| 13 | contacts | with-opening | 0.993 [0.986, 0.998] | 0.986 [0.975, 0.996] | 0.969 (tickets) | timesheets->contacts | 0 / 847 |
| 13 | contacts | episode | 0.880 [0.861, 0.899] | 0.776 [0.741, 0.812] | 0.833 (travel) | scheduling->drafting | 0 / 758 |
| 14 | search | message | 0.900 [0.879, 0.918] | 0.840 [0.808, 0.871] | 0.809 (invoices) | approvals->inventory | 1 / 851 |
| 14 | search | with-opening | 0.995 [0.990, 0.999] | 0.991 [0.982, 0.998] | 0.972 (reminders) | tickets->approvals | 0 / 940 |
| 14 | search | episode | 0.869 [0.850, 0.889] | 0.755 [0.719, 0.791] | 0.823 (travel) | scheduling->drafting | 0 / 833 |

## logistic-refit, unseen

| routes | added | strategy | turn acc [95% CI] | episode acc [95% CI] | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | scheduling | message | 0.987 [0.959, 1.000] | 0.979 [0.938, 1.000] | 0.974 (drafting) | drafting->scheduling | – |
| 2 | scheduling | with-opening | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 (drafting) | – | – |
| 2 | scheduling | episode | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 (drafting) | – | – |
| 3 | invoices | message | 0.696 [0.610, 0.778] | 0.556 [0.431, 0.667] | 0.500 (drafting) | scheduling->invoices | 37 / 76 |
| 3 | invoices | with-opening | 0.664 [0.566, 0.752] | 0.514 [0.403, 0.625] | 0.368 (drafting) | drafting->invoices | 42 / 77 |
| 3 | invoices | episode | 0.880 [0.797, 0.952] | 0.875 [0.792, 0.944] | 0.763 (drafting) | drafting->invoices | 15 / 77 |
| 4 | tickets | message | 0.746 [0.684, 0.800] | 0.582 [0.480, 0.673] | 0.487 (scheduling) | drafting->tickets | 7 / 87 |
| 4 | tickets | with-opening | 0.825 [0.766, 0.877] | 0.704 [0.612, 0.796] | 0.579 (drafting) | scheduling->tickets | 2 / 83 |
| 4 | tickets | episode | 0.932 [0.877, 0.976] | 0.918 [0.857, 0.969] | 0.795 (scheduling) | scheduling->tickets | 3 / 110 |
| 5 | rooms | message | 0.772 [0.723, 0.822] | 0.611 [0.532, 0.690] | 0.487 (scheduling) | scheduling->rooms | 3 / 132 |
| 5 | rooms | with-opening | 0.845 [0.791, 0.897] | 0.762 [0.690, 0.833] | 0.410 (scheduling) | scheduling->rooms | 9 / 146 |
| 5 | rooms | episode | 0.940 [0.897, 0.974] | 0.913 [0.857, 0.960] | 0.795 (scheduling) | scheduling->rooms | 2 / 165 |
| 6 | expenses | message | 0.701 [0.654, 0.748] | 0.500 [0.423, 0.571] | 0.550 (expenses) | expenses->rooms | 12 / 179 |
| 6 | expenses | with-opening | 0.872 [0.829, 0.910] | 0.795 [0.731, 0.859] | 0.410 (scheduling) | scheduling->rooms | 1 / 196 |
| 6 | expenses | episode | 0.795 [0.726, 0.858] | 0.782 [0.712, 0.846] | 0.500 (expenses) | expenses->rooms | 19 / 218 |
| 7 | approvals | message | 0.731 [0.691, 0.774] | 0.553 [0.484, 0.622] | 0.564 (scheduling) | scheduling->rooms | 7 / 209 |
| 7 | approvals | with-opening | 0.870 [0.830, 0.907] | 0.793 [0.739, 0.851] | 0.410 (scheduling) | scheduling->rooms | 7 / 260 |
| 7 | approvals | episode | 0.825 [0.770, 0.878] | 0.793 [0.734, 0.851] | 0.571 (invoices) | expenses->rooms | 0 / 237 |
| 8 | reminders | message | 0.749 [0.711, 0.790] | 0.577 [0.514, 0.640] | 0.564 (scheduling) | scheduling->rooms | 5 / 264 |
| 8 | reminders | with-opening | 0.896 [0.862, 0.929] | 0.833 [0.784, 0.883] | 0.513 (scheduling) | scheduling->rooms | 0 / 314 |
| 8 | reminders | episode | 0.831 [0.780, 0.877] | 0.784 [0.725, 0.838] | 0.569 (invoices) | expenses->rooms | 0 / 298 |
| 9 | summaries | message | 0.786 [0.754, 0.818] | 0.628 [0.570, 0.682] | 0.590 (scheduling) | scheduling->rooms | 2 / 323 |
| 9 | summaries | with-opening | 0.898 [0.866, 0.927] | 0.841 [0.795, 0.884] | 0.421 (drafting) | scheduling->rooms | 6 / 386 |
| 9 | summaries | episode | 0.860 [0.821, 0.897] | 0.791 [0.740, 0.841] | 0.652 (expenses) | expenses->rooms | 0 / 358 |
| 10 | travel | message | 0.772 [0.743, 0.802] | 0.591 [0.537, 0.645] | 0.484 (travel) | travel->expenses | 1 / 392 |
| 10 | travel | with-opening | 0.896 [0.865, 0.923] | 0.831 [0.787, 0.875] | 0.447 (drafting) | scheduling->rooms | 1 / 448 |
| 10 | travel | episode | 0.850 [0.818, 0.885] | 0.774 [0.730, 0.821] | 0.516 (travel) | travel->expenses | 0 / 429 |
| 11 | inventory | message | 0.784 [0.756, 0.814] | 0.613 [0.562, 0.664] | 0.500 (travel) | travel->expenses | 1 / 444 |
| 11 | inventory | with-opening | 0.899 [0.873, 0.923] | 0.833 [0.795, 0.872] | 0.474 (drafting) | scheduling->rooms | 3 / 515 |
| 11 | inventory | episode | 0.844 [0.814, 0.874] | 0.753 [0.708, 0.798] | 0.516 (travel) | travel->expenses | 0 / 489 |
| 12 | timesheets | message | 0.787 [0.762, 0.813] | 0.624 [0.579, 0.672] | 0.500 (travel) | travel->expenses | 11 / 513 |
| 12 | timesheets | with-opening | 0.916 [0.894, 0.937] | 0.854 [0.817, 0.889] | 0.421 (drafting) | scheduling->rooms | 4 / 588 |
| 12 | timesheets | episode | 0.820 [0.787, 0.853] | 0.720 [0.675, 0.765] | 0.500 (travel) | travel->expenses | 14 / 552 |
| 13 | contacts | message | 0.794 [0.768, 0.820] | 0.637 [0.595, 0.680] | 0.485 (travel) | expenses->timesheets | 5 / 578 |
| 13 | contacts | with-opening | 0.926 [0.904, 0.945] | 0.872 [0.839, 0.905] | 0.447 (drafting) | drafting->summaries | 2 / 672 |
| 13 | contacts | episode | 0.807 [0.776, 0.837] | 0.699 [0.656, 0.742] | 0.485 (travel) | travel->expenses | 7 / 602 |
| 14 | search | message | 0.781 [0.755, 0.809] | 0.622 [0.579, 0.669] | 0.443 (travel) | search->inventory | 2 / 651 |
| 14 | search | with-opening | 0.931 [0.913, 0.947] | 0.880 [0.850, 0.908] | 0.447 (drafting) | drafting->summaries | 0 / 759 |
| 14 | search | episode | 0.778 [0.743, 0.807] | 0.660 [0.613, 0.701] | 0.403 (search) | search->inventory | 0 / 662 |
