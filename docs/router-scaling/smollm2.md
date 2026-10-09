# Router scaling: lm encoder (layer-1)

836 fit cases; evaluation cases {'test': 556, 'unseen': 468}, turns {'test': 1040, 'unseen': 909}; 2646 texts, features in 0 s. Intervals: 1,000 case-level bootstrap draws.

## centroids-refit, test

| routes | added | strategy | turn acc [95% CI] | episode acc [95% CI] | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | scheduling | message | 0.926 [0.889, 0.959] | 0.890 [0.831, 0.941] | 0.920 (drafting) | drafting->scheduling | – |
| 2 | scheduling | with-opening | 0.980 [0.959, 0.995] | 0.971 [0.941, 0.993] | 0.962 (scheduling) | scheduling->drafting | – |
| 2 | scheduling | episode | 0.980 [0.959, 0.995] | 0.971 [0.941, 0.993] | 0.962 (scheduling) | scheduling->drafting | – |
| 3 | invoices | message | 0.872 [0.833, 0.909] | 0.800 [0.731, 0.863] | 0.840 (drafting) | drafting->invoices | 17 / 189 |
| 3 | invoices | with-opening | 0.984 [0.967, 0.996] | 0.975 [0.950, 0.994] | 0.962 (scheduling) | scheduling->drafting | 0 / 200 |
| 3 | invoices | episode | 0.984 [0.967, 0.996] | 0.975 [0.950, 0.994] | 0.962 (scheduling) | scheduling->drafting | 0 / 200 |
| 4 | tickets | message | 0.913 [0.886, 0.940] | 0.860 [0.812, 0.903] | 0.833 (invoices) | scheduling->drafting | 8 / 218 |
| 4 | tickets | with-opening | 0.980 [0.963, 0.993] | 0.968 [0.941, 0.989] | 0.962 (scheduling) | scheduling->drafting | 0 / 246 |
| 4 | tickets | episode | 0.980 [0.963, 0.993] | 0.968 [0.941, 0.989] | 0.962 (scheduling) | scheduling->drafting | 0 / 246 |
| 5 | rooms | message | 0.890 [0.859, 0.922] | 0.836 [0.790, 0.883] | 0.729 (tickets) | tickets->rooms | 16 / 272 |
| 5 | rooms | with-opening | 0.975 [0.958, 0.991] | 0.958 [0.930, 0.986] | 0.958 (tickets) | scheduling->drafting | 0 / 292 |
| 5 | rooms | episode | 0.972 [0.955, 0.989] | 0.953 [0.925, 0.981] | 0.958 (tickets) | scheduling->drafting | 0 / 292 |
| 6 | expenses | message | 0.890 [0.857, 0.918] | 0.836 [0.787, 0.881] | 0.740 (tickets) | tickets->rooms | 0 / 315 |
| 6 | expenses | with-opening | 0.951 [0.930, 0.970] | 0.918 [0.881, 0.951] | 0.885 (invoices) | scheduling->drafting | 4 / 345 |
| 6 | expenses | episode | 0.961 [0.940, 0.978] | 0.934 [0.902, 0.963] | 0.940 (expenses) | scheduling->drafting | 0 / 344 |
| 7 | approvals | message | 0.880 [0.848, 0.909] | 0.819 [0.768, 0.862] | 0.712 (tickets) | tickets->rooms | 7 / 365 |
| 7 | approvals | with-opening | 0.943 [0.924, 0.962] | 0.902 [0.870, 0.935] | 0.852 (invoices) | invoices->expenses | 1 / 390 |
| 7 | approvals | episode | 0.950 [0.932, 0.969] | 0.913 [0.884, 0.946] | 0.923 (expenses) | scheduling->drafting | 0 / 394 |
| 8 | reminders | message | 0.857 [0.827, 0.886] | 0.781 [0.735, 0.826] | 0.722 (tickets) | tickets->rooms | 10 / 419 |
| 8 | reminders | with-opening | 0.939 [0.920, 0.959] | 0.894 [0.858, 0.929] | 0.893 (invoices) | scheduling->drafting | 0 / 449 |
| 8 | reminders | episode | 0.910 [0.881, 0.937] | 0.868 [0.832, 0.903] | 0.655 (rooms) | rooms->reminders | 15 / 452 |
| 9 | summaries | message | 0.791 [0.757, 0.823] | 0.705 [0.656, 0.751] | 0.467 (rooms) | approvals->summaries | 52 / 467 |
| 9 | summaries | with-opening | 0.927 [0.907, 0.947] | 0.870 [0.832, 0.905] | 0.862 (invoices) | scheduling->drafting | 2 / 512 |
| 9 | summaries | episode | 0.883 [0.852, 0.912] | 0.832 [0.795, 0.873] | 0.467 (rooms) | rooms->reminders | 11 / 496 |
| 10 | travel | message | 0.803 [0.773, 0.832] | 0.714 [0.667, 0.758] | 0.562 (approvals) | approvals->summaries | 1 / 489 |
| 10 | travel | with-opening | 0.918 [0.900, 0.938] | 0.854 [0.820, 0.888] | 0.867 (invoices) | scheduling->drafting | 4 / 573 |
| 10 | travel | episode | 0.894 [0.870, 0.918] | 0.828 [0.792, 0.865] | 0.677 (rooms) | rooms->reminders | 0 / 546 |
| 11 | inventory | message | 0.811 [0.781, 0.839] | 0.722 [0.677, 0.764] | 0.561 (approvals) | approvals->summaries | 4 / 551 |
| 11 | inventory | with-opening | 0.908 [0.890, 0.928] | 0.833 [0.800, 0.870] | 0.806 (invoices) | scheduling->drafting | 3 / 630 |
| 11 | inventory | episode | 0.888 [0.865, 0.911] | 0.811 [0.774, 0.849] | 0.719 (rooms) | rooms->reminders | 0 / 613 |
| 12 | timesheets | message | 0.799 [0.774, 0.827] | 0.706 [0.665, 0.747] | 0.574 (approvals) | approvals->summaries | 7 / 625 |
| 12 | timesheets | with-opening | 0.890 [0.870, 0.909] | 0.798 [0.762, 0.835] | 0.812 (invoices) | scheduling->drafting | 6 / 700 |
| 12 | timesheets | episode | 0.879 [0.857, 0.901] | 0.790 [0.749, 0.826] | 0.742 (rooms) | rooms->reminders | 0 / 685 |
| 13 | contacts | message | 0.780 [0.753, 0.807] | 0.678 [0.637, 0.720] | 0.586 (approvals) | contacts->summaries | 14 / 681 |
| 13 | contacts | with-opening | 0.884 [0.865, 0.903] | 0.784 [0.749, 0.820] | 0.803 (invoices) | reminders->contacts | 2 / 758 |
| 13 | contacts | episode | 0.874 [0.854, 0.894] | 0.771 [0.737, 0.808] | 0.794 (rooms) | rooms->reminders | 0 / 749 |
| 14 | search | message | 0.769 [0.743, 0.794] | 0.664 [0.624, 0.701] | 0.479 (search) | contacts->summaries | 0 / 739 |
| 14 | search | with-opening | 0.867 [0.847, 0.885] | 0.757 [0.721, 0.791] | 0.648 (search) | search->timesheets | 0 / 837 |
| 14 | search | episode | 0.827 [0.801, 0.852] | 0.727 [0.689, 0.763] | 0.268 (search) | search->timesheets | 0 / 828 |

## centroids-refit, unseen

| routes | added | strategy | turn acc [95% CI] | episode acc [95% CI] | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | scheduling | message | 0.675 [0.551, 0.803] | 0.604 [0.458, 0.750] | 0.342 (drafting) | drafting->scheduling | – |
| 2 | scheduling | with-opening | 0.571 [0.430, 0.723] | 0.542 [0.396, 0.688] | 0.132 (drafting) | drafting->scheduling | – |
| 2 | scheduling | episode | 0.623 [0.487, 0.775] | 0.625 [0.479, 0.771] | 0.237 (drafting) | drafting->scheduling | – |
| 3 | invoices | message | 0.608 [0.504, 0.703] | 0.486 [0.375, 0.597] | 0.289 (drafting) | scheduling->invoices | 29 / 52 |
| 3 | invoices | with-opening | 0.600 [0.488, 0.697] | 0.486 [0.375, 0.597] | 0.289 (drafting) | scheduling->invoices | 25 / 44 |
| 3 | invoices | episode | 0.744 [0.637, 0.840] | 0.722 [0.611, 0.819] | 0.447 (drafting) | drafting->scheduling | 11 / 48 |
| 4 | tickets | message | 0.627 [0.556, 0.698] | 0.469 [0.378, 0.561] | 0.395 (drafting) | scheduling->tickets | 16 / 76 |
| 4 | tickets | with-opening | 0.667 [0.588, 0.743] | 0.551 [0.459, 0.643] | 0.436 (scheduling) | scheduling->tickets | 13 / 75 |
| 4 | tickets | episode | 0.695 [0.604, 0.784] | 0.684 [0.592, 0.776] | 0.632 (drafting) | invoices->tickets | 15 / 93 |
| 5 | rooms | message | 0.625 [0.567, 0.686] | 0.444 [0.365, 0.532] | 0.410 (scheduling) | scheduling->rooms | 16 / 111 |
| 5 | rooms | with-opening | 0.724 [0.661, 0.785] | 0.603 [0.516, 0.683] | 0.333 (scheduling) | scheduling->rooms | 15 / 118 |
| 5 | rooms | episode | 0.720 [0.637, 0.793] | 0.698 [0.619, 0.770] | 0.538 (invoices) | invoices->rooms | 11 / 123 |
| 6 | expenses | message | 0.597 [0.545, 0.650] | 0.391 [0.321, 0.462] | 0.296 (tickets) | tickets->expenses | 19 / 145 |
| 6 | expenses | with-opening | 0.718 [0.658, 0.772] | 0.577 [0.500, 0.647] | 0.436 (scheduling) | invoices->expenses | 11 / 168 |
| 6 | expenses | episode | 0.648 [0.572, 0.724] | 0.615 [0.538, 0.686] | 0.056 (tickets) | tickets->expenses | 34 / 167 |
| 7 | approvals | message | 0.526 [0.475, 0.576] | 0.324 [0.261, 0.388] | 0.071 (tickets) | tickets->expenses | 25 / 178 |
| 7 | approvals | with-opening | 0.643 [0.585, 0.703] | 0.495 [0.420, 0.569] | 0.436 (scheduling) | invoices->expenses | 18 / 214 |
| 7 | approvals | episode | 0.598 [0.527, 0.665] | 0.559 [0.484, 0.622] | 0.000 (tickets) | tickets->expenses | 7 / 193 |
| 8 | reminders | message | 0.517 [0.472, 0.563] | 0.311 [0.252, 0.369] | 0.069 (tickets) | rooms->reminders | 25 / 190 |
| 8 | reminders | with-opening | 0.659 [0.605, 0.715] | 0.500 [0.437, 0.568] | 0.414 (tickets) | approvals->reminders | 10 / 232 |
| 8 | reminders | episode | 0.555 [0.491, 0.617] | 0.509 [0.446, 0.572] | 0.000 (tickets) | rooms->reminders | 36 / 216 |
| 9 | summaries | message | 0.443 [0.400, 0.484] | 0.240 [0.186, 0.291] | 0.067 (tickets) | rooms->reminders | 50 / 223 |
| 9 | summaries | with-opening | 0.673 [0.626, 0.723] | 0.523 [0.461, 0.585] | 0.367 (tickets) | invoices->expenses | 9 / 284 |
| 9 | summaries | episode | 0.519 [0.456, 0.578] | 0.465 [0.403, 0.523] | 0.000 (tickets) | rooms->reminders | 15 / 239 |
| 10 | travel | message | 0.459 [0.419, 0.498] | 0.267 [0.216, 0.318] | 0.065 (tickets) | rooms->reminders | 1 / 221 |
| 10 | travel | with-opening | 0.683 [0.637, 0.731] | 0.547 [0.490, 0.605] | 0.387 (tickets) | travel->rooms | 2 / 336 |
| 10 | travel | episode | 0.550 [0.496, 0.607] | 0.490 [0.436, 0.547] | 0.000 (tickets) | rooms->reminders | 2 / 259 |
| 11 | inventory | message | 0.511 [0.472, 0.550] | 0.318 [0.271, 0.366] | 0.062 (tickets) | tickets->expenses | 7 / 264 |
| 11 | inventory | with-opening | 0.691 [0.653, 0.730] | 0.542 [0.491, 0.592] | 0.375 (tickets) | travel->rooms | 8 / 393 |
| 11 | inventory | episode | 0.601 [0.552, 0.651] | 0.524 [0.473, 0.574] | 0.000 (tickets) | tickets->expenses | 1 / 316 |
| 12 | timesheets | message | 0.503 [0.466, 0.538] | 0.323 [0.272, 0.368] | 0.061 (tickets) | travel->rooms | 18 / 334 |
| 12 | timesheets | with-opening | 0.666 [0.626, 0.705] | 0.519 [0.463, 0.566] | 0.167 (tickets) | tickets->timesheets | 31 / 452 |
| 12 | timesheets | episode | 0.597 [0.549, 0.644] | 0.521 [0.471, 0.569] | 0.000 (tickets) | tickets->expenses | 4 / 393 |
| 13 | contacts | message | 0.456 [0.422, 0.487] | 0.277 [0.235, 0.318] | 0.045 (contacts) | contacts->reminders | 9 / 369 |
| 13 | contacts | with-opening | 0.633 [0.594, 0.670] | 0.474 [0.429, 0.521] | 0.147 (tickets) | contacts->timesheets | 3 / 489 |
| 13 | contacts | episode | 0.544 [0.497, 0.586] | 0.467 [0.419, 0.514] | 0.000 (contacts) | contacts->reminders | 0 / 438 |
| 14 | search | message | 0.441 [0.410, 0.471] | 0.256 [0.218, 0.295] | 0.044 (contacts) | search->inventory | 2 / 374 |
| 14 | search | with-opening | 0.586 [0.550, 0.621] | 0.423 [0.378, 0.468] | 0.090 (search) | contacts->timesheets | 3 / 519 |
| 14 | search | episode | 0.506 [0.463, 0.548] | 0.425 [0.382, 0.468] | 0.000 (contacts) | search->inventory | 2 / 446 |

## centroids-frozen, test

| routes | added | strategy | turn acc [95% CI] | episode acc [95% CI] | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | scheduling | message | 0.926 [0.889, 0.959] | 0.890 [0.831, 0.941] | 0.920 (drafting) | drafting->scheduling | – |
| 2 | scheduling | with-opening | 0.980 [0.959, 0.995] | 0.971 [0.941, 0.993] | 0.962 (scheduling) | scheduling->drafting | – |
| 2 | scheduling | episode | 0.980 [0.959, 0.995] | 0.971 [0.941, 0.993] | 0.962 (scheduling) | scheduling->drafting | – |
| 3 | invoices | message | 0.856 [0.816, 0.894] | 0.775 [0.706, 0.838] | 0.820 (drafting) | drafting->invoices | 21 / 189 |
| 3 | invoices | with-opening | 0.972 [0.950, 0.992] | 0.956 [0.919, 0.988] | 0.962 (scheduling) | scheduling->drafting | 3 / 200 |
| 3 | invoices | episode | 0.984 [0.967, 0.996] | 0.975 [0.950, 0.994] | 0.962 (scheduling) | scheduling->drafting | 0 / 200 |
| 4 | tickets | message | 0.842 [0.808, 0.876] | 0.747 [0.683, 0.806] | 0.810 (drafting) | drafting->tickets | 9 / 214 |
| 4 | tickets | with-opening | 0.950 [0.925, 0.973] | 0.919 [0.882, 0.957] | 0.920 (drafting) | drafting->tickets | 6 / 243 |
| 4 | tickets | episode | 0.980 [0.963, 0.993] | 0.968 [0.941, 0.989] | 0.962 (scheduling) | scheduling->drafting | 0 / 246 |
| 5 | rooms | message | 0.808 [0.773, 0.842] | 0.701 [0.640, 0.757] | 0.729 (tickets) | tickets->rooms | 16 / 251 |
| 5 | rooms | with-opening | 0.912 [0.884, 0.940] | 0.855 [0.808, 0.902] | 0.827 (scheduling) | scheduling->rooms | 13 / 283 |
| 5 | rooms | episode | 0.972 [0.955, 0.989] | 0.953 [0.925, 0.981] | 0.958 (tickets) | scheduling->drafting | 0 / 292 |
| 6 | expenses | message | 0.812 [0.777, 0.845] | 0.705 [0.643, 0.762] | 0.692 (invoices) | tickets->rooms | 3 / 286 |
| 6 | expenses | with-opening | 0.907 [0.881, 0.932] | 0.844 [0.799, 0.885] | 0.827 (scheduling) | scheduling->rooms | 4 / 323 |
| 6 | expenses | episode | 0.954 [0.933, 0.973] | 0.926 [0.893, 0.955] | 0.885 (invoices) | scheduling->drafting | 3 / 344 |
| 7 | approvals | message | 0.773 [0.740, 0.809] | 0.649 [0.591, 0.707] | 0.643 (rooms) | drafting->approvals | 32 / 333 |
| 7 | approvals | with-opening | 0.897 [0.874, 0.921] | 0.822 [0.783, 0.866] | 0.827 (scheduling) | scheduling->rooms | 7 / 372 |
| 7 | approvals | episode | 0.945 [0.924, 0.965] | 0.909 [0.877, 0.942] | 0.889 (invoices) | scheduling->drafting | 0 / 391 |
| 8 | reminders | message | 0.760 [0.729, 0.790] | 0.619 [0.565, 0.674] | 0.655 (rooms) | drafting->approvals | 4 / 368 |
| 8 | reminders | with-opening | 0.899 [0.876, 0.923] | 0.823 [0.784, 0.868] | 0.827 (scheduling) | scheduling->rooms | 0 / 427 |
| 8 | reminders | episode | 0.934 [0.913, 0.955] | 0.887 [0.852, 0.923] | 0.875 (invoices) | scheduling->drafting | 0 / 450 |
| 9 | summaries | message | 0.749 [0.720, 0.781] | 0.607 [0.552, 0.659] | 0.643 (tickets) | drafting->approvals | 16 / 414 |
| 9 | summaries | with-opening | 0.883 [0.860, 0.907] | 0.792 [0.751, 0.835] | 0.827 (scheduling) | drafting->summaries | 8 / 490 |
| 9 | summaries | episode | 0.922 [0.902, 0.943] | 0.864 [0.827, 0.902] | 0.862 (invoices) | scheduling->drafting | 0 / 509 |
| 10 | travel | message | 0.757 [0.730, 0.785] | 0.617 [0.568, 0.664] | 0.655 (tickets) | drafting->approvals | 3 / 463 |
| 10 | travel | with-opening | 0.876 [0.854, 0.899] | 0.779 [0.737, 0.820] | 0.788 (scheduling) | scheduling->travel | 5 / 546 |
| 10 | travel | episode | 0.913 [0.893, 0.933] | 0.844 [0.810, 0.880] | 0.870 (travel) | scheduling->drafting | 0 / 570 |
| 11 | inventory | message | 0.767 [0.740, 0.792] | 0.627 [0.580, 0.670] | 0.667 (tickets) | drafting->approvals | 4 / 519 |
| 11 | inventory | with-opening | 0.872 [0.850, 0.895] | 0.767 [0.726, 0.809] | 0.798 (scheduling) | scheduling->travel | 1 / 601 |
| 11 | inventory | episode | 0.901 [0.881, 0.921] | 0.821 [0.785, 0.858] | 0.857 (travel) | scheduling->drafting | 0 / 626 |
| 12 | timesheets | message | 0.763 [0.737, 0.789] | 0.627 [0.582, 0.670] | 0.652 (rooms) | scheduling->travel | 9 / 591 |
| 12 | timesheets | with-opening | 0.864 [0.842, 0.886] | 0.751 [0.710, 0.792] | 0.808 (scheduling) | scheduling->travel | 1 / 672 |
| 12 | timesheets | episode | 0.890 [0.869, 0.909] | 0.798 [0.760, 0.835] | 0.845 (travel) | scheduling->drafting | 0 / 695 |
| 13 | contacts | message | 0.752 [0.727, 0.777] | 0.610 [0.569, 0.649] | 0.646 (timesheets) | timesheets->contacts | 12 / 650 |
| 13 | contacts | with-opening | 0.857 [0.838, 0.877] | 0.735 [0.698, 0.771] | 0.808 (scheduling) | scheduling->travel | 6 / 736 |
| 13 | contacts | episode | 0.873 [0.854, 0.893] | 0.771 [0.735, 0.808] | 0.787 (contacts) | contacts->reminders | 0 / 758 |
| 14 | search | message | 0.745 [0.721, 0.767] | 0.594 [0.554, 0.630] | 0.563 (search) | search->timesheets | 2 / 712 |
| 14 | search | with-opening | 0.831 [0.810, 0.852] | 0.694 [0.656, 0.730] | 0.535 (search) | search->timesheets | 1 / 812 |
| 14 | search | episode | 0.826 [0.801, 0.852] | 0.727 [0.689, 0.764] | 0.268 (search) | search->timesheets | 0 / 827 |

## centroids-frozen, unseen

| routes | added | strategy | turn acc [95% CI] | episode acc [95% CI] | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | scheduling | message | 0.675 [0.551, 0.803] | 0.604 [0.458, 0.750] | 0.342 (drafting) | drafting->scheduling | – |
| 2 | scheduling | with-opening | 0.571 [0.430, 0.723] | 0.542 [0.396, 0.688] | 0.132 (drafting) | drafting->scheduling | – |
| 2 | scheduling | episode | 0.623 [0.487, 0.775] | 0.625 [0.479, 0.771] | 0.237 (drafting) | drafting->scheduling | – |
| 3 | invoices | message | 0.512 [0.402, 0.621] | 0.417 [0.292, 0.528] | 0.000 (drafting) | drafting->invoices | 36 / 52 |
| 3 | invoices | with-opening | 0.512 [0.402, 0.621] | 0.417 [0.292, 0.528] | 0.000 (drafting) | drafting->invoices | 28 / 44 |
| 3 | invoices | episode | 0.592 [0.479, 0.709] | 0.556 [0.444, 0.667] | 0.000 (drafting) | drafting->invoices | 22 / 48 |
| 4 | tickets | message | 0.576 [0.489, 0.661] | 0.469 [0.367, 0.561] | 0.000 (drafting) | drafting->tickets | 14 / 64 |
| 4 | tickets | with-opening | 0.605 [0.518, 0.689] | 0.500 [0.408, 0.592] | 0.000 (drafting) | drafting->tickets | 8 / 64 |
| 4 | tickets | episode | 0.633 [0.543, 0.729] | 0.592 [0.500, 0.694] | 0.000 (drafting) | drafting->tickets | 12 / 74 |
| 5 | rooms | message | 0.517 [0.443, 0.595] | 0.397 [0.317, 0.484] | 0.000 (drafting) | scheduling->rooms | 30 / 102 |
| 5 | rooms | with-opening | 0.586 [0.502, 0.672] | 0.508 [0.429, 0.595] | 0.000 (drafting) | scheduling->rooms | 23 / 107 |
| 5 | rooms | episode | 0.552 [0.468, 0.644] | 0.492 [0.413, 0.579] | 0.000 (drafting) | scheduling->rooms | 35 / 112 |
| 6 | expenses | message | 0.530 [0.466, 0.595] | 0.372 [0.295, 0.449] | 0.000 (drafting) | scheduling->rooms | 13 / 120 |
| 6 | expenses | with-opening | 0.628 [0.554, 0.696] | 0.538 [0.462, 0.615] | 0.000 (drafting) | scheduling->rooms | 7 / 136 |
| 6 | expenses | episode | 0.547 [0.467, 0.625] | 0.474 [0.397, 0.551] | 0.000 (drafting) | invoices->expenses | 25 / 128 |
| 7 | approvals | message | 0.457 [0.406, 0.512] | 0.261 [0.202, 0.324] | 0.000 (drafting) | scheduling->rooms | 40 / 158 |
| 7 | approvals | with-opening | 0.618 [0.555, 0.681] | 0.495 [0.425, 0.564] | 0.000 (drafting) | scheduling->rooms | 9 / 187 |
| 7 | approvals | episode | 0.535 [0.464, 0.605] | 0.463 [0.388, 0.532] | 0.000 (drafting) | invoices->expenses | 0 / 163 |
| 8 | reminders | message | 0.469 [0.422, 0.515] | 0.266 [0.212, 0.324] | 0.000 (drafting) | rooms->approvals | 15 / 165 |
| 8 | reminders | with-opening | 0.626 [0.568, 0.684] | 0.500 [0.437, 0.568] | 0.000 (drafting) | drafting->approvals | 13 / 223 |
| 8 | reminders | episode | 0.548 [0.483, 0.615] | 0.464 [0.401, 0.532] | 0.000 (drafting) | invoices->expenses | 15 / 193 |
| 9 | summaries | message | 0.457 [0.412, 0.501] | 0.256 [0.205, 0.310] | 0.000 (drafting) | invoices->expenses | 15 / 202 |
| 9 | summaries | with-opening | 0.627 [0.577, 0.681] | 0.488 [0.430, 0.550] | 0.000 (drafting) | scheduling->rooms | 0 / 270 |
| 9 | summaries | episode | 0.531 [0.470, 0.588] | 0.438 [0.376, 0.496] | 0.000 (drafting) | invoices->expenses | 1 / 236 |
| 10 | travel | message | 0.445 [0.402, 0.483] | 0.243 [0.196, 0.294] | 0.000 (drafting) | tickets->expenses | 3 / 228 |
| 10 | travel | with-opening | 0.610 [0.563, 0.660] | 0.476 [0.422, 0.537] | 0.000 (drafting) | scheduling->rooms | 1 / 313 |
| 10 | travel | episode | 0.513 [0.458, 0.572] | 0.426 [0.372, 0.483] | 0.000 (drafting) | invoices->expenses | 5 / 265 |
| 11 | inventory | message | 0.495 [0.454, 0.534] | 0.304 [0.250, 0.351] | 0.000 (drafting) | tickets->expenses | 5 / 256 |
| 11 | inventory | with-opening | 0.636 [0.595, 0.682] | 0.503 [0.452, 0.557] | 0.000 (drafting) | scheduling->rooms | 4 / 351 |
| 11 | inventory | episode | 0.558 [0.506, 0.604] | 0.455 [0.405, 0.506] | 0.000 (drafting) | tickets->expenses | 2 / 295 |
| 12 | timesheets | message | 0.488 [0.450, 0.525] | 0.296 [0.251, 0.341] | 0.000 (drafting) | tickets->expenses | 7 / 324 |
| 12 | timesheets | with-opening | 0.621 [0.578, 0.661] | 0.474 [0.421, 0.524] | 0.000 (drafting) | scheduling->rooms | 24 / 416 |
| 12 | timesheets | episode | 0.531 [0.484, 0.578] | 0.426 [0.376, 0.476] | 0.000 (drafting) | invoices->expenses | 2 / 365 |
| 13 | contacts | message | 0.451 [0.416, 0.484] | 0.258 [0.220, 0.296] | 0.000 (drafting) | contacts->reminders | 12 / 358 |
| 13 | contacts | with-opening | 0.578 [0.538, 0.617] | 0.405 [0.358, 0.453] | 0.000 (drafting) | invoices->expenses | 11 / 456 |
| 13 | contacts | episode | 0.490 [0.441, 0.535] | 0.386 [0.336, 0.431] | 0.000 (contacts) | contacts->reminders | 1 / 390 |
| 14 | search | message | 0.438 [0.405, 0.469] | 0.229 [0.194, 0.265] | 0.000 (drafting) | contacts->reminders | 3 / 370 |
| 14 | search | with-opening | 0.549 [0.511, 0.586] | 0.361 [0.318, 0.404] | 0.000 (drafting) | search->inventory | 4 / 474 |
| 14 | search | episode | 0.461 [0.418, 0.504] | 0.355 [0.310, 0.400] | 0.000 (contacts) | contacts->reminders | 0 / 402 |

## logistic-refit, test

| routes | added | strategy | turn acc [95% CI] | episode acc [95% CI] | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | scheduling | message | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 (drafting) | – | – |
| 2 | scheduling | with-opening | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 (drafting) | – | – |
| 2 | scheduling | episode | 0.980 [0.959, 0.995] | 0.971 [0.941, 0.993] | 0.962 (scheduling) | scheduling->drafting | – |
| 3 | invoices | message | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 (drafting) | – | 0 / 204 |
| 3 | invoices | with-opening | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 (drafting) | – | 0 / 204 |
| 3 | invoices | episode | 0.984 [0.967, 0.996] | 0.975 [0.950, 0.994] | 0.962 (scheduling) | scheduling->drafting | 0 / 200 |
| 4 | tickets | message | 0.966 [0.945, 0.984] | 0.946 [0.914, 0.973] | 0.875 (invoices) | invoices->tickets | 6 / 250 |
| 4 | tickets | with-opening | 0.993 [0.983, 1.000] | 0.989 [0.973, 1.000] | 0.978 (tickets) | tickets->invoices | 0 / 250 |
| 4 | tickets | episode | 0.980 [0.963, 0.993] | 0.968 [0.941, 0.989] | 0.962 (scheduling) | scheduling->drafting | 0 / 246 |
| 5 | rooms | message | 0.949 [0.925, 0.970] | 0.921 [0.883, 0.953] | 0.800 (invoices) | invoices->tickets | 5 / 288 |
| 5 | rooms | with-opening | 0.994 [0.986, 1.000] | 0.991 [0.977, 1.000] | 0.960 (invoices) | invoices->tickets | 0 / 296 |
| 5 | rooms | episode | 0.972 [0.955, 0.989] | 0.953 [0.925, 0.981] | 0.958 (tickets) | scheduling->drafting | 0 / 292 |
| 6 | expenses | message | 0.941 [0.917, 0.964] | 0.914 [0.877, 0.947] | 0.788 (invoices) | invoices->tickets | 1 / 336 |
| 6 | expenses | with-opening | 0.990 [0.980, 0.998] | 0.984 [0.967, 0.996] | 0.960 (expenses) | invoices->tickets | 0 / 352 |
| 6 | expenses | episode | 0.961 [0.940, 0.978] | 0.934 [0.902, 0.963] | 0.940 (expenses) | scheduling->drafting | 0 / 344 |
| 7 | approvals | message | 0.926 [0.898, 0.951] | 0.891 [0.848, 0.928] | 0.759 (invoices) | invoices->approvals | 8 / 386 |
| 7 | approvals | with-opening | 0.994 [0.986, 1.000] | 0.989 [0.975, 1.000] | 0.963 (invoices) | tickets->approvals | 0 / 406 |
| 7 | approvals | episode | 0.950 [0.932, 0.969] | 0.913 [0.884, 0.946] | 0.923 (expenses) | scheduling->drafting | 0 / 394 |
| 8 | reminders | message | 0.919 [0.896, 0.941] | 0.874 [0.839, 0.906] | 0.768 (invoices) | invoices->approvals | 1 / 441 |
| 8 | reminders | with-opening | 0.983 [0.972, 0.993] | 0.971 [0.952, 0.987] | 0.949 (reminders) | tickets->approvals | 1 / 473 |
| 8 | reminders | episode | 0.938 [0.917, 0.958] | 0.890 [0.855, 0.926] | 0.907 (expenses) | scheduling->drafting | 0 / 452 |
| 9 | summaries | message | 0.914 [0.891, 0.937] | 0.864 [0.827, 0.899] | 0.776 (invoices) | invoices->approvals | 0 / 501 |
| 9 | summaries | with-opening | 0.987 [0.978, 0.995] | 0.977 [0.960, 0.991] | 0.951 (reminders) | tickets->approvals | 0 / 536 |
| 9 | summaries | episode | 0.926 [0.905, 0.945] | 0.867 [0.832, 0.902] | 0.893 (expenses) | scheduling->drafting | 0 / 511 |
| 10 | travel | message | 0.914 [0.892, 0.934] | 0.862 [0.826, 0.893] | 0.783 (invoices) | invoices->approvals | 0 / 565 |
| 10 | travel | with-opening | 0.987 [0.978, 0.994] | 0.977 [0.961, 0.990] | 0.952 (reminders) | tickets->approvals | 1 / 610 |
| 10 | travel | episode | 0.913 [0.893, 0.933] | 0.844 [0.810, 0.880] | 0.870 (travel) | scheduling->drafting | 0 / 572 |
| 11 | inventory | message | 0.900 [0.877, 0.920] | 0.842 [0.804, 0.875] | 0.790 (invoices) | approvals->inventory | 8 / 627 |
| 11 | inventory | with-opening | 0.988 [0.980, 0.995] | 0.979 [0.965, 0.991] | 0.954 (reminders) | tickets->approvals | 0 / 677 |
| 11 | inventory | episode | 0.901 [0.881, 0.921] | 0.821 [0.785, 0.858] | 0.857 (travel) | scheduling->drafting | 0 / 626 |
| 12 | timesheets | message | 0.903 [0.882, 0.923] | 0.845 [0.813, 0.878] | 0.797 (invoices) | approvals->inventory | 0 / 694 |
| 12 | timesheets | with-opening | 0.985 [0.977, 0.992] | 0.972 [0.957, 0.985] | 0.955 (reminders) | timesheets->reminders | 0 / 762 |
| 12 | timesheets | episode | 0.890 [0.869, 0.909] | 0.798 [0.760, 0.835] | 0.845 (travel) | scheduling->drafting | 0 / 695 |
| 13 | contacts | message | 0.902 [0.882, 0.920] | 0.843 [0.812, 0.871] | 0.803 (invoices) | approvals->inventory | 0 / 769 |
| 13 | contacts | with-opening | 0.987 [0.980, 0.994] | 0.976 [0.963, 0.988] | 0.969 (expenses) | timesheets->reminders | 0 / 839 |
| 13 | contacts | episode | 0.880 [0.861, 0.899] | 0.776 [0.741, 0.812] | 0.833 (travel) | scheduling->drafting | 0 / 758 |
| 14 | search | message | 0.903 [0.883, 0.921] | 0.844 [0.813, 0.874] | 0.809 (invoices) | approvals->inventory | 3 / 854 |
| 14 | search | with-opening | 0.987 [0.979, 0.992] | 0.975 [0.960, 0.986] | 0.955 (timesheets) | timesheets->search | 0 / 935 |
| 14 | search | episode | 0.869 [0.850, 0.889] | 0.755 [0.719, 0.791] | 0.823 (travel) | scheduling->drafting | 0 / 833 |

## logistic-refit, unseen

| routes | added | strategy | turn acc [95% CI] | episode acc [95% CI] | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | scheduling | message | 0.636 [0.513, 0.753] | 0.542 [0.396, 0.688] | 0.395 (drafting) | drafting->scheduling | – |
| 2 | scheduling | with-opening | 0.636 [0.507, 0.765] | 0.542 [0.396, 0.688] | 0.263 (drafting) | drafting->scheduling | – |
| 2 | scheduling | episode | 0.675 [0.532, 0.811] | 0.667 [0.521, 0.792] | 0.342 (drafting) | drafting->scheduling | – |
| 3 | invoices | message | 0.552 [0.440, 0.654] | 0.444 [0.319, 0.556] | 0.132 (drafting) | drafting->invoices | 29 / 49 |
| 3 | invoices | with-opening | 0.696 [0.603, 0.788] | 0.569 [0.458, 0.681] | 0.237 (drafting) | drafting->scheduling | 10 / 49 |
| 3 | invoices | episode | 0.656 [0.542, 0.766] | 0.625 [0.514, 0.736] | 0.211 (drafting) | drafting->invoices | 19 / 52 |
| 4 | tickets | message | 0.588 [0.509, 0.667] | 0.429 [0.337, 0.520] | 0.132 (drafting) | drafting->invoices | 11 / 69 |
| 4 | tickets | with-opening | 0.729 [0.644, 0.802] | 0.643 [0.551, 0.735] | 0.158 (drafting) | drafting->tickets | 14 / 87 |
| 4 | tickets | episode | 0.672 [0.583, 0.766] | 0.633 [0.551, 0.724] | 0.263 (drafting) | drafting->invoices | 11 / 82 |
| 5 | rooms | message | 0.612 [0.543, 0.674] | 0.444 [0.365, 0.524] | 0.211 (drafting) | scheduling->rooms | 18 / 104 |
| 5 | rooms | with-opening | 0.763 [0.693, 0.827] | 0.690 [0.611, 0.770] | 0.308 (scheduling) | drafting->tickets | 13 / 129 |
| 5 | rooms | episode | 0.659 [0.575, 0.740] | 0.619 [0.532, 0.698] | 0.342 (drafting) | invoices->rooms | 28 / 119 |
| 6 | expenses | message | 0.594 [0.538, 0.650] | 0.397 [0.321, 0.474] | 0.263 (drafting) | invoices->expenses | 24 / 142 |
| 6 | expenses | with-opening | 0.742 [0.682, 0.798] | 0.628 [0.558, 0.705] | 0.308 (scheduling) | scheduling->rooms | 18 / 177 |
| 6 | expenses | episode | 0.584 [0.505, 0.670] | 0.545 [0.468, 0.622] | 0.222 (invoices) | invoices->expenses | 47 / 153 |
| 7 | approvals | message | 0.551 [0.496, 0.603] | 0.340 [0.271, 0.404] | 0.158 (drafting) | invoices->expenses | 19 / 177 |
| 7 | approvals | with-opening | 0.723 [0.663, 0.778] | 0.606 [0.537, 0.676] | 0.026 (drafting) | drafting->approvals | 16 / 221 |
| 7 | approvals | episode | 0.515 [0.444, 0.586] | 0.479 [0.410, 0.548] | 0.107 (invoices) | invoices->expenses | 13 / 174 |
| 8 | reminders | message | 0.573 [0.523, 0.620] | 0.351 [0.293, 0.415] | 0.158 (drafting) | invoices->expenses | 18 / 199 |
| 8 | reminders | with-opening | 0.738 [0.689, 0.789] | 0.608 [0.545, 0.676] | 0.053 (drafting) | scheduling->rooms | 7 / 261 |
| 8 | reminders | episode | 0.541 [0.479, 0.603] | 0.486 [0.423, 0.554] | 0.138 (invoices) | invoices->expenses | 20 / 186 |
| 9 | summaries | message | 0.607 [0.563, 0.653] | 0.399 [0.345, 0.457] | 0.158 (drafting) | invoices->expenses | 4 / 247 |
| 9 | summaries | with-opening | 0.762 [0.718, 0.802] | 0.640 [0.581, 0.694] | 0.079 (drafting) | scheduling->rooms | 3 / 318 |
| 9 | summaries | episode | 0.569 [0.511, 0.627] | 0.508 [0.446, 0.566] | 0.200 (tickets) | invoices->expenses | 7 / 233 |
| 10 | travel | message | 0.637 [0.598, 0.674] | 0.426 [0.372, 0.486] | 0.158 (drafting) | invoices->expenses | 5 / 303 |
| 10 | travel | with-opening | 0.777 [0.737, 0.816] | 0.662 [0.605, 0.716] | 0.105 (drafting) | invoices->expenses | 2 / 380 |
| 10 | travel | episode | 0.612 [0.556, 0.669] | 0.541 [0.483, 0.601] | 0.148 (approvals) | invoices->expenses | 6 / 284 |
| 11 | inventory | message | 0.645 [0.606, 0.681] | 0.455 [0.402, 0.506] | 0.158 (drafting) | invoices->expenses | 12 / 366 |
| 11 | inventory | with-opening | 0.786 [0.746, 0.822] | 0.667 [0.613, 0.717] | 0.053 (drafting) | invoices->expenses | 10 / 447 |
| 11 | inventory | episode | 0.613 [0.564, 0.660] | 0.530 [0.476, 0.583] | 0.143 (approvals) | invoices->expenses | 17 / 352 |
| 12 | timesheets | message | 0.693 [0.660, 0.726] | 0.511 [0.458, 0.558] | 0.184 (drafting) | invoices->expenses | 6 / 422 |
| 12 | timesheets | with-opening | 0.772 [0.738, 0.807] | 0.648 [0.600, 0.698] | 0.079 (drafting) | invoices->expenses | 30 / 514 |
| 12 | timesheets | episode | 0.659 [0.614, 0.700] | 0.566 [0.519, 0.616] | 0.185 (approvals) | invoices->expenses | 8 / 401 |
| 13 | contacts | message | 0.689 [0.657, 0.720] | 0.500 [0.453, 0.547] | 0.211 (drafting) | invoices->expenses | 19 / 509 |
| 13 | contacts | with-opening | 0.789 [0.759, 0.820] | 0.666 [0.621, 0.711] | 0.158 (drafting) | invoices->expenses | 8 / 567 |
| 13 | contacts | episode | 0.649 [0.606, 0.692] | 0.550 [0.502, 0.595] | 0.194 (approvals) | invoices->expenses | 19 / 484 |
| 14 | search | message | 0.694 [0.664, 0.728] | 0.511 [0.464, 0.560] | 0.132 (drafting) | invoices->expenses | 8 / 565 |
| 14 | search | with-opening | 0.772 [0.742, 0.802] | 0.635 [0.592, 0.679] | 0.184 (drafting) | invoices->expenses | 12 / 647 |
| 14 | search | episode | 0.628 [0.590, 0.671] | 0.524 [0.481, 0.568] | 0.211 (drafting) | invoices->expenses | 8 / 532 |
