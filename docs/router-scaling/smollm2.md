# Router scaling: lm encoder

836 fit cases; evaluation {'test': 556, 'unseen': 420}; features in 1239 s.

## centroids-refit, test

| routes | added | strategy | turn acc | episode acc | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | scheduling | message | 0.926 | 0.890 | 0.920 (drafting) | drafting->scheduling | – |
| 2 | scheduling | with-opening | 0.980 | 0.971 | 0.962 (scheduling) | scheduling->drafting | – |
| 2 | scheduling | episode | 0.980 | 0.971 | 0.962 (scheduling) | scheduling->drafting | – |
| 3 | invoices | message | 0.872 | 0.800 | 0.840 (drafting) | drafting->invoices | 17 / 189 |
| 3 | invoices | with-opening | 0.984 | 0.975 | 0.962 (scheduling) | scheduling->drafting | 0 / 200 |
| 3 | invoices | episode | 0.984 | 0.975 | 0.962 (scheduling) | scheduling->drafting | 0 / 200 |
| 4 | tickets | message | 0.913 | 0.860 | 0.833 (invoices) | scheduling->drafting | 8 / 218 |
| 4 | tickets | with-opening | 0.980 | 0.968 | 0.962 (scheduling) | scheduling->drafting | 0 / 246 |
| 4 | tickets | episode | 0.980 | 0.968 | 0.962 (scheduling) | scheduling->drafting | 0 / 246 |
| 5 | rooms | message | 0.890 | 0.836 | 0.729 (tickets) | tickets->rooms | 16 / 272 |
| 5 | rooms | with-opening | 0.975 | 0.958 | 0.958 (tickets) | scheduling->drafting | 0 / 292 |
| 5 | rooms | episode | 0.972 | 0.953 | 0.958 (tickets) | scheduling->drafting | 0 / 292 |
| 6 | expenses | message | 0.890 | 0.836 | 0.740 (tickets) | tickets->rooms | 0 / 315 |
| 6 | expenses | with-opening | 0.951 | 0.918 | 0.885 (invoices) | scheduling->drafting | 4 / 345 |
| 6 | expenses | episode | 0.961 | 0.934 | 0.940 (expenses) | scheduling->drafting | 0 / 344 |
| 7 | approvals | message | 0.880 | 0.819 | 0.712 (tickets) | tickets->rooms | 7 / 365 |
| 7 | approvals | with-opening | 0.943 | 0.902 | 0.852 (invoices) | invoices->expenses | 1 / 390 |
| 7 | approvals | episode | 0.950 | 0.913 | 0.923 (expenses) | scheduling->drafting | 0 / 394 |
| 8 | reminders | message | 0.857 | 0.781 | 0.722 (tickets) | tickets->rooms | 10 / 419 |
| 8 | reminders | with-opening | 0.939 | 0.894 | 0.893 (invoices) | scheduling->drafting | 0 / 449 |
| 8 | reminders | episode | 0.910 | 0.868 | 0.655 (rooms) | rooms->reminders | 15 / 452 |
| 9 | summaries | message | 0.791 | 0.705 | 0.467 (rooms) | approvals->summaries | 52 / 467 |
| 9 | summaries | with-opening | 0.927 | 0.870 | 0.862 (invoices) | scheduling->drafting | 2 / 512 |
| 9 | summaries | episode | 0.883 | 0.832 | 0.467 (rooms) | rooms->reminders | 11 / 496 |
| 10 | travel | message | 0.803 | 0.714 | 0.562 (approvals) | approvals->summaries | 1 / 489 |
| 10 | travel | with-opening | 0.918 | 0.854 | 0.867 (invoices) | scheduling->drafting | 4 / 573 |
| 10 | travel | episode | 0.894 | 0.828 | 0.677 (rooms) | rooms->reminders | 0 / 546 |
| 11 | inventory | message | 0.811 | 0.722 | 0.561 (approvals) | approvals->summaries | 4 / 551 |
| 11 | inventory | with-opening | 0.908 | 0.833 | 0.806 (invoices) | scheduling->drafting | 3 / 630 |
| 11 | inventory | episode | 0.888 | 0.811 | 0.719 (rooms) | rooms->reminders | 0 / 613 |
| 12 | timesheets | message | 0.799 | 0.706 | 0.574 (approvals) | approvals->summaries | 7 / 625 |
| 12 | timesheets | with-opening | 0.890 | 0.798 | 0.812 (invoices) | scheduling->drafting | 6 / 700 |
| 12 | timesheets | episode | 0.879 | 0.790 | 0.742 (rooms) | rooms->reminders | 0 / 685 |
| 13 | contacts | message | 0.780 | 0.678 | 0.586 (approvals) | contacts->summaries | 14 / 681 |
| 13 | contacts | with-opening | 0.884 | 0.784 | 0.803 (invoices) | reminders->contacts | 2 / 758 |
| 13 | contacts | episode | 0.874 | 0.771 | 0.794 (rooms) | rooms->reminders | 0 / 749 |
| 14 | search | message | 0.769 | 0.664 | 0.479 (search) | contacts->summaries | 0 / 739 |
| 14 | search | with-opening | 0.867 | 0.757 | 0.648 (search) | search->timesheets | 0 / 837 |
| 14 | search | episode | 0.827 | 0.727 | 0.268 (search) | search->timesheets | 0 / 828 |

## centroids-refit, unseen

| routes | added | strategy | turn acc | episode acc | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | invoices | message | 1.000 | 1.000 | 1.000 (invoices) | – | 0 / 0 |
| 3 | invoices | with-opening | 1.000 | 1.000 | 1.000 (invoices) | – | 0 / 0 |
| 3 | invoices | episode | 1.000 | 1.000 | 1.000 (invoices) | – | 0 / 0 |
| 4 | tickets | message | 0.800 | 0.680 | 0.700 (invoices) | invoices->tickets | 15 / 48 |
| 4 | tickets | with-opening | 0.840 | 0.780 | 0.740 (invoices) | invoices->tickets | 12 / 48 |
| 4 | tickets | episode | 0.730 | 0.720 | 0.720 (invoices) | invoices->tickets | 13 / 48 |
| 5 | rooms | message | 0.716 | 0.538 | 0.615 (invoices) | invoices->rooms | 16 / 80 |
| 5 | rooms | with-opening | 0.806 | 0.705 | 0.577 (invoices) | invoices->rooms | 11 / 84 |
| 5 | rooms | episode | 0.729 | 0.692 | 0.538 (invoices) | invoices->rooms | 11 / 73 |
| 6 | expenses | message | 0.652 | 0.435 | 0.296 (tickets) | tickets->expenses | 19 / 111 |
| 6 | expenses | with-opening | 0.751 | 0.602 | 0.537 (invoices) | invoices->expenses | 10 / 125 |
| 6 | expenses | episode | 0.629 | 0.574 | 0.056 (tickets) | tickets->expenses | 34 / 113 |
| 7 | approvals | message | 0.553 | 0.343 | 0.071 (tickets) | tickets->expenses | 24 / 144 |
| 7 | approvals | with-opening | 0.676 | 0.543 | 0.518 (invoices) | invoices->expenses | 10 / 166 |
| 7 | approvals | episode | 0.574 | 0.514 | 0.000 (tickets) | tickets->expenses | 6 / 139 |
| 8 | reminders | message | 0.537 | 0.322 | 0.069 (tickets) | rooms->reminders | 25 / 157 |
| 8 | reminders | with-opening | 0.684 | 0.534 | 0.414 (tickets) | approvals->reminders | 10 / 192 |
| 8 | reminders | episode | 0.525 | 0.460 | 0.000 (tickets) | rooms->reminders | 36 / 163 |
| 9 | summaries | message | 0.445 | 0.233 | 0.067 (tickets) | rooms->reminders | 50 / 190 |
| 9 | summaries | with-opening | 0.704 | 0.562 | 0.367 (tickets) | invoices->expenses | 6 / 242 |
| 9 | summaries | episode | 0.488 | 0.414 | 0.000 (tickets) | rooms->reminders | 15 / 186 |
| 10 | travel | message | 0.462 | 0.262 | 0.065 (tickets) | rooms->reminders | 1 / 188 |
| 10 | travel | with-opening | 0.705 | 0.581 | 0.387 (tickets) | travel->rooms | 2 / 297 |
| 10 | travel | episode | 0.526 | 0.448 | 0.000 (tickets) | rooms->reminders | 2 / 206 |
| 11 | inventory | message | 0.518 | 0.323 | 0.062 (tickets) | tickets->expenses | 7 / 230 |
| 11 | inventory | with-opening | 0.712 | 0.573 | 0.375 (tickets) | travel->rooms | 7 / 351 |
| 11 | inventory | episode | 0.584 | 0.490 | 0.000 (tickets) | tickets->expenses | 1 / 262 |
| 12 | timesheets | message | 0.508 | 0.327 | 0.061 (tickets) | travel->rooms | 18 / 299 |
| 12 | timesheets | with-opening | 0.680 | 0.539 | 0.167 (tickets) | tickets->timesheets | 31 / 411 |
| 12 | timesheets | episode | 0.581 | 0.491 | 0.000 (tickets) | tickets->expenses | 4 / 337 |
| 13 | contacts | message | 0.456 | 0.275 | 0.045 (contacts) | contacts->reminders | 9 / 334 |
| 13 | contacts | with-opening | 0.641 | 0.484 | 0.147 (tickets) | contacts->timesheets | 3 / 447 |
| 13 | contacts | episode | 0.525 | 0.433 | 0.000 (contacts) | contacts->reminders | 0 / 382 |
| 14 | search | message | 0.441 | 0.252 | 0.044 (contacts) | search->inventory | 1 / 339 |
| 14 | search | with-opening | 0.589 | 0.426 | 0.090 (search) | contacts->timesheets | 3 / 476 |
| 14 | search | episode | 0.488 | 0.393 | 0.000 (contacts) | search->inventory | 0 / 390 |

## centroids-frozen, test

| routes | added | strategy | turn acc | episode acc | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | scheduling | message | 0.926 | 0.890 | 0.920 (drafting) | drafting->scheduling | – |
| 2 | scheduling | with-opening | 0.980 | 0.971 | 0.962 (scheduling) | scheduling->drafting | – |
| 2 | scheduling | episode | 0.980 | 0.971 | 0.962 (scheduling) | scheduling->drafting | – |
| 3 | invoices | message | 0.856 | 0.775 | 0.820 (drafting) | drafting->invoices | 21 / 189 |
| 3 | invoices | with-opening | 0.972 | 0.956 | 0.962 (scheduling) | scheduling->drafting | 3 / 200 |
| 3 | invoices | episode | 0.984 | 0.975 | 0.962 (scheduling) | scheduling->drafting | 0 / 200 |
| 4 | tickets | message | 0.842 | 0.747 | 0.810 (drafting) | drafting->tickets | 9 / 214 |
| 4 | tickets | with-opening | 0.950 | 0.919 | 0.920 (drafting) | drafting->tickets | 6 / 243 |
| 4 | tickets | episode | 0.980 | 0.968 | 0.962 (scheduling) | scheduling->drafting | 0 / 246 |
| 5 | rooms | message | 0.808 | 0.701 | 0.729 (tickets) | tickets->rooms | 16 / 251 |
| 5 | rooms | with-opening | 0.912 | 0.855 | 0.827 (scheduling) | scheduling->rooms | 13 / 283 |
| 5 | rooms | episode | 0.972 | 0.953 | 0.958 (tickets) | scheduling->drafting | 0 / 292 |
| 6 | expenses | message | 0.812 | 0.705 | 0.692 (invoices) | tickets->rooms | 3 / 286 |
| 6 | expenses | with-opening | 0.907 | 0.844 | 0.827 (scheduling) | scheduling->rooms | 4 / 323 |
| 6 | expenses | episode | 0.954 | 0.926 | 0.885 (invoices) | scheduling->drafting | 3 / 344 |
| 7 | approvals | message | 0.773 | 0.649 | 0.643 (rooms) | drafting->approvals | 32 / 333 |
| 7 | approvals | with-opening | 0.897 | 0.822 | 0.827 (scheduling) | scheduling->rooms | 7 / 372 |
| 7 | approvals | episode | 0.945 | 0.909 | 0.889 (invoices) | scheduling->drafting | 0 / 391 |
| 8 | reminders | message | 0.760 | 0.619 | 0.655 (rooms) | drafting->approvals | 4 / 368 |
| 8 | reminders | with-opening | 0.899 | 0.823 | 0.827 (scheduling) | scheduling->rooms | 0 / 427 |
| 8 | reminders | episode | 0.934 | 0.887 | 0.875 (invoices) | scheduling->drafting | 0 / 450 |
| 9 | summaries | message | 0.749 | 0.607 | 0.643 (tickets) | drafting->approvals | 16 / 414 |
| 9 | summaries | with-opening | 0.883 | 0.792 | 0.827 (scheduling) | drafting->summaries | 8 / 490 |
| 9 | summaries | episode | 0.922 | 0.864 | 0.862 (invoices) | scheduling->drafting | 0 / 509 |
| 10 | travel | message | 0.757 | 0.617 | 0.655 (tickets) | drafting->approvals | 3 / 463 |
| 10 | travel | with-opening | 0.876 | 0.779 | 0.788 (scheduling) | scheduling->travel | 5 / 546 |
| 10 | travel | episode | 0.913 | 0.844 | 0.870 (travel) | scheduling->drafting | 0 / 570 |
| 11 | inventory | message | 0.767 | 0.627 | 0.667 (tickets) | drafting->approvals | 4 / 519 |
| 11 | inventory | with-opening | 0.872 | 0.767 | 0.798 (scheduling) | scheduling->travel | 1 / 601 |
| 11 | inventory | episode | 0.901 | 0.821 | 0.857 (travel) | scheduling->drafting | 0 / 626 |
| 12 | timesheets | message | 0.763 | 0.627 | 0.652 (rooms) | scheduling->travel | 9 / 591 |
| 12 | timesheets | with-opening | 0.864 | 0.751 | 0.808 (scheduling) | scheduling->travel | 1 / 672 |
| 12 | timesheets | episode | 0.890 | 0.798 | 0.845 (travel) | scheduling->drafting | 0 / 695 |
| 13 | contacts | message | 0.752 | 0.610 | 0.646 (timesheets) | timesheets->contacts | 12 / 650 |
| 13 | contacts | with-opening | 0.857 | 0.735 | 0.808 (scheduling) | scheduling->travel | 6 / 736 |
| 13 | contacts | episode | 0.873 | 0.771 | 0.787 (contacts) | contacts->reminders | 0 / 758 |
| 14 | search | message | 0.745 | 0.594 | 0.563 (search) | search->timesheets | 2 / 712 |
| 14 | search | with-opening | 0.831 | 0.694 | 0.535 (search) | search->timesheets | 1 / 812 |
| 14 | search | episode | 0.826 | 0.727 | 0.268 (search) | search->timesheets | 0 / 827 |

## centroids-frozen, unseen

| routes | added | strategy | turn acc | episode acc | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | invoices | message | 1.000 | 1.000 | 1.000 (invoices) | – | 0 / 0 |
| 3 | invoices | with-opening | 1.000 | 1.000 | 1.000 (invoices) | – | 0 / 0 |
| 3 | invoices | episode | 1.000 | 1.000 | 1.000 (invoices) | – | 0 / 0 |
| 4 | tickets | message | 0.860 | 0.800 | 0.720 (invoices) | invoices->tickets | 14 / 48 |
| 4 | tickets | with-opening | 0.910 | 0.860 | 0.820 (invoices) | invoices->tickets | 8 / 48 |
| 4 | tickets | episode | 0.860 | 0.840 | 0.740 (invoices) | invoices->tickets | 12 / 48 |
| 5 | rooms | message | 0.774 | 0.641 | 0.615 (invoices) | invoices->rooms | 14 / 86 |
| 5 | rooms | with-opening | 0.877 | 0.821 | 0.673 (invoices) | invoices->rooms | 7 / 91 |
| 5 | rooms | episode | 0.826 | 0.795 | 0.538 (invoices) | invoices->rooms | 9 / 86 |
| 6 | expenses | message | 0.715 | 0.537 | 0.574 (invoices) | invoices->expenses | 13 / 120 |
| 6 | expenses | with-opening | 0.846 | 0.778 | 0.593 (invoices) | invoices->expenses | 7 / 136 |
| 6 | expenses | episode | 0.738 | 0.685 | 0.463 (invoices) | invoices->expenses | 25 / 128 |
| 7 | approvals | message | 0.581 | 0.350 | 0.339 (tickets) | rooms->approvals | 40 / 158 |
| 7 | approvals | with-opening | 0.785 | 0.664 | 0.571 (invoices) | invoices->expenses | 9 / 187 |
| 7 | approvals | episode | 0.680 | 0.621 | 0.446 (invoices) | invoices->expenses | 0 / 163 |
| 8 | reminders | message | 0.571 | 0.339 | 0.310 (tickets) | rooms->approvals | 15 / 165 |
| 8 | reminders | with-opening | 0.763 | 0.638 | 0.534 (invoices) | invoices->expenses | 13 / 223 |
| 8 | reminders | episode | 0.667 | 0.592 | 0.431 (invoices) | invoices->expenses | 15 / 193 |
| 9 | summaries | message | 0.540 | 0.314 | 0.317 (tickets) | invoices->expenses | 15 / 202 |
| 9 | summaries | with-opening | 0.742 | 0.600 | 0.533 (invoices) | invoices->expenses | 0 / 270 |
| 9 | summaries | episode | 0.628 | 0.538 | 0.411 (summaries) | invoices->expenses | 1 / 236 |
| 10 | travel | message | 0.514 | 0.290 | 0.306 (tickets) | tickets->expenses | 3 / 228 |
| 10 | travel | with-opening | 0.705 | 0.569 | 0.516 (invoices) | invoices->expenses | 1 / 313 |
| 10 | travel | episode | 0.592 | 0.508 | 0.355 (invoices) | invoices->expenses | 5 / 265 |
| 11 | inventory | message | 0.562 | 0.354 | 0.312 (tickets) | tickets->expenses | 5 / 256 |
| 11 | inventory | with-opening | 0.721 | 0.587 | 0.516 (invoices) | invoices->expenses | 4 / 351 |
| 11 | inventory | episode | 0.633 | 0.531 | 0.400 (summaries) | tickets->expenses | 2 / 295 |
| 12 | timesheets | message | 0.545 | 0.339 | 0.333 (tickets) | tickets->expenses | 7 / 324 |
| 12 | timesheets | with-opening | 0.694 | 0.542 | 0.500 (tickets) | invoices->expenses | 24 / 416 |
| 12 | timesheets | episode | 0.594 | 0.488 | 0.274 (timesheets) | invoices->expenses | 2 / 365 |
| 13 | contacts | message | 0.498 | 0.291 | 0.136 (contacts) | contacts->reminders | 12 / 358 |
| 13 | contacts | with-opening | 0.638 | 0.457 | 0.318 (contacts) | invoices->expenses | 11 / 456 |
| 13 | contacts | episode | 0.541 | 0.436 | 0.000 (contacts) | contacts->reminders | 1 / 390 |
| 14 | search | message | 0.478 | 0.255 | 0.132 (contacts) | contacts->reminders | 3 / 370 |
| 14 | search | with-opening | 0.600 | 0.402 | 0.284 (search) | search->inventory | 4 / 474 |
| 14 | search | episode | 0.504 | 0.395 | 0.000 (contacts) | contacts->reminders | 0 / 402 |

## logistic-refit, test

| routes | added | strategy | turn acc | episode acc | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | scheduling | message | 1.000 | 1.000 | 1.000 (drafting) | – | – |
| 2 | scheduling | with-opening | 1.000 | 1.000 | 1.000 (drafting) | – | – |
| 2 | scheduling | episode | 0.980 | 0.971 | 0.962 (scheduling) | scheduling->drafting | – |
| 3 | invoices | message | 1.000 | 1.000 | 1.000 (drafting) | – | 0 / 204 |
| 3 | invoices | with-opening | 1.000 | 1.000 | 1.000 (drafting) | – | 0 / 204 |
| 3 | invoices | episode | 0.984 | 0.975 | 0.962 (scheduling) | scheduling->drafting | 0 / 200 |
| 4 | tickets | message | 0.966 | 0.946 | 0.875 (invoices) | invoices->tickets | 6 / 250 |
| 4 | tickets | with-opening | 0.993 | 0.989 | 0.978 (tickets) | tickets->invoices | 0 / 250 |
| 4 | tickets | episode | 0.980 | 0.968 | 0.962 (scheduling) | scheduling->drafting | 0 / 246 |
| 5 | rooms | message | 0.949 | 0.921 | 0.800 (invoices) | invoices->tickets | 5 / 288 |
| 5 | rooms | with-opening | 0.994 | 0.991 | 0.960 (invoices) | invoices->tickets | 0 / 296 |
| 5 | rooms | episode | 0.972 | 0.953 | 0.958 (tickets) | scheduling->drafting | 0 / 292 |
| 6 | expenses | message | 0.941 | 0.914 | 0.788 (invoices) | invoices->tickets | 1 / 336 |
| 6 | expenses | with-opening | 0.990 | 0.984 | 0.960 (expenses) | invoices->tickets | 0 / 352 |
| 6 | expenses | episode | 0.961 | 0.934 | 0.940 (expenses) | scheduling->drafting | 0 / 344 |
| 7 | approvals | message | 0.926 | 0.891 | 0.759 (invoices) | invoices->approvals | 8 / 386 |
| 7 | approvals | with-opening | 0.994 | 0.989 | 0.963 (invoices) | tickets->approvals | 0 / 406 |
| 7 | approvals | episode | 0.950 | 0.913 | 0.923 (expenses) | scheduling->drafting | 0 / 394 |
| 8 | reminders | message | 0.919 | 0.874 | 0.768 (invoices) | invoices->approvals | 1 / 441 |
| 8 | reminders | with-opening | 0.983 | 0.971 | 0.949 (reminders) | tickets->approvals | 1 / 473 |
| 8 | reminders | episode | 0.938 | 0.890 | 0.907 (expenses) | scheduling->drafting | 0 / 452 |
| 9 | summaries | message | 0.914 | 0.864 | 0.776 (invoices) | invoices->approvals | 0 / 501 |
| 9 | summaries | with-opening | 0.987 | 0.977 | 0.951 (reminders) | tickets->approvals | 0 / 536 |
| 9 | summaries | episode | 0.926 | 0.867 | 0.893 (expenses) | scheduling->drafting | 0 / 511 |
| 10 | travel | message | 0.914 | 0.862 | 0.783 (invoices) | invoices->approvals | 0 / 565 |
| 10 | travel | with-opening | 0.987 | 0.977 | 0.952 (reminders) | tickets->approvals | 1 / 610 |
| 10 | travel | episode | 0.913 | 0.844 | 0.870 (travel) | scheduling->drafting | 0 / 572 |
| 11 | inventory | message | 0.900 | 0.842 | 0.790 (invoices) | approvals->inventory | 8 / 627 |
| 11 | inventory | with-opening | 0.988 | 0.979 | 0.954 (reminders) | tickets->approvals | 0 / 677 |
| 11 | inventory | episode | 0.901 | 0.821 | 0.857 (travel) | scheduling->drafting | 0 / 626 |
| 12 | timesheets | message | 0.903 | 0.845 | 0.797 (invoices) | approvals->inventory | 0 / 694 |
| 12 | timesheets | with-opening | 0.985 | 0.972 | 0.955 (reminders) | timesheets->reminders | 0 / 762 |
| 12 | timesheets | episode | 0.890 | 0.798 | 0.845 (travel) | scheduling->drafting | 0 / 695 |
| 13 | contacts | message | 0.902 | 0.843 | 0.803 (invoices) | approvals->inventory | 0 / 769 |
| 13 | contacts | with-opening | 0.987 | 0.976 | 0.969 (expenses) | timesheets->reminders | 0 / 839 |
| 13 | contacts | episode | 0.880 | 0.776 | 0.833 (travel) | scheduling->drafting | 0 / 758 |
| 14 | search | message | 0.903 | 0.844 | 0.809 (invoices) | approvals->inventory | 3 / 854 |
| 14 | search | with-opening | 0.987 | 0.975 | 0.955 (timesheets) | timesheets->search | 0 / 935 |
| 14 | search | episode | 0.869 | 0.755 | 0.823 (travel) | scheduling->drafting | 0 / 833 |

## logistic-refit, unseen

| routes | added | strategy | turn acc | episode acc | min recall (route) | worst confusion | lost / kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | invoices | message | 1.000 | 1.000 | 1.000 (invoices) | – | 0 / 0 |
| 3 | invoices | with-opening | 1.000 | 1.000 | 1.000 (invoices) | – | 0 / 0 |
| 3 | invoices | episode | 1.000 | 1.000 | 1.000 (invoices) | – | 0 / 0 |
| 4 | tickets | message | 0.830 | 0.720 | 0.820 (invoices) | invoices->tickets | 9 / 48 |
| 4 | tickets | with-opening | 0.990 | 0.980 | 0.980 (tickets) | tickets->invoices | 0 / 48 |
| 4 | tickets | episode | 0.830 | 0.820 | 0.800 (invoices) | invoices->tickets | 9 / 48 |
| 5 | rooms | message | 0.794 | 0.628 | 0.673 (invoices) | invoices->rooms | 13 / 83 |
| 5 | rooms | with-opening | 0.987 | 0.974 | 0.981 (invoices) | tickets->invoices | 0 / 99 |
| 5 | rooms | episode | 0.787 | 0.756 | 0.673 (invoices) | invoices->rooms | 20 / 83 |
| 6 | expenses | message | 0.697 | 0.481 | 0.463 (invoices) | invoices->expenses | 24 / 123 |
| 6 | expenses | with-opening | 0.887 | 0.796 | 0.630 (invoices) | invoices->expenses | 18 / 153 |
| 6 | expenses | episode | 0.633 | 0.593 | 0.222 (invoices) | invoices->expenses | 47 / 122 |
| 7 | approvals | message | 0.627 | 0.400 | 0.393 (invoices) | invoices->expenses | 13 / 154 |
| 7 | approvals | with-opening | 0.873 | 0.771 | 0.607 (invoices) | invoices->expenses | 4 / 196 |
| 7 | approvals | episode | 0.535 | 0.493 | 0.107 (invoices) | invoices->expenses | 7 / 140 |
| 8 | reminders | message | 0.633 | 0.402 | 0.345 (tickets) | invoices->expenses | 18 / 178 |
| 8 | reminders | with-opening | 0.856 | 0.741 | 0.586 (invoices) | invoices->expenses | 7 / 248 |
| 8 | reminders | episode | 0.551 | 0.489 | 0.138 (invoices) | invoices->expenses | 20 / 152 |
| 9 | summaries | message | 0.666 | 0.452 | 0.300 (tickets) | invoices->expenses | 3 / 224 |
| 9 | summaries | with-opening | 0.863 | 0.757 | 0.583 (invoices) | invoices->expenses | 2 / 303 |
| 9 | summaries | episode | 0.588 | 0.519 | 0.200 (tickets) | invoices->expenses | 5 / 195 |
| 10 | travel | message | 0.687 | 0.476 | 0.355 (tickets) | invoices->expenses | 5 / 281 |
| 10 | travel | with-opening | 0.857 | 0.754 | 0.613 (invoices) | invoices->expenses | 2 / 364 |
| 10 | travel | episode | 0.627 | 0.548 | 0.148 (approvals) | invoices->expenses | 6 / 248 |
| 11 | inventory | message | 0.693 | 0.503 | 0.375 (tickets) | invoices->expenses | 10 / 342 |
| 11 | inventory | with-opening | 0.865 | 0.757 | 0.594 (invoices) | invoices->expenses | 4 / 427 |
| 11 | inventory | episode | 0.633 | 0.542 | 0.143 (approvals) | invoices->expenses | 13 / 312 |
| 12 | timesheets | message | 0.740 | 0.558 | 0.446 (approvals) | invoices->expenses | 6 / 400 |
| 12 | timesheets | with-opening | 0.836 | 0.718 | 0.636 (invoices) | invoices->expenses | 30 / 499 |
| 12 | timesheets | episode | 0.680 | 0.579 | 0.185 (approvals) | invoices->expenses | 8 / 365 |
| 13 | contacts | message | 0.728 | 0.540 | 0.471 (tickets) | invoices->expenses | 19 / 486 |
| 13 | contacts | with-opening | 0.840 | 0.725 | 0.632 (invoices) | invoices->expenses | 8 / 549 |
| 13 | contacts | episode | 0.664 | 0.556 | 0.194 (approvals) | invoices->expenses | 19 / 447 |
| 14 | search | message | 0.733 | 0.550 | 0.471 (tickets) | invoices->expenses | 5 / 541 |
| 14 | search | with-opening | 0.817 | 0.686 | 0.627 (search) | invoices->expenses | 10 / 624 |
| 14 | search | episode | 0.645 | 0.533 | 0.246 (approvals) | invoices->expenses | 3 / 493 |
