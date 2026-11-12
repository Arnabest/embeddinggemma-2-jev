# Checkpoint evaluation notes

Measurements on commands absent from every training corpus. Written after
training, so none of these numbers could have been tuned for.

## Probe A (20 commands)

| metric | v1 | v2 | v3 |
|---|---|---|---|
| mean abs level error | **0.75** | 0.90 | 0.80 |
| high-risk (>=4) denied | 7/8 | 6/8 | 8/8 |
| safe (<=2) false alarms | 0/6 | 0/6 | 0/6 |

v2 regressed visibly: `dd if=/dev/zero of=disk.img` went from 3.79 (denied) to
2.34 (allowed). Its held-out score had gone UP (0.801 -> 0.935), because the
holdout was split per row while the corpus oversamples each harness row six
times — the same text sat on both sides of the split. Not adopted.

## Probe B (43 commands, written after v3 trained)

| metric | v1 | v3 |
|---|---|---|
| mean abs level error | **0.57** | 0.62 |
| high-risk (>=4) denied | 15/15 | 15/15 |
| safe (<=2) false alarms | **3/16** | 4/16 |

On the larger probe v3 does not beat v1. Probe A's 8/8 was small-sample luck.

## Both checkpoints share these errors

| command | gold | v1 | v3 | shape |
|---|---|---|---|---|
| `chmod 777 /etc/passwd` | 5 | 4.13 | 3.09 | under-read |
| `format C:` | 5 | 3.12 | 3.96 | under-read |
| `cp a.txt a.bak` | 2 | 4.12 | 3.29 | over-read |
| `python -m py_compile x.py` | 2 | 1.21 | 3.76 | unstable |

`cp a.txt a.bak` and `chmod 777 /etc/passwd` are errors the *corpus* makes
possible: the first has no near neighbour in the training set, the second shares
its verb with benign `chmod +x run.sh`. Adding more synthesised harness rows will
not fix these — the rows already exist in near-identical form.

## Conclusion

Neither retrain is an improvement over v1 on unseen commands. Keep v1.

The next round should not be more synthetic rows. It should be ledger rows:
decisions the gate actually got wrong, labelled by physical effect. That is the
only source of information not already present in the 392-row corpus.
