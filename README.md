# Markov Chains and Their Evolution

A small, from-scratch tour of the Markov chain as a model of text: from Markov's
1913 vowel/consonant count, through Shannon's n-gram approximations of English,
to the order-N chains with backoff that powered language modeling for decades.

**The task**: generate mythology names (Greek, Norse, Celtic) one character at a time.

| Step | Model | Key idea | File |
|------|-------|----------|------|
| 1 | Order-1 Markov chain | Count letter pairs, sample from the counts | `name_generator.py` |
| 2 | Order-N Markov chain | Use the last N characters as the state; back off when a context was never seen | `name_generator_higher_order.py` |

The explainer page (`index.html`) puts these in historical context (Markov, Shannon,
the smoothing methods of the n-gram era, and other branches like MCMC, HMMs and
PageRank) and has a live playground: an order slider, a copy-rate measurement,
and a view of the next-character distribution for any context.

Pure Python standard library. No dependencies.

## Quick start

```bash
python3 name_generator.py              # order-1
python3 name_generator_higher_order.py # orders 1-3 side by side, with backoff
```

Browser playground (needs a local server so it can load the JSON):

```bash
python3 -m http.server 8888
```

Then open `http://localhost:8888/index.html`.

## Regenerating the chain data

`markov_chains.json` holds the order-1 to order-4 chains for each tradition plus the
training names (used to measure how often each order copies one verbatim):

```bash
python3 -c "import name_generator; name_generator.export_markov_chains()"
```

## What the order buys

Raising the order trades gibberish for memorization. Rough numbers for the Greek set
(363 names), from the playground's 500-sample measurement:

| Order | Contexts | Copies of training names |
|------:|---------:|-------------------------:|
| 1 | ~40 | ~0% |
| 2 | ~365 | ~2% |
| 3 | ~1,220 | ~8% |
| 4 | ~2,120 | ~15% |

That trade-off, and the fact that no fixed window can remember anything older than
N characters, is where this repo stops.

## Where the story continues

Neural sequence models (RNN, LSTM, transformer) used to live here too, generating
the same names. They were removed: name generation is the right showcase for a
Markov chain, but it exercises almost none of what those models are for. They are
in this repo's history (the initial commit) for anyone curious.

## Training data

- **Greek** (363 names): Achilles, Persephone, Prometheus, Eurydice...
- **Norse** (169 names): Aegir, Brunhild, Fenric, Yggdrasil...
- **Celtic** (168 names): Abandinus, Branwen, Cernunnos, Taliesin...

Source: [repushko/mythology_names_dataset](https://github.com/repushko/mythology_names_dataset)
(CC0), parsed from [Godchecker](https://www.godchecker.com/).

## Inspiration

The order-1 generator follows [donjon's Markov Name Generator](https://donjon.bin.sh/code/name/),
a public-domain RPG name generator that builds frequency tables for name length and
letter pairs from sample names.

## License

Code is MIT licensed (see `LICENSE`). The training names come from a CC0 dataset.
