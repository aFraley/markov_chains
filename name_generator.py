"""
=============================================================================
Markov Chain Name Generator — Python Edition
=============================================================================

WHAT IS A MARKOV CHAIN?
-----------------------
A Markov chain is a statistical model where the probability of the NEXT state
depends ONLY on the CURRENT state — not on the sequence of states before it.
This is called the "Markov property" or "memorylessness."

In this name generator, each CHARACTER is a state. Given the current letter,
we look at a probability distribution of what letter comes next, learned from
real training names.

HOW THIS COMPARES TO THE JS VERSION:
-------------------------------------
This is a 1:1 port of the JavaScript implementation. The algorithm is identical:
  1. Build a chain by counting character transitions from training names
  2. Scale counts with a power function (a "temperature" knob)
  3. Generate names by weighted random sampling from the chain

Python dicts behave like JS objects here — both are hash maps under the hood.
The main difference is Python gives us defaultdict, random.choices, and other
niceties that make the code more concise.

THE EVOLUTION FROM HERE:
------------------------
  Order-1 Markov Chain (this file)
    -> You're manually building a transition table from data
    -> Context is the last 1 char only
    -> No "learning" in the gradient sense: just counting

  Higher-Order Markov (name_generator_higher_order.py)
    -> Use last N chars as context instead of just 1
    -> Better results, but the state space explodes: 26^N possible contexts
    -> Unseen contexts handled by BACKOFF to shorter contexts

  Smoothed n-gram models (Katz 1987, Kneser-Ney 1995)
    -> Principled ways to share probability mass with unseen contexts
    -> The state of the art in language modeling until ~2010

Every step is still a fixed window of the last N symbols. What comes after
that window (RNNs, LSTMs, transformers) is outside the scope of this repo.
=============================================================================
"""

import math
import random
from collections import defaultdict


# =============================================================================
# TRAINING DATA
# =============================================================================
# Mythology names from three traditions with maximally different phonetics.
# Greek: vowel-heavy, -eus/-os/-ne endings (363 names)
# Norse: hard consonants, compact compounds (169 names)
# Celtic: consonant clusters, gw/ll/ch patterns (168 names)
#
# Source: repushko/mythology_names_dataset (CC0 public domain)
#         parsed from godchecker.com
#
# Original Markov chain name generation concept inspired by donjon
# (https://donjon.bin.sh/code/name/) — a public domain RPG name generator.
# =============================================================================

NAME_SETS = {
    # Greek mythology — vowel-rich, classical suffixes (-eus, -os, -ia, -ne)
    "greek": [
        "Achelois", "Achelous", "Acheron", "Achilles", "Acidalia", "Adamanthea",
        "Adastreia", "Adephagia", "Adonis", "Aeacos", "Aedos", "Aegaeon",
        "Aegina", "Aegle", "Aello", "Aeolos", "Aer", "Aesculapius",
        "Aethalides", "Aether", "Aethon", "Aetna", "Agave", "Agdistis",
        "Aglaea", "Aglauros", "Agrotara", "Alcides", "Alcmena", "Alcyone",
        "Alecto", "Alectrona", "Alexandra", "Aloadae", "Alpheos", "Amalthea",
        "Ampelos", "Amphion", "Amphitrite", "Amphitryon", "Amymone", "Ananke",
        "Andromeda", "Antaeus", "Anteros", "Anticlea", "Antiope", "Apate",
        "Apollo", "Arachne", "Arcas", "Areon", "Ares", "Arethusa",
        "Argeos", "Argus", "Ariadne", "Arion", "Aristeas", "Asopus",
        "Asteria", "Astraea", "Astraeus", "Atalanta", "Ate", "Athamas",
        "Athena", "Atlantides", "Atlas", "Atropos", "Attis", "Aurai",
        "Autolycus", "Auxesia", "Bacchae", "Balios", "Bellerophon", "Bia",
        "Boreads", "Boreas", "Bromios", "Cadmus", "Caeneus", "Calchas",
        "Calliope", "Callisto", "Calypso", "Castor", "Cecrops", "Celaeno",
        "Ceneus", "Cerberus", "Cercopes", "Ceto", "Chaos", "Charites",
        "Charon", "Charybdis", "Cheiron", "Chelone", "Chimaera", "Chione",
        "Chloris", "Chronos", "Chthonia", "Circe", "Clio", "Clotho",
        "Clymene", "Coeus", "Coltus", "Comus", "Cotys", "Cratus",
        "Crius", "Cronos", "Cybele", "Cyclopes", "Cyclops", "Cyrene",
        "Danae", "Daphnaie", "Deimos", "Deino", "Delphine", "Demphredo",
        "Despoena", "Deucalion", "Dice", "Dione", "Dioscuri", "Doris",
        "Dryades", "Echidna", "Echo", "Eileithyia", "Eirene", "Electra",
        "Elpis", "Empousa", "Enosichthon", "Enyo", "Eos", "Epaphos",
        "Ephialtes", "Epimeliades", "Epimetheus", "Epiona", "Erato", "Erebos",
        "Erechtheus", "Erichthoneus", "Erinyes", "Erinys", "Eris", "Eros",
        "Erytheia", "Eunomia", "Euphrosyne", "Europa", "Euros", "Euryale",
        "Eurybia", "Eurydice", "Eurynome", "Eurystheus", "Euterpe", "Fates",
        "Galeotes", "Ganymede", "Geryon", "Glaucus", "Gorgons", "Graeae",
        "Gyes", "Hades", "Hamadryades", "Harmonia", "Harpies", "Harpocrates",
        "Hebe", "Hecate", "Hecatoncheires", "Helen", "Heliades", "Helios",
        "Hemera", "Hephaestus", "Hera", "Hermaphroditos", "Hermes", "Hespera",
        "Hesperides", "Hesperis", "Hestia", "Hippolyta", "Hippolytos", "Horae",
        "Hyacinth", "Hyades", "Hydra", "Hydriades", "Hygeia", "Hymen",
        "Hyperion", "Hypnos", "Iacchus", "Iambe", "Iapetos", "Icelos",
        "Inachus", "Ino", "Ion", "Iphicles", "Iris", "Keres",
        "Kore", "Lachesis", "Laertes", "Lamia", "Lampetia", "Leda",
        "Leimoniades", "Lelantos", "Lethe", "Leto", "Linus", "Maia",
        "Maiandros", "Maliades", "Medea", "Medousa", "Melpomene", "Memnon",
        "Menoetius", "Merope", "Metis", "Minos", "Minotaur", "Mnemosyne",
        "Momos", "Mopsus", "Moros", "Morpheus", "Mousai", "Myiagros",
        "Naiades", "Neaera", "Nemesis", "Nephelai", "Nephele", "Nereides",
        "Nereus", "Nike", "Niobe", "Nix", "Notos", "Nymphai",
        "Oannes", "Oceanides", "Oceanus", "Ocypete", "Odysseus", "Oeager",
        "Oenomaus", "Omphale", "Ondines", "Oreades", "Oreithuia", "Orion",
        "Orpheus", "Orphus", "Ossa", "Otus", "Ourania", "Ouranos",
        "Pallas", "Pan", "Panacea", "Pandora", "Pasiphae", "Pasithea",
        "Pegasos", "Pelops", "Penia", "Persa", "Perses", "Perseus",
        "Perso", "Petesuchos", "Phaethousa", "Phaeton", "Phantasos", "Phema",
        "Philammon", "Philomelus", "Philyra", "Phobos", "Phoebe", "Phorcys",
        "Phospheros", "Pleione", "Ploutos", "Podarge", "Pollux", "Polyhymnia",
        "Pontos", "Poros", "Potamoi", "Priapos", "Prometheus", "Proteus",
        "Psyche", "Pyrrha", "Python", "Rhadamanthus", "Rhea", "Sabazius",
        "Salmoneus", "Sarapis", "Sarpedon", "Scamander", "Scylla", "Seilenos",
        "Seirenes", "Selene", "Semele", "Sibyls", "Sisyphus", "Spercheios",
        "Sphinx", "Sterope", "Stheno", "Styx", "Syrinx", "Tantalus",
        "Tartaros", "Taygete", "Telchines", "Terpsichore", "Tethys", "Thalassa",
        "Thaleia", "Thamyras", "Thanatos", "Thaumas", "Thea", "Thebe",
        "Thelxinoe", "Themis", "Theseus", "Thetis", "Titanes", "Titanides",
        "Tithonus", "Triptolemos", "Triton", "Tyche", "Typhoeus", "Xanthos",
        "Zelos", "Zephyros", "Zethus",
    ],

    # Norse mythology — hard consonants, compact, compound names
    "norse": [
        "Aasgard", "Aegir", "Aesir", "Alfheim", "Amleth", "Amma", "Andhrimnir",
        "Andvari", "Angerbda", "Asabru", "Ask", "Astrild", "Audhumbla",
        "Aurgelmir", "Aurvandil", "Balder", "Besla", "Beyla", "Biort",
        "Blid", "Bor", "Brage", "Brono", "Brunhild", "Buri",
        "Byggvir", "Bylgia", "Disir", "Donar", "Drofn", "Eir",
        "Elli", "Embla", "Fafnir", "Fenric", "Fjorgyn", "Forseti",
        "Frey", "Freya", "Frigg", "Fulla", "Gangleri", "Garm",
        "Gefion", "Geirolul", "Geirrod", "Geirskogul", "Gerd", "Gersemi",
        "Gertrude", "Ginnungagap", "Gjalp", "Gladsheim", "Glut", "Goll",
        "Gondul", "Gotterdammerung", "Greip", "Grid", "Groa", "Gullintani",
        "Gullveig", "Gunlod", "Gur", "Guthorm", "Har", "Hardgreip",
        "Hati", "Havamal", "Heidrun", "Hel", "Helgardh", "Herfjoturr",
        "Hermod", "Hild", "Hladgunnr", "Hlin", "Hlokk", "Hnoss",
        "Hod", "Hoenir", "Hraesvelg", "Hreidmar", "Hrim", "Hrist",
        "Hrungnir", "Hulder", "Hymir", "Hyndla", "Hyrokkin", "Idun",
        "Jafenhar", "Jarnsaaxa", "Jormungand", "Jotunheim", "Kara", "Kvasir",
        "Laufey", "Lif", "Lifthrasir", "Lofn", "Loke", "Magni",
        "Manheimr", "Mani", "Mimir", "Mist", "Mjollner", "Modgud",
        "Modi", "Mokerkialfi", "Muspelheim", "Nanna", "Narfi", "Nehalennia",
        "Nibelungs", "Nidavellir", "Nidhogg", "Niflheim", "Njoerd", "Norns",
        "Nott", "Oller", "Otr", "Radgrid", "Ran", "Ratatosk",
        "Regin", "Rig", "Rind", "Rota", "Saga", "Sibilja",
        "Siegfried", "Sif", "Sigyn", "Sjofn", "Skadi", "Skeggiold",
        "Skirnir", "Skogul", "Skoll", "Skuld", "Sleipner", "Snotra",
        "Sokkvabekk", "Sol", "Surt", "Suttung", "Syn", "Thialfi",
        "Thiassi", "Thridi", "Thrud", "Thrudheim", "Tyr", "Urd",
        "Vafthrudnir", "Vaganhope", "Valhalla", "Vali", "Valkyries", "Vanaheim",
        "Vanir", "Verdandi", "Vidar", "Vili", "Volsung", "Yggdrasil",
    ],

    # Celtic mythology — consonant clusters, Welsh/Irish phonetics
    "celtic": [
        "Abandinus", "Abarta", "Abred", "Aebh", "Aedh", "Aengus", "Afagddu",
        "Agrona", "Aibell", "Aichleach", "Aife", "Aillen", "Aillinn",
        "Aimend", "Aine", "Ainge", "Airmed", "Alisanos", "Amaethon",
        "Amairgen", "Ana", "Andarta", "Andrasta", "Angau", "Anluan",
        "Anociticus", "Araius", "Aranrhod", "Arawen", "Arduinna", "Argetlamh",
        "Arnamentia", "Arthur", "Aywell", "Badb", "Balar", "Balaticaurus",
        "Banba", "Banshee", "Baudihillie", "Bebhinn", "Bel", "Belisama",
        "Beltaine", "Benadonner", "Bendigeidfran", "Birog", "Blaithene",
        "Blodeuedd", "Boand", "Boobach", "Bormana", "Bormanus", "Branwen",
        "Bres", "Brian", "Bricriu", "Brig", "Brigantia", "Buanann",
        "Buggane", "Cael", "Caer", "Caibell", "Cailitin", "Cailleach",
        "Camulos", "Cano", "Caradawg", "Caridwen", "Carman", "Caswallawn",
        "Cathbad", "Cathubodua", "Celtchair", "Cernunnos", "Ciabhan", "Cian",
        "Cigfa", "Cleena", "Clota", "Cocidius", "Coincheann", "Coinchend",
        "Conall", "Conaran", "Condatis", "Conlai", "Conlaoch", "Credne",
        "Creirwy", "Cronnchu", "Culann", "Cumall", "Daire", "Damona",
        "Dechtire", "Deirdre", "Delbchaem", "Diarmait", "Dirona", "Domnu",
        "Don", "Donn", "Dubh", "Dylan", "Eadaoin", "Ealadha",
        "Echdae", "Ecne", "Efnisien", "Eimhear", "Eire", "Eithlinn",
        "Elffin", "Epona", "Esus", "Fand", "Fea", "Findabair",
        "Finegas", "Fodhla", "Fomhoraigh", "Fraoch", "Fuamnach", "Gaibhne",
        "Grannos", "Gwri", "Gwyddien", "Icovellauna", "Ildanach", "Irusan",
        "Laha", "Ler", "Llwyd", "Lugus", "Mabon", "Macha",
        "Maeve", "Maga", "Manandan", "Manawydan", "Math", "Matunos",
        "Merlin", "Midhir", "Mongan", "Nantosuelta", "Neamhan", "Nechtan",
        "Nemglan", "Niam", "Oghma", "Ogmios", "Oisin", "Pwyll",
        "Rhiannon", "Rosmerta", "Ruadan", "Sadb", "Sequana", "Succellos",
        "Taillte", "Taliesin", "Taranis", "Teutates", "Wogan", "Xulsigiae",
    ],
}


# =============================================================================
# CHAIN CACHE
# =============================================================================
# Same caching strategy as JS — build once, reuse forever.
# =============================================================================

_chain_cache: dict[str, dict] = {}


# =============================================================================
# CORE ALGORITHM: Building the Markov Chain
# =============================================================================

def construct_chain(name_list: list[str]) -> dict:
    """
    Build a Markov chain (transition probability table) from a list of names.

    This is the HEART of the system. We walk through each name character by
    character, counting how often each character follows each other character.

    The resulting chain is a nested dict:
        {
            "parts":    {1: count, 2: count},       # word count distribution
            "name_len": {3: count, 5: count, ...},   # name length distribution
            "initial":  {"A": count, "B": count},    # first-letter frequencies
            "A":        {"l": count, "r": count},    # transitions from 'A'
            "l":        {"e": count, "i": count},    # transitions from 'l'
            ...
            "table_len": {"parts": total, "A": total} # pre-computed totals
        }

    Args:
        name_list: Training names to learn patterns from.

    Returns:
        The scaled chain ready for name generation.
    """
    chain: dict[str, dict] = {}

    for full_name in name_list:
        # Split multi-word names: "Samwise Gamgee" -> ["Samwise", "Gamgee"]
        # Most of our training names are single words, but the chain can
        # learn to generate multi-part names if the training data has them.
        parts = full_name.split()
        _incr(chain, "parts", len(parts))

        for name in parts:
            # Record this name's length so generated names have similar lengths.
            # Without this, the chain would just keep going until it hits a
            # dead end, producing wildly variable-length names.
            _incr(chain, "name_len", len(name))

            # Record the starting character separately.
            # Starting letters have very different distributions than mid-word
            # letters. In English, 'T' is a common starter but rare mid-word.
            first_char = name[0]
            _incr(chain, "initial", first_char)

            # THE CORE MARKOV COUNTING LOOP
            #
            # Walk through the name recording each pair of adjacent characters.
            # For "Achilles":
            #   A->c, c->h, h->i, i->l, l->l, l->e, e->s
            #
            # Each pair becomes a count in our transition table:
            #   chain["G"]["a"] += 1
            #   chain["a"]["n"] += 1
            #   chain["n"]["d"] += 1
            #   ... and so on
            #
            # After processing ALL names, chain["a"] might look like:
            #   {"n": 5, "l": 8, "r": 12, "g": 3}
            # This tells us: after 'a', 'r' is the most likely next character.
            last_c = first_char
            for c in name[1:]:
                _incr(chain, last_c, c)
                last_c = c

    # Scale the raw counts to shape the probability distribution
    return _scale(chain)


def _incr(chain: dict, key: str, token) -> None:
    """
    Increment chain[key][token] by 1, initializing if needed.

    This is the counting primitive. Every time we see character 'b' follow
    character 'a', we call _incr(chain, 'a', 'b').

    We use a plain dict instead of defaultdict here to match the JS version's
    structure exactly, making it easier to compare the two implementations.

    Args:
        chain: The chain being built.
        key:   The "from" state (e.g., the current character, or "initial").
        token: The "to" state (e.g., the next character, or a name length).
    """
    if key not in chain:
        chain[key] = {}
    if token not in chain[key]:
        chain[key][token] = 0
    chain[key][token] += 1


def _scale(chain: dict) -> dict:
    """
    Apply power scaling to the chain's raw counts and pre-compute totals.

    WHY SCALE?
    The power of 1.3 amplifies the difference between common and rare transitions:
        count=10 -> floor(10^1.3) = 19  (common patterns get boosted)
        count=2  -> floor(2^1.3)  = 2   (rare patterns stay low)

    This works like a "temperature" knob on the sampling distribution:
        - Power > 1.0 (like temperature < 1): more conservative, predictable
        - Power = 1.0 (like temperature = 1): faithful to training distribution
        - Power < 1.0 (like temperature > 1): more random, creative

    EXPERIMENT: Try changing 1.3 to other values!
        - 0.5: very random, weird names
        - 1.0: proportional to training data
        - 2.0: very conservative, almost copies training names

    We also pre-compute table_len[key] = sum of all weights for that key.
    This lets select_link() sample efficiently without re-summing every time.

    Args:
        chain: The chain with raw counts.

    Returns:
        The same chain with scaled weights and table_len added.
    """
    table_len = {}

    for key in chain:
        table_len[key] = 0
        for token in chain[key]:
            count = chain[key][token]
            # The magic number: try changing 1.3 to see how it affects output
            weighted = math.floor(count ** 1.3)
            chain[key][token] = weighted
            table_len[key] += weighted

    chain["table_len"] = table_len
    return chain


# =============================================================================
# CORE ALGORITHM: Generating Names from the Chain
# =============================================================================

def markov_name(chain: dict) -> str:
    """
    Generate a single name by "walking" the Markov chain.

    The generation process mirrors how all autoregressive models work:
        1. Start with some initial state
        2. Sample the next token from a probability distribution
        3. Use that token as the new state
        4. Repeat until done

    Here, step 2 is: look up the current char in our table, sample from its
    recorded transitions. Every later sequence model keeps this same loop and
    only swaps out the table.

    Args:
        chain: A scaled Markov chain built by construct_chain().

    Returns:
        A generated name string.
    """
    # Step 1: How many words should this name have?
    # Our training data is mostly single-word names, so this will usually be 1.
    num_parts = _select(chain, "parts")
    name_parts = []

    for _ in range(num_parts):
        # Step 2a: How long should this word be?
        # Sampled from the length distribution of training names.
        target_len = _select(chain, "name_len")

        # Step 2b: Pick a starting character.
        # This uses the "initial" distribution, not the general transition table,
        # because name-initial letters have very different frequencies.
        c = _select(chain, "initial")
        name = c
        last_c = c

        # Step 2c: Generate the rest of the name character by character.
        # This is the "autoregressive loop" — each character depends on the last.
        #
        # KEY LIMITATION: We only look at the LAST character (order-1).
        # An order-2 chain would look at the last TWO characters, capturing
        # patterns like "th" -> likely followed by "e", "o", "a", "i".
        while len(name) < target_len:
            c = _select(chain, last_c)

            # Dead end: this character never appeared before another character
            # in training. It only ever appeared at the END of names.
            # Higher-order chains hit this more often; see backoff in
            # name_generator_higher_order.py.
            if c is None:
                break

            name += c
            last_c = c

        name_parts.append(name)

    return " ".join(name_parts)


def _select(chain: dict, key: str):
    """
    Weighted random sampling from a probability distribution.

    This is the fundamental "dice roll" of any generative model.

    HOW IT WORKS:
    Imagine the weights laid out on a number line:

        chain["a"] = {"n": 5, "l": 8, "r": 12}
        total = 25

        |--n(5)--|----l(8)----|-------r(12)--------|
        0        5            13                    25

    We pick a random number 0-24. Wherever it lands determines our choice:
        0-4   -> "n"  (20% chance)
        5-12  -> "l"  (32% chance)
        13-24 -> "r"  (48% chance)

    This is exactly the same as numpy.random.choice with weights, or
    torch.multinomial in PyTorch — just implemented from scratch so you
    can see the mechanics.

    EFFICIENCY NOTE: For large vocabularies, you'd use the "alias method"
    for O(1) sampling instead of this O(n) linear scan. But for our small
    character set (~50 unique chars), this is fine.

    Args:
        chain: The Markov chain.
        key:   Which distribution to sample from (a character, "initial", etc.)

    Returns:
        The sampled token, or None if the distribution is empty.
    """
    total = chain["table_len"].get(key, 0)
    if not total:
        return None

    # Pick a random point in the cumulative weight range
    idx = random.randint(0, total - 1)

    # Walk through tokens, accumulating weight until we pass our random point
    acc = 0
    for token, weight in chain[key].items():
        acc += weight
        if acc > idx:
            return token

    return None


# =============================================================================
# PUBLIC API
# =============================================================================

def get_chain(name_type: str) -> dict | None:
    """Retrieve or build+cache the chain for a given name type."""
    if name_type in _chain_cache:
        return _chain_cache[name_type]

    name_list = NAME_SETS.get(name_type)
    if name_list:
        chain = construct_chain(name_list)
        _chain_cache[name_type] = chain
        return chain

    return None


def generate_name(name_type: str) -> str:
    """Generate a single name of the given type."""
    chain = get_chain(name_type)
    if chain:
        return markov_name(chain)
    return ""


def generate_names(name_type: str, count: int) -> list[str]:
    """Generate multiple names of the given type."""
    return [generate_name(name_type) for _ in range(count)]


# =============================================================================
# DEMO: Run with `python name_generator.py`
# =============================================================================

def demo():
    print("=== Markov Chain Name Generator (Python) ===\n")
    print("This generator learns character-by-character transition")
    print("probabilities from training names, then 'walks' the chain")
    print("to produce new names that feel similar but are novel.\n")

    for name_type, names in NAME_SETS.items():
        print(f"--- {name_type.upper()} names ---")
        print(f"Training set: {len(names)} names")

        generated = generate_names(name_type, 10)
        for name in generated:
            print(f"  {name}")

        print()

    # Peek inside the chain to understand the data structure
    print("--- PEEK INSIDE THE CHAIN (greek, letter 'a') ---")
    chain = get_chain("greek")
    if chain and "a" in chain:
        print("After 'a', the possible next characters and their weights:")
        total = chain["table_len"]["a"]
        for char, weight in chain["a"].items():
            pct = (weight / total) * 100
            print(f"  '{char}' -> weight {weight} ({pct:.1f}%)")

    print("\n--- WHAT'S NEXT? ---")
    print("This order-1 chain only remembers the LAST character.")
    print("To capture more context, you could:")
    print("  1. Increase the order (bigrams -> trigrams -> n-grams)")
    print("  2. Back off to shorter contexts when a long one was never seen")
    print("Run name_generator_higher_order.py to see both.\n")


def export_markov_chains(path: str = "markov_chains.json") -> None:
    """
    Export Markov chains of orders 1-4 for all categories to JSON.

    Order-1 comes from construct_chain() in this module; orders 2-4 come
    from name_generator_higher_order. The browser page (index.html) loads
    this file to generate names client-side. The training names are included
    so the page can measure how often each order copies one verbatim.
    """
    import json
    import name_generator_higher_order as ho

    data = {f"order{o}": {} for o in range(1, 5)}
    data["names"] = {k: list(v) for k, v in NAME_SETS.items()}

    for name_type, name_list in NAME_SETS.items():
        data["order1"][name_type] = construct_chain(list(name_list))
        for o in range(2, 5):
            data[f"order{o}"][name_type] = ho.construct_chain(list(name_list), o)

    with open(path, "w") as f:
        json.dump(data, f)

    print(f"Wrote {path} ({len(open(path).read())} bytes)")


if __name__ == "__main__":
    demo()
