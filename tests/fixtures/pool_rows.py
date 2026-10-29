"""Hand-made rows for the pool tests: no network, no `datasets`."""

from s1d1_bench.keys import normalize, prompt_key

DOLLY, OASST2, ARENA = "cc-by-sa-3.0", "apache-2.0", "apache-2.0"


def _row(source: str, source_id: str, prompt: str, lang: str = "eng", licence: str | None = None, label: str = "x"):
    default = {"dolly": DOLLY, "oasst2": OASST2, "arena_hard": ARENA}[source]
    return {
        "source": source,
        "source_id": source_id,
        "prompt": prompt,
        "lang": lang,
        "licence": licence or default,
        "source_label": label,
    }


ROW_0 = _row("dolly", "0", "What is the capital of France?", label="open_qa")
KEY_OF_ROW_0 = prompt_key(normalize(ROW_0["prompt"]))

ROW_DUP_FIRST = _row("oasst2", "a1", "Write a haiku about autumn.", label="prompter")
# same prompt as above once normalized (CRLF, surrounding whitespace): the first occurrence wins
ROW_DUP_SECOND = _row("oasst2", "a2", "  Write a haiku about autumn.\r\n", label="prompter")

ROW_EMPTY = _row("dolly", "e", " \r\n\t ", label="open_qa")
ROW_9000_CHARS = _row("arena_hard", "long", "x" * 9000, label="cluster-1")

ROWS = [
    ROW_0,
    _row("dolly", "1", "Summarize this.\n\nThe cat sat on the mat.", label="summarization"),
    _row("dolly", "2", "Name three primary colours.", label="brainstorming"),
    _row("oasst2", "a0", "Bonjour, peux-tu m'aider ?", lang="fra", label="prompter"),
    ROW_DUP_FIRST,
    ROW_DUP_SECOND,
    _row("oasst2", "a3", "Explain recursion to a ten year old.", label="prompter"),
    _row("oasst2", "a4", "Como se dice hola en ingles?", lang="spa", label="prompter"),
    _row("arena_hard", "q1", "Write a Python function that merges two sorted lists.", label="cluster-1"),
    _row("arena_hard", "q2", "Design a rate limiter for a multi-tenant API.", label="cluster-2"),
    _row("arena_hard", "q3", "Prove that the square root of 2 is irrational.", label="cluster-3"),
]

# 5 OASST2 messages: 2 validated root prompts, 1 root prompt that failed review, 1 assistant reply,
# 1 deleted root prompt
OASST_FIXTURE = [
    {
        "message_id": "m1",
        "parent_id": None,
        "text": "How do I bake bread?",
        "role": "prompter",
        "lang": "en",
        "review_result": True,
        "deleted": False,
    },
    {
        "message_id": "m2",
        "parent_id": None,
        "text": "Quelle heure est-il à Tokyo ?",
        "role": "prompter",
        "lang": "fr",
        "review_result": True,
        "deleted": False,
    },
    {
        "message_id": "m3",
        "parent_id": None,
        "text": "spam spam",
        "role": "prompter",
        "lang": "en",
        "review_result": False,
        "deleted": False,
    },
    {
        "message_id": "m4",
        "parent_id": "m1",
        "text": "Mix flour and water.",
        "role": "assistant",
        "lang": "en",
        "review_result": True,
        "deleted": False,
    },
    {
        "message_id": "m5",
        "parent_id": None,
        "text": "removed prompt",
        "role": "prompter",
        "lang": "en",
        "review_result": True,
        "deleted": True,
    },
]

# region codes occur in the dataset (pt-BR, uk-UA); codes we do not know map to `und`
OASST_LANG_FIXTURE = [
    {
        "message_id": f"l{i}",
        "parent_id": None,
        "text": f"prompt {i}",
        "role": "prompter",
        "lang": code,
        "review_result": True,
        "deleted": False,
    }
    for i, code in enumerate(["pt-BR", "uk-UA", "zh", "xx"])
]

# extra English rows so that sampling has something to choose from; one non-English row per source
_EXTRA = [
    _row(s, f"{s}-{i}", f"English prompt number {i} from {s}")
    for s in ("dolly", "oasst2", "arena_hard")
    for i in range(6)
]
_EXTRA.append(_row("dolly", "fr", "Une question en français", lang="fra"))
POOL_ROWS = ROWS + _EXTRA
