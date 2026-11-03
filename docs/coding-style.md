<!-- vendored: coding-style.md @ 09d16fa02fda (master: ~/.claude/resources) — edit the master, then `lib sync pull` -->
# Code style

Much of this code produces numbers that end up in a paper or a thesis. So the
rules below are not sorted by craft — they are sorted by **what a violation
costs you**. The first four can turn a bug into a wrong number that nobody
notices until the defense. The rest keep the code readable while it grows, and
the last four are about the world outside the process, where the failures are
quiet.

One idea runs through all of them: **nothing important stays implicit.** Types,
decisions, assumptions, side effects and errors must be visible in the code —
the reader (a supervisor, a reviewer, you in three months) should never have to
guess.

These rules are binding. If a request conflicts with one of them, **flag the
conflict** instead of silently violating it.

Two words that recur below:

- **The protocol** — the experiment's own record. `PROTOCOL-<task>.md` in a
  thesis repo, the area's protocol file in a lab repo.
- **The shared module** — the one home for logic used in more than one place.
  `shared.py` at the repo root, or whatever the repo's own `CLAUDE.md` names.

---

## 1. Fail loud — never swallow an error

A swallowed error doesn't disappear. It comes back as a wrong result, far away
from its cause, with nothing pointing at it.

```python
# Bad — 3 malformed rows vanish, your N is silently 197 instead of 200
for row in rows:
    try:
        items.append(parse(row))
    except Exception:
        pass

# Good — a bad row stops you now, while you can still see why
for row in rows:
    items.append(parse(row))     # raises on malformed input
```

- No bare `except:` and no `except Exception: pass`. Catch only the specific
  exception you actually know how to handle.
- Don't skip bad data quietly. If an input violates your expectations, **raise**.
  A clear error at the source beats a silently wrong accuracy downstream.
- **Validate at the boundary** — right where data enters (the prepare script,
  the answer parser). After that, the rest of the code can trust its inputs.
- **Convert types deliberately.** `int(...)` or `str(...)` sprinkled in to make
  something run turns a real type error into a silently wrong value.
- If you *do* want to tolerate bad entries, that is a real decision: make it an
  explicit flag (`skip_invalid: bool`), never a silent `except`.

If you catch something, never discard the context — re-raise or log the
original error with it.

---

## 2. Make assumptions explicit

When a choice materially changes your numbers, it must be visible in the code —
not buried in a line you'll forget you wrote.

The classic example is one you will actually hit: "average the score across the
classes" is ambiguous.

- **Macro:** average the per-class score, every class weighted equally. A class
  with 3 items counts as much as one with 300.
- **Micro:** pool all instances first, every instance weighted equally. Large
  classes dominate.

On unbalanced data these give very different numbers. So don't hardcode one —
name it:

```python
def average_score(
    per_class: dict[str, ClassStats],
    method: Literal["macro", "micro"] = "macro",
) -> float:
    """Average a score across classes.

    Args:
        per_class: Stats per class label.
        method: "macro" weights each class equally; "micro" weights each
            instance equally. These differ when class sizes are uneven.
    """
```

Now the assumption is named, documented, and changeable. Write which one you
chose — and why — into the protocol. If the choice is big and hard to reverse,
ask before you build on it.

---

## 3. Test the functions that produce numbers

Every function whose output can become a reported number — metrics, answer
parsers, label normalization — gets a small test with hand-checked examples in
`tests/`. This is not general test discipline; it is the one thing standing
between you and a **parser bug that looks exactly like a model error**.

```python
def test_parse_answer_rejects_ambiguous_output() -> None:
    assert parse_answer("pro, or maybe con") == "?"

def test_average_score_macro_weights_classes_equally() -> None:
    stats = {"a": ClassStats(score=1.0, n=1), "b": ClassStats(score=0.0, n=99)}
    assert average_score(stats, method="macro") == 0.5
```

Cover the normal case **and** the edges: empty input, boundaries, and the errors
from §1 — assert that bad input really does raise. Run `python3 -m pytest tests/`
**before** you report any number, and update tests in the same change that
changes behavior.

Scale the tests to the change, though — don't gold-plate. Cover the edges that
can realistically occur for *this* code, not every edge the language permits.

---

## 4. One home per piece of logic

Before writing a function, check whether it already exists — search, don't
assume. The failure mode is two slightly different `compute_f1` helpers that
drift apart, so the same experiment reports two different numbers depending on
which script ran.

- If it exists, call it. If it almost fits, extend it rather than clone it.
- The **second** time you need something, move it to the shared module and
  import it from both places. Never copy-paste between experiments.
- Don't generalize on speculation — extract once the repetition is real.

A single shared function also means one signature and one behavior everywhere
it is used, which is what makes §5 hold.

---

## 5. Keep interfaces consistent between functions

When one function's output feeds another's input, the **order, names and shape**
stay identical along the chain.

```python
def pre_filter(doc: Doc) -> tuple[list[Flag], bool]:
    return flags, needs_judge

def resume_filter(needs_judge: bool, flags: list[Flag]) -> Result:   # WRONG ORDER
```

Same two values, opposite order — the type checker won't always catch it, and
the call site looks fine. Once more than about two values cross a boundary, stop
using a tuple and give them names:

```python
@dataclass(frozen=True)
class FilterOutcome:
    flags: list[Flag]
    needs_judge: bool
```

Now position is irrelevant, and adding a third field later doesn't reshuffle
every call site.

---

## 6. One function, one thing — and keep I/O at the edges

A function does exactly one thing, at one level of abstraction. Warning signs
that it has grown too big:

- The name needs an "and" to be honest (`load_and_score`).
- It mixes orchestration with byte-level detail.
- It has blocks separated by `# now do X` comments.
- It's hard to name, or hard to test without elaborate setup.

Prefer **pure** functions for anything that computes: output depends only on the
inputs, nothing outside changes. Don't mutate arguments, don't touch global
state, and keep disk/network/randomness at the edges of the script. That is what
makes §3 easy — a pure metric function is a two-line test; one that reads a file
is not.

If a side effect is genuinely needed, isolate it and say so:

```python
def write_report(report: Report, path: Path) -> None:
    """Render `report` and write it to disk.

    Side effects:
        Creates or overwrites the file at `path`.
    """
```

---

## 7. Type hints and docstrings

- **Type hints** on every signature — parameters and return. Prefer precise
  types (`list[str]`, `Path`, a dataclass) over `Any` or a bare `dict`.
- A **docstring** on every function that isn't trivially obvious: what it does
  and *why it exists*, not a line-by-line how. Document `Args`, `Returns` and
  `Raises`; for non-obvious return shapes describe each element, so the caller
  never has to read the body to use it right.
- **Names use the field's standard term.** An identifier, a field name or a
  label says what the lecture notes call the thing — `feature_vector`,
  `hypothesis_space`, `gold_label` — never a house coinage or a private
  synonym; a coinage may appear as a second name in a comment. Look the term
  up before you coin one: the Webis ML notation
  ([Syntax Overview](https://downloads.webis.de/lecturenotes/machine-learning/unit-en-ml-syntax-overview.pdf),
  [Elements](https://downloads.webis.de/lecturenotes/machine-learning/unit-en-ml-elements.pdf)),
  in lab repos `lib term <term>`. A reader who knows the field should never
  have to translate your names.

---

## 8. Run the linter and the type checker

```bash
ruff check .      # lint
ruff format .     # format
pyright           # type-check — catches the swapped-argument bug from §5
```

Fix what they report rather than silencing it; a `# noqa` or `# type: ignore`
needs a stated reason. A change isn't finished until both run clean.

**Know when to stop.** If the checker floods you with errors from a cause
outside your control — most often an untyped third-party library — do not
refactor for days chasing them. Say what is happening and what the likely cause
is, propose the scoped options (a targeted `# type: ignore` at the boundary, or
excluding the library in the pyright config), and let the decision be made.

---

## 9. Record every dependency

Adding a dependency is a real decision — it affects everyone who runs the
project. Don't pull one in silently.

- Before installing a new package, **say so and say why**. A new dependency
  shouldn't appear unannounced.
- The moment you `pip install` something, add it to `requirements.txt` (or the
  `pyproject.toml` dependencies). A package that is installed but unrecorded
  works on your laptop and breaks on the cluster — usually the week before
  submission.
- Prefer the standard library, or something already in the project (§4), over a
  new package.

---

## 10. Read the docs first, and keep them in sync

- **Before changing anything**, check the `README` and any relevant docs to
  understand how the affected part is meant to work. Don't infer the design
  from the code alone when it's already written down.
- **After changing a module**, update the documentation if your change made it
  inaccurate — new or changed behavior, interfaces, configuration, usage. Code
  and docs must not drift apart.
- If a change is significant enough that no doc covers it yet, add a short note
  where a reader would look for it.
- Keep this distinct from the notes file (see *Where the why goes*): docs
  explain *how the system works* for people using it; the notes record *why* a
  tricky problem was solved a particular way.

---

## 11. Verify against the real system, not only against your tests

Tests prove the code does what *you* expect. They cannot tell you whether that
expectation matches reality. Before calling something done, put one real
artifact through the real path and look at what comes out.

- **Exercise the actual integration once, by hand.** Drop a real file into the
  watched folder and confirm it was picked up. Call the real endpoint. Start the
  real container. Assumptions about someone else's system are the ones that turn
  out to be wrong, and they fail silently — a folder watcher that never fires
  looks exactly like a user who hasn't scanned anything yet.
- **Look at the artifact, not just the exit code.** A file writer can return
  plausible bytes, pass every structural assertion, and still produce pages that
  render blank, inverted, or with the wrong aspect ratio. Open the output.
- **Clean up what your verification created.** A test document left in a real
  archive is indistinguishable from real data six months later.
- **Say what you verified and what you did not.** An untested path presented as
  finished is worse than an untested path labelled as such — mark it where a
  reader will look, not only in the reply:

```markdown
3. ~~Scanner per eSCL → `./consume/`~~ — `scripts/scan.py` steht, gegen einen
   nachgebauten eSCL-Server getestet; **am echten Gerät noch nicht erprobt**.
```

---

## 12. Fake the external system — including the ways it misbehaves

When code talks to a device, service, or API you don't control, write a small
fake that speaks its protocol and test against that. It turns "works while the
hardware is on my desk" into something anyone can run, on any machine, later.

The fake earns its keep by reproducing **known misbehavior**, not the happy
path. Those are the branches that otherwise only ever execute in production:

```python
def do_GET(self) -> None:
    if self.path.endswith("/NextDocument"):
        if not cls.hiccuped:          # real devices return busy mid-job
            cls.hiccuped = True
            return self._send(503)
        ...

def do_POST(self) -> None:
    # Deliberately wrong host, as some firmwares really do send.
    return self._send(201, b"", extra={"Location": "http://192.0.2.99/eSCL/ScanJobs/x1"})
```

- Keep the fake beside the tests, in the same language, with no new dependency
  (§9) — a fake that needs its own toolchain stops being run.
- Every quirk the fake simulates should carry a comment saying it is real, not
  invented. Otherwise the next reader deletes it as noise.
- The fake covers the protocol. It does **not** cover the hardware — see the
  labelling rule in §11.

---

## 13. Publish state atomically

Anything that watches a location — a consume folder, a queue directory, a config
path another process reads — will eventually read your file while you are still
writing it. A partially written file is usually accepted as a whole one.

Write to a staging path on the **same filesystem**, then rename:

```python
def deliver(pdf: bytes, target: Path) -> Path:
    """Write the PDF elsewhere, then atomically rename it into place.

    Side effects:
        Creates TMPDIR if missing; creates or replaces the file at `target`.
    """
    TMPDIR.mkdir(exist_ok=True)
    staged = TMPDIR / f".{target.name}.{os.getpid()}{random.randrange(1 << 20):05x}"
    staged.write_bytes(pdf)
    os.replace(staged, target)   # rename is atomic; a write is not
    return target
```

- Same filesystem is the requirement, not a detail: across devices the rename
  degrades into a copy, and the atomicity is gone.
- Stage **outside** the watched directory. A temp file inside it may match the
  watcher's patterns and be consumed half-written — the exact failure you were
  avoiding.
- Assert in a test that a successful hand-off leaves no staging file behind.

---

## 14. Shedding a dependency is as legitimate a fix as adding one

§9 governs taking a dependency on. This is the counterpart: when one breaks, or
turns out to be heavy for what you actually use it for, the first question is
not "how do I work around this" but **"what do I actually need from it?"** Often
the honest answer is a fraction of what it offers.

- A handful of flat scalar fields does not need a full XML parser. A single
  timestamp format does not need a date library.
- This weighs more when the code must run in **more than one environment** —
  this laptop today, a cluster node later. Every dependency is a thing that can
  be missing or broken over there.
- Don't do it by feel: shed a dependency when the replacement is genuinely small
  and testable. Reimplementing something subtle (crypto, timezones, real
  parsing) to avoid an import is the worse trade.
- **Record why the plain-looking code is plain**, or the next reader will
  helpfully restore the import:

```python
def xml_text(doc: bytes, tag: str) -> str:
    """First text value of <ns:tag>, namespace-agnostic.

    Deliberately a regex and not ElementTree: the replies we care about are a
    handful of flat scalar fields, and this keeps the script free of pyexpat —
    which is broken in some Homebrew Python builds and would otherwise take the
    whole scanner down with an ImportError.
    """
```

---

## Where the *why* goes

Code can show what it does, never why it ended up that way. When you solve a
non-obvious problem — a bug that took a day, a surprising constraint, a design
decision with non-obvious trade-offs — write it down where the future reader
will look. They see the working code, but not the three approaches that failed
first.

Keep each entry short:

- **Problem:** what was hard, and why the obvious approach didn't work.
- **Solution:** the approach you landed on.
- **Notes:** any gotcha or constraint to watch out for later.

```markdown
## 2026-06-04 — Filter ordering between pre_filter and resume_filter

**Problem:** The two stages passed (flags, needs_judge) in opposite order,
causing a silent mismatch that the type checker didn't flag.

**Solution:** Introduced a frozen `FilterOutcome` dataclass so the values are
named and order no longer matters across the boundary.

**Notes:** Any new filter stage must consume/produce `FilterOutcome`, not a tuple.
```

One place per repo, not three: the protocol's "Notes" section while the problem
belongs to one task, a `NOTES.md` (or `docs/notes.md`) when it belongs to the
codebase, its own file in `notes/` once it outgrows either.

Only record what is worth remembering — this is not a changelog of every edit.
When you do add an entry, **say so in one line at the end of your reply**,
naming what you recorded, so the framing is easy to correct (e.g. "Added a
NOTES.md entry on the filter-ordering fix.").

---

## Self-check before you report a number

- [ ] No swallowed errors; bad input raises instead of being skipped, boundaries
      validate, and type conversions are deliberate rather than reflexive.
- [ ] Every choice that moves the number (averaging, thresholds, filtering) is
      a named parameter, and the choice is written in the protocol.
- [ ] `python3 -m pytest tests/` passes, and covers the parser/metric you used.
- [ ] No duplicated logic — shared code lives in the shared module.
- [ ] Data passed between functions keeps a consistent shape, or has names.
- [ ] Each function does one thing; side effects are isolated and documented.
- [ ] New/changed functions have type hints and a docstring.
- [ ] Names are the field's standard terms; a house coinage is at most a comment.
- [ ] `ruff check`, `ruff format`, `pyright` run clean — or an external cause
      was reported instead of endlessly refactored against.
- [ ] New packages were announced and are in `requirements.txt`.
- [ ] Relevant docs/README were read beforehand, and updated afterward if the
      change made them inaccurate.
- [ ] One real artifact went through the real path, the output was looked at
      (not just its exit code), leftovers were cleaned up, and anything still
      unverified is labelled where a reader will find it.
- [ ] Code talking to a system you don't control has a fake, and the fake
      reproduces that system's known misbehavior, each quirk commented as real.
- [ ] Files handed to a watcher or another process are staged outside the
      watched location and renamed into place on the same filesystem.
- [ ] A dependency that broke or proved oversized was reconsidered, not just
      worked around — and a deliberately plain replacement says why it is plain.
- [ ] Any hard problem solved is captured where the *why* goes, and the entry is
      mentioned in the reply.
