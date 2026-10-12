"""Build the MLA-C01 question pool.

Reads only exam-datasets/MLA. The Specialty folder is a different exam and is
never opened.

Official MLA-C01 blueprint (scored content):
  Data Preparation for ML                              28%
  ML Model Development                                 26%
  Deployment and Orchestration of ML Workflows         22%
  ML Solution Monitoring, Maintenance, and Security    24%

The published exam is 65 questions and 130 minutes, with a 720/1000 pass mark.
A sitting uses that size when every domain pool is large enough. Otherwise it
uses the largest size that still matches the weights.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "exam-datasets" / "MLA"
OUT_PATH = ROOT / "assets" / "data" / "mla-c01.json"

EXAM_CODE = "MLA-C01"
OFFICIAL_QUESTIONS = 65
OFFICIAL_MINUTES = 130
PASS_MARK = 0.72
OFFICIAL_PASS = "720/1000"

DOMAINS = [
    {
        "id": "data-preparation",
        "name": "Data Preparation",
        "officialName": "Data Preparation for Machine Learning (ML)",
        "weight": 0.28,
    },
    {
        "id": "model-development",
        "name": "Model Development",
        "officialName": "ML Model Development",
        "weight": 0.26,
    },
    {
        "id": "deployment",
        "name": "Deployment & Orchestration",
        "officialName": "Deployment and Orchestration of ML Workflows",
        "weight": 0.22,
    },
    {
        "id": "monitoring-security",
        "name": "Monitoring & Security",
        "officialName": "ML Solution Monitoring, Maintenance, and Security",
        "weight": 0.24,
    },
]

# Checked against the question being asked (the last sentence), in this order.
ASK_RULES: list[tuple[str, re.Pattern[str], str]] = [
    (
        "monitoring-security",
        re.compile(
            r"drift|\bmodel monitor\b|data quality monitor|bias drift|concept drift|"
            r"cloudtrail|cloudwatch|\balarm\b|monitor (?:the )?(?:model|performance|endpoint)|"
            r"network isolation|no internet|without (?:using )?the internet|private subnet|"
            r"vpc endpoint|interface endpoint|encrypt|\bkms\b|least privilege|\biam\b|"
            r"security group|network acl|network configuration|block traffic|"
            r"resource polic|bucket polic|permissions?|unauthorized|appropriate access|"
            r"vulnerab|complian|\baudit\b|cloudwatch logs|detect (?:changes|anomal)|"
            r"secrets manager|rotate .{0,40}token|blocked phrase|content filter|guardrail|"
            r"explainab|clarif|explanation for the model|compute optimizer|\bbudgets?\b",
            re.IGNORECASE,
        ),
        "asked about monitoring or security",
    ),
    (
        "deployment",
        re.compile(
            r"inference|endpoint|deploy|model registry|model package|ci/cd|continuous integration|"
            r"codepipeline|pipeline|batch transform|production variant|multi-model|"
            r"serverless inference|asynchronous inference|real-time inference|"
            r"shadow|blue/green|invoke the model|host(?:ing)? the model|inference component|"
            r"auto scaling|autoscaling",
            re.IGNORECASE,
        ),
        "asked about deployment",
    ),
    (
        "model-development",
        re.compile(
            r"hyperparameter|which algorithm|built-in algorithm|sagemaker algorithm|training job|"
            r"debugger|sagemaker experiments?|automatic model tuning|warm pools?|"
            r"overfit|underfit|evaluation metric|objective metric|\bmetric\b|"
            r"accuracy|precision|recall|\bf1\b|\bauc\b|rmse|loss function|"
            r"epochs?|learning rate|early stopping|spot training|managed spot|"
            r"modeling approach|distributed training|communication overhead",
            re.IGNORECASE,
        ),
        "asked about model development",
    ),
    (
        "data-preparation",
        re.compile(
            r"feature store|feature engineering|ground truth|\blabel|"
            r"data wrangler|\bglue\b|lake formation|\bingest\b|\bdataset\b|"
            r"missing value|class imbalance|imbalanced|aggregate the data|"
            r"prepare (?:the |this )?data|store the data|data quality|"
            r"\bschema\b|parquet|\bcsv\b|tokeniz|normaliz|one-hot|categorical|"
            r"split (?:the )?data|data source|knowledge base|"
            r"extract entit|textract|comprehend|transcribe|translate|"
            r"data-processing|processing job|key phrase",
            re.IGNORECASE,
        ),
        "asked about data preparation",
    ),
]

# Phrases that decide the domain wherever they appear in the stem or correct choice.
STRONG: list[tuple[str, re.Pattern[str], str]] = [
    ("monitoring-security", re.compile(r"\bmodel monitor\b|data drift|concept drift|bias drift|attribute drift|\bclarify\b", re.I), "drift, monitor, or clarify"),
    ("monitoring-security", re.compile(r"network isolation|s3 gateway vpc endpoint|interface vpc endpoint", re.I), "private network path"),
    ("monitoring-security", re.compile(r"server-side encryption|sse-kms|aws kms|\bkms key|aws budgets|compute optimizer|blocked phrase|secrets manager", re.I), "encryption or operations control"),
    ("deployment", re.compile(r"model registry|model package group|model groups?", re.I), "model registry"),
    ("deployment", re.compile(r"asynchronous inference|serverless inference|real-time inference|batch transform", re.I), "inference option"),
    ("deployment", re.compile(r"multi-model endpoint|multi-container endpoint|inference component", re.I), "endpoint hosting"),
    ("deployment", re.compile(r"sagemaker pipelines|codepipeline|codebuild", re.I), "pipeline"),
    ("data-preparation", re.compile(r"feature store|data wrangler|ground truth|sagemaker processing|textract|comprehend|transcribe", re.I), "data prep service"),
    ("data-preparation", re.compile(r"lake formation|aws glue|amazon emr|kinesis data firehose", re.I), "data movement"),
    ("data-preparation", re.compile(r"class imbalance|oversampling|undersampling|smote", re.I), "class imbalance"),
    ("model-development", re.compile(r"automatic model tuning|hyperparameter (?:optimization|tuning)|sagemaker debugger|managed warm pools?", re.I), "tuning, debugger, or warm pool"),
    ("model-development", re.compile(r"xgboost|linear learner|blazingtext|object2vec|deepar|random cut forest|k-means", re.I), "built-in algorithm"),
]

KEYWORD: list[tuple[str, int, re.Pattern[str], str]] = [
    ("data-preparation", 6, re.compile(r"feature engineering|feature store|ground truth|data wrangler", re.I), "features or labeling"),
    ("data-preparation", 5, re.compile(r"aws glue|lake formation|amazon emr|athena|data brew", re.I), "data integration"),
    ("data-preparation", 4, re.compile(r"\bcsv\b|\bjson\b|parquet|missing value|outlier|schema|tokeniz|normaliz", re.I), "data format"),
    ("data-preparation", 4, re.compile(r"training data|dataset|data source|ingest|aggregate|data-processing|processing job", re.I), "dataset"),
    ("model-development", 6, re.compile(r"hyperparameter|automatic model tuning|debugger", re.I), "tuning"),
    ("model-development", 5, re.compile(r"xgboost|pytorch|tensorflow|hugging face|built-in algorithm", re.I), "framework or algorithm"),
    ("model-development", 4, re.compile(r"overfit|underfit|accuracy|precision|recall|f1 score|\bauc\b|rmse", re.I), "metric"),
    ("model-development", 3, re.compile(r"training job|train the model|model training", re.I), "training"),
    ("deployment", 6, re.compile(r"endpoint|batch transform|serverless inference|asynchronous inference", re.I), "inference"),
    ("deployment", 5, re.compile(r"model registry|sagemaker pipelines|codepipeline|deploy", re.I), "release"),
    ("deployment", 4, re.compile(r"production variant|blue/green|shadow variant|auto scaling", re.I), "traffic"),
    ("monitoring-security", 6, re.compile(r"model monitor|drift|cloudtrail", re.I), "monitor"),
    ("monitoring-security", 5, re.compile(r"\biam\b|kms|encrypt|vpc endpoint|private subnet|security group", re.I), "security control"),
    ("monitoring-security", 4, re.compile(r"cloudwatch|alarm|log group|audit", re.I), "operations signal"),
]


def source_files() -> list[Path]:
    if SOURCE.name != "MLA" or SOURCE.parent.name != "exam-datasets":
        raise SystemExit(f"Refusing to read a dataset outside exam-datasets/MLA: {SOURCE}")
    if not SOURCE.is_dir():
        raise SystemExit(f"MLA dataset not found: {SOURCE}")
    files = [
        path
        for path in SOURCE.glob("*.json")
        if "specialty" not in path.name.lower() and "MLA-C01" in path.name
    ]
    return sorted(files, key=lambda path: int(path.stem.rsplit("_", 1)[-1]))


def parse_letters(value: str) -> list[str]:
    seen: list[str] = []
    for letter in re.findall(r"[A-F]", str(value).upper()):
        if letter not in seen:
            seen.append(letter)
    return seen


def discussion_vote(question: dict) -> list[str]:
    votes: Counter[str] = Counter()
    for comment in question.get("discussion") or []:
        match = re.search(r"Selected Answer:\s*([A-F][A-F,\s/&]*)", comment.get("content") or "", re.I)
        if not match:
            continue
        key = "".join(parse_letters(match.group(1)))
        if not key:
            continue
        try:
            weight = int(float(comment.get("upvote_count") or 1))
        except (TypeError, ValueError):
            weight = 1
        votes[key] += max(weight, 1)
    if not votes:
        return []
    winner, _score = votes.most_common(1)[0]
    return parse_letters(winner)


def resolve_answer(question: dict) -> tuple[list[str], str]:
    marked = parse_letters(question.get("answer") or "")
    marked_et = parse_letters(question.get("answer_ET") or "")
    voted = discussion_vote(question)
    if marked:
        note = ""
        if marked_et and marked_et != marked:
            note = f"q{question.get('question_id')}: marked {''.join(marked)}, examtopics {''.join(marked_et)}"
        return marked, note
    if voted:
        return voted, ""
    if marked_et:
        return marked_et, ""
    return [], ""


def explanation_for(question: dict, answers: list[str]) -> str:
    target = "".join(answers)
    best = ""
    best_votes = -1
    for comment in question.get("discussion") or []:
        content = (comment.get("content") or "").strip()
        match = re.search(r"Selected Answer:\s*([A-F][A-F,\s/&]*)", content, re.I)
        if not match or "".join(parse_letters(match.group(1))) != target:
            continue
        try:
            votes = int(float(comment.get("upvote_count") or 0))
        except (TypeError, ValueError):
            votes = 0
        if votes > best_votes:
            best = content
            best_votes = votes
    if not best:
        return (question.get("answer_description") or "").strip()
    text = re.sub(r"^Selected Answer:[^\n]*\n?", "", best, count=1, flags=re.I).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:1600]


GENERIC_ASK = re.compile(
    r"^(?:which|what|how|select)\b.{0,140}\b(?:requirements?|this issue|this goal|this problem)\b",
    re.IGNORECASE,
)


def focus_text(stem: str) -> str:
    """The requirement being asked, without the case-study introduction."""
    parts = [part.strip() for part in re.split(r"(?<=[?.])\s+", stem.strip()) if part.strip()]
    while len(parts) > 1 and GENERIC_ASK.search(parts[-1]):
        parts.pop()
    if not parts:
        return stem
    focus = parts[-1]
    if len(parts) > 1 and len(focus) < 140:
        focus = f"{parts[-2]} {focus}"
    return focus


def score_keywords(text: str, multiplier: float) -> tuple[dict[str, float], list[str]]:
    scores = {domain["id"]: 0.0 for domain in DOMAINS}
    hits: list[tuple[float, str, str]] = []
    for domain, weight, pattern, label in KEYWORD:
        if not pattern.search(text):
            continue
        points = weight * multiplier
        scores[domain] += points
        hits.append((points, domain, label))
    hits.sort(reverse=True)
    return scores, [label for _points, _domain, label in hits[:4]]


def classify(stem: str, correct_text: str) -> tuple[str, list[str], float]:
    focus = focus_text(stem)
    focused = f"{focus}\n{correct_text}"
    for domain, pattern, label in ASK_RULES:
        if pattern.search(focus):
            return domain, [label], 10.0
    for domain, pattern, label in STRONG:
        if pattern.search(focused):
            return domain, [label], 8.0
    focus_scores, hits = score_keywords(focus, 3.0)
    answer_scores, answer_hits = score_keywords(correct_text, 2.0)
    scores = {domain["id"]: focus_scores[domain["id"]] + answer_scores[domain["id"]] for domain in DOMAINS}
    if max(scores.values()) <= 0:
        scores, hits = score_keywords(stem, 1.0)
    else:
        hits = hits + answer_hits
    ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_domain, top_score = ranking[0]
    second = ranking[1][1]
    if top_score <= 0:
        return "model-development", ["unscored, kept with model development"], 0.0
    return top_domain, [label for label in hits if label][:4] or ["keyword"], round(top_score - second, 1)


def stem_key(value: str) -> str:
    flattened = re.sub(r"\s+", " ", value).strip().lower()
    return re.sub(r"[^a-z0-9]+", " ", flattened).strip()


def question_id(key: str, answers: list[str]) -> str:
    digest = hashlib.sha1(f"{key}|{''.join(answers)}".encode("utf-8")).hexdigest()[:12]
    return f"mla_{digest}"


def apportion(size: int, domains: list[dict]) -> dict[str, int]:
    raw = {domain["id"]: size * domain["weight"] for domain in domains}
    floors = {domain_id: int(value) for domain_id, value in raw.items()}
    left = size - sum(floors.values())
    order = sorted(raw, key=lambda domain_id: (raw[domain_id] - floors[domain_id], raw[domain_id]), reverse=True)
    for domain_id in order[:left]:
        floors[domain_id] += 1
    return floors


def fit_exam(pool_counts: dict[str, int]) -> tuple[int, dict[str, int]]:
    for size in range(OFFICIAL_QUESTIONS, 3, -1):
        counts = apportion(size, DOMAINS)
        if all(pool_counts.get(domain["id"], 0) >= counts[domain["id"]] >= 1 for domain in DOMAINS):
            return size, counts
    raise SystemExit("MLA pools are too small for a weighted exam.")


def choices_of(raw: dict) -> list[dict]:
    items = []
    for letter, text in raw.items():
        if letter not in list("ABCDEF"):
            continue
        cleaned = re.sub(r"\s+", " ", str(text)).strip()
        if cleaned:
            items.append({"letter": letter, "text": cleaned})
    items.sort(key=lambda item: item["letter"])
    return items


def load_questions(files: list[Path]) -> tuple[list[dict], list[str], list[str]]:
    parsed: list[dict] = []
    conflicts: list[str] = []
    skipped: list[str] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        page = payload.get("pageProps") or {}
        exam_name = ((page.get("exam") or {}).get("name") or "")
        if "Specialty" in exam_name or "MLA-C01" not in exam_name:
            raise SystemExit(f"Refusing file that is not MLA-C01: {path.name} ({exam_name})")
        for question in page.get("questions") or []:
            number = question.get("question_id")
            stem = re.sub(r"\s+\n", "\n", (question.get("question_text") or "")).strip()
            if stem.upper().startswith("HOTSPOT") or question.get("question_images") or question.get("answer_images"):
                skipped.append(f"q{number}: image or hotspot, no text key")
                continue
            answers, note = resolve_answer(question)
            choices = choices_of(question.get("choices") or {})
            letters = {choice["letter"] for choice in choices}
            if not stem or not answers or not set(answers).issubset(letters):
                skipped.append(f"q{number}: missing stem, choices, or answer")
                continue
            if note:
                conflicts.append(note)
            parsed.append(
                {
                    "exam": int(page.get("currentPage") or 0),
                    "number": int(number),
                    "stem": stem,
                    "choices": choices,
                    "answers": answers,
                    "explanation": explanation_for(question, answers),
                    "source": path.name,
                }
            )
    return parsed, conflicts, skipped


def build() -> dict:
    files = source_files()
    parsed, conflicts, skipped = load_questions(files)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for question in parsed:
        grouped[stem_key(question["stem"])].append(question)
    unique: list[dict] = []
    removed = 0
    for key, group in grouped.items():
        group.sort(key=lambda item: (-len(item["explanation"]), item["number"]))
        keeper = group[0]
        correct = " ".join(
            choice["text"] for choice in keeper["choices"] if choice["letter"] in keeper["answers"]
        )
        domain, signals, margin = classify(keeper["stem"], correct)
        keeper["domain"] = domain
        keeper["signals"] = signals
        keeper["margin"] = margin
        keeper["id"] = question_id(key, keeper["answers"])
        keeper["stemKey"] = key
        unique.append(keeper)
        removed += len(group) - 1
    unique.sort(key=lambda item: item["number"])
    pool_counts = Counter(item["domain"] for item in unique)
    exam_size, counts = fit_exam(pool_counts)
    minutes = OFFICIAL_MINUTES if exam_size == OFFICIAL_QUESTIONS else round(OFFICIAL_MINUTES * exam_size / OFFICIAL_QUESTIONS)
    domains_out = []
    for domain in DOMAINS:
        domains_out.append({**domain, "examCount": counts[domain["id"]], "poolCount": pool_counts[domain["id"]]})
    return {
        "meta": {
            "examCode": EXAM_CODE,
            "examName": "Machine Learning Engineer — Associate",
            "level": "Associate",
            "examSize": exam_size,
            "passMark": PASS_MARK,
            "officialQuestions": OFFICIAL_QUESTIONS,
            "officialMinutes": OFFICIAL_MINUTES,
            "officialPass": OFFICIAL_PASS,
            "minutes": minutes,
            "sourceFiles": len({question["exam"] for question in unique}),
            "parsed": len(parsed),
            "unique": len(unique),
            "duplicatesRemoved": removed,
            "skipped": skipped,
            "answerConflicts": conflicts,
            "domains": domains_out,
        },
        "questions": [
            {
                "id": question["id"],
                "exam": question["exam"],
                "number": question["number"],
                "stem": question["stem"],
                "choices": question["choices"],
                "answers": question["answers"],
                "explanation": question["explanation"],
                "domain": question["domain"],
                "stemKey": question["stemKey"],
                "signals": question["signals"],
                "source": question["source"],
            }
            for question in unique
        ],
    }


def main() -> None:
    payload = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    meta = payload["meta"]
    print(f"wrote {OUT_PATH}")
    print(
        f"unique {meta['unique']} skipped {len(meta['skipped'])} "
        f"exam {meta['examSize']} in {meta['minutes']} min pass {meta['officialPass']}"
    )
    for domain in meta["domains"]:
        print(f"  {domain['id']:<22} pool {domain['poolCount']:>3}  exam {domain['examCount']:>2}")
    for item in payload["questions"]:
        print(f"{item['number']:>4}  {item['domain']:<22}  {item['stem'][:110].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
