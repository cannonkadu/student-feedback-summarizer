"""Milestone 1: Synthetic student feedback & summary dataset structured for lfllm-lflora training."""
from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "Summarize the following student course feedback into a concise, actionable insight for the instructor.\n\n"
    "### Feedback:\n"
    "{feedback}\n\n"
    "### Summary:\n"
)


@dataclass(frozen=True)
class FeedbackSample:
    id: str
    course: str
    category: str
    sentiment: str
    feedback: str
    summary: str

    def to_prompt(self) -> str:
        return PROMPT_TEMPLATE.format(feedback=self.feedback.strip())

    def to_training_text(self) -> str:
        return f"{self.to_prompt()}{self.summary.strip()}"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["prompt"] = self.to_prompt()
        data["text"] = self.to_training_text()
        return data


SYNTHETIC_FEEDBACK_DATA: list[FeedbackSample] = [
    FeedbackSample(
        id="fb-01",
        course="CS101 - Intro to Python",
        category="Lectures",
        sentiment="mixed",
        feedback=(
            "The live coding demos during lectures were super engaging and helped me understand loops and functions. "
            "However, the professor sometimes scrolled through the IDE too fast, making it hard to follow along on our laptops."
        ),
        summary="Live coding demos are effective, but slow down IDE scrolling so students can follow along.",
    ),
    FeedbackSample(
        id="fb-02",
        course="CS101 - Intro to Python",
        category="Assignments",
        sentiment="negative",
        feedback=(
            "Weekly homework assignments took way longer than the estimated 3 hours, especially the recursion problem set. "
            "Having a few starter test cases or hints would have saved hours of debugging unclear edge cases."
        ),
        summary="Homework workload exceeds estimates; add starter test cases and hints for difficult topics like recursion.",
    ),
    FeedbackSample(
        id="fb-03",
        course="CS204 - Data Structures",
        category="Pacing",
        sentiment="negative",
        feedback=(
            "We spent three weeks on linked lists and stacks, then rushed through balanced trees and graph algorithms in the last week. "
            "Rebalancing the schedule would make the final exam feel much fairer."
        ),
        summary="Rebalance course pacing: spend less time on basic lists and more time on trees and graphs.",
    ),
    FeedbackSample(
        id="fb-04",
        course="CS204 - Data Structures",
        category="Labs",
        sentiment="positive",
        feedback=(
            "The visualizers used in lab sessions for BFS, DFS, and Dijkstra's algorithm made everything click immediately. "
            "The TAs were also very patient and walked through time complexity step by step."
        ),
        summary="Algorithm visualizers and step-by-step TA walkthroughs in labs significantly improved comprehension.",
    ),
    FeedbackSample(
        id="fb-05",
        course="ML301 - Machine Learning",
        category="Materials",
        sentiment="mixed",
        feedback=(
            "The lecture slides are packed with great math derivations, but they are posted only an hour before class. "
            "Uploading slides 24 hours in advance would let us review the notation beforehand."
        ),
        summary="Lecture slides have strong mathematical depth; publish them at least 24 hours before class.",
    ),
    FeedbackSample(
        id="fb-06",
        course="ML301 - Machine Learning",
        category="Grading",
        sentiment="negative",
        feedback=(
            "Grading on the midterm project felt inconsistent across TAs. "
            "A detailed rubric showing exact point breakdowns for model evaluation and write-up quality is really needed."
        ),
        summary="Standardize TA grading by publishing a detailed rubric with clear point breakdowns.",
    ),
    FeedbackSample(
        id="fb-07",
        course="ML301 - Machine Learning",
        category="Projects",
        sentiment="positive",
        feedback=(
            "The capstone project where we fine-tuned a small model on a real dataset was the highlight of the semester. "
            "It connected theory from gradient descent and regularization directly to practical engineering."
        ),
        summary="The hands-on capstone project effectively bridged ML theory with practical engineering skills.",
    ),
    FeedbackSample(
        id="fb-08",
        course="DS210 - Applied Statistics",
        category="Support",
        sentiment="negative",
        feedback=(
            "Office hours were always packed right before Friday deadlines, and many students never got their questions answered. "
            "Adding a second TA slot on Thursday evenings or an active Q&A forum would help a lot."
        ),
        summary="Expand pre-deadline support by adding Thursday office hours or an active Q&A discussion board.",
    ),
    FeedbackSample(
        id="fb-09",
        course="DS210 - Applied Statistics",
        category="Lectures",
        sentiment="positive",
        feedback=(
            "Using real-world A/B testing case studies from tech and healthcare made hypothesis testing genuinely interesting. "
            "Please keep those case studies in future semesters."
        ),
        summary="Real-world A/B testing case studies boosted engagement with hypothesis testing concepts.",
    ),
    FeedbackSample(
        id="fb-10",
        course="WEB220 - Full-Stack Dev",
        category="Labs",
        sentiment="mixed",
        feedback=(
            "Building a full React and FastAPI app in lab was awesome, but half the first lab was wasted fixing local Node and Python version mismatches. "
            "Providing a Docker devcontainer or setup script before week one would prevent this."
        ),
        summary="Full-stack labs are valuable; provide a pre-configured setup script or devcontainer to avoid environment issues.",
    ),
    FeedbackSample(
        id="fb-11",
        course="WEB220 - Full-Stack Dev",
        category="Assignments",
        sentiment="positive",
        feedback=(
            "Automated GitHub Actions feedback on every push made it super clear whether my API endpoints passed the spec. "
            "It felt like working on a real engineering team."
        ),
        summary="Automated CI checks on assignments gave fast, clear feedback and simulated real-world workflows.",
    ),
    FeedbackSample(
        id="fb-12",
        course="NLP402 - Natural Language Processing",
        category="Materials",
        sentiment="mixed",
        feedback=(
            "The reading list includes classic papers and recent LLM breakthroughs, which is great, "
            "but some readings assume prior PyTorch knowledge that was not listed in the prerequisites."
        ),
        summary="Reading list is strong, but clarify PyTorch prerequisites or add a brief PyTorch refresher module.",
    ),
    FeedbackSample(
        id="fb-13",
        course="NLP402 - Natural Language Processing",
        category="Pacing",
        sentiment="mixed",
        feedback=(
            "The transformer architecture and LoRA fine-tuning modules were fantastic, "
            "though having two major paper reviews due the same week as the coding lab created an unnecessary crunch."
        ),
        summary="Transformer and LoRA modules were well received; stagger paper review deadlines away from coding labs.",
    ),
    FeedbackSample(
        id="fb-14",
        course="DB310 - Database Systems",
        category="Lectures",
        sentiment="negative",
        feedback=(
            "The audio in recorded lectures was muffled whenever the instructor stepped away from the podium to draw on the whiteboard. "
            "A clip-on microphone or digital tablet would make recordings much more accessible."
        ),
        summary="Improve lecture recording audio quality using a clip-on mic or digital whiteboard.",
    ),
    FeedbackSample(
        id="fb-15",
        course="DB310 - Database Systems",
        category="Grading",
        sentiment="mixed",
        feedback=(
            "Exam questions were fair and matched the practice problems, but it took almost four weeks to get midterm grades back. "
            "Faster grading turnaround would help us know what to improve before the drop deadline."
        ),
        summary="Exams align well with practice material, but reduce midterm grading turnaround time.",
    ),
    FeedbackSample(
        id="fb-16",
        course="SYS350 - Operating Systems",
        category="Support",
        sentiment="positive",
        feedback=(
            "The instructor's detailed Piazza responses and debugging guides for kernel panics saved me multiple times. "
            "This is the most responsive teaching staff I have had."
        ),
        summary="Responsive teaching staff and kernel debugging guides provided strong student support.",
    ),
    FeedbackSample(
        id="fb-17",
        course="SYS350 - Operating Systems",
        category="Assignments",
        sentiment="mixed",
        feedback=(
            "The virtual memory lab taught me more than any lecture, but the specification PDF had contradictory details about page table flags. "
            "Updating the spec with a single source of truth would eliminate confusion."
        ),
        summary="The virtual memory lab is highly educational; resolve contradictory details in the specification PDF.",
    ),
    FeedbackSample(
        id="fb-18",
        course="UX150 - Human-Computer Interaction",
        category="Projects",
        sentiment="mixed",
        feedback=(
            "Team critique sessions gave us great design ideas, but some group members did not contribute equally. "
            "Adding mid-semester peer evaluations would keep teams accountable."
        ),
        summary="Design critiques are helpful; introduce mid-semester peer evaluations to improve team accountability.",
    ),
    FeedbackSample(
        id="fb-19",
        course="UX150 - Human-Computer Interaction",
        category="Lectures",
        sentiment="positive",
        feedback=(
            "Short 5-minute interactive polls during lectures kept everyone awake and sparked great discussions about accessibility."
        ),
        summary="Interactive lecture polls effectively maintained engagement and encouraged accessibility discussions.",
    ),
    FeedbackSample(
        id="fb-20",
        course="SEC410 - Cybersecurity",
        category="Labs",
        sentiment="positive",
        feedback=(
            "Capture-the-flag style labs for SQL injection and XSS were thrilling and made security concepts unforgettable. "
            "Would love one more lab focused on cloud IAM misconfigurations."
        ),
        summary="CTF security labs were highly engaging; consider adding a lab on cloud IAM misconfigurations.",
    ),
    FeedbackSample(
        id="fb-21",
        course="CS101 - Intro to Python",
        category="Materials",
        sentiment="positive",
        feedback=(
            "The concise cheat sheets for list comprehensions and dictionaries were super helpful during lab quizzes."
        ),
        summary="Concise syntax cheat sheets were a valuable study aid for lab quizzes.",
    ),
    FeedbackSample(
        id="fb-22",
        course="ML301 - Machine Learning",
        category="Pacing",
        sentiment="negative",
        feedback=(
            "Moving directly from linear regression to backpropagation in one lecture felt too abrupt for students without matrix calculus."
        ),
        summary="Add a transitional matrix calculus review before introducing backpropagation.",
    ),
    FeedbackSample(
        id="fb-23",
        course="NLP402 - Natural Language Processing",
        category="Projects",
        sentiment="positive",
        feedback=(
            "Evaluating our LoRA adapters with ROUGE scores and comparing merged vs unmerged latency made deployment trade-offs crystal clear."
        ),
        summary="Hands-on ROUGE evaluation and LoRA adapter merging clarified real-world deployment trade-offs.",
    ),
    FeedbackSample(
        id="fb-24",
        course="DS210 - Applied Statistics",
        category="Grading",
        sentiment="positive",
        feedback=(
            "Allowing one dropped quiz and offering bonus points for clear data visualizations reduced stress while rewarding quality work."
        ),
        summary="Flexible quiz policies and visualization bonuses reduced stress and encouraged higher-quality work.",
    ),
]


def get_synthetic_dataset() -> list[dict]:
    """Return the synthetic student feedback-summary dataset structured for lfllm-lflora training."""
    return [sample.to_dict() for sample in SYNTHETIC_FEEDBACK_DATA]


def split_dataset(
    records: list[dict],
    val_ratio: float = 0.2,
) -> tuple[list[dict], list[dict]]:
    """Deterministically split records into train and validation partitions."""
    if not records:
        return [], []
    if len(records) == 1:
        return list(records), []
    val_count = max(1, int(round(len(records) * val_ratio)))
    val_count = min(val_count, len(records) - 1)
    return list(records[:-val_count]), list(records[-val_count:])


def format_for_lora(feedback: str, summary: str | None = None) -> str:
    """Format a single feedback comment (and optional summary) using the lfllm-lflora prompt template."""
    prompt = PROMPT_TEMPLATE.format(feedback=feedback.strip())
    if summary is not None:
        return f"{prompt}{summary.strip()}"
    return prompt


def parse_uploaded_feedback(filename: str, content: bytes) -> list[str]:
    """Parse uploaded .txt, .csv, or .json feedback files into a list of feedback strings."""
    text = content.decode("utf-8", errors="replace").strip()
    if not text:
        return []

    lower = filename.lower()
    if lower.endswith(".json"):
        data = json.loads(text)
        if isinstance(data, list):
            results = []
            for item in data:
                if isinstance(item, str) and item.strip():
                    results.append(item.strip())
                elif isinstance(item, dict):
                    fb = item.get("feedback") or item.get("comment") or item.get("text")
                    if fb and str(fb).strip():
                        results.append(str(fb).strip())
            return results
        if isinstance(data, dict):
            fb = data.get("feedback") or data.get("comment") or data.get("text")
            return [str(fb).strip()] if fb else []
        return []

    if lower.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames:
            for col in ("feedback", "comment", "comments", "text", "review"):
                for field in reader.fieldnames:
                    if field and field.strip().lower() == col:
                        return [row[field].strip() for row in reader if row.get(field, "").strip()]
            first_col = reader.fieldnames[0]
            return [row[first_col].strip() for row in reader if row.get(first_col, "").strip()]

    # Plain text: one comment per non-empty line
    return [line.strip() for line in text.splitlines() if line.strip()]


def export_jsonl(records: Iterable[dict]) -> str:
    """Serialize dataset records to JSONL format."""
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in records)


def save_dataset_jsonl(records: Iterable[dict], path: str | Path) -> Path:
    """Write dataset records to a JSONL file on disk."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(export_jsonl(records), encoding="utf-8")
    return out
