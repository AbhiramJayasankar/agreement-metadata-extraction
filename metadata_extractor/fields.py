"""Output formatting and arithmetic on extracted facts."""

from calendar import monthrange
from collections.abc import Mapping
from datetime import date, timedelta
import re

FIELDS = (
    "Agreement Value",
    "Agreement Start Date",
    "Agreement End Date",
    "Renewal Notice (Days)",
    "Party One",
    "Party Two",
)
CSV_COLUMNS = (
    "File Name",
    "Aggrement Value",
    "Aggrement Start Date",
    "Aggrement End Date",
    "Renewal Notice (Days)",
    "Party One",
    "Party Two",
)
FIELD_TO_COLUMN = dict(zip(FIELDS, CSV_COLUMNS[1:]))
FACT_KEYS = (
    "monthly_rent", "start_date", "end_date", "term_months",
    "notice_amount", "notice_unit", "lessor", "lessee",
)


def format_answer(field: str, answer: object) -> str:
    """Format only the answer; never consult the source document."""
    value = "" if answer is None else str(answer).strip()
    if field == "Agreement Value":
        value = re.sub(r"(?<=\d)\.\d+(?=\D*$)", "", value)
    if field in ("Agreement Value", "Renewal Notice (Days)"):
        return "".join(character for character in value if character.isdecimal())
    if field in ("Agreement Start Date", "Agreement End Date"):
        parts = re.fullmatch(r"(\d{1,2})\s*[./-]\s*(\d{1,2})\s*[./-]\s*(\d{4})", value)
        if parts:
            day, month, year = parts.groups()
            return f"{day.zfill(2)}.{month.zfill(2)}.{year}"
    return value


def format_fields(answers: Mapping[str, object]) -> dict[str, str]:
    return {field: format_answer(field, answers.get(field)) for field in FIELDS}


def infer_end_date(start: str, months: str) -> str:
    """Add calendar months with month-end clamping, then subtract one day."""
    if not months.isdecimal() or int(months) <= 0:
        return ""
    try:
        day, month, year = map(int, start.split("."))
        beginning = date(year, month, day)
        year, month_index = divmod(beginning.year * 12 + beginning.month - 1 + int(months), 12)
        month = month_index + 1
        last_day = min(beginning.day, monthrange(year, month)[1])
        end = date(year, month, last_day) - timedelta(days=1)
        return f"{end.day:02}.{end.month:02}.{end.year:04}"
    except (ValueError, OverflowError):
        return ""


def clean_party(name: str) -> str:
    name = re.sub(r"^(?:Sri|Shri|Smt|Mr|Mrs|Ms|Dr|Prof)(?:\.\s*|\s+|$)", "", name.strip(), count=1, flags=re.IGNORECASE)
    return name.rstrip(".,;:").strip()


def facts_to_fields(facts: Mapping[str, object]) -> dict[str, str]:
    """Use model answers only; the source document is never consulted."""
    values = {key: "" if facts.get(key) is None else str(facts[key]).strip() for key in FACT_KEYS}
    start = format_answer("Agreement Start Date", values["start_date"])
    end = format_answer("Agreement End Date", values["end_date"])
    if not end:
        end = infer_end_date(start, values["term_months"])
    notice = ""
    amount, unit = values["notice_amount"], values["notice_unit"].casefold().removesuffix("s")
    if amount.isdecimal() and unit in ("day", "month"):
        notice = str(int(amount) * (30 if unit == "month" else 1))
    return {
        "Agreement Value": format_answer("Agreement Value", values["monthly_rent"]),
        "Agreement Start Date": start,
        "Agreement End Date": end,
        "Renewal Notice (Days)": notice,
        "Party One": clean_party(values["lessor"]),
        "Party Two": clean_party(values["lessee"]),
    }
