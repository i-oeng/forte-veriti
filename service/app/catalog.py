"""Violation catalogue loader (see catalog.yaml)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import config


@dataclass
class Rule:
    document: str = ""   # full title of the act
    short: str = ""      # how the act is cited in descriptions
    url: str = ""
    reg: str = ""        # justice registration and edition
    about: str = ""      # what the act covers and why we use it
    section: str = ""
    summary: str = ""    # the requirement, close to the act's wording

    @property
    def filled(self) -> bool:
        return bool(self.document.strip())

    def as_text(self) -> str:
        return ", ".join(p for p in ((self.short or self.document).strip(), self.section.strip()) if p)


@dataclass
class ViolationType:
    id: str
    name: str
    definition: str
    why: str
    tier: str = "B"
    name_ru: str = ""                 # display name for the Russian UI
    why_ru: str = ""
    detect: str = "rules"            # rules | review | reference
    aliases: list[str] = field(default_factory=list)
    proposal_kinds: list[str] = field(default_factory=list)
    gate: dict = field(default_factory=dict)
    rule: Rule = field(default_factory=Rule)

    @property
    def label(self) -> str:
        """What the API reports as `type`: readable name plus the catalogue ID."""
        return f"{self.name} ({self.id})"

    @property
    def label_ru(self) -> str:
        return f"{self.name_ru} ({self.id})"


@dataclass
class Catalog:
    types: list[ViolationType]

    def __post_init__(self) -> None:
        self.by_id = {t.id: t for t in self.types}

    def subset(self, ids: list[str]) -> "Catalog":
        return Catalog([t for t in self.types if t.id in set(ids)])

    def canonical(self, name: str) -> str | None:
        """Catalogue ID for an ID or alias, case-insensitive (used to compare with the labels sheet)."""
        key = name.strip().lower()
        for t in self.types:
            if key == t.id.lower() or key in (a.lower() for a in t.aliases):
                return t.id
        return None

    @classmethod
    def load(cls, path: Path | None = None) -> "Catalog":
        data = yaml.safe_load(Path(path or config.CATALOG_PATH).read_text(encoding="utf-8"))
        types = []
        fields = set(Rule.__dataclass_fields__)
        for t in data["types"]:
            rule = Rule(**{k: str(v or "") for k, v in (t.get("rule") or {}).items() if k in fields})
            types.append(ViolationType(
                id=t["id"], name=t["name"], definition=" ".join(str(t["definition"]).split()), why=t["why"],
                tier=str(t.get("tier", "B")), name_ru=str(t.get("name_ru") or t["name"]),
                why_ru=str(t.get("why_ru") or t["why"]), detect=t.get("detect", "rules"), aliases=list(t.get("aliases") or []),
                proposal_kinds=list(t.get("proposal_kinds") or []), gate=dict(t.get("gate") or {}), rule=rule))
        return cls(types)
