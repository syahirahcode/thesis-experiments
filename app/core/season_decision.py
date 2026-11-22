"""Keputusan season deterministik dari kb_rules.json (kontrak final P0).

Aturan yang disetujui:
1. Cari season yang cocok dengan ketiga atribut (undertone, skin tone, saturation).
   - Tepat satu: SINGLE_MATCH.
   - Lebih dari satu: MULTI_MATCH (kandidat setara, urutan mengikuti kb_rules.json).
2. Jika tidak ada yang cocok (NO_RULE_MATCH):
   - Undertone tidak pernah diubah.
   - Skin tone tetap.
   - Saturation digeser ke tingkat terdekat yang memiliki aturan (Low, Medium, High).
   - Jika dua season sama dekat, keduanya menjadi kandidat setara.
   - Hasil ditandai sebagai perkiraan dan disarankan verifikasi pakar.
LLM tidak memilih season; LLM hanya menulis penjelasan dari keputusan ini.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

MatchStatus = Literal["SINGLE_MATCH", "MULTI_MATCH", "NO_RULE_MATCH"]

SATURATION_ORDER = ["low", "medium", "high"]


class SeasonDecision(BaseModel):
    match_status: MatchStatus
    candidate_seasons: list[str]
    relaxed_attributes: list[str]
    is_tie: bool
    primary_season_code: str
    fixed_sister_season_code: str | None
    status_sentence: str


def _norm(value: Any) -> str:
    return str(value).strip().casefold()


def _values(rule: dict[str, Any], key: str) -> list[str]:
    return [_norm(v) for v in rule.get(key, [])]


def _status_sentence(status: MatchStatus, is_tie: bool) -> str:
    if status == "SINGLE_MATCH":
        return "Atribut kamu cocok langsung dengan satu aturan season CFY."
    if status == "MULTI_MATCH":
        return (
            "Atribut kamu cocok dengan lebih dari satu aturan season CFY; kandidat season "
            "ditampilkan setara dan disarankan verifikasi pakar."
        )
    if is_tie:
        return (
            "Atribut kamu tidak cocok dengan aturan season CFY mana pun. Dengan menggeser "
            "saturation ke tingkat terdekat, terdapat dua season yang sama dekat sehingga "
            "keduanya ditampilkan setara sebagai perkiraan; disarankan verifikasi pakar."
        )
    return (
        "Atribut kamu tidak cocok dengan aturan season CFY mana pun. Season ini adalah "
        "perkiraan dengan menggeser saturation ke tingkat terdekat; disarankan verifikasi pakar."
    )


def decide_season(
    undertone: str, skintone: str, saturation: str, kb_rules: dict[str, Any]
) -> SeasonDecision:
    u, t, s = _norm(undertone), _norm(skintone), _norm(saturation)
    if s not in SATURATION_ORDER:
        raise ValueError(f"Saturation tidak dikenal: {saturation!r}")

    exact = [
        code
        for code, rule in kb_rules.items()
        if _norm(rule.get("undertone")) == u
        and t in _values(rule, "target_skintone")
        and s in _values(rule, "target_saturation")
    ]
    relaxed: list[str] = []
    if len(exact) == 1:
        status: MatchStatus = "SINGLE_MATCH"
        candidates = exact
    elif len(exact) > 1:
        status = "MULTI_MATCH"
        candidates = exact
    else:
        status = "NO_RULE_MATCH"
        relaxed = ["saturation"]
        same = [
            (code, rule)
            for code, rule in kb_rules.items()
            if _norm(rule.get("undertone")) == u and t in _values(rule, "target_skintone")
        ]
        if not same:
            raise ValueError(
                f"Tidak ada aturan dengan undertone={undertone} dan skin tone={skintone}; "
                "kombinasi ini harus diarahkan ke verifikasi pakar."
            )
        target = SATURATION_ORDER.index(s)

        def distance(rule: dict[str, Any]) -> int:
            return min(
                abs(SATURATION_ORDER.index(v) - target)
                for v in _values(rule, "target_saturation")
                if v in SATURATION_ORDER
            )

        best = min(distance(rule) for _, rule in same)
        candidates = [code for code, rule in same if distance(rule) == best]

    is_tie = len(candidates) > 1
    return SeasonDecision(
        match_status=status,
        candidate_seasons=candidates,
        relaxed_attributes=relaxed,
        is_tie=is_tie,
        primary_season_code=candidates[0],
        fixed_sister_season_code=candidates[1] if is_tie else None,
        status_sentence=_status_sentence(status, is_tie),
    )


def validate_reasoning(output: Any, decision: SeasonDecision) -> list[str]:
    """Kembalikan daftar pelanggaran kontrak; list kosong berarti output sah."""
    errors: list[str] = []
    if getattr(output, "match_status", None) != decision.match_status:
        errors.append(f"match_status harus {decision.match_status}")
    if list(getattr(output, "candidate_seasons", [])) != decision.candidate_seasons:
        errors.append(f"candidate_seasons harus {decision.candidate_seasons}")
    if list(getattr(output, "relaxed_attributes", [])) != decision.relaxed_attributes:
        errors.append(f"relaxed_attributes harus {decision.relaxed_attributes}")
    if output.season_code != decision.primary_season_code:
        errors.append(f"season_code harus {decision.primary_season_code}")
    if output.secondary_season_code == output.season_code:
        errors.append("secondary_season_code tidak boleh sama dengan season_code")
    if decision.fixed_sister_season_code and (
        output.secondary_season_code != decision.fixed_sister_season_code
    ):
        errors.append(f"secondary_season_code harus {decision.fixed_sister_season_code}")
    expected_ambiguous = decision.match_status != "SINGLE_MATCH"
    if output.is_ambiguous != expected_ambiguous:
        errors.append(f"is_ambiguous harus {expected_ambiguous}")
    return errors