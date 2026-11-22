"""Season matching helpers for evaluation and tests."""

from typing import Any


def _norm(value: Any) -> str:
    return str(value).strip().casefold()


def _get(rule: Any, key: str, default: Any = None) -> Any:
    return rule.get(key, default) if isinstance(rule, dict) else getattr(rule, key, default)


def rule_matches(
    rule: Any, undertone: str, skintone: str, saturation: str
) -> bool:
    """Match visual attributes against a dict or SeasonRule KB entry."""
    if _norm(_get(rule, "undertone", "")) != _norm(undertone):
        return False

    combinations = _get(rule, "target_combinations", [])
    if combinations:
        return any(
            len(pair) == 2
            and _norm(pair[0]) == _norm(skintone)
            and _norm(pair[1]) == _norm(saturation)
            for pair in combinations
        )

    target_skintone = _get(rule, "target_skintone", [])
    target_saturation = _get(rule, "target_saturation", [])
    return any(_norm(value) == _norm(skintone) for value in target_skintone) and any(
        _norm(value) == _norm(saturation) for value in target_saturation
    )


def matching_seasons(
    undertone: str, skintone: str, saturation: str, kb_rules: dict
) -> list[str]:
    """Return matching season codes in knowledge-base order."""
    return [
        str(season_code)
        for season_code, rule in kb_rules.items()
        if rule_matches(rule, undertone, skintone, saturation)
    ]


def reasoning_violations(reasoning: Any, attributes: Any, kb_rules: dict) -> list[str]:
    """Return season-contract violations for a generated reasoning result."""
    expected = matching_seasons(
        attributes.undertone.value,
        attributes.skintone.value,
        attributes.saturation.value,
        kb_rules,
    )
    violations: list[str] = []
    if len(expected) != 1:
        violations.append("KB tidak memetakan kombinasi ini ke tepat satu season")
    elif reasoning.season_code != expected[0]:
        violations.append(
            f"season_code {reasoning.season_code} tidak sesuai; "
            f"seharusnya {expected[0]}"
        )

    if reasoning.secondary_season_code == reasoning.season_code:
        violations.append("secondary_season_code tidak boleh sama dengan season_code")
    if reasoning.secondary_season_code not in kb_rules:
        violations.append("secondary_season_code tidak ada di KB")
    if reasoning.is_ambiguous:
        violations.append("is_ambiguous harus false")
    return violations
