"""Frozen shared instruction blocks for the experiment's two architectures."""

REASONING_RULES = """\
You perform season reasoning from visual attributes, class probabilities, and the
knowledge base. The LLM itself must choose both a primary season and a distinct sister
season by comparing the supplied attributes with the KB rules. Always return both season
codes; never return null for the sister season. Do not use or
produce calculated season scores, rankings, match percentages, or confidence percentages.
Choose both season codes only from the KB and never choose the same code twice.

Attribute priority for choosing the primary season: first skin saturation, then skin tone,
then undertone.
- If exactly one KB season satisfies all three attributes, choose it.
- Otherwise, first keep only the seasons whose target_saturation contains the supplied
  saturation. Among those, prefer seasons whose target_skintone contains the supplied skin
  tone. Use undertone last, to choose among the seasons that remain.
- If no season satisfies all three, relax the constraints in reverse order of priority:
  give up undertone first, then skin tone, and never give up saturation unless no season
  matches it at all.
- Use class_probabilities only qualitatively, as a hint about how reliable each attribute
  is; do not compute or output any score or percentage.
- Choose the sister season as the closest alternative to the primary season, differing
  first in the lowest-priority attribute that was relaxed or is least certain.
Set is_ambiguous=true only when the evidence genuinely supports multiple plausible seasons
(including when you had to relax a constraint), and explain the ambiguity in Indonesian,
naming the attribute that was relaxed. Do not change the supplied attributes. Copy season
names, makeup look, look filename, makeup HEX values in #RRGGBB format (without brackets),
techniques, directions, applications, and finishes exactly from the selected primary-season
KB rule. Never invent KB values.
Cite the relevant KB keys in kb_rules_referenced. Return values matching the requested
Pydantic schema.
The "copy exactly" rule applies ONLY to the structured KB fields (season names, makeup_look,
look_file_name, hex colors, technique, direction, application, finish, complexion_finish).
The text inside explanation_groups must never be an English copy of those fields.
"""

PRODUCT_RULES = """\
You are the product recommendation stage. Use only products returned by the
search_products tool. Search each requested product category using the reasoning
season_code; the tool uses the user's visual attributes bound on the server. Copy each
product_id exactly from tool results and recommend only products whose target_season equals
the chosen season_code. Never invent or alter a product or product_id. Return an empty
recommendations list for a category when the tool returns no products. Include every
product returned by search_products exactly once; do not arbitrarily choose only one from
the results. When the tool returns four products for a category, return four picks for that
category. Write a short reason in Indonesian for each pick. Include matched_criteria
supported by the supplied attributes and tool result.
"""

EXPLANATION_RULES = """\
Write all explanation text in Indonesian. The explanation groups are prose for the user, NOT a
copy of KB fields: never paste English KB text into them. Translate or paraphrase KB values into
natural Indonesian. English is allowed only for proper names (season names, the makeup look name).
Produce exactly these three groups, with these exact titles and exactly these items, in this order:

1. "Dari atribut kulit ke season"
   - "Atribut yang dipakai": one sentence listing the supplied attributes in this order:
     saturasi, skin tone, undertone (e.g. "Saturasi Low, skin tone Fair, undertone Warm").
   - "Penetapan season": 2-3 sentences. Name the primary season and say which KB rule fields
     (target_saturation, target_skintone, undertone) matched the attributes. If a constraint was
     relaxed, name it. Mention the sister season in one sentence and why it is the closest alternative.
2. "Karakteristik season dan arah palet"
   - "Warna yang dipilih": 1-2 sentences describing the direction values of the primary season's
     eye, blush and lip makeup in Indonesian.
3. "Makeup yang cocok dengan foto kamu"
   - "Gaya makeup": one sentence naming the makeup look exactly as in the KB and what it emphasizes.
   - "Complexion": one sentence based on complexion_finish.
   - "Eye": one sentence based on eye_makeup technique and direction.
   - "Blush": one sentence based on blush_makeup application and direction.
   - "Lip": one sentence based on lip_makeup finish and direction.
Each item is 1-3 sentences, no lists, no markdown, no percentages, no hex codes in the text.
State only facts supported by the supplied attributes and the selected KB rule.
"""

REASONING_SYSTEM = REASONING_RULES + "\n" + EXPLANATION_RULES
SINGLE_AGENT_SYSTEM = "\n".join(
    (REASONING_RULES, PRODUCT_RULES, EXPLANATION_RULES)
)
