"""Prompts for the two LLM steps of graph retrieval.

LLM1 turns a question into entity groups (OR inside a group, AND across groups).
LLM2 maps one entity word onto the L3/L2 taxonomy of its entity type, or sends it to
the evidence (surface-form) channel.
"""

ENTITY_TYPES = (
    "Pollutant",
    "Wastewater_Treatment_Process",
    "Reactor",
    "Treatment_Parameter",
    "Microorganism",
    "Dosed_Material",
)

LLM1_PROMPT = """You are an intent parser for wastewater-treatment literature retrieval. Break the user's natural-language question into atomic entity conditions that the graph can retrieve, and group them into "entity groups".

The graph can retrieve six entity types: Pollutant (pollutants), Wastewater_Treatment_Process (processes), Treatment_Parameter (parameters), Reactor (reactors), Microorganism (microorganisms), Dosed_Material (dosed materials).

Rules:
1. Find every entity that must actually be retrieved (specific pollutants, processes, materials, reactors, microorganisms, parameters). Do not miss entities connected by "and/with/simultaneously".
2. Group by SEMANTIC ROLE, not by entity type:
   - PARALLEL ALTERNATIVES go into the SAME group (OR): interchangeable substitutes listed as "A, B or C". Examples: "Fe-based single-atom sites, Fe-doped g-C3N4, Co-Mn spinel oxides, Ag2Se..." are all candidate catalysts (one Dosed_Material group); "peracids, peroxymonosulfate, persulfate" are all oxidants (one group).
   - ENTITIES THAT MUST CO-OCCUR go into SEPARATE groups (AND): "simultaneous removal of A and B", "A and B both present", "co-removal of A and B" — put A and B in DIFFERENT groups even if they share the same entity type (e.g. both Pollutants). Otherwise a paper hitting only one of them would wrongly satisfy the group.
     Example: "Which studies achieved removal of both nitrate and phosphate in a single biological reactor?" — nitrate and phosphate are both Pollutants but MUST be two separate groups (AND).
3. Give each entity word an entity_type (one of the six). "remove/removal", research purposes and relational phrases are NOT entities.
4. For each entity word, source_text uses the verbatim word from the question; search_text gives a normalized phrase suitable for retrieval.
5. Put relational/background phrasing into relation_handling, marked co_mention_all or ignore_as_non_entity, not as an entity.
6. If the question asks for causal/numeric/performance conclusions the graph cannot provide, still keep the retrievable co-occurrence conditions; use unsupported only if no atomic condition can be determined.
7. Entity fidelity: decompose the words that appear in the question. You MAY include synonym/near-synonym rewrites of the same entity (e.g. "cadmium" -> Cd, Cd(II), Cd2+; "bisphenol A" -> BPA) as members or within a member's search_text — these are how the same substance appears in papers. You MUST NOT invent entities the question does not imply: do NOT expand "iron-based materials" into iron oxides, ferrihydrite, goethite, hematite, magnetite, zero-valent iron etc. as separate members, and do NOT turn "arsenic" into arsenate/arsenite as separate members unless the question names them. A rewrite is allowed only if it is clearly the SAME entity under another name (Cd/Cd2+ for cadmium); it is forbidden if it is a different species, a broader class, or an unmentioned sub-material.

Output ONLY JSON (only output JSON):
{"intent": "find_papers|...|unsupported", "condition_operator": "all|any",
 "entity_groups": [{"label": "...", "entity_type": "one of the six", "required": true,
   "members": [{"source_text": "verbatim word", "search_text": "normalized phrase"}]}],
 "relation_handling": [...], "year_start": null, "year_end": null, "request_evidence": true, "reason": "explanation"}
"""

LLM2_PROMPT = """You are responsible for mapping a single entity word to the classification hierarchy (L2/L3) of the wastewater-treatment entity graph, or deciding that it cannot be placed into any category.

You handle only ONE entity word, whose entity_type is {entity_type}. Below are all L3 major categories and L2 subcategories for that type (each line: L3 / L2 / number of canonical entities):

{taxonomy_list}

Task:
1. Decide whether this entity word "fully corresponds" to some L2 subclass (or L3 major category):
   - If the entity word IS the name of that L2/L3, or is semantically clearly equivalent, return that category.
   - Broad class words (emerging contaminants, micropollutants) may correspond to several L3/L2 — list them all.
   - If the entity word is a concrete entity (antibiotics, biochar, e. coli), it is NOT a category name — such words go to evidence.
2. Return category ONLY when the entity word itself is a category name; concrete substances/materials/processes return evidence; hyper-specific forms are necessarily evidence.

Output ONLY JSON (only output JSON):
{"layer": "category" or "evidence",
 "matches": [{"entity_type": "{entity_type}", "l3": "L3", "l2": "L2 or empty"}],
 "reason": "brief explanation"}
If layer=evidence, matches is an empty array.
"""

LLM2_USER = "Entity word: {word}"
