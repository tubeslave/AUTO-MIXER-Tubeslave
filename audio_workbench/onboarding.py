from __future__ import annotations
from typing import Any

DOMAINS=("drums","bass","guitars","vocals","space","stereo","macro","master")

def intake_questions(track_count: int | None=None, inferred_intent: dict[str,Any] | None=None) -> dict[str,Any]:
    inferred=inferred_intent or {}
    questions=[
      {"id":"references","required":False,
       "question":"Есть ли 1–3 референса? Для каждого: что именно нравится (drums/bass/guitars/vocals/space/stereo/macro/master)?",
       "why":"Reference DNA берет выбранные свойства, а не копирует весь трек."},
      {"id":"hierarchy","required":True,
       "question":"Кто должен быть впереди в куплете, припеве, соло/проигрыше и финале?",
       "why":"Это задает section-specific protect/candidate priorities для masking и баланса."},
      {"id":"energy","required":True,
       "question":"Как должна развиваться энергия: насколько припев больше куплета, где кульминация, должен ли финал быть максимальным?",
       "why":"Macro Director проверяет драматургию, а не только локальный баланс."},
      {"id":"space","required":True,
       "question":"Как должна меняться глубина: сухой/близкий куплет, средний припев, большой соло/финал или другая схема?",
       "why":"Sectional Space Director строит пространство по структуре песни."},
      {"id":"character","required":True,
       "question":"Назови 2–5 приоритетов характера микса: punch, плотность, грязь/чистота, яркость, ширина, близость, натуральность, агрессия и т.п.",
       "why":"Это protected objectives и направления экспериментов."},
      {"id":"do_not_break","required":True,
       "question":"Что нельзя испортить ни при каких коррекциях?",
       "why":"Эти свойства становятся protected regression gates."},
      {"id":"delivery","required":False,
       "question":"Нужен только premaster или также loud master? Есть ли целевая площадка/формат?",
       "why":"Микс и мастеринг оцениваются раздельно."},
    ]
    unresolved=[q for q in questions if q["id"] not in inferred or inferred.get(q["id"]) in (None,[],"")]
    return {"track_count":track_count,"questions":unresolved,"inferred":inferred,
            "recommended_reference_count":"1–3; один достаточен, 2–3 полезны только если у каждого назначена роль",
            "policy":"сначала вывести максимум намерения из аудио и референсов; задать только короткие прямые вопросы по реально неоднозначным решениям"}

def build_intent(answers: dict[str,Any]) -> dict[str,Any]:
    missing=[q for q in ("hierarchy","energy","space","character","do_not_break") if not answers.get(q)]
    refs=answers.get("references",[])
    normalized=[]
    for r in refs:
        domains=r.get("domains",[])
        bad=[d for d in domains if d not in DOMAINS]
        if bad: raise ValueError(f"unknown reference domains: {bad}")
        normalized.append({"name":r.get("name"),"path":r.get("path"),"domains":domains,
                           "notes":r.get("notes","")})
    return {"ready":not missing,"missing":missing,"references":normalized,
            "hierarchy":answers.get("hierarchy"),"energy":answers.get("energy"),
            "space":answers.get("space"),"character":answers.get("character"),
            "protected":answers.get("do_not_break"),"delivery":answers.get("delivery"),
            "policy":"intent guides hypotheses and gates; it is not a license to force metrics"}


def infer_from_reference(reference_traits: dict[str,Any], confidence_threshold: float=.75) -> dict[str,Any]:
    """Convert sufficiently confident reference observations into proposed MixIntent fields.
    These are proposals, not hidden requirements; ambiguity stays unresolved.
    """
    out={}
    for field in ("hierarchy","energy","space","character","delivery"):
        item=reference_traits.get(field)
        if isinstance(item,dict) and float(item.get("confidence",0))>=confidence_threshold:
            out[field]=item.get("value")
    return out
