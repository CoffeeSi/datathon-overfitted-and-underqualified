"""Separate visible facts, proposed catalog mappings and conditional rule references.

Catalog membership is not model support or a legal finding. These gates do not
read binary labels and never fill missing facts from a model's confidence score.
"""
import math
import re


# Shared by annotation proposals and the actual action-data preparation path.
# A stored, explicit per-type review remains a separate source of supervision.
SEMANTIC_GAPS = {
    'unattended_bag_contents_unknown': 'Содержимое сумки и факт предыдущего приёма наличных не установлены.',
    'employee_handles_suspected_cash_without_client': 'Работа с предполагаемыми наличными не устанавливает нарушение хранения принятых денег.',
    'suspected_accepted_cash_left_after_client_departure': 'Не установлены предмет (наличные) и факт его предыдущего приёма.',
    'employee_steps_away_suspected_cash_on_desk': 'Уход от стола не подтверждает, что обслуживание клиента ещё продолжалось.',
}

VISUAL_REQUIREMENTS = {
    'unsecured_accepted_cash': ('banknotes_visible', 'prior_receipt_observed', 'cash_remains_after_service_transition', 'accepted_cash_outside_lockable_storage_visible'),
    'cash_counted_without_client_present': ('cash_count_observed', 'client_receipt_context_observed', 'client_absent_during_count'),
    'cashier_leaves_during_client_service': ('cashier_departure_observed', 'ongoing_client_service_observed'),
    'client_identity_check_not_visible': ('cash_deposit_observed', 'identity_check_context_covered', 'identity_check_not_visible'),
    'employee_personal_items_in_cash_unit': ('personal_item_visible', 'employee_ownership_observed', 'cash_unit_context_observed', 'permitted_item_exceptions_checked'),
    'recount_without_visible_controller': ('cash_count_observed', 'internal_recount_context_observed', 'controller_context_covered', 'controller_not_visible'),
    'collection_cash_amount_registration_not_observed': ('collection_cash_receipt_observed', 'registration_context_covered', 'registration_not_visible'),
    'cash_bag_received_without_visible_identity_authority_check': ('cash_bag_receipt_observed', 'identity_authority_context_covered', 'identity_authority_check_not_visible'),
    'safe_key_left_in_lock': ('safe_lock_identified', 'key_visible_in_lock', 'key_left_in_lock_sequence_observed'),
    'suspected_stamp_or_seal_left_unattended': ('stamp_or_seal_identified', 'item_left_unattended_sequence_observed'),
}

FACT_DESCRIPTIONS = {
    'banknotes_visible': 'Предмет различим как банкноты, а не просто прямоугольник.',
    'prior_receipt_observed': 'Видно предшествующее получение этих денег; одного наличия денег недостаточно.',
    'cash_remains_after_service_transition': 'Деньги остаются при наблюдаемом уходе/смене клиента.',
    'identity_check_context_covered': 'Просмотрен достаточный контекст до и во время приёма; короткий фрагмент не подтверждает отсутствие проверки.',
    'registration_context_covered': 'В кадре различим именно релевантный процесс регистрации, а не просто работа за компьютером.',
    'identity_authority_context_covered': 'Достаточен обзор процесса проверки личности и полномочий передающего лица.',
    'controller_context_covered': 'Достаточен обзор возможного присутствия контролирующего работника.',
}


def proposed_mapping(observed_tag):
    """A tag suggests a candidate only; no automatic confirmation of conditions."""
    from .catalog import supported_events
    candidates = sorted(e['event_type'] for e in supported_events()
                        if observed_tag in e.get('maps_from_observed_event_type', []))
    reason = SEMANTIC_GAPS.get(observed_tag)
    return dict(observed_event_type=observed_tag, catalog_candidates=candidates,
                candidate_types=[] if reason else candidates,
                status='needs_context_review' if reason else 'candidate_only' if candidates else 'unmapped',
                reason=reason, confirmed_type=None, requires_review=True)


def grounded_rule_match(event_type, facts, *, evidence=None):
    """Return a candidate type/reference only with explicit facts and traceable evidence.

    ``facts`` values are True/False/None. An absent or unknown defining fact blocks
    emission. The caller owns factual verification: this helper is not an action
    recognizer. A binary model score is never accepted as an evidence method.
    Even complete visible facts do not establish all legal applicability conditions.
    """
    from .catalog import event_by_type, source_links_for
    if not isinstance(facts, dict) or any(v is not None and type(v) is not bool for v in facts.values()):
        raise ValueError('Facts must explicitly be true, false or unknown')
    event = event_by_type(event_type)
    required = VISUAL_REQUIREMENTS.get(event_type)
    supported = bool(event and event.get('api_supported') and required)
    required = required or ()
    unknown = [name for name in required if facts.get(name) is None]
    contradicted = [name for name in required if facts.get(name) is False]
    evidence = {} if evidence is None else evidence
    if not isinstance(evidence,dict):
        raise ValueError('Evidence provenance must be an object')
    method = evidence.get('method')
    provenance_ok = (method in {'reviewed_frame_sequence', 'evaluated_action_model', 'evaluated_object_sequence'}
        and bool(re.fullmatch(r'[0-9a-f]{64}', str(evidence.get('video_sha256', ''))))
        and isinstance(evidence.get('basis_id'), str) and bool(evidence['basis_id'].strip()))
    times = evidence.get('timestamps_sec')
    provenance_ok = bool(provenance_ok and isinstance(times, list) and len(times)>=2
        and all(type(t) in (int,float) and math.isfinite(t) and t >= 0 for t in times)
        and all(a<b for a,b in zip(times,times[1:])))
    if method in {'evaluated_action_model', 'evaluated_object_sequence'}:
        provenance_ok = bool(provenance_ok and re.fullmatch(r'[0-9a-f]{64}',str(evidence.get('checkpoint_sha256','')))
                             and evidence.get('quality_approved') is True)
    allowed = supported and not unknown and not contradicted and provenance_ok
    links = source_links_for(event_type) if allowed else []
    allowed = bool(allowed and links)
    return dict(candidate_event_type=event_type, event_type=event_type if allowed else None,
                visual_basis_sufficient=allowed, required_visual_facts=list(required),
                unknown_visual_facts=unknown, contradicted_visual_facts=contradicted,
                evidence_provenance_valid=provenance_ok,
                source_links=links if allowed else [],
                rule_status='conditional_reference' if allowed else 'unresolved',
                applicability_to_verify=list(event.get('applicability', [])) if event else [],
                video_cannot_establish=list(event.get('video_cannot_establish', [])) if event else [],
                legal_conclusion=None, requires_review=True,
                reason=('Visible candidate supported; rule applicability still requires human review.' if allowed
                    else 'Unsupported type, missing/contradicted visual facts, or unverified evidence.'))
