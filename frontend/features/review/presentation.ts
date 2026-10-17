import type { Finding } from "@/lib/contracts";
const names: Record<string, string> = {
  unsecured_accepted_cash: "Хранение наличных",
  cash_counted_without_client_present: "Пересчёт без клиента в кадре",
  cashier_leaves_during_client_service: "Рабочее место без сотрудника",
  client_identity_check_not_visible: "Проверка документа не видна",
  employee_personal_items_in_cash_unit: "Личные вещи в кассовой зоне",
  recount_without_visible_controller: "Контролёр не виден",
  collection_cash_amount_registration_not_observed: "Регистрация суммы не видна",
  cash_bag_received_without_visible_identity_authority_check: "Приём сумки",
  safe_key_left_in_lock: "Ключ в замке",
  suspected_stamp_or_seal_left_unattended: "Предметы без присмотра",
};
export const actionName = (type: string) => Object.hasOwn(names, type) ? names[type] : "Событие для проверки";
export function reviewEvents(findings: Finding[]) {
  return findings.map((finding, originalIndex) => {
    let hash = 2166136261;
    for (const char of `${finding.type}|${finding.timecode_sec}|${finding.description}`) hash = Math.imul(hash ^ char.charCodeAt(0), 16777619);
    return { ...finding, id: `f-${originalIndex}-${(hash >>> 0).toString(36)}`, originalIndex, title: actionName(finding.type) };
  }).sort((a, b) => a.timecode_sec - b.timecode_sec || a.originalIndex - b.originalIndex);
}
export type ReviewEvent = ReturnType<typeof reviewEvents>[number];
