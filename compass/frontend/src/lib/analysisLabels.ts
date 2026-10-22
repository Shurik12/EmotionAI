/**
 * Russian labels for the codes the audio analyses return.
 *
 * The backend deliberately emits machine codes (`low`, `INSUFFICIENT_DATA`,
 * `voice_monotony_detected`, …) and the landing resolves them through
 * `frontend/src/utils/translations.js`. The Compass screen carries its own copy
 * so the prototype does not depend on the landing bundle; wording is kept
 * identical to the landing so both surfaces say the same thing.
 */

const label = (map: Record<string, string>, code: unknown, fallback = '—'): string => {
  if (typeof code !== 'string' || !code) return fallback;
  return map[code] ?? code;
};

export const burnoutLevel = (code: unknown): string =>
  label(
    {
      low: 'Низкий риск',
      moderate: 'Умеренный риск',
      high: 'Высокий риск',
      severe: 'Очень высокий риск',
      unknown: 'Недостаточно данных',
    },
    code,
  );

export const burnoutComponent = (code: string): string =>
  label(
    {
      exhaustion: 'Эмоциональное истощение',
      negative_activation: 'Эмоциональное напряжение',
      pause_tempo: 'Темп речи и паузы',
      positive_affect_loss: 'Снижение положительной окраски',
      prosodic_flattening: 'Интонационная выразительность',
      voice_activity_drop: 'Речевая активность',
    },
    code,
    code,
  );

export const burnoutRecommendation = (code: string): string =>
  label(
    {
      severe_risk_immediate_action: 'Очень высокий риск — требуется немедленное внимание',
      contact_employee_same_day: 'Свяжитесь с сотрудником в тот же рабочий день',
      check_wellbeing_and_workload: 'Уточните самочувствие и возможность продолжать текущую работу',
      remove_non_urgent_tasks: 'По возможности снимите необязательные срочные задачи',
      offer_support_resources: 'Предложите корпоративные ресурсы поддержки',
      high_risk_contact_supervisor: 'Свяжитесь с сотрудником в ближайший рабочий день',
      discuss_workload_and_deadlines: 'Обсудите нагрузку, сроки, переработки и необходимую поддержку',
      check_dynamics_after_actions: 'После принятых мер проверьте динамику',
      moderate_risk_preventive_measures:
        'Выявлены отдельные признаки повышенной нагрузки. Рекомендуются профилактические меры и наблюдение за динамикой.',
      discuss_workload_with_supervisor:
        'Проведите конфиденциальную беседу о текущей нагрузке, сроках, приоритетах и переработках.',
      schedule_regular_breaks:
        'Обсудите с сотрудником возможность регулярных перерывов и восстановления в течение рабочего дня.',
      repeat_assessment_7_14_days: 'Повторите оценку через 7–14 дней.',
      normal_emotional_state_detected:
        'Значимых признаков повышенного эмоционального напряжения не выявлено.',
      no_additional_measures: 'Дополнительные меры не требуются.',
      maintain_balanced_schedule:
        'Рекомендуется сохранять сбалансированный режим работы и восстановления.',
      regular_check_ups:
        'Следующую оценку рекомендуется провести по установленному графику наблюдения.',
      emotional_exhaustion_detected:
        'Признаки снижения энергии. Рекомендуется обратить внимание на режим работы и восстановления.',
      voice_monotony_detected:
        'Выявлено изменение интонационной выразительности речи относительно обычного. Оценивайте показатель только вместе с другими сигналами и динамикой.',
      speech_pattern_changes:
        'Выявлены изменения темпа речи или пауз. Уточните контекст записи и повторите оценку в сопоставимых условиях.',
      high_negative_activation:
        'Выявлен повышенный уровень напряжения. Рекомендуется обсудить возможные источники стресса.',
      reduced_positive_affect:
        'Выявлено снижение положительной эмоциональной окраски речи. Это неспецифический сигнал.',
      reduced_speech_activity:
        'Выявлено снижение речевой активности относительно обычного для сотрудника.',
      analysis_failed_retry: 'Ошибка анализа записи. Попробуйте другой фрагмент.',
    },
    code,
    code,
  );

export const burnoutComment = (code: string): string =>
  label(
    {
      need_history_to_distinguish_short_vs_chronic:
        'Без истории нескольких записей нельзя надёжно различить кратковременное напряжение и хроническое выгорание.',
      need_dynamics_for_sustained_stress:
        'Для разграничения кратковременного напряжения и устойчивого риска нужна динамика нескольких сопоставимых записей.',
      burnout_compatible_not_diagnosis:
        'Результат совместим с профессиональным выгоранием, но не является диагнозом.',
      low_affect_nonspecific_signal:
        'Сниженная эмоциональная выразительность — неспецифический сигнал, требующий сопоставления с контекстом.',
    },
    code,
    code,
  );

export const externalStatus = (code: unknown): string =>
  label(
    {
      INSUFFICIENT_DATA: 'Недостаточно данных для оценки',
      LOW: 'Выраженных признаков не выявлено',
      ELEVATED_TENSION: 'Повышенное напряжение',
      POSSIBLE_EXTERNAL_PRESSURE: 'Возможное внешнее давление',
      PROBABLE_EXTERNAL_INFLUENCE: 'Вероятное внешнее воздействие',
      HIGH_EXTERNAL_INFLUENCE_RISK: 'Высокий риск внешнего воздействия',
    },
    code,
  );

export const externalAction = (code: unknown): string =>
  label(
    {
      STANDARD_PROCESS: 'Обрабатывать в стандартном порядке',
      CONTINUE_NO_BLOCK: 'Продолжить без блокировки',
      ASK_CONTROL_QUESTIONS: 'Задать контрольные вопросы',
      ENHANCED_ANTIFRAUD_CHECK: 'Провести усиленную антифрод-проверку',
      PRIORITY_ANTIFRAUD_REVIEW: 'Провести приоритетную антифрод-проверку',
    },
    code,
  );

export const externalFactor = (code: string): string =>
  label(
    {
      NEGATIVE_ACTIVATION_HIGH: 'Повышенное эмоциональное напряжение',
      PAUSE_TEMPO_DEVIATION: 'Отклонение темпа речи и пауз',
      PROSODIC_DEVIATION: 'Отклонение интонационной выразительности',
      POSITIVE_AFFECT_LOSS_HIGH: 'Снижение положительной эмоциональной окраски',
      PERSISTENT_PATTERN: 'Устойчивое сочетание признаков (не менее 2 из 3 фрагментов)',
      CONTEXT_CONFIRMED: 'Подтверждено контекстными признаками',
    },
    code,
    code,
  );

export const burnoutState = (code: unknown): string =>
  label(
    {
      NORMAL: 'Норма',
      SHORT_STRESS: 'Кратковременное напряжение',
      SUSTAINED_STRESS: 'Устойчивое напряжение',
      BURNOUT_LIKE: 'Признаки, совместимые с выгоранием',
      LOW_AFFECT_UNSPECIFIC: 'Сниженная выразительность (неспецифичный сигнал)',
      INSUFFICIENT_DATA: 'Недостаточно данных',
      UNKNOWN: 'Недостаточно данных',
    },
    code,
  );

/** `top_factor` is an English display name chosen by the backend, not a code. */
export const burnoutTopFactor = (name: string): string =>
  label(
    {
      'Emotional Exhaustion': 'Эмоциональное истощение',
      'Prosodic Flattening': 'Интонационная выразительность',
      'Pause/Tempo Changes': 'Темп речи и паузы',
      'Negative Activation': 'Эмоциональное напряжение',
      'Positive Affect Loss': 'Снижение положительной окраски',
      'Reduced Speech Activity': 'Речевая активность',
      Unknown: 'Недостаточно данных',
    },
    name,
    name,
  );
