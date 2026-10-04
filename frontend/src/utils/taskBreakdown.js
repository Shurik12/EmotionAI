// Local, deterministic task breakdown for the Razuma Focus page (EMO-17).
//
// Since EMO-19 the primary source is the AI model via POST /api/focus/breakdown.
// This module is the offline fallback used when the model is disabled or
// unavailable: a small keyword match picks a plan template, and the function
// returns translation KEYS (not human-readable strings). The UI resolves them
// through t(), so the fallback plan follows the selected language — the same
// "content as keys" approach used by BurnoutAnalyzer.

const TEMPLATES = {
  presentation: [
    'focus.plan.steps.presentation.goal',
    'focus.plan.steps.presentation.structure',
    'focus.plan.steps.presentation.draft',
    'focus.plan.steps.presentation.visual',
    'focus.plan.steps.presentation.rehearse',
  ],
  letter: [
    'focus.plan.steps.letter.goal',
    'focus.plan.steps.letter.points',
    'focus.plan.steps.letter.draft',
    'focus.plan.steps.letter.send',
  ],
  study: [
    'focus.plan.steps.study.topic',
    'focus.plan.steps.study.materials',
    'focus.plan.steps.study.firstPass',
    'focus.plan.steps.study.selfCheck',
  ],
  generic: [
    'focus.plan.steps.generic.clarify',
    'focus.plan.steps.generic.plan',
    'focus.plan.steps.generic.firstStep',
    'focus.plan.steps.generic.review',
  ],
};

// Keywords are checked low-cased and work for both RU and EN input.
const KEYWORDS = {
  presentation: ['презентац', 'слайд', 'доклад', 'выступл', 'защит', 'pitch', 'deck', 'presentation', 'slide'],
  letter: ['письм', 'сообщен', 'ответ', 'letter', 'email', 'e-mail', 'message', 'reply'],
  study: ['учёб', 'учеб', 'изуч', 'выуч', 'курс', 'экзам', 'конспект', 'study', 'learn', 'course', 'exam', 'homework'],
};

export const detectTaskType = (text = '') => {
  const normalized = text.toLowerCase();
  for (const [type, words] of Object.entries(KEYWORDS)) {
    if (words.some((word) => normalized.includes(word))) return type;
  }
  return 'generic';
};

export const breakdownTask = (text = '') => TEMPLATES[detectTaskType(text)];
