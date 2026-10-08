// Razuma Focus intervention catalog (EMO-17).
//
// Data model for the 16 scenarios in `docs/focus/FOCUS_INTERVENTION_SPEC.md`.
// Rows 1–4 are automatic offers (driven by `emotionPolicy.decideEmotionOffer`);
// rows 9–16 are the user-driven and edge cases the component layer plays out.
//
// The same answer id can carry a different label per scenario (row 1's
// "continue" reads «Да, продолжаю», row 3's reads «Продолжить работу»), so each
// button pairs an answer id with its own translation key. Every string lives in
// `translations.js` (RU + EN).

// Answer ids referenced by the spec's `answer = ...` notation.
export const FOCUS_ANSWERS = Object.freeze({
  CONTINUE: 'continue',
  HELP_RETURN: 'help_return',
  SHOW_CURRENT_ACTION: 'show_current_action',
  HELP: 'help',
  PAUSE: 'pause',
  SPLIT_INTO_STEPS: 'split_into_steps',
  USE_STEPS: 'use_steps',
  EDIT_STEPS: 'edit_steps',
  BACK: 'back',
  RESUME: 'resume',
});

const continueBtn = { answer: 'continue', label: 'focus.action.continue' };
const continueSelfBtn = { answer: 'continue', label: 'focus.intervention.continueSelf' };
const continueWorkBtn = { answer: 'continue', label: 'focus.intervention.continueWork' };
const pauseBtn = { answer: 'pause', label: 'focus.action.pause' };

// Automatic offers (rows 1–4).
export const focusOffers = Object.freeze({
  check: {
    scenario: 1,
    gesture: 'check',
    sound: 'tu',
    messageKey: 'focus.intervention.check.message',
    buttons: [{ answer: 'continue', label: 'focus.intervention.check.continue' }, { answer: 'help_return', label: 'focus.action.helpReturn' }, pauseBtn],
  },
  help: {
    scenario: 2,
    gesture: 'help',
    sound: null,
    messageKey: 'focus.intervention.help.message',
    buttons: [{ answer: 'split_into_steps', label: 'focus.action.split' }, continueSelfBtn, pauseBtn],
  },
  support: {
    scenario: 3,
    gesture: 'support',
    sound: null,
    messageKey: 'focus.intervention.support.message',
    buttons: [continueWorkBtn, { answer: 'help', label: 'focus.action.help' }, pauseBtn],
  },
  point: {
    scenario: 4,
    gesture: 'point',
    sound: null,
    messageKey: 'focus.intervention.point.message',
    buttons: [{ answer: 'show_current_action', label: 'focus.action.showAction' }, continueSelfBtn, pauseBtn],
  },
});

// Follow-up cards (rows 9–16). `static: true` means Numa does not move.
export const focusCards = Object.freeze({
  action: {
    scenario: 9,
    gesture: 'point',
    sound: null,
    messageKey: 'focus.intervention.action.message',
    buttons: [continueBtn, { answer: 'split_into_steps', label: 'focus.action.split' }, pauseBtn],
  },
  helpMenu: {
    scenario: 11,
    gesture: null,
    sound: null,
    messageKey: 'focus.intervention.helpMenu.message',
    buttons: [
      { answer: 'show_current_action', label: 'focus.action.showActionLong' },
      { answer: 'split_into_steps', label: 'focus.action.split' },
      pauseBtn,
    ],
  },
  steps: {
    scenario: 12,
    gesture: null,
    sound: null,
    static: true,
    messageKey: 'focus.intervention.steps.message',
    buttons: [
      { answer: 'use_steps', label: 'focus.action.useSteps' },
      { answer: 'edit_steps', label: 'focus.action.edit' },
      { answer: 'back', label: 'focus.action.back' },
    ],
  },
  pause: {
    scenario: 13,
    gesture: 'pause',
    sound: null,
    messageKey: 'focus.intervention.pause.message',
    buttons: [{ answer: 'resume', label: 'focus.action.resume' }],
  },
  done: {
    scenario: 14,
    gesture: 'done',
    sound: 'tu-du',
    messageKey: 'focus.intervention.done.message',
    buttons: [],
  },
  unavailable: {
    scenario: 16,
    gesture: null,
    sound: null,
    static: true,
    messageKey: 'focus.intervention.unavailable.message',
    buttons: [],
  },
});

// Timing for the disappearing cards (rows 10 and 15).
export const DISMISS_MS = 150;
export const TIMEOUT_MS = 20000;
export const TIMEOUT_DISMISS_MS = 200;

// Which follow-up (if any) an answer leads to. `null` closes the card.
export const answerRouting = Object.freeze({
  continue: null, // row 10
  help_return: 'action', // row 9
  show_current_action: 'action', // row 9
  help: 'helpMenu', // row 11
  split_into_steps: 'steps', // row 12
  pause: 'pause', // row 13
  resume: null,
  use_steps: null,
  edit_steps: null,
  back: 'helpMenu',
});

// Resolve the offer/card definition for a stable id.
export const resolveIntervention = (id) => focusOffers[id] || focusCards[id] || null;
