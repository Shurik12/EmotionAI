import React, { useCallback, useState } from 'react';
import { useLanguage } from '../hooks/useLanguage';
import { breakdownTask } from '../utils/taskBreakdown';
import { FocusSession } from './FocusSession';
import { FocusCamera } from './FocusCamera';

const makeSteps = (keys) =>
  keys.map((key, index) => ({ id: `step-${Date.now()}-${index}`, key }));

const PRESET_KEYS = ['focus.presets.presentation', 'focus.presets.letter', 'focus.presets.study'];

export const Focus = () => {
  const { t } = useLanguage();
  const [activeTab, setActiveTab] = useState('movement');
  const [task, setTask] = useState('');
  const [plan, setPlan] = useState([]);
  const [timeLimit, setTimeLimit] = useState(false);
  const [numaNearby, setNumaNearby] = useState(true);

  // Session state shared by the timer and the camera offer card.
  const [stepIndex, setStepIndex] = useState(0);
  const [sessionRunning, setSessionRunning] = useState(false);
  const [completedSteps, setCompletedSteps] = useState([]);
  const [highlightStep, setHighlightStep] = useState(null);
  const [planNote, setPlanNote] = useState('');

  const tabs = [
    { id: 'movement', label: t('focus.tabs.movement') },
    { id: 'session', label: t('focus.tabs.session') },
    { id: 'how', label: t('focus.tabs.how') },
  ];

  const generate = (text) => setPlan(makeSteps(breakdownTask(text)));

  const handleGenerate = () => {
    if (!task.trim()) return;
    generate(task);
  };

  const handlePreset = (key) => {
    const text = t(key);
    setTask(text);
    generate(text);
  };

  const handleAddStep = () =>
    setPlan((prev) => [...prev, { id: `step-${Date.now()}`, key: 'focus.plan.manual' }]);

  const handleRemoveStep = (id) =>
    setPlan((prev) => prev.filter((step) => step.id !== id));

  const handlePrepare = () => {
    if (plan.length) setActiveTab('session');
  };

  const currentStep = plan[Math.min(stepIndex, Math.max(plan.length - 1, 0))];
  const hasStep = plan.length > 0 && completedSteps.length < plan.length;

  const scrollToStep = useCallback((id) => {
    if (!id) return;
    const el = document.getElementById(`focus-step-${id}`);
    if (el) {
      const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      el.scrollIntoView({ block: 'center', behavior: reduced ? 'auto' : 'smooth' });
    }
  }, []);

  const handlePoint = useCallback(() => {
    const id = currentStep?.id;
    scrollToStep(id);
    setHighlightStep(id);
    window.setTimeout(() => setHighlightStep(null), 2400);
  }, [currentStep, scrollToStep]);

  const handleSmaller = useCallback(() => {
    handlePoint();
    setPlanNote(t('focus.emotion.smallerHint'));
    window.setTimeout(() => setPlanNote(''), 6000);
  }, [handlePoint, t]);

  const renderPlan = () => (
    <div className="focus-plan">
      <div className="focus-plan-head">
        <h3>{t('focus.plan.heading')}</h3>
        <span className="focus-plan-count">{t('focus.plan.count', { count: plan.length })}</span>
      </div>

      {plan.length === 0 ? (
        <p className="focus-empty">{t('focus.plan.empty')}</p>
      ) : (
        <ol className="focus-steps">
          {plan.map((step, index) => (
            <li key={step.id} className="focus-step">
              <span className="focus-step-index">{index + 1}</span>
              <span className="focus-step-title">{t(step.key)}</span>
              <button
                type="button"
                className="focus-step-remove"
                onClick={() => handleRemoveStep(step.id)}
                aria-label={t('focus.plan.remove')}
              >
                ×
              </button>
            </li>
          ))}
        </ol>
      )}

      <button type="button" className="focus-btn ghost" onClick={handleAddStep}>
        + {t('focus.plan.addStep')}
      </button>
    </div>
  );

  return (
    <div className="focus-page">
      <section className="focus-hero">
        <span className="focus-badge">{t('focus.badge')}</span>
        <p className="focus-kicker">{t('focus.kicker')}</p>
        <h1 className="focus-title">{t('focus.title')}</h1>
        <p className="focus-subtitle">{t('focus.subtitle')}</p>
      </section>

      <div className="focus-tabs" role="tablist">
        {tabs.map(({ id, label }) => (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={activeTab === id}
            className={`focus-tab ${activeTab === id ? 'active' : ''}`}
            onClick={() => setActiveTab(id)}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="focus-panel" role="tabpanel">
        {activeTab === 'movement' && (
          <>
            <div className="focus-card">
              <h2>{t('focus.movement.title')}</h2>
              <p className="focus-prompt">{t('focus.movement.prompt')}</p>
              <p className="focus-hint">{t('focus.movement.hint')}</p>

              <div className="focus-presets">
                {PRESET_KEYS.map((key) => (
                  <button
                    key={key}
                    type="button"
                    className="focus-preset"
                    onClick={() => handlePreset(key)}
                  >
                    {t(key)}
                  </button>
                ))}
              </div>

              <textarea
                className="focus-field"
                value={task}
                onChange={(e) => setTask(e.target.value)}
                placeholder={t('focus.movement.placeholder')}
                rows={3}
              />

              <div className="focus-actions">
                <button type="button" className="focus-btn primary" onClick={handleGenerate}>
                  {plan.length ? t('focus.movement.regenerate') : t('focus.movement.generate')}
                </button>
              </div>
            </div>

            <div className="focus-card">
              {renderPlan()}

              <div className="focus-options">
                <label className="focus-option">
                  <input
                    type="checkbox"
                    checked={timeLimit}
                    onChange={(e) => setTimeLimit(e.target.checked)}
                  />
                  {t('focus.plan.timeLimit')}
                </label>
                <label className="focus-option">
                  <input
                    type="checkbox"
                    checked={numaNearby}
                    onChange={(e) => setNumaNearby(e.target.checked)}
                  />
                  {t('focus.plan.numaNearby')}
                </label>
              </div>

              <div className="focus-actions">
                <button
                  type="button"
                  className="focus-btn primary"
                  onClick={handlePrepare}
                  disabled={!plan.length}
                >
                  {t('focus.plan.prepare')}
                </button>
              </div>
            </div>
          </>
        )}

        {activeTab === 'session' && (
          <>
            <div className="focus-card">
              <h2>{t('focus.session.title')}</h2>
              <p>{t('focus.session.text')}</p>
              {planNote && <p className="focus-plan-note">{planNote}</p>}
              <FocusSession
                plan={plan}
                index={stepIndex}
                onIndexChange={setStepIndex}
                running={sessionRunning}
                onRunningChange={setSessionRunning}
                completed={completedSteps}
                onCompletedChange={setCompletedSteps}
                highlightId={highlightStep}
              />
            </div>

            <div className="focus-card">
              <FocusCamera
                numaNearby={numaNearby}
                stepId={hasStep ? currentStep?.id ?? null : null}
                hasStep={hasStep}
                running={sessionRunning}
                onPoint={handlePoint}
                onSmaller={handleSmaller}
                onPause={() => setSessionRunning(false)}
                onResume={() => setSessionRunning(true)}
              />
            </div>
          </>
        )}

        {activeTab === 'how' && (
          <div className="focus-card">
            <h2>{t('focus.how.title')}</h2>
            <p>{t('focus.how.text')}</p>
          </div>
        )}
      </div>

      <p className="focus-privacy">{t('focus.privacy')}</p>
    </div>
  );
};
