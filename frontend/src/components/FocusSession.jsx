import React, { useEffect, useState } from 'react';
import { useLanguage } from '../hooks/useLanguage';

const DURATIONS = [5, 15, 25];

const formatTime = (totalSeconds) => {
  const minutes = Math.floor(totalSeconds / 60).toString().padStart(2, '0');
  const seconds = (totalSeconds % 60).toString().padStart(2, '0');
  return `${minutes}:${seconds}`;
};

export const FocusSession = ({ plan }) => {
  const { t } = useLanguage();
  const [index, setIndex] = useState(0);
  const [durationMin, setDurationMin] = useState(DURATIONS[0]);
  const [secondsLeft, setSecondsLeft] = useState(DURATIONS[0] * 60);
  const [running, setRunning] = useState(false);
  const [completed, setCompleted] = useState([]);

  // A new plan (e.g. regenerated) starts the session from scratch.
  useEffect(() => {
    setIndex(0);
    setRunning(false);
    setCompleted([]);
  }, [plan]);

  // Changing the duration or the current step resets the countdown.
  useEffect(() => {
    setSecondsLeft(durationMin * 60);
  }, [durationMin, index]);

  useEffect(() => {
    if (!running) return undefined;
    const id = setInterval(() => {
      setSecondsLeft((prev) => Math.max(0, prev - 1));
    }, 1000);
    return () => clearInterval(id);
  }, [running]);

  useEffect(() => {
    if (secondsLeft === 0 && running) setRunning(false);
  }, [secondsLeft, running]);

  if (!plan.length) {
    return <p className="focus-empty">{t('focus.plan.empty')}</p>;
  }

  const allDone = completed.length === plan.length;
  const current = plan[Math.min(index, plan.length - 1)];
  const isFresh = secondsLeft === durationMin * 60;

  const markCompleted = (stepId) =>
    setCompleted((prev) => (prev.includes(stepId) ? prev : [...prev, stepId]));

  const handleNext = () => {
    markCompleted(current.id);
    setRunning(false);
    setIndex((prev) => Math.min(prev + 1, plan.length - 1));
  };

  const handleSelect = (stepIndex) => {
    setRunning(false);
    setIndex(stepIndex);
  };

  const handleDuration = (minutes) => {
    setRunning(false);
    setDurationMin(minutes);
  };

  const handleReset = () => {
    setRunning(false);
    setIndex(0);
    setCompleted([]);
    setSecondsLeft(durationMin * 60);
  };

  return (
    <div className="focus-session">
      <div className="focus-progress">
        <div
          className="focus-progress-bar"
          style={{ width: `${(completed.length / plan.length) * 100}%` }}
        />
      </div>

      {allDone ? (
        <div className="focus-done">
          <p className="focus-done-title">{t('focus.session.done')}</p>
          <p className="focus-done-text">{t('focus.session.doneText')}</p>
          <div className="focus-actions">
            <button type="button" className="focus-btn ghost" onClick={handleReset}>
              {t('focus.session.reset')}
            </button>
          </div>
        </div>
      ) : (
        <>
          <p className="focus-progress-label">
            {t('focus.session.progress', { current: index + 1, total: plan.length })}
          </p>

          <div className="focus-timer">{formatTime(secondsLeft)}</div>

          <div className="focus-durations">
            {DURATIONS.map((minutes) => (
              <button
                key={minutes}
                type="button"
                className={`focus-preset ${durationMin === minutes ? 'active' : ''}`}
                onClick={() => handleDuration(minutes)}
              >
                {t('focus.session.minutes', { count: minutes })}
              </button>
            ))}
          </div>

          <div className="focus-actions">
            <button
              type="button"
              className="focus-btn primary"
              onClick={() => setRunning((prev) => !prev)}
            >
              {running
                ? t('focus.session.pause')
                : isFresh
                  ? t('focus.session.start')
                  : t('focus.session.resume')}
            </button>
            <button type="button" className="focus-btn ghost" onClick={handleNext}>
              {t('focus.session.next')}
            </button>
          </div>

          <ol className="focus-steps">
            {plan.map((step, stepIndex) => {
              const isDone = completed.includes(step.id);
              return (
                <li
                  key={step.id}
                  className={`focus-step ${stepIndex === index ? 'current' : ''} ${isDone ? 'done' : ''}`}
                >
                  <button
                    type="button"
                    className="focus-step-pick"
                    onClick={() => handleSelect(stepIndex)}
                  >
                    <span className="focus-step-index">{isDone ? '✓' : stepIndex + 1}</span>
                    <span className="focus-step-title">{t(step.key)}</span>
                  </button>
                </li>
              );
            })}
          </ol>
        </>
      )}
    </div>
  );
};
