import { useEffect, useRef, useState } from 'react';
import { modeLabel, type DetectorMode, type ProgressPayload } from '@/lib/detectorApi';
import {
  burnoutComment,
  burnoutComponent,
  burnoutLevel,
  burnoutRecommendation,
  burnoutState,
  burnoutTopFactor,
  externalAction,
  externalFactor,
  externalStatus,
} from '@/lib/analysisLabels';

const EMOTION_LABELS: Record<string, string> = {
  anger: 'Гнев',
  contempt: 'Презрение',
  disgust: 'Отвращение',
  enthusiasm: 'Восторг',
  fear: 'Страх',
  happiness: 'Радость',
  joy: 'Радость',
  neutral: 'Нейтрально',
  sadness: 'Грусть',
  surprise: 'Удивление',
  valence: 'Валентность',
  arousal: 'Интенсивность',
};

const VERDICTS: Record<string, { text: string; tone: string }> = {
  low: { text: 'Низкая вероятность депрессии', tone: 'ok' },
  monitor: { text: 'Требуется наблюдение', tone: 'warn' },
  high: { text: 'Высокая вероятность депрессии', tone: 'danger' },
};

const REPORT_TITLE_ID = 'rz-report-title';

const emotionLabel = (key: string): string => EMOTION_LABELS[key] || key;

const splitEmotions = (source?: Record<string, number | string> | null) => {
  const emotions: Array<[string, number]> = [];
  const features: Array<[string, number]> = [];
  Object.entries(source || {}).forEach(([key, value]) => {
    const num = typeof value === 'number' ? value : parseFloat(String(value));
    if (!Number.isFinite(num)) return;
    if (key === 'valence' || key === 'arousal') features.push([key, num]);
    else emotions.push([key, num]);
  });
  emotions.sort((a, b) => b[1] - a[1]);
  return { emotions, features };
};

const asPercent = (value?: number): string =>
  typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';

/** Burnout components are 0..1 (higher = stronger signal); sort the strongest up. */
const componentRows = (components: unknown): Array<[string, number]> => {
  if (!components || typeof components !== 'object') return [];
  return Object.entries(components as Record<string, unknown>)
    .map(([key, value]) => [key, Number(value)] as [string, number])
    .filter(([, value]) => Number.isFinite(value))
    .sort((a, b) => b[1] - a[1]);
};

const textList = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];

interface Props {
  payload: ProgressPayload;
  fileName: string;
  mode: DetectorMode;
  previewUrl: string | null;
  onClose: () => void;
}

export function DetectorReport({ payload, fileName, mode, previewUrl, onClose }: Props) {
  // Collapsed by default: the AI verdict and the PDF export are the priority, the
  // raw source image and emotion numbers are one click away.
  const [showData, setShowData] = useState(false);
  const reportRef = useRef<HTMLElement>(null);

  // The upload panel is hidden while the report is open, so the report must take
  // focus itself: otherwise keyboard focus stays on a display:none button and the
  // next Tab lands nowhere useful.
  useEffect(() => {
    reportRef.current?.focus();
  }, []);

  const data = payload.result && typeof payload.result === 'object' ? payload.result : payload;
  const type = payload.type || data.type || '';
  const isVideo = type === 'video';
  const isExternal =
    type === 'audio_external_influence' || payload.mode === 'external_influence';

  const ai = payload.ai_analysis || data?.ai_analysis || data?.result?.ai_analysis || null;
  const verdict = ai?.verdict ? VERDICTS[ai.verdict] : null;

  const emotionSource = isVideo
    ? payload.average_emotions
    : data?.additional_probs ?? data?.result?.additional_probs;
  const { emotions, features } = splitEmotions(emotionSource);
  const main = isVideo
    ? payload.average_main_emotion
    : data?.main_prediction ?? data?.result?.main_prediction;
  // The server signals "no usable face" with a sentinel row instead of an error,
  // so the report must not print `error · 0.0%` as an emotion.
  const recognizedMain =
    main?.label && String(main.label) !== 'error' ? String(main.label) : null;

  const frames: string[] = (payload.results || [])
    .map((frame) => frame?.image_url)
    .filter((url): url is string => Boolean(url));
  const sourceImage = payload.image_url || frames[0] || null;

  const burnout = data?.burnout_analysis || data?.result?.burnout_analysis || null;
  const burnoutComponents = componentRows(burnout?.components);
  const burnoutRecommendations = textList(burnout?.recommendations);
  const external = isExternal ? data?.result || null : null;
  const externalFactors = textList(external?.topFactors);

  const meta = [
    {
      label: 'Дата',
      value: new Date().toLocaleString('ru-RU', { dateStyle: 'medium', timeStyle: 'short' }),
    },
    { label: 'Файл', value: fileName || '—' },
    { label: 'Режим', value: modeLabel(mode) },
    ...(isVideo && payload.frames_processed
      ? [{ label: 'Кадров обработано', value: String(payload.frames_processed) }]
      : []),
  ];

  const exportPdf = () => {
    // The source image and the numbers are collapsed by default; a PDF of the
    // report should include them, so expand before the print dialog opens.
    setShowData(true);
    window.setTimeout(() => window.print(), 50);
  };

  return (
    <section
      ref={reportRef}
      className="rz-report"
      role="dialog"
      aria-labelledby={REPORT_TITLE_ID}
      tabIndex={-1}
    >
      <div className="rz-report-head">
        <h2 id={REPORT_TITLE_ID}>Отчёт по анализу</h2>
        <button
          type="button"
          className="rz-report-close"
          onClick={onClose}
          aria-label="Закрыть отчёт и вернуться к загрузке"
          title="Закрыть отчёт"
        >
          ×
        </button>
      </div>

      <dl className="rz-report-meta">
        {meta.map((row) => (
          <div className="rz-report-meta-row" key={row.label}>
            <dt>{row.label}</dt>
            <dd>{row.value}</dd>
          </div>
        ))}
      </dl>

      <p className="rz-report-hint">
        Сохраните документ в PDF, выбрав «Сохранить как PDF» в диалоге печати.
      </p>

      {ai?.verdict ? (
        <div className={`rz-ai rz-ai-${verdict?.tone ?? 'ok'}`}>
          <h3>ИИ-интерпретация</h3>
          <div className="rz-ai-verdict">
            <span className="rz-ai-dot" aria-hidden="true" />
            <strong>{verdict?.text ?? ai.verdict}</strong>
            {typeof ai.probability === 'number' && (
              <span className="rz-small rz-muted">вероятность {asPercent(ai.probability)}</span>
            )}
          </div>
          {ai.reasoning && <p className="rz-ai-reasoning">{ai.reasoning}</p>}
          {ai.scores && Object.keys(ai.scores).length > 0 && (
            <details className="rz-ai-scores">
              <summary>Баллы</summary>
              <ul>
                {Object.entries(ai.scores as Record<string, unknown>).map(([key, value]) => (
                  <li key={key}>
                    <span>{key}</span>
                    <span>{String(value)}</span>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      ) : (
        <p className="rz-note neutral rz-report-noai">
          ИИ-интерпретация недоступна: второй этап анализа отключён на сервере. Данные эмоций — ниже.
        </p>
      )}

      <div className="rz-actions rz-report-actions">
        <button type="button" className="rz-btn primary" onClick={exportPdf}>
          Выгрузить в PDF
        </button>
      </div>

      <p className="rz-report-summary">
        <span className="rz-muted">Главная эмоция:</span>{' '}
        <strong>{recognizedMain ? emotionLabel(recognizedMain) : 'лицо не распознано'}</strong>
        {recognizedMain && typeof main?.probability === 'number'
          ? ` · ${asPercent(main.probability)}`
          : ''}
      </p>

      <button
        type="button"
        className="rz-report-toggle"
        aria-expanded={showData}
        onClick={() => setShowData((value) => !value)}
      >
        {showData ? 'Свернуть данные анализа' : 'Показать данные анализа'}
      </button>

      {showData && (
        <div className="rz-report-body">
          {sourceImage ? (
            <figure className="rz-report-source">
              <img src={sourceImage} alt="Источник: обработанный кадр" loading="lazy" />
              <figcaption className="rz-small rz-muted">
                Источник: обработанный кадр от сервера
                {isVideo && frames.length > 1 ? ' (первый кадр)' : ''}
              </figcaption>
            </figure>
          ) : previewUrl ? (
            <figure className="rz-report-source">
              <img src={previewUrl} alt="Источник: выбранный файл" />
              <figcaption className="rz-small rz-muted">Источник: выбранный файл</figcaption>
            </figure>
          ) : null}

          {emotions.length > 0 && (
            <div className="rz-report-emotions">
              <h4>Эмоции</h4>
              {emotions.map(([key, value]) => (
                <div className="rz-emotion" key={key}>
                  <div className="rz-emotion-label">
                    <span>{emotionLabel(key)}</span>
                    <span>{asPercent(value)}</span>
                  </div>
                  <div className="rz-emotion-track">
                    <span style={{ width: `${Math.min(100, Math.max(0, value * 100))}%` }} />
                  </div>
                </div>
              ))}
            </div>
          )}

          {features.length > 0 && (
            <dl className="rz-report-kv">
              {features.map(([key, value]) => (
                <div key={key}>
                  <dt>{emotionLabel(key)}</dt>
                  <dd>{value.toFixed(2)}</dd>
                </div>
              ))}
            </dl>
          )}

          {burnout && (
            <div className="rz-report-block">
              <h4>Анализ состояния человека</h4>
              <dl className="rz-report-kv">
                <div>
                  <dt>Состояние</dt>
                  <dd>{burnoutState(burnout.state)}</dd>
                </div>
                <div>
                  <dt>Уровень риска</dt>
                  <dd>{burnoutLevel(burnout.level)}</dd>
                </div>
                <div>
                  <dt>Индекс риска</dt>
                  <dd>
                    {typeof burnout.score === 'number' ? `${burnout.score.toFixed(1)} / 100` : '—'}
                  </dd>
                </div>
                <div>
                  <dt>Надёжность оценки</dt>
                  <dd>{asPercent(burnout.confidence)}</dd>
                </div>
                {burnout.top_factor && (
                  <div>
                    <dt>Основной сигнал</dt>
                    <dd>{burnoutTopFactor(String(burnout.top_factor))}</dd>
                  </div>
                )}
              </dl>

              {burnoutComponents.length > 0 && (
                <div className="rz-report-emotions rz-report-components">
                  <h4>Из чего складывается оценка</h4>
                  {burnoutComponents.map(([key, value]) => (
                    <div className="rz-emotion" key={key}>
                      <div className="rz-emotion-label">
                        <span>{burnoutComponent(key)}</span>
                        <span>{asPercent(value)}</span>
                      </div>
                      <div className="rz-emotion-track">
                        <span style={{ width: `${Math.min(100, Math.max(0, value * 100))}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {burnoutRecommendations.length > 0 && (
                <div className="rz-report-block">
                  <h4>Рекомендации</h4>
                  <ul className="rz-report-recs">
                    {burnoutRecommendations.map((code) => (
                      <li key={code}>{burnoutRecommendation(code)}</li>
                    ))}
                  </ul>
                </div>
              )}

              {burnout.comment && (
                <p className="rz-report-note">{burnoutComment(String(burnout.comment))}</p>
              )}
            </div>
          )}

          {external && (
            <div className="rz-report-block">
              <h4>Внешнее влияние</h4>
              <dl className="rz-report-kv">
                <div>
                  <dt>Статус</dt>
                  <dd>{externalStatus(external.status)}</dd>
                </div>
                <div>
                  <dt>Индекс выраженности сигнала</dt>
                  <dd>
                    {typeof external.score === 'number' ? external.score.toFixed(2) : '—'}
                  </dd>
                </div>
                <div>
                  <dt>Уверенность</dt>
                  <dd>{asPercent(external.confidence)}</dd>
                </div>
                <div>
                  <dt>Устойчивое сочетание признаков</dt>
                  <dd>{external.persistentPattern ? 'да' : 'нет'}</dd>
                </div>
                <div>
                  <dt>Подтверждено контекстом</dt>
                  <dd>{external.contextConfirmed ? 'да' : 'нет'}</dd>
                </div>
                {externalFactors.length > 0 && (
                  <div>
                    <dt>Ключевые сигналы</dt>
                    <dd>{externalFactors.map(externalFactor).join(', ')}</dd>
                  </div>
                )}
                {external.managerAction && (
                  <div>
                    <dt>Действие менеджера</dt>
                    <dd>{externalAction(external.managerAction)}</dd>
                  </div>
                )}
              </dl>
              <p className="rz-report-note">
                Сигнал не является вероятностью мошенничества и не должен сам по себе блокировать
                операции.
              </p>
            </div>
          )}

          {emotions.length === 0 && !burnout && !external && (
            <p className="rz-small rz-muted">
              Распознанных эмоций нет: скорее всего, на кадре не найдено лицо. Попробуйте другое фото
              или видео, где лицо крупнее и лучше освещено.
            </p>
          )}
        </div>
      )}
    </section>
  );
}
