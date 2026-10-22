import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { DetectorReport } from '@/components/DetectorReport';
import { ACCEPT, MODES, SUPPORTED_FORMATS, fileKind, formatBytes, modeAcceptsKind } from '@/lib/detectorApi';
import {
  attachSession,
  getSnapshot,
  resetSession,
  selectFile,
  selectMode,
  startAnalysis,
  subscribe,
} from '@/lib/detectorSession';

/**
 * P08: the landing detector ported into the Compass shell.
 *
 * Deliberate differences from `frontend/src/components/Detector.jsx`:
 * - the consent checkbox stays in the DOM, checked and enabled, but hidden —
 *   the prototype shows the upload flow, not the legal gate;
 * - the report opens above the uploader and the close control resets the whole
 *   flow (file, preview, mode, progress), not just hides the report;
 * - video is allowed in burnout and external influence: the backend extracts the
 *   audio track itself, only images are rejected there.
 *
 * All flow state lives in `lib/detectorSession` (not in this component) because
 * the engine remounts the page on publish — see the comment there.
 */
export function DepressionAnalyzer() {
  const { file, previewUrl, mode, busy, percent, error, payload } = useSyncExternalStore(
    subscribe,
    getSnapshot,
    getSnapshot,
  );
  const [dragging, setDragging] = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);
  const chooseRef = useRef<HTMLButtonElement>(null);

  // Drop the session when the screen is actually left; an engine re-render
  // unmounts and remounts within the same tick and keeps it.
  useEffect(() => attachSession(), []);

  // The store owns the file. When it is cleared (report closed) the native input
  // must be cleared too, otherwise re-picking the same file fires no change event.
  useEffect(() => {
    if (!file && inputRef.current) inputRef.current.value = '';
  }, [file]);

  const closeReport = () => {
    resetSession();
    window.requestAnimationFrame(() => chooseRef.current?.focus());
  };

  const kind = fileKind(file);
  const modeAllowed = modeAcceptsKind(mode, kind);
  const canAnalyze = Boolean(file) && !busy && modeAllowed;

  return (
    <div className={`rz-analyzer${payload ? ' has-report' : ''}`}>
      <div className="rz-panel rz-analyzer-panel">
        <h2>Загрузка файла</h2>

        <div
          className={`rz-dropzone${dragging && !busy ? ' is-dragover' : ''}`}
          aria-disabled={busy || undefined}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            if (busy) return;
            const dropped = event.dataTransfer.files?.[0];
            if (dropped) selectFile(dropped);
          }}
          onDragOver={(event) => {
            event.preventDefault();
            if (!busy) setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
        >
          <div className="rz-dropzone-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M12 16V4m0 0L8 8m4-4 4 4" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" strokeLinecap="round" />
            </svg>
          </div>
          <strong>
            {busy ? 'Идёт анализ' : dragging ? 'Отпустите файл' : 'Перетащите файл сюда'}
          </strong>
          <span className="rz-muted">или</span>
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPT}
            className="rz-file-input"
            onChange={(event) => selectFile(event.target.files?.[0] ?? null)}
            disabled={busy}
            tabIndex={-1}
          />
          <button
            ref={chooseRef}
            type="button"
            className="rz-btn"
            onClick={() => inputRef.current?.click()}
            disabled={busy}
          >
            Выбрать файл
          </button>
          <p className="rz-small rz-muted">{SUPPORTED_FORMATS}</p>
          {file && (
            <div className="rz-file-info">
              <span>
                {file.name} · {formatBytes(file.size)}
              </span>
              <button
                type="button"
                className="rz-file-remove"
                onClick={() => selectFile(null)}
                disabled={busy}
                aria-label="Убрать файл"
                title="Убрать файл"
              >
                ×
              </button>
            </div>
          )}
        </div>

        {previewUrl && (
          <img className="rz-analyzer-preview" src={previewUrl} alt="Предпросмотр выбранного файла" />
        )}

        <label className="rz-field rz-analyzer-mode">
          <span>Режим обработки</span>
          <select
            value={mode}
            onChange={(event) => selectMode(event.target.value as typeof mode)}
            disabled={busy}
          >
            {MODES.map((option) => {
              const allowed = modeAcceptsKind(option.value, kind);
              return (
                <option key={option.value} value={option.value} disabled={!allowed}>
                  {option.label}
                  {allowed ? '' : ' — нужен файл со звуком'}
                </option>
              );
            })}
          </select>
        </label>

        {/* Согласие скрыто в прототипе, но остаётся в форме включённым (checked, не disabled),
            чтобы логика загрузки не считала его отсутствующим. */}
        <input
          type="checkbox"
          className="rz-consent-hidden"
          checked
          readOnly
          hidden
          aria-hidden="true"
          tabIndex={-1}
          onChange={() => undefined}
        />

        <div className="rz-actions">
          <button
            type="button"
            className="rz-btn primary"
            onClick={() => void startAnalysis()}
            disabled={!canAnalyze}
          >
            {busy ? 'Анализ…' : 'Анализировать'}
          </button>
        </div>

        {!modeAllowed && (
          <p className="rz-note warning">
            Для изображения доступен только «Стандартный анализ»: в нём нет звуковой дорожки.
          </p>
        )}

        {error && (
          <p className="rz-error" role="alert">
            {error}
          </p>
        )}

        {busy && (
          <div className="rz-progress" role="status" aria-live="polite">
            <div className="rz-progress-track">
              <span style={{ width: `${percent ?? 0}%` }} />
            </div>
            <span className="rz-small rz-muted">
              {percent === null ? 'Обработка…' : `Обработка: ${percent}%`}
            </span>
          </div>
        )}
      </div>

      {payload && (
        <DetectorReport
          payload={payload}
          fileName={file?.name || ''}
          mode={mode}
          previewUrl={previewUrl}
          onClose={closeReport}
        />
      )}
    </div>
  );
}
