import { useEffect, useRef, useState } from 'react';
import { resetSession, selectFile, selectMode, startAnalysis } from '@/lib/detectorSession';

export function Recorder() {
  const [mode, setMode] = useState<'audio' | 'video'>('video');
  const [consent, setConsent] = useState(false);
  const [status, setStatus] = useState(
    'Запись хранится в этой вкладке. На анализ она уйдёт только по кнопке «Выполнить анализ».',
  );
  const [active, setActive] = useState(false);
  const [busy, setBusy] = useState(false);
  const [url, setUrl] = useState('');
  const [ext, setExt] = useState('webm');

  const stream = useRef<MediaStream | null>(null);
  const rec = useRef<MediaRecorder | null>(null);
  const video = useRef<HTMLVideoElement | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const live = useRef(true);
  const objectUrl = useRef('');
  const blob = useRef<Blob | null>(null);

  const release = () => {
    stream.current?.getTracks().forEach((track) => track.stop());
    stream.current = null;
    if (timer.current) clearTimeout(timer.current);
  };

  useEffect(() => {
    live.current = true;
    return () => {
      live.current = false;
      if (rec.current?.state === 'recording') rec.current.stop();
      release();
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    };
  }, []);

  const clear = () => {
    blob.current = null;
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    objectUrl.current = '';
    setUrl('');
  };

  const analyze = () => {
    const recorded = blob.current;
    if (!recorded) return;
    const type = recorded.type || (mode === 'video' ? 'video/webm' : 'audio/webm');
    const file = new File([recorded], `razuma-recording-${Date.now()}.${ext}`, { type });
    resetSession();
    selectFile(file);
    selectMode('standard');
    void startAnalysis();
    window.location.hash = '#/screens/P08';
  };

  async function start() {
    if (!consent) return;
    setBusy(true);
    clear();
    try {
      if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
        throw Error('Этот браузер не поддерживает запись. Используйте HTTPS или localhost.');
      }
      const media = await navigator.mediaDevices.getUserMedia({ audio: true, video: mode === 'video' });
      if (!live.current) {
        media.getTracks().forEach((track) => track.stop());
        return;
      }
      stream.current = media;
      if (video.current) {
        video.current.srcObject = media;
        await video.current.play();
      }
      const candidates =
        mode === 'video' ? ['video/webm;codecs=vp8,opus', 'video/mp4'] : ['audio/webm;codecs=opus', 'audio/mp4'];
      const mimeType = candidates.find((type) => MediaRecorder.isTypeSupported(type));
      const recorder = new MediaRecorder(media, mimeType ? { mimeType } : undefined);
      rec.current = recorder;
      const chunks: BlobPart[] = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size) chunks.push(event.data);
      };
      recorder.onstop = () => {
        release();
        if (video.current) video.current.srcObject = null;
        if (!live.current) return;
        const recorded = new Blob(chunks, { type: recorder.mimeType });
        blob.current = recorded;
        objectUrl.current = URL.createObjectURL(recorded);
        setUrl(objectUrl.current);
        setExt(recorder.mimeType.includes('mp4') ? 'mp4' : 'webm');
        setActive(false);
        setStatus('Запись готова. Можно выполнить анализ, прослушать, скачать или удалить.');
      };
      recorder.start();
      setActive(true);
      setStatus('Идёт запись. Автоматическая остановка через 60 секунд.');
      timer.current = setTimeout(() => {
        if (recorder.state === 'recording') recorder.stop();
      }, 60000);
    } catch (cause) {
      release();
      setStatus(cause instanceof Error ? cause.message : 'Не удалось получить доступ к устройствам.');
    } finally {
      if (live.current) setBusy(false);
    }
  }

  return (
    <main className="capture">
      <a href="#/screens/P06">← В дневник</a>
      <h1>Голос или видеокружок</h1>
      <p>Реальная запись устройства. Используйте учебные данные.</p>
      <label>
        Формат{' '}
        <select
          disabled={active || busy}
          value={mode}
          onChange={(event) => {
            clear();
            setMode(event.target.value as 'audio' | 'video');
          }}
        >
          <option value="video">Видео</option>
          <option value="audio">Аудио</option>
        </select>
      </label>
      <label>
        <input
          type="checkbox"
          checked={consent}
          disabled={active || busy}
          onChange={(event) => setConsent(event.target.checked)}
        />{' '}
        Разрешаю запись в этой вкладке
      </label>
      <video
        ref={video}
        muted
        playsInline
        className={mode === 'video' ? 'circle' : 'hidden'}
        aria-label="Предпросмотр камеры"
      />
      {url &&
        (mode === 'video' ? (
          <video className="circle" src={url} controls playsInline />
        ) : (
          <audio src={url} controls />
        ))}
      <p role="status">{status}</p>
      <div className="capture-actions">
        <button disabled={!consent || active || busy} onClick={start}>
          {busy ? 'Запрашиваем доступ…' : 'Начать запись'}
        </button>
        <button disabled={!active} onClick={() => rec.current?.stop()}>
          Остановить
        </button>
        {url && (
          <>
            <button onClick={analyze}>Выполнить анализ</button>
            <a download={`razuma-recording.${ext}`} href={url}>
              Скачать запись
            </a>
            <button onClick={clear}>Удалить</button>
          </>
        )}
      </div>
    </main>
  );
}
