import { useEffect, useRef, useState } from 'react';
import { selectFile, selectMode, startAnalysis } from '@/lib/detectorSession';
import { bumpRoute } from '@/App';

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
  /** Set by onstop after the webm→mp4 transcode promise resolves. */
  const remux = useRef<Promise<Blob> | null>(null);

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
    remux.current = null;
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    objectUrl.current = '';
    setUrl('');
  };

  const analyze = () => {
    const recorded = blob.current;
    if (!recorded || remux.current) return;
    const type = recorded.type || (mode === 'video' ? 'video/webm' : 'audio/webm');
    const file = new File([recorded], `razuma-recording-${Date.now()}.${ext}`, { type });
    // Keep the upload alive for P08: resetSession() invalidates the run token
    // and aborts the in-flight fetch, and selectFile() returns early while
    // busy. The session store is module-scoped, so set the file BEFORE
    // navigating; P08's DepressionAnalyzer mounts, attaches, and picks the
    // state up.
    selectFile(file);
    selectMode('standard');
    void startAnalysis();
    history.replaceState({}, '', '/compass/P08');
    bumpRoute();
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
        objectUrl.current = URL.createObjectURL(recorded);
        setUrl(objectUrl.current);
        setExt(recorder.mimeType.includes('mp4') ? 'mp4' : 'webm');
        setActive(false);
        setStatus('Запись готова. Можно выполнить анализ, прослушать, скачать или удалить.');
        // The server's video pipeline (OpenCV/FFmpeg) decodes mp4 reliably but
        // webm is flaky; MediaRecorder emits webm on Chrome/Firefox. Remux in
        // the browser so P08 receives a format the server can open. Falls back
        // to the webm blob if encoding is unsupported.
        blob.current = recorded;
        if (mode === 'video' && recorder.mimeType.includes('webm')) {
          remux.current = transcodeToMp4(recorded).then((result) => {
            blob.current = result;
            if (result.type.includes('mp4')) setExt('mp4');
            remux.current = null;
            return result;
          });
        }
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
      <a href="/compass/P06">← В дневник</a>
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

/**
 * Encode a recorded webm blob to mp4 in the browser using MediaEncoder
 * (Chrome). Returns the original blob when MediaEncoder is unavailable or
 * encoding fails, so the flow degrades gracefully instead of breaking.
 */
async function transcodeToMp4(source: Blob): Promise<Blob> {
  type MediaEncoderLike = {
    state: string;
    videoTrack: MediaStreamTrack;
    audioTrack?: MediaStreamTrack;
    ondataavailable: ((event: { data: ArrayBuffer }) => void) | null;
    onended: (() => void) | null;
    start: () => void;
    flush: () => Promise<void>;
    close: () => void;
  };
  const Ctor = (window as unknown as {
    MediaEncoder?: new (stream: MediaStream, opts: object) => MediaEncoderLike;
  }).MediaEncoder;
  if (!Ctor) return source;

  const videoEl = document.createElement('video');
  const srcUrl = URL.createObjectURL(source);
  videoEl.src = srcUrl;
  videoEl.muted = true;
  videoEl.playsInline = true;
  try {
    await videoEl.play().catch(() => undefined);
    const canvas = document.createElement('canvas');
    canvas.width = videoEl.videoWidth || 640;
    canvas.height = videoEl.videoHeight || 360;
    const ctx = canvas.getContext('2d');
    if (!ctx) return source;
    const canvasStream = canvas.captureStream(30);
    let audioStream: MediaStream | null = null;
    try {
      const arrayBuffer = await source.arrayBuffer();
      const audioCtx = new AudioContext();
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      const dest = audioCtx.createMediaStreamDestination();
      const srcNode = audioCtx.createBufferSource();
      srcNode.buffer = audioBuffer;
      srcNode.connect(dest);
      srcNode.start();
      audioStream = dest.stream;
    } catch {
      audioStream = null;
    }
    const out = new MediaStream([...canvasStream.getVideoTracks(), ...(audioStream?.getAudioTracks() ?? [])]);
    const encoder = new Ctor(out, {
      video: { codec: 'avc1.42001f', bitrate: 2_000_000 },
      ...(audioStream ? { audio: { codec: 'mp4a.40.2', bitrate: 128_000 } } : {}),
    });
    const parts: BlobPart[] = [];
    const done = new Promise<void>((resolve) => {
      encoder.ondataavailable = (e) => {
        if (e.data?.byteLength) parts.push(new Uint8Array(e.data).buffer);
      };
      encoder.onended = () => resolve();
    });
    encoder.start();
    const stopDrawing = () => {
      videoEl.pause();
      void encoder.flush().catch(() => undefined);
      for (const t of out.getTracks()) t.stop();
      encoder.close?.();
      URL.revokeObjectURL(srcUrl);
    };
    const draw = () => {
      if (videoEl.ended || videoEl.paused) {
        stopDrawing();
        return;
      }
      ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
      requestAnimationFrame(draw);
    };
    draw();
    await Promise.race([done, new Promise((r) => setTimeout(r, 90_000))]);
    const mp4 = new Blob(parts, { type: 'video/mp4' });
    return mp4.size > 0 ? mp4 : source;
  } catch {
    URL.revokeObjectURL(srcUrl);
    return source;
  }
}
