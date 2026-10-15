// Anonymous client identity + traffic source for the server-side analytics
// event log (contest requirement 5.1). The id is random and only ever sent to
// our own backend, which hashes it before writing anything to disk — no
// personal data leaves the browser.

const CLIENT_ID_KEY = 'razuma_client_id';

export const getClientId = () => {
  try {
    if (typeof localStorage === 'undefined') return '';
    let id = localStorage.getItem(CLIENT_ID_KEY);
    if (!id) {
      id =
        (typeof crypto !== 'undefined' && crypto.randomUUID && crypto.randomUUID()) ||
        `c-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
      localStorage.setItem(CLIENT_ID_KEY, id);
    }
    return id;
  } catch {
    return '';
  }
};

// Full page URL (carries ?utm_source=... when the visitor arrived from a
// campaign). The server also reads the standard Referer header, so UTM data
// survives even when this header is absent.
export const getTrafficSource = () => {
  try {
    return typeof window !== 'undefined' ? window.location.href : '';
  } catch {
    return '';
  }
};

export const getReferrer = () => {
  try {
    return typeof document !== 'undefined' ? document.referrer || '' : '';
  } catch {
    return '';
  }
};

export const analyticsHeaders = () => ({
  'X-Client-Id': getClientId(),
  'X-Traffic-Source': getTrafficSource(),
  'X-Referrer': getReferrer(),
});
