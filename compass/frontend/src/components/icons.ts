import {
  Bell,
  BookOpen,
  CalendarCheck,
  Circle,
  CircleCheck,
  Inbox,
  ListChecks,
  MessagesSquare,
  Mic,
  MicOff,
  NotebookPen,
  Pause,
  Play,
  Plus,
  Settings2,
  ShieldCheck,
  Square,
  Sun,
  UserRound,
  Users,
  Video,
  WifiOff,
  type LucideIcon,
} from 'lucide-react';

/**
 * Explicit icon map. The prototype engine emits `data-lucide="<kebab-case>"`, and a
 * namespace import (`import { icons } from 'lucide-react'`) defeats tree-shaking: it
 * pulled all ~1500 icons into a 1 MB bundle for these twenty-odd glyphs.
 *
 * Only add an entry when `src/prototype/engine.js` starts emitting that name.
 * `tools/check-engine.mjs` fails the build if the engine renders an unmapped icon,
 * which would otherwise degrade silently into an invisible `<span/>`.
 */
export const ICONS: Record<string, LucideIcon> = {
  'bell': Bell,
  'book-open': BookOpen,
  'calendar-check': CalendarCheck,
  'circle': Circle,
  'circle-check': CircleCheck,
  'inbox': Inbox,
  'list-checks': ListChecks,
  'messages-square': MessagesSquare,
  'mic': Mic,
  'mic-off': MicOff,
  'notebook-pen': NotebookPen,
  'pause': Pause,
  'play': Play,
  'plus': Plus,
  'settings-2': Settings2,
  'shield-check': ShieldCheck,
  'square': Square,
  'sun': Sun,
  'user-round': UserRound,
  'users': Users,
  'video': Video,
  'wifi-off': WifiOff,
};
