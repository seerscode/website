/**
 * AI Personas for the social feed: normal people and enlightened (Zen, Advaita, etc.)
 */

export const personas = [
  // Normal people
  {
    id: 'dave',
    name: 'Dave',
    handle: '@dave_irl',
    avatar: '👨‍💻',
    type: 'normal',
    tradition: null,
    color: 'bg-sky-100 text-sky-800'
  },
  {
    id: 'sarah',
    name: 'Sarah',
    handle: '@sarah_coffee',
    avatar: '👩',
    type: 'normal',
    tradition: null,
    color: 'bg-rose-100 text-rose-800'
  },
  {
    id: 'mike',
    name: 'Mike',
    handle: '@mike_lifts',
    avatar: '🧔',
    type: 'normal',
    tradition: null,
    color: 'bg-amber-100 text-amber-800'
  },
  {
    id: 'jen',
    name: 'Jen',
    handle: '@jen_reads',
    avatar: '👩‍🦰',
    type: 'normal',
    tradition: null,
    color: 'bg-violet-100 text-violet-800'
  },
  // Enlightened - Zen
  {
    id: 'koan',
    name: 'Koan',
    handle: '@koan_zen',
    avatar: '🧘',
    type: 'enlightened',
    tradition: 'Zen',
    color: 'bg-stone-200 text-stone-800'
  },
  {
    id: 'satori',
    name: 'Satori',
    handle: '@satori_mind',
    avatar: '☸️',
    type: 'enlightened',
    tradition: 'Zen',
    color: 'bg-stone-200 text-stone-800'
  },
  // Enlightened - Advaita
  {
    id: 'vidya',
    name: 'Vidya',
    handle: '@vidya_advaita',
    avatar: '🕉️',
    type: 'enlightened',
    tradition: 'Advaita',
    color: 'bg-amber-100 text-amber-900'
  },
  {
    id: 'atman',
    name: 'Atman',
    handle: '@atman_only',
    avatar: '🙏',
    type: 'enlightened',
    tradition: 'Advaita',
    color: 'bg-amber-100 text-amber-900'
  },
  // Enlightened - Taoist
  {
    id: 'wei',
    name: 'Wei',
    handle: '@wei_tao',
    avatar: '☯️',
    type: 'enlightened',
    tradition: 'Tao',
    color: 'bg-emerald-100 text-emerald-800'
  }
];

export function getPersonaById(id) {
  return personas.find((p) => p.id === id) || null;
}

export const normalPersonas = personas.filter((p) => p.type === 'normal');
export const masterPersonas = personas.filter((p) => p.type === 'enlightened');
