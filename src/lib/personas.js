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

// Phrase banks: normal people (everyday, slightly stressed, opinionated)
const normalPhrases = [
  "Honestly though, why is Monday a thing.",
  "Coffee count: 3. Still not enough.",
  "That meeting could've been an email. Again.",
  "Just realized I've been staring at the same tab for 20 mins.",
  "Nobody talks about how exhausting it is to exist sometimes.",
  "Trying to adult today. It's going okay. Ish.",
  "Why do we all pretend we have it together.",
  "The algorithm knows me better than my mom at this point.",
  "Small wins only today. Got out of bed. That's it.",
  "Anyone else feel like they're just making it up as they go?",
  "Unpopular opinion: we need more naps.",
  "My brain at 3am: here are 47 things to worry about.",
  "Productivity hack: lower your expectations. You're welcome.",
  "Sometimes the best response is no response.",
  "Real talk: who has their life actually figured out?"
];

// Zen-style phrases (short, paradoxical, present)
const zenPhrases = [
  "The finger pointing at the moon is not the moon.",
  "Before enlightenment, chop wood. After enlightenment, chop wood.",
  "What is the sound of one hand clapping?",
  "When walking, just walk. When sitting, just sit.",
  "No snowflake ever falls in the wrong place.",
  "The obstacle is the path.",
  "Let go or be dragged.",
  "In the beginner's mind there are many possibilities. In the expert's, few.",
  "When the pupil is ready, the teacher appears.",
  "Sitting quietly, doing nothing, spring comes and the grass grows by itself.",
  "Not knowing is most intimate.",
  "The way out is through.",
  "Empty your cup.",
  "Wherever you go, there you are.",
  "This too."
];

// Advaita-style phrases (self, awareness, non-duality)
const advaitaPhrases = [
  "You are what you seek.",
  "The seeker is the sought.",
  "There is only awareness, and you are that.",
  "What is looking out through your eyes?",
  "The sense of 'I' is the only thing that never changes.",
  "Before the thought 'I am' — what are you?",
  "There is no one to become enlightened. Only the illusion of someone.",
  "Rest in the one who knows.",
  "You are not in the world. The world is in you.",
  "The wave is not separate from the ocean.",
  "Turn attention back upon the one who is aware.",
  "Peace is not something you get. It is what you are when the noise stops.",
  "Who were you before your parents were born?",
  "Reality is not dual. The split is only in thought.",
  "There is nothing to do. Only something to see."
];

// Tao-style phrases (flow, simplicity, nature)
const taoPhrases = [
  "The Tao that can be spoken is not the eternal Tao.",
  "Water does not argue. It flows.",
  "A journey of a thousand miles begins with a single step.",
  "Nature does not hurry, yet everything is accomplished.",
  "When I let go of what I am, I become what I might be.",
  "The wise one does not push. The river reaches the sea.",
  "Soft overcomes hard. Slow overcomes fast.",
  "Be still and know.",
  "The usefulness of a cup is in its emptiness.",
  "Do nothing, and nothing is left undone.",
  "Return to the root and you will find the meaning.",
  "Those who know do not speak. Those who speak do not know.",
  "The sage stays behind, thus is ahead.",
  "Allow things to unfold. No forcing.",
  "In the midst of movement, find stillness."
];

const traditionPhrases = {
  Zen: zenPhrases,
  Advaita: advaitaPhrases,
  Tao: taoPhrases
};

/**
 * Get a random phrase for a persona, optionally influenced by topic.
 */
export function getPhraseForPersona(persona, topic = '') {
  const topicLower = topic.trim().toLowerCase();
  if (persona.type === 'normal') {
    const list = [...normalPhrases];
    if (topicLower) {
      list.push(
        `About ${topic}: honestly no idea.`,
        `Re: ${topic} — I have thoughts. Many thoughts.`,
        `${topic}? Yeah it's a lot.`,
        `Can we talk about ${topic} without spiraling? Asking for a friend.`,
        `My take on ${topic}: it's complicated.`
      );
    }
    return list[Math.floor(Math.random() * list.length)];
  }
  if (persona.type === 'enlightened' && traditionPhrases[persona.tradition]) {
    const list = [...traditionPhrases[persona.tradition]];
    if (topicLower) {
      list.push(
        `On ${topic}: look not at the problem, but at the one who sees it.`,
        `Regarding ${topic} — the question and the questioner are one.`,
        `${topic} appears in awareness. What is aware of that?`,
        `In ${topic}, find the unchanging.`
      );
    }
    return list[Math.floor(Math.random() * list.length)];
  }
  return "…";
}

export function getPersonaById(id) {
  return personas.find((p) => p.id === id) || null;
}
