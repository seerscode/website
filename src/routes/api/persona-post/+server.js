import { json } from '@sveltejs/kit';
import { personas, getPersonaById } from '$lib/personas.js';

function buildSystemPrompt(persona, topic = '') {
  const topicLine = topic.trim()
    ? ` The current topic of conversation is: "${topic}". You may respond to this theme if it fits your voice.`
    : '';

  if (persona.type === 'normal') {
    return `You are ${persona.name}, a regular person posting on a social feed. You're relatable, a bit tired, sometimes sarcastic or self-deprecating. Post exactly one short tweet-style message (1-2 sentences max, under 280 characters). No hashtags, no quotes. Write only the post text, nothing else.${topicLine}`;
  }

  if (persona.type === 'enlightened') {
    const tradition = persona.tradition || '';
    if (tradition === 'Zen') {
      return `You are ${persona.name}, a voice rooted in Zen. Write exactly one short, crisp tweet-style post. Use paradox, presence, simplicity. Under 280 characters. No hashtags. Write only the post text.${topicLine}`;
    }
    if (tradition === 'Advaita') {
      return `You are ${persona.name}, a voice rooted in Advaita Vedanta. Write exactly one short tweet-style post about awareness, the self, or non-duality. Under 280 characters. No hashtags. Write only the post text.${topicLine}`;
    }
    if (tradition === 'Tao') {
      return `You are ${persona.name}, a voice rooted in Taoism. Write exactly one short tweet-style post: flow, nature, simplicity. Under 280 characters. No hashtags. Write only the post text.${topicLine}`;
    }
  }

  return `You are ${persona.name}. Post one short tweet-style message (under 280 characters). No hashtags. Write only the post text.${topicLine}`;
}

export async function POST({ request }) {
  try {
    const body = await request.json().catch(() => ({}));
    const { personaId, topic = '' } = body;

    if (!personaId || typeof personaId !== 'string') {
      return json({ error: 'personaId is required' }, { status: 400 });
    }

    const persona = getPersonaById(personaId);
    if (!persona) {
      return json({ error: 'Unknown persona' }, { status: 400 });
    }

    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      return json({
        error: 'OpenAI API key not configured. Set OPENAI_API_KEY.'
      }, { status: 500 });
    }

    const systemPrompt = buildSystemPrompt(persona, topic);
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: 'Write your next post.' }
        ],
        max_tokens: 120,
        temperature: 0.8
      })
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      console.error('OpenAI persona-post error:', err);
      if (response.status === 429) {
        return json({ error: 'Rate limit exceeded. Try again in a moment.' }, { status: 429 });
      }
      return json({
        error: err.error?.message || 'OpenAI request failed'
      }, { status: 500 });
    }

    const data = await response.json();
    const content = data.choices?.[0]?.message?.content?.trim() || '…';
    return json({ post: content });
  } catch (error) {
    console.error('Persona post API error:', error);
    return json({ error: 'An unexpected error occurred.' }, { status: 500 });
  }
}
