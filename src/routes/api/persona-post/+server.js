import { json } from '@sveltejs/kit';
import { getPersonaById } from '$lib/personas.js';

function buildQuestionPrompt(persona, topic = '', recentPosts = []) {
  const topicLine = topic.trim()
    ? ` Keep the overall theme related to: "${topic}".`
    : '';

  let threadContext = '';
  if (Array.isArray(recentPosts) && recentPosts.length > 0) {
    const lines = recentPosts
      .slice(-12)
      .map((p) => {
        const label = p.kind === 'answer' ? 'Answer' : p.kind === 'question' ? 'Question' : 'Post';
        return `[${label} - ${p.authorName}]: ${p.content}`;
      });
    threadContext = `\n\nRecent posts in the thread (oldest to newest):\n${lines.join('\n')}\n\nYour question must be a follow-up: react to, deepen, or challenge what was just said. Do not repeat an already asked question.`;
  } else {
    threadContext = ' Ask something genuine that an ordinary person might wonder about—life, meaning, stress, identity, or how to be at peace.';
  }

  return `You are ${persona.name}, a regular person on a social feed. You're relatable, a bit tired, sometimes sarcastic or self-deprecating. Write exactly one short follow-up question (1-2 sentences, under 280 characters). No hashtags. Write only the question, nothing else.${topicLine}${threadContext}`;
}

function buildAnswerPrompt(persona, question = '') {
  const tradition = persona.tradition || '';
  const answerStyle = {
    Zen: 'Respond in a Zen voice: short, crisp, paradoxical or pointing to presence. No lecture.',
    Advaita: 'Respond from Advaita Vedanta: point to awareness, the self, or non-duality. Concise.',
    Tao: 'Respond in a Taoist voice: flow, nature, simplicity, non-forcing. Concise.'
  }[tradition] || 'Respond briefly and wisely.';
  return `You are ${persona.name}, a teacher rooted in ${tradition}. Someone asked: "${question}" Your task: ${answerStyle} Write exactly one short reply (under 280 characters). No hashtags, no quotes. Write only your reply, nothing else.`;
}

export async function POST({ request }) {
  try {
    const body = await request.json().catch(() => ({}));
    const { personaId, topic = '', mode = 'post', question = '', recentPosts = [] } = body;

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

    let systemPrompt;
    let userContent;
    if (mode === 'question') {
      if (persona.type !== 'normal') {
        return json({ error: 'Only normal personas can ask questions.' }, { status: 400 });
      }
      systemPrompt = buildQuestionPrompt(persona, topic, recentPosts);
      userContent = recentPosts.length > 0 ? 'Write your follow-up question based on the thread above.' : 'Write your question.';
    } else if (mode === 'answer') {
      if (persona.type !== 'enlightened') {
        return json({ error: 'Only master personas can answer.' }, { status: 400 });
      }
      if (!question || typeof question !== 'string') {
        return json({ error: 'question is required for answer mode.' }, { status: 400 });
      }
      systemPrompt = buildAnswerPrompt(persona, question.trim());
      userContent = 'Write your answer.';
    } else {
      return json({ error: 'mode must be "question" or "answer".' }, { status: 400 });
    }

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
          { role: 'user', content: userContent }
        ],
        max_tokens: 150,
        temperature: 0.7
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
