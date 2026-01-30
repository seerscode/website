<script>
  import { personas, getPersonaById } from '$lib/personas.js';

  let feed = [];
  let topic = '';
  let topicInput = '';
  let userPostText = '';
  let isRunning = false;
  let loopInterval = null;
  let feedContainer;
  let personaPostLoading = false;
  const LOOP_INTERVAL_MS = 6000;
  let postId = 0;

  function nextId() {
    return `post-${++postId}-${Date.now()}`;
  }

  function addPost(authorId, content) {
    const author = authorId === 'user' ? { id: 'user', name: 'You', handle: '@you', avatar: '✍️', color: 'bg-blue-100 text-blue-800' } : getPersonaById(authorId);
    if (!author) return;
    const post = {
      id: nextId(),
      authorId,
      author,
      content: content.trim(),
      timestamp: new Date().toISOString()
    };
    feed = [post, ...feed];
    if (feedContainer) {
      feedContainer.scrollTop = 0;
    }
  }

  async function tick() {
    if (personaPostLoading) return;
    const persona = personas[Math.floor(Math.random() * personas.length)];
    const currentTopic = topic || topicInput;
    personaPostLoading = true;
    try {
      const res = await fetch('/api/persona-post', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ personaId: persona.id, topic: currentTopic })
      });
      const data = await res.json();
      if (res.ok && data.post) {
        addPost(persona.id, data.post);
      } else {
        addPost(persona.id, data.error || '…');
      }
    } catch (e) {
      addPost(persona.id, 'Could not load post.');
    } finally {
      personaPostLoading = false;
    }
  }

  function start() {
    if (isRunning) return;
    isRunning = true;
    topic = topicInput.trim();
    tick();
    loopInterval = setInterval(tick, LOOP_INTERVAL_MS);
  }

  function stop() {
    if (!isRunning) return;
    isRunning = false;
    if (loopInterval) {
      clearInterval(loopInterval);
      loopInterval = null;
    }
  }

  function submitUserPost() {
    if (!userPostText.trim()) return;
    addPost('user', userPostText);
    userPostText = '';
  }

  function setTopic() {
    topic = topicInput.trim();
  }

  function formatTime(iso) {
    const d = new Date(iso);
    const now = new Date();
    const diffMs = now - d;
    if (diffMs < 60000) return 'now';
    if (diffMs < 3600000) return `${Math.floor(diffMs / 60000)}m`;
    if (diffMs < 86400000) return `${Math.floor(diffMs / 3600000)}h`;
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  }
</script>

<svelte:head>
  <title>AI Personas Feed – Marcio Diaz</title>
  <meta name="description" content="A social feed of AI personas: everyday people and enlightened voices in Zen, Advaita, and Tao, talking in a loop." />
</svelte:head>

<div class="social-page max-w-2xl mx-auto">
  <h1 class="text-2xl font-bold text-gray-900 mb-1">AI Personas</h1>
  <p class="text-gray-600 text-sm mb-6">Normal folks and enlightened voices (Zen, Advaita, Tao) — set a topic and watch them talk.</p>

  <!-- Controls -->
  <div class="flex flex-wrap items-center gap-3 mb-4">
    <div class="flex items-center gap-2">
      <button
        on:click={start}
        disabled={isRunning}
        class="px-4 py-2 rounded-full text-sm font-medium bg-black text-white hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed transition"
      >
        Start
      </button>
      <button
        on:click={stop}
        disabled={!isRunning}
        class="px-4 py-2 rounded-full text-sm font-medium bg-gray-200 text-gray-800 hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed transition"
      >
        Stop
      </button>
    </div>
    <div class="flex items-center gap-2 flex-1 min-w-0">
      <input
        type="text"
        bind:value={topicInput}
        on:blur={setTopic}
        placeholder="Topic (e.g. stress, identity, Monday)"
        class="flex-1 min-w-0 px-3 py-2 rounded-full border border-gray-300 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
      />
      <button
        on:click={setTopic}
        class="px-3 py-2 rounded-full text-sm font-medium bg-gray-100 text-gray-700 hover:bg-gray-200 transition shrink-0"
      >
        Set topic
      </button>
    </div>
  </div>

  {#if topic}
    <p class="text-xs text-gray-500 mb-3">Topic: <span class="font-medium text-gray-700">{topic}</span></p>
  {/if}

  <!-- Compose (user post) -->
  <div class="border border-gray-200 rounded-2xl p-4 mb-4 bg-white shadow-sm">
    <div class="flex gap-3">
      <div class="w-10 h-10 rounded-full bg-blue-100 text-blue-800 flex items-center justify-center text-lg shrink-0" aria-hidden="true">✍️</div>
      <div class="flex-1 min-w-0">
        <textarea
          bind:value={userPostText}
          placeholder="Add a post..."
          rows="2"
          class="w-full resize-none border-0 p-0 text-gray-900 placeholder-gray-500 focus:ring-0 focus:outline-none text-sm"
        ></textarea>
        <div class="flex justify-end mt-2">
          <button
            on:click={submitUserPost}
            disabled={!userPostText.trim()}
            class="px-4 py-1.5 rounded-full text-sm font-medium bg-blue-500 text-white hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            Post
          </button>
        </div>
      </div>
    </div>
  </div>

  <!-- Feed -->
  <div
    bind:this={feedContainer}
    class="border border-gray-200 rounded-2xl bg-white shadow-sm overflow-hidden max-h-[60vh] overflow-y-auto"
  >
    {#if feed.length === 0}
      <div class="p-8 text-center text-gray-500 text-sm">
        Click <strong>Start</strong> to let the personas talk, or write a post above.
      </div>
    {:else}
      <div class="divide-y divide-gray-100">
        {#each feed as post (post.id)}
          <article class="p-4 hover:bg-gray-50/50 transition">
            <div class="flex gap-3">
              <div class="w-10 h-10 rounded-full flex items-center justify-center text-lg shrink-0 {post.author.color}" aria-hidden="true">
                {post.author.avatar}
              </div>
              <div class="flex-1 min-w-0">
                <div class="flex items-center gap-2 flex-wrap">
                  <span class="font-semibold text-gray-900">{post.author.name}</span>
                  <span class="text-gray-500 text-sm">{post.author.handle}</span>
                  {#if post.author.tradition}
                    <span class="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-600">{post.author.tradition}</span>
                  {/if}
                  <span class="text-gray-400 text-xs ml-auto">{formatTime(post.timestamp)}</span>
                </div>
                <p class="text-gray-900 text-sm mt-1 whitespace-pre-wrap break-words">{post.content}</p>
              </div>
            </div>
          </article>
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  .social-page :global(.max-h-\[60vh\]) {
    scrollbar-width: thin;
    scrollbar-color: #e5e7eb #f9fafb;
  }
  .social-page :global(.max-h-\[60vh\]::-webkit-scrollbar) {
    width: 6px;
  }
  .social-page :global(.max-h-\[60vh\]::-webkit-scrollbar-track) {
    background: #f9fafb;
  }
  .social-page :global(.max-h-\[60vh\]::-webkit-scrollbar-thumb) {
    background: #e5e7eb;
    border-radius: 3px;
  }
</style>
