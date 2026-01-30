<script>
  import { normalPersonas, masterPersonas, getPersonaById } from '$lib/personas.js';

  let feed = [];
  let topic = '';
  let topicInput = '';
  let userPostText = '';
  let isRunning = false;
  let loopInterval = null;
  let feedContainer;
  let personaPostLoading = false;
  const LOOP_INTERVAL_MS = 8000;
  let postId = 0;

  function nextId() {
    return `post-${++postId}-${Date.now()}`;
  }

  function addPost(authorId, content, kind = 'post') {
    const author = authorId === 'user' ? { id: 'user', name: 'You', handle: '@you', avatar: '✍️', color: 'bg-blue-100 text-blue-800' } : getPersonaById(authorId);
    if (!author) return;
    const post = {
      id: nextId(),
      authorId,
      author,
      content: content.trim(),
      kind,
      timestamp: new Date().toISOString()
    };
    feed = [post, ...feed];
    if (feedContainer) {
      feedContainer.scrollTop = 0;
    }
  }

  async function tick() {
    if (personaPostLoading || normalPersonas.length === 0 || masterPersonas.length === 0) return;
    const asker = normalPersonas[Math.floor(Math.random() * normalPersonas.length)];
    const master = masterPersonas[Math.floor(Math.random() * masterPersonas.length)];
    const currentTopic = topic || topicInput;
    personaPostLoading = true;
    try {
      const recentPosts = feed
        .slice(0, 12)
        .reverse()
        .map((p) => ({
          authorName: p.author.name,
          content: p.content,
          kind: p.kind || 'post'
        }));
      const questionRes = await fetch('/api/persona-post', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          personaId: asker.id,
          topic: currentTopic,
          mode: 'question',
          recentPosts
        })
      });
      const questionData = await questionRes.json();
      const questionText = questionRes.ok && questionData.post ? questionData.post : questionData.error || '…';

      const answerRes = await fetch('/api/persona-post', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ personaId: master.id, mode: 'answer', question: questionText })
      });
      const answerData = await answerRes.json();
      const answerText = answerRes.ok && answerData.post ? answerData.post : answerData.error || '…';

      addPost(master.id, answerText, 'answer');
      addPost(asker.id, questionText, 'question');
    } catch (e) {
      addPost(asker.id, 'Could not load question.');
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
  <title>Q&A Feed – Marcio Diaz</title>
  <meta name="description" content="A question-and-answer feed: normal people ask, masters (Zen, Advaita, Tao) answer." />
</svelte:head>

<div class="social-page max-w-2xl mx-auto">
  <h1 class="text-2xl font-bold text-gray-900 mb-1">Q&A Feed</h1>
  <p class="text-gray-600 text-sm mb-6">Set a topic — a normal person asks a question, then a master (Zen, Advaita, Tao) answers.</p>

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
        placeholder="Topic for questions (e.g. stress, identity, meaning)"
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
        Click <strong>Start</strong> to generate a question (from a normal person) and an answer (from a master), or add your own post above.
      </div>
    {:else}
      <div class="divide-y divide-gray-100">
        {#each feed as post (post.id)}
          <article class="p-4 hover:bg-gray-50/50 transition {post.kind === 'answer' ? 'bg-amber-50/30' : ''}">
            <div class="flex gap-3">
              <div class="w-10 h-10 rounded-full flex items-center justify-center text-lg shrink-0 {post.author.color}" aria-hidden="true">
                {post.author.avatar}
              </div>
              <div class="flex-1 min-w-0">
                <div class="flex items-center gap-2 flex-wrap">
                  {#if post.kind === 'question'}
                    <span class="text-xs font-medium text-blue-600 uppercase tracking-wide">Question</span>
                  {:else if post.kind === 'answer'}
                    <span class="text-xs font-medium text-amber-700 uppercase tracking-wide">Answer</span>
                  {/if}
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
