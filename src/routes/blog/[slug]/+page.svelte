<script>
  export let data;

  function fmtDate(d) {
    return new Date(d).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
  }
</script>

<svelte:head>
  <title>{data.meta.title} | Marcio Diaz</title>
  <meta name="description" content={data.meta.excerpt} />
</svelte:head>

<article>
  <header class="mb-12 pb-8 border-b border-stone-200">
    <p class="eyebrow mb-6">
      {data.meta.category === 'therapeutics' ? 'Therapeutics' : 'Machine Consciousness'}
      <span class="ml-3 text-stone-400">·</span>
      <span class="ml-3 normal-case tracking-normal text-stone-500">{fmtDate(data.meta.date)}</span>
    </p>
    <h1 class="font-display text-3xl sm:text-4xl font-semibold tracking-tight text-stone-900 leading-[1.1] mb-5">
      {data.meta.title}
    </h1>
    {#if data.meta.excerpt}
      <p class="text-lg text-stone-600 leading-relaxed max-w-2xl">
        {data.meta.excerpt}
      </p>
    {/if}
  </header>

  <div class="blog-content">
    <svelte:component this={data.content} />
  </div>

  <footer class="mt-16 pt-8 border-t border-stone-200">
    <a href="/blog" class="inline-flex items-center gap-1.5 text-sm text-stone-500 hover:text-stone-900 transition">
      <svg class="w-3.5 h-3.5" viewBox="0 0 14 14" fill="none"><path d="M11 3L3 11M3 11H9M3 11V5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
      All writings
    </a>
  </footer>
</article>
