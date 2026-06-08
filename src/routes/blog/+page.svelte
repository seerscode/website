<script>
  export let data;

  const SECTIONS = [
    {
      id: 'therapeutics',
      label: 'Therapeutics',
      blurb: 'Current focus. CYFIP1 ASO program at Longinus Therapeutics — rare-disease drug development, regulatory economics, and the engineering of antisense oligonucleotides for 15q11.2 BP1-BP2 microduplication.',
      match: (p) => p.meta.category === 'therapeutics',
    },
    {
      id: 'consciousness',
      label: 'Machine Consciousness',
      blurb: 'Earlier work. Integrated information theory, global workspace theory, transformer interpretability, and the question of what it would take for a machine to have an inside.',
      match: (p) => p.meta.category !== 'therapeutics',
    },
  ];

  $: grouped = SECTIONS.map((s) => ({
    ...s,
    posts: data.posts.filter(s.match),
  })).filter((s) => s.posts.length > 0);

  function fmtDate(d) {
    return new Date(d).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
  }
</script>

<svelte:head>
  <title>Research | Marcio Diaz</title>
</svelte:head>

<div class="space-y-16">
  <header>
    <p class="eyebrow mb-6">Research & writings</p>
    <h1 class="font-display text-3xl sm:text-4xl font-semibold tracking-tight text-stone-900 leading-tight mb-4">
      Notes from a <span class="font-serif-italic">two-act</span> career.
    </h1>
    <p class="text-stone-600 max-w-xl text-base leading-relaxed">
      Currently in act two — biotech, antisense oligonucleotides, rare disease drug development. Act one was machine consciousness research. Both sections live below.
    </p>
  </header>

  {#each grouped as section, sIdx}
    <section class="space-y-8">
      <div class="border-t border-stone-200 pt-10">
        <p class="eyebrow mb-3">{section.label}</p>
        <p class="text-sm text-stone-500 leading-relaxed max-w-xl">{section.blurb}</p>
      </div>

      <ul class="divide-y divide-stone-100">
        {#each section.posts as post, i}
          <li>
            <a
              href={post.path}
              class="group grid grid-cols-[auto_1fr_auto] gap-x-6 items-baseline py-5 -mx-2 px-2 rounded-lg hover:bg-stone-50 transition"
            >
              <span class="font-mono text-xs text-stone-400 tabular-nums">
                {fmtDate(post.meta.date)}
              </span>
              <span class="font-display text-base sm:text-[1.0625rem] font-medium text-stone-900 group-hover:text-stone-950 leading-snug tracking-tight">
                {post.meta.title}
              </span>
              <svg
                class="w-3.5 h-3.5 text-stone-300 group-hover:text-stone-600 group-hover:translate-x-0.5 transition shrink-0"
                viewBox="0 0 14 14" fill="none"
              ><path d="M3 11L11 3M11 3H5M11 3V9" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
            </a>
          </li>
        {/each}
      </ul>
    </section>
  {/each}
</div>
