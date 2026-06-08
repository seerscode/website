
<script>
  export let data;

  // Section mapping. Anything tagged "therapeutics" is current work;
  // everything else is the older consciousness archive.
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
</script>

<svelte:head>
  <title>Research | Marcio Diaz</title>
</svelte:head>

<h1 class="text-3xl font-serif font-bold text-slate-900 mb-12">Research & Writings</h1>

<div class="space-y-16">
  {#each grouped as section}
    <section>
      <div class="border-b border-slate-300 pb-4 mb-8">
        <p class="text-xs font-medium text-slate-500 uppercase tracking-widest mb-1">{section.label}</p>
        <p class="text-slate-600 text-sm leading-relaxed max-w-2xl">{section.blurb}</p>
      </div>

      <div class="space-y-10">
        {#each section.posts as post}
          <article>
            <p class="text-sm text-slate-500 mb-2 font-mono">
              {new Date(post.meta.date).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}
            </p>
            <h2 class="text-xl font-serif font-semibold text-slate-900 mb-3">
              <a href={post.path} class="hover:text-slate-800 transition border-b border-slate-300 hover:border-slate-600">
                {post.meta.title}
              </a>
            </h2>
            <p class="text-slate-600">{post.meta.excerpt}</p>
          </article>
        {/each}
      </div>
    </section>
  {/each}
</div>
