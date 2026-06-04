
<script>
  export let data;

  const CATEGORY_META = {
    therapeutics: {
      title: 'Therapeutics & Longinus',
      description: 'Antisense, rare disease, CYFIP1, and notes from building a biotech.'
    },
    consciousness: {
      title: 'Machine Consciousness',
      description: 'Earlier work on integrated information, transformers, and the question of phenomenal experience in artificial systems.'
    },
    other: {
      title: 'Other',
      description: ''
    }
  };

  const ORDER = ['therapeutics', 'consciousness', 'other'];

  $: grouped = (() => {
    const buckets = {};
    for (const post of data.posts) {
      const cat = (post.meta && post.meta.category) || 'other';
      if (!buckets[cat]) buckets[cat] = [];
      buckets[cat].push(post);
    }
    return ORDER.filter((k) => buckets[k] && buckets[k].length).map((k) => ({
      key: k,
      meta: CATEGORY_META[k] || { title: k, description: '' },
      posts: buckets[k]
    }));
  })();
</script>

<svelte:head>
  <title>Research | Marcio Diaz</title>
</svelte:head>

<h1 class="text-3xl font-serif font-bold text-slate-900 mb-4">Research & Writings</h1>
<p class="text-slate-600 mb-12 max-w-2xl">
  Two threads run through this page. The current work is on therapeutics — antisense oligonucleotides for a rare chromosomal microduplication. The earlier work is on machine consciousness and transformer interpretability. Both are kept here.
</p>

<div class="space-y-16">
  {#each grouped as section}
    <section>
      <header class="mb-8 pb-3 border-b border-slate-200">
        <h2 class="text-2xl font-serif font-semibold text-slate-900">{section.meta.title}</h2>
        {#if section.meta.description}
          <p class="text-slate-600 mt-2 max-w-2xl">{section.meta.description}</p>
        {/if}
      </header>

      <div class="space-y-10">
        {#each section.posts as post}
          <article>
            <p class="text-sm text-slate-500 mb-2 font-mono">
              {new Date(post.meta.date).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}
            </p>
            <h3 class="text-xl font-serif font-semibold text-slate-900 mb-3">
              <a href={post.path} class="hover:text-slate-800 transition border-b border-slate-300 hover:border-slate-600">
                {post.meta.title}
              </a>
            </h3>
            <p class="text-slate-600">{post.meta.excerpt}</p>
          </article>
        {/each}
      </div>
    </section>
  {/each}
</div>
