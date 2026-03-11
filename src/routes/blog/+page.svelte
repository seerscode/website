
<script>
  export let data;
</script>

<svelte:head>
  <title>Research | Marcio Diaz</title>
</svelte:head>

<h1 class="text-3xl font-serif font-bold text-slate-900 mb-10">Research & Writings</h1>

<div class="space-y-10">
  {#each data.posts as post}
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
