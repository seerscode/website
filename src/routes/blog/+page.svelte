
<script>
  export let data;
</script>

<svelte:head>
  <title>Blog | Marcio Diaz</title>
</svelte:head>

<h1 class="text-3xl font-bold text-gray-900 mb-10">Writings & Teachings</h1>

<div class="space-y-10">
  {#each data.posts as post}
    <article>
      <p class="text-sm text-gray-500 mb-2">
        {new Date(post.meta.date).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}
      </p>
      <h2 class="text-2xl font-semibold text-gray-900 mb-3">
        <a href={post.path} class="hover:text-blue-600 transition hover:underline">
          {post.meta.title}
        </a>
      </h2>
      <p class="text-gray-600">{post.meta.excerpt}</p>
    </article>
  {/each}
</div>
