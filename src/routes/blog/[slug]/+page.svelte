
<script>
  export let data;
</script>

<svelte:head>
  <title>{data.meta.title} | Marcio Diaz</title>
  <meta name="description" content={data.meta.excerpt} />
</svelte:head>

<article>
  <header class="mb-10 border-b border-gray-100 pb-6">
    <h1 class="text-4xl font-bold text-gray-900 mb-4">{data.meta.title}</h1>
    <p class="text-gray-500">
      Published on {new Date(data.meta.date).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}
    </p>
  </header>
  <!-- Using 'prose' for beautiful typography on the markdown content -->
  <div class="prose lg:prose-lg max-w-none">
    <!-- Render the markdown content -->
    <svelte:component this={data.content} />
  </div>

    <div class="mt-12">
        <a href="/blog" class="text-blue-600 hover:text-blue-800 transition hover:underline">&larr; Back to blog</a>
    </div>
</article>
