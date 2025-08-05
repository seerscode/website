
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
  <div class="prose lg:prose-lg max-w-none" style="word-break: break-word; overflow-wrap: break-word;">
    <!-- Render the markdown content -->
    <svelte:component this={data.content} />
  </div>

    <div class="mt-12">
        <a href="/blog" class="text-blue-600 hover:text-blue-800 transition hover:underline">&larr; Back to blog</a>
    </div>
</article>

<style>
  /* Additional styles to handle long URLs and improve readability */
  :global(.prose a) {
    word-break: break-all;
    overflow-wrap: break-word;
    hyphens: auto;
  }
  
  /* Specifically target the Works cited section */
  :global(.prose p:has(a[href*="arxiv.org"]),
          .prose p:has(a[href*="aclanthology.org"]),
          .prose p:has(a[href*="openreview.net"]),
          .prose p:has(a[href*="medium.com"]),
          .prose p:has(a[href*="lesswrong.com"]),
          .prose p:has(a[href*="reddit.com"])) {
    word-break: break-all;
    overflow-wrap: break-word;
    line-height: 1.6;
  }
</style>
