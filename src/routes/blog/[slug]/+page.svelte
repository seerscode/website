
<script>
  export let data;
</script>

<svelte:head>
  <title>{data.meta.title} | Marcio Diaz</title>
  <meta name="description" content={data.meta.excerpt} />
</svelte:head>

<article class="blog-content px-4 sm:px-6 lg:px-8">
  <header class="mb-10 border-b border-gray-100 pb-6">
    <h1 class="text-4xl font-bold text-gray-900 mb-4">{data.meta.title}</h1>
    <p class="text-gray-500">
      Published on {new Date(data.meta.date).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}
    </p>
  </header>
  
  <!-- Enhanced content with responsive tables -->
  <div class="prose lg:prose-lg max-w-none responsive-content blog-content" style="word-break: break-word; overflow-wrap: break-word;">
    <!-- Render the markdown content as HTML -->
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

  /* Enhanced responsive table styles */
  :global(.responsive-content table) {
    display: block;
    width: 100%;
    overflow-x: auto;
    white-space: nowrap;
    border-radius: 0.5rem;
    border: 1px solid #e5e7eb;
    margin: 2rem 0;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
  }

  :global(.responsive-content table thead) {
    background-color: #f9fafb;
    border-bottom: 2px solid #e5e7eb;
  }

  :global(.responsive-content table tbody tr:nth-child(even)) {
    background-color: #fafafa;
  }

  :global(.responsive-content table tbody tr:hover) {
    background-color: #f3f4f6;
    transition: background-color 0.2s ease;
  }

  :global(.responsive-content table th),
  :global(.responsive-content table td) {
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid #e5e7eb;
    min-width: 120px;
    vertical-align: top;
  }

  :global(.responsive-content table th) {
    font-weight: 600;
    color: #111827;
    position: sticky;
    top: 0;
    background-color: #f9fafb;
    z-index: 10;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  }

  :global(.responsive-content table td) {
    color: #374151;
  }

  /* Mobile-first responsive design */
  @media (max-width: 768px) {
    :global(.responsive-content table) {
      font-size: 0.75rem;
      margin: 1rem 0;
    }

    :global(.responsive-content table th),
    :global(.responsive-content table td) {
      padding: 0.5rem 0.75rem;
      min-width: 100px;
    }
  }

  /* Enhanced typography for better readability */
  :global(.responsive-content h1) {
    font-size: 2.25rem;
    font-weight: 700;
    margin-top: 2rem;
    margin-bottom: 1rem;
    color: #111827;
  }

  :global(.responsive-content h2) {
    font-size: 1.875rem;
    font-weight: 600;
    margin-top: 1.75rem;
    margin-bottom: 0.75rem;
    color: #111827;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 0.5rem;
  }

  :global(.responsive-content h3) {
    font-size: 1.5rem;
    font-weight: 600;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
    color: #111827;
  }

  :global(.responsive-content h4) {
    font-size: 1.25rem;
    font-weight: 600;
    margin-top: 1.25rem;
    margin-bottom: 0.5rem;
    color: #111827;
  }

  :global(.responsive-content p) {
    margin-bottom: 1rem;
    line-height: 1.7;
    color: #374151;
  }

  :global(.responsive-content ul),
  :global(.responsive-content ol) {
    margin-bottom: 1rem;
    padding-left: 1.5rem;
  }

  :global(.responsive-content li) {
    margin-bottom: 0.5rem;
    line-height: 1.6;
  }

  :global(.responsive-content blockquote) {
    border-left: 4px solid #3b82f6;
    padding-left: 1rem;
    margin: 1.5rem 0;
    font-style: italic;
    color: #6b7280;
    background-color: #f9fafb;
    padding: 1rem;
    border-radius: 0.375rem;
  }

  :global(.responsive-content code) {
    background-color: #f3f4f6;
    padding: 0.125rem 0.25rem;
    border-radius: 0.25rem;
    font-size: 0.875em;
    color: #dc2626;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  }

  :global(.responsive-content pre) {
    background-color: #1f2937;
    color: #f9fafb;
    padding: 1rem;
    border-radius: 0.5rem;
    overflow-x: auto;
    margin: 1.5rem 0;
    border: 1px solid #374151;
  }

  :global(.responsive-content pre code) {
    background-color: transparent;
    padding: 0;
    color: inherit;
  }

  /* Math formula styling */
  :global(.responsive-content .math) {
    overflow-x: auto;
    margin: 1rem 0;
  }

  /* Image styling */
  :global(.responsive-content img) {
    max-width: 100%;
    height: auto;
    border-radius: 0.5rem;
    margin: 1.5rem 0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  }

  /* Horizontal rule styling */
  :global(.responsive-content hr) {
    border: none;
    border-top: 2px solid #e5e7eb;
    margin: 2rem 0;
  }

  /* Link styling */
  :global(.responsive-content a) {
    color: #2563eb;
    text-decoration: none;
    border-bottom: 1px solid transparent;
    transition: all 0.2s ease;
  }

  :global(.responsive-content a:hover) {
    color: #1d4ed8;
    border-bottom-color: #1d4ed8;
  }

  /* Strong and emphasis styling */
  :global(.responsive-content strong) {
    font-weight: 600;
    color: #111827;
  }

  :global(.responsive-content em) {
    font-style: italic;
    color: #4b5563;
  }

  /* Table of contents styling */
  :global(.responsive-content .toc) {
    background-color: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 0.5rem;
    padding: 1.5rem;
    margin: 2rem 0;
  }

  :global(.responsive-content .toc h2) {
    margin-top: 0;
    margin-bottom: 1rem;
    font-size: 1.25rem;
  }

  :global(.responsive-content .toc ul) {
    margin: 0;
    padding-left: 1rem;
  }

  :global(.responsive-content .toc li) {
    margin-bottom: 0.5rem;
  }

  /* Definition list styling */
  :global(.responsive-content dl) {
    margin: 1.5rem 0;
  }

  :global(.responsive-content dt) {
    font-weight: 600;
    color: #111827;
    margin-top: 1rem;
  }

  :global(.responsive-content dd) {
    margin-left: 1rem;
    margin-top: 0.5rem;
    color: #374151;
  }
</style>
