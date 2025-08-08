
<script>
  import { onMount } from 'svelte';
  export let data;
  
  let toc = [];
  let contentElement;
  let references = [];
  
  // Function to extract references from the markdown content
  function extractReferences() {
    if (!contentElement) {
      return [];
    }
    
    // Look for the "Works Cited" section
    const headings = contentElement.querySelectorAll('h2, h3');
    let worksCitedSection = null;
    
    for (let heading of headings) {
      if (heading.textContent.includes('Works Cited')) {
        worksCitedSection = heading;
        break;
      }
    }
    
    if (!worksCitedSection) {
      return [];
    }
    
    const refs = [];
    let currentElement = worksCitedSection.nextElementSibling;
    
    while (currentElement && currentElement.tagName !== 'H2' && currentElement.tagName !== 'H3') {
      if (currentElement.tagName === 'P') {
        const text = currentElement.textContent;
        const match = text.match(/^\[(\d+)\]\s*(.+?)\s*Accessed.*?\[(https?:\/\/[^\]]+)\]\(([^)]+)\)/);
        if (match) {
          refs.push({
            number: parseInt(match[1]),
            title: match[2].trim(),
            url: match[4]
          });
        }
      }
      currentElement = currentElement.nextElementSibling;
    }
    
    return refs;
  }
  
  // Function to make citation numbers clickable using a more direct approach
  function makeCitationsClickable() {
    if (!contentElement) {
      return;
    }
    
    // Find all citation numbers [1], [2], etc. and replace them with clickable links
    const citationRegex = /\[(\d+)\]/g;
    
    // Process all text nodes in the content
    const processNode = (node) => {
      if (node.nodeType === Node.TEXT_NODE) {
        const text = node.textContent;
        if (citationRegex.test(text)) {
          const parent = node.parentNode;
          if (parent.tagName === 'A') return; // Skip if already a link
          
          const newHTML = text.replace(citationRegex, (match, number) => {
            const ref = references.find(r => r.number === parseInt(number));
            if (ref) {
              return `<a href="#ref-${number}" class="citation-link" data-ref-number="${number}">${match}</a>`;
            }
            return match;
          });
          
          if (newHTML !== text) {
            const wrapper = document.createElement('span');
            wrapper.innerHTML = newHTML;
            parent.replaceChild(wrapper, node);
          }
        }
      } else if (node.nodeType === Node.ELEMENT_NODE) {
        // Recursively process child nodes
        const children = Array.from(node.childNodes);
        children.forEach(processNode);
      }
    };
    
    // Process all nodes in the content
    const allNodes = Array.from(contentElement.childNodes);
    allNodes.forEach(processNode);
  }
  
  // Function to add IDs to reference entries
  function addReferenceIds() {
    if (!contentElement) {
      return;
    }
    
    const headings = contentElement.querySelectorAll('h2, h3');
    let worksCitedSection = null;
    
    for (let heading of headings) {
      if (heading.textContent.includes('Works Cited')) {
        worksCitedSection = heading;
        break;
      }
    }
    
    if (!worksCitedSection) {
      return;
    }
    
    let currentElement = worksCitedSection.nextElementSibling;
    
    while (currentElement && currentElement.tagName !== 'H2' && currentElement.tagName !== 'H3') {
      if (currentElement.tagName === 'P') {
        const text = currentElement.textContent;
        const match = text.match(/^\[(\d+)\]/);
        if (match) {
          currentElement.id = `ref-${match[1]}`;
        }
      }
      currentElement = currentElement.nextElementSibling;
    }
  }
  
  // Function to handle citation clicks
  function handleCitationClick(event) {
    const refNumber = event.target.getAttribute('data-ref-number');
    const refElement = document.getElementById(`ref-${refNumber}`);
    
    if (refElement) {
      refElement.scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
      });
      
      // Add a brief highlight effect
      refElement.style.backgroundColor = '#fef3c7';
      setTimeout(() => {
        refElement.style.backgroundColor = '';
      }, 2000);
    }
  }
  
  // Function to generate TOC from the rendered content
  function generateTOC() {
    if (!contentElement) {
      return [];
    }
    
    // Wait a bit more for the content to be fully rendered
    const headings = contentElement.querySelectorAll('h3, h4, h5');
    
    const tocItems = [];
    
    headings.forEach((heading, index) => {
      const level = parseInt(heading.tagName.charAt(1));
      const text = heading.textContent;
      const id = `heading-${index}`;
      
      // Add ID to heading for anchor links
      heading.id = id;
      
      tocItems.push({
        id,
        text,
        level,
        href: `#${id}`
      });
    });
    
    return tocItems;
  }
  
  // Function to handle smooth scrolling to sections
  function scrollToSection(href) {
    const element = document.querySelector(href);
    if (element) {
      element.scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
      });
    }
  }
  
  // Generate TOC and handle citations after component mounts and content is rendered
  onMount(() => {
    console.log('Component mounted, content element:', contentElement);
    
    // Use a longer delay to ensure content is fully rendered
    setTimeout(() => {
      console.log('Processing content...');
      toc = generateTOC();
      console.log('TOC generated:', toc.length, 'items');
      
      references = extractReferences();
      console.log('References extracted:', references.length, 'items');
      
      addReferenceIds();
      makeCitationsClickable();
      
      // Add event listeners for citation clicks
      const citationLinks = contentElement.querySelectorAll('.citation-link');
      console.log('Citation links found:', citationLinks.length);
      citationLinks.forEach(link => {
        link.addEventListener('click', handleCitationClick);
      });
    }, 1000);
  });
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
  
  <!-- Table of Contents -->
  {#if toc.length > 0}
    <div class="toc-container mb-8 p-6 bg-gray-50 border border-gray-200 rounded-lg">
      <h2 class="text-lg font-semibold text-gray-900 mb-4 flex items-center">
        <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 10h16M4 14h16M4 18h16"></path>
        </svg>
        Table of Contents ({toc.length} sections)
      </h2>
      <nav class="toc-nav">
        <ul class="space-y-2">
          {#each toc as item}
            <li class="toc-item" style="padding-left: {(item.level - 3) * 1.5}rem;">
              <a 
                href={item.href}
                class="text-blue-600 hover:text-blue-800 transition-colors duration-200 text-sm leading-relaxed block py-1"
                on:click|preventDefault={() => scrollToSection(item.href)}
              >
                {item.text}
              </a>
            </li>
          {/each}
        </ul>
      </nav>
    </div>
  {/if}
  
  <!-- Enhanced content with responsive tables -->
  <div 
    bind:this={contentElement}
    class="prose lg:prose-lg max-w-none responsive-content blog-content" 
    style="word-break: break-word; overflow-wrap: break-word;"
  >
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

  /* Blog content styles */
  .blog-content {
    max-width: 4rem;
    margin: 0 auto;
  }

  /* Enhanced prose styling */
  .prose {
    color: #374151;
    line-height: 1.6;
  }

  /* Enhanced responsive table styles */
  .responsive-table {
    @apply block w-full overflow-x-auto whitespace-nowrap border border-gray-200 rounded-lg my-8;
  }

  .responsive-table table {
    @apply w-full border-collapse;
  }

  .responsive-table thead {
    @apply bg-gray-50 border-b-2 border-gray-200;
  }

  .responsive-table tbody tr:nth-child(even) {
    @apply bg-gray-50;
  }

  .responsive-table tbody tr:hover {
    @apply bg-gray-100;
  }

  .responsive-table th,
  .responsive-table td {
    @apply px-4 py-3 text-left border-b border-gray-200 min-w-[120px];
  }

  .responsive-table th {
    @apply font-semibold text-gray-900 sticky top-0 bg-gray-50 z-10;
  }

  .responsive-table td {
    @apply text-gray-700;
  }

  /* Enhanced heading styles with scroll margin */
  :global(.responsive-content h3) {
    font-size: 1.5rem;
    font-weight: 700;
    margin-top: 2rem;
    margin-bottom: 1rem;
    color: #111827;
    scroll-margin-top: 2rem;
  }

  :global(.responsive-content h4) {
    font-size: 1.25rem;
    font-weight: 600;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
    color: #111827;
    scroll-margin-top: 2rem;
  }

  :global(.responsive-content h5) {
    font-size: 1.125rem;
    font-weight: 600;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    color: #111827;
    scroll-margin-top: 2rem;
  }

  /* Citation link styles */
  :global(.citation-link) {
    color: #3b82f6;
    text-decoration: none;
    font-weight: 500;
    transition: all 0.2s ease;
    cursor: pointer;
  }

  :global(.citation-link:hover) {
    color: #1d4ed8;
    text-decoration: underline;
    transform: scale(1.05);
  }

  /* TOC Styles */
  .toc-container {
    position: relative;
  }

  .toc-nav ul {
    list-style: none;
    padding: 0;
    margin: 0;
  }

  .toc-item {
    border-left: 2px solid transparent;
    transition: all 0.2s ease;
  }

  .toc-item:hover {
    border-left-color: #3b82f6;
  }

  .toc-item a {
    display: block;
    text-decoration: none;
    transition: all 0.2s ease;
  }

  .toc-item a:hover {
    transform: translateX(2px);
  }
</style>
