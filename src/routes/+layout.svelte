<script>
  import "../app.css";
  import 'katex/dist/katex.min.css';
  import { page } from '$app/stores';
  $: path = $page.url.pathname;
</script>

<div class="min-h-screen flex flex-col bg-white text-stone-900 antialiased">
  <header class="sticky top-0 z-40 border-b border-stone-200/60 bg-white/70 backdrop-blur-md">
    <div class="max-w-5xl mx-auto px-6 sm:px-8 h-14 flex items-center justify-between">
      <a href="/" class="group flex items-center gap-2.5">
        <!-- Personal mark: a single offset square — minimal, abstract -->
        <svg class="w-5 h-5 text-stone-900" viewBox="0 0 20 20" fill="none">
          <rect x="2" y="2" width="11" height="11" stroke="currentColor" stroke-width="1.4"/>
          <rect x="7" y="7" width="11" height="11" fill="currentColor"/>
        </svg>
        <span class="font-display font-medium tracking-tight text-[0.9375rem] text-stone-900">Marcio Diaz</span>
      </a>
      <nav class="flex items-center gap-1">
        <a
          href="/blog"
          class="px-3 py-1.5 text-sm font-medium rounded-md transition-colors {path.startsWith('/blog') ? 'text-stone-900 bg-stone-100' : 'text-stone-500 hover:text-stone-900 hover:bg-stone-50'}"
        >Research</a>
        <a
          href="/about"
          class="px-3 py-1.5 text-sm font-medium rounded-md transition-colors {path.startsWith('/about') ? 'text-stone-900 bg-stone-100' : 'text-stone-500 hover:text-stone-900 hover:bg-stone-50'}"
        >About</a>
      </nav>
    </div>
  </header>

  <main class="flex-1 w-full max-w-3xl mx-auto px-6 sm:px-8 py-16 sm:py-24">
    <slot />
  </main>

  <footer class="border-t border-stone-200/60 py-10 mt-16">
    <div class="max-w-5xl mx-auto px-6 sm:px-8 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 text-xs text-stone-500">
      <div class="flex items-center gap-2">
        <span>© {new Date().getFullYear()} Marcio Diaz</span>
        <span class="text-stone-300">·</span>
        <span>Madrid</span>
      </div>
      <div class="flex items-center gap-5">
        <a href="https://longinus.bio" target="_blank" rel="noopener" class="hover:text-stone-900 transition">longinus.bio</a>
        <a href="https://x.com/marciodiaz_ai" target="_blank" rel="noopener" class="hover:text-stone-900 transition">x.com</a>
        <a href="mailto:marcio@longinus.bio" class="hover:text-stone-900 transition">email</a>
      </div>
    </div>
  </footer>
</div>
