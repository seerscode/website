
<script>
  import "../app.css";
  import 'katex/dist/katex.min.css'
</script>

<div class="min-h-screen bg-white">
  <!-- Header -->
  <header class="py-8">
    <div class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
      <nav class="flex justify-between items-center">
        <a href="/" class="text-xl font-semibold text-gray-900 hover:text-gray-700 transition">Marcio Diaz</a>
        <div class="space-x-6">
          <a href="/blog" class="text-gray-600 hover:text-gray-900 transition duration-150">Blog</a>
          <a href="/about" class="text-gray-600 hover:text-gray-900 transition duration-150">About</a>
          <a href="/chat" class="text-gray-600 hover:text-gray-900 transition duration-150">NonDualGPT</a>
        </div>
      </nav>
    </div>
  </header>

  <!-- Main Content -->
  <main class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
    <slot />
  </main>

  <!-- Footer -->
  <footer class="py-8 mt-16 border-t border-gray-100">
     <div class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-sm text-gray-500">
      &copy; {new Date().getFullYear()} Marcio Diaz.
    </div>
  </footer>
</div>
