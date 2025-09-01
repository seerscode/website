<script>
  import { onMount } from 'svelte';
  
  let messages = [];
  let inputValue = '';
  let isLoading = false;
  let chatContainer;
  
  // Load messages from localStorage on mount
  onMount(() => {
    const savedMessages = localStorage.getItem('chat-messages');
    if (savedMessages) {
      messages = JSON.parse(savedMessages);
      scrollToBottom();
    }
  });
  
  // Save messages to localStorage whenever messages change
  $: if (typeof window !== 'undefined') {
    localStorage.setItem('chat-messages', JSON.stringify(messages));
  }
  
  function scrollToBottom() {
    setTimeout(() => {
      if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
      }
    }, 100);
  }
  
  async function sendMessage() {
    if (!inputValue.trim() || isLoading) return;
    
    const userMessage = {
      role: 'user',
      content: inputValue.trim(),
      timestamp: new Date().toISOString()
    };
    
    messages = [...messages, userMessage];
    inputValue = '';
    isLoading = true;
    scrollToBottom();
    
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: messages.map(msg => ({
            role: msg.role,
            content: msg.content
          }))
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      const assistantMessage = {
        role: 'assistant',
        content: data.message,
        timestamp: new Date().toISOString()
      };
      
      messages = [...messages, assistantMessage];
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please make sure you have set up your OpenAI API key in the environment variables.',
        timestamp: new Date().toISOString(),
        isError: true
      };
      messages = [...messages, errorMessage];
    } finally {
      isLoading = false;
      scrollToBottom();
    }
  }
  
  function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }
  
  function clearChat() {
    messages = [];
    localStorage.removeItem('chat-messages');
  }
  
  function formatTime(timestamp) {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  }
</script>

<svelte:head>
  <title>NonDualGPT Chat - Marcio Diaz</title>
  <meta name="description" content="Chat with NonDualGPT: a unique and powerful AI model that is able to reason and understand the world in a non-dualistic way." />
</svelte:head>

<div class="min-h-screen bg-gray-50">
  <div class="max-w-4xl mx-auto px-4 py-8">
    <!-- Header -->
    <div class="mb-6">
      <h1 class="text-3xl font-bold text-gray-900 mb-2">NonDualGPT Chat</h1>
      <p class="text-gray-600">Chat with NonDualGPT: a unique and powerful AI model that is able to reason and understand the world in a non-dualistic way.</p>
    </div>
    
    <!-- Chat Container -->
    <div class="bg-white rounded-lg shadow-lg overflow-hidden">
      <!-- Chat Messages -->
      <div 
        bind:this={chatContainer}
        class="h-96 overflow-y-auto p-6 space-y-4"
      >
        {#if messages.length === 0}
          <div class="text-center text-gray-500 py-8">
            <div class="text-4xl mb-4">🤖</div>
            <p>Start a conversation with NonDualGPT!</p>
            <p class="text-sm mt-2">Ask me anything - I can help with questions, creative writing, coding, and more.</p>
          </div>
        {:else}
          {#each messages as message (message.timestamp)}
            <div class="flex {message.role === 'user' ? 'justify-end' : 'justify-start'}">
              <div class="max-w-xs lg:max-w-md px-4 py-2 rounded-lg {
                message.role === 'user' 
                  ? 'bg-blue-500 text-white' 
                  : message.isError 
                    ? 'bg-red-100 text-red-800 border border-red-200'
                    : 'bg-gray-100 text-gray-800'
              }">
                <div class="whitespace-pre-wrap">{message.content}</div>
                <div class="text-xs opacity-70 mt-1">
                  {formatTime(message.timestamp)}
                </div>
              </div>
            </div>
          {/each}
          
          {#if isLoading}
            <div class="flex justify-start">
              <div class="bg-gray-100 text-gray-800 px-4 py-2 rounded-lg">
                <div class="flex items-center space-x-2">
                  <div class="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600"></div>
                  <span>AI is thinking...</span>
                </div>
              </div>
            </div>
          {/if}
        {/if}
      </div>
      
      <!-- Input Area -->
      <div class="border-t bg-gray-50 p-4">
        <div class="flex space-x-2">
          <textarea
            bind:value={inputValue}
            on:keypress={handleKeyPress}
            placeholder="Type your message here..."
            class="flex-1 resize-none border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            rows="2"
            disabled={isLoading}
          ></textarea>
          <button
            on:click={sendMessage}
            disabled={!inputValue.trim() || isLoading}
            class="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? 'Sending...' : 'Send'}
          </button>
        </div>
        
        <!-- Clear Chat Button -->
        {#if messages.length > 0}
          <div class="mt-2 flex justify-end">
            <button
              on:click={clearChat}
              class="text-sm text-gray-500 hover:text-gray-700 underline"
            >
              Clear chat history
            </button>
          </div>
        {/if}
      </div>
    </div>
    
    <!-- Instructions -->
    <div class="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
      <h3 class="font-semibold text-blue-900 mb-2">💡 Tips for better conversations:</h3>
      <ul class="text-sm text-blue-800 space-y-1">
        <li>• Be specific in your questions for more accurate responses</li>
        <li>• You can ask for help with coding, writing, analysis, or general knowledge</li>
        <li>• The conversation history is saved locally in your browser</li>
      </ul>
    </div>
  </div>
</div>

<style>
  /* Custom scrollbar for chat container */
  :global(.h-96) {
    scrollbar-width: thin;
    scrollbar-color: #cbd5e0 #f7fafc;
  }
  
  :global(.h-96::-webkit-scrollbar) {
    width: 6px;
  }
  
  :global(.h-96::-webkit-scrollbar-track) {
    background: #f7fafc;
  }
  
  :global(.h-96::-webkit-scrollbar-thumb) {
    background: #cbd5e0;
    border-radius: 3px;
  }
  
  :global(.h-96::-webkit-scrollbar-thumb:hover) {
    background: #a0aec0;
  }
</style>
