
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{html,js,svelte,ts,md}'],
  theme: {
    extend: {
      // Using Inter font for a clean, modern look
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
      // Customize typography colors for a cleaner look
      typography: (theme) => ({
        DEFAULT: {
          css: {
            color: theme('colors.gray.800'),
            a: {
              color: theme('colors.blue.600'),
              'text-decoration': 'none',
              '&:hover': {
                'text-decoration': 'underline',
                color: theme('colors.blue.800'),
              },
            },
            h1: {
              color: theme('colors.gray.900'),
            },
            h2: {
              color: theme('colors.gray.900'),
            },
            h3: {
              color: theme('colors.gray.900'),
            },
            h4: {
              color: theme('colors.gray.900'),
            },
            h5: {
              color: theme('colors.gray.900'),
            },
            h6: {
              color: theme('colors.gray.900'),
            },
            // Enhanced table styling
            table: {
              'border-collapse': 'collapse',
              'width': '100%',
              'margin': '2rem 0',
              'font-size': '0.875rem',
              'line-height': '1.25rem',
            },
            'thead': {
              'border-bottom': '2px solid #e5e7eb',
              'background-color': '#f9fafb',
            },
            'tbody tr': {
              'border-bottom': '1px solid #e5e7eb',
              '&:hover': {
                'background-color': '#f9fafb',
              },
            },
            'th, td': {
              'padding': '0.75rem 1rem',
              'text-align': 'left',
              'vertical-align': 'top',
            },
            th: {
              'font-weight': '600',
              'color': theme('colors.gray.900'),
            },
            td: {
              'color': theme('colors.gray.700'),
            },
            // Code block styling
            'pre': {
              'background-color': '#1f2937',
              'color': '#f9fafb',
              'padding': '1rem',
              'border-radius': '0.5rem',
              'overflow-x': 'auto',
              'margin': '1.5rem 0',
            },
            'code': {
              'background-color': '#f3f4f6',
              'padding': '0.125rem 0.25rem',
              'border-radius': '0.25rem',
              'font-size': '0.875em',
              'color': '#dc2626',
            },
            'pre code': {
              'background-color': 'transparent',
              'padding': '0',
              'color': 'inherit',
            },
            // Blockquote styling
            blockquote: {
              'border-left': '4px solid #3b82f6',
              'padding-left': '1rem',
              'margin': '1.5rem 0',
              'font-style': 'italic',
              'color': theme('colors.gray.600'),
            },
            // List styling
            'ul, ol': {
              'padding-left': '1.5rem',
            },
            li: {
              'margin': '0.5rem 0',
            },
          },
        },
      }),
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}
