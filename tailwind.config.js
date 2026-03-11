
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{html,js,svelte,ts,md}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        serif: ['Source Serif 4', 'Georgia', 'serif'],
      },
      // Customize typography colors for a cleaner look
      typography: (theme) => ({
        DEFAULT: {
          css: {
            color: theme('colors.slate.700'),
            a: {
              color: theme('colors.slate.700'),
              'text-decoration': 'none',
              'border-bottom': '1px solid',
              'border-color': theme('colors.slate.300'),
              '&:hover': {
                color: theme('colors.slate.900'),
                'border-color': theme('colors.slate.600'),
              },
            },
            h1: {
              color: theme('colors.slate.900'),
              'font-family': theme('fontFamily.serif').join(', '),
            },
            h2: {
              color: theme('colors.slate.900'),
              'font-family': theme('fontFamily.serif').join(', '),
            },
            h3: {
              color: theme('colors.slate.900'),
              'font-family': theme('fontFamily.serif').join(', '),
            },
            h4: {
              color: theme('colors.slate.900'),
              'font-family': theme('fontFamily.serif').join(', '),
            },
            h5: {
              color: theme('colors.slate.900'),
              'font-family': theme('fontFamily.serif').join(', '),
            },
            h6: {
              color: theme('colors.slate.900'),
              'font-family': theme('fontFamily.serif').join(', '),
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
              'background-color': '#f1f5f9',
              'padding': '0.125rem 0.25rem',
              'border-radius': '0.25rem',
              'font-size': '0.875em',
              'color': '#334155',
            },
            'pre code': {
              'background-color': 'transparent',
              'padding': '0',
              'color': 'inherit',
            },
            // Blockquote styling
            blockquote: {
              'border-left': '4px solid #94a3b8',
              'padding-left': '1rem',
              'margin': '1.5rem 0',
              'font-style': 'italic',
              'color': theme('colors.slate.600'),
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
