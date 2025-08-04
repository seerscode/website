
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
          },
        },
      }),
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}
