module.exports = {
  content: [
    "./index.html",
    "./components/**/*.html",
    "./js/**/*.js"
  ],
  theme: {
    extend: {},
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography')
  ],
};