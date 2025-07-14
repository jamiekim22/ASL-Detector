module.exports = {
  content: [
    "./index.html",
    "./components/**/*.html",
    "./js/**/*.js",
    "./styles/main.css"
  ],
  theme: {
    extend: {},
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography')
  ],
};