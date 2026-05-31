// Updated Data
const models = ["CodeLlama-7B", "Llama-3-8B", "Qwen-7B", "Gemma2-9B"];
const categories = ["Robo-Instruct", "Evol-Instruct", "Self-Instruct", "Base Model"];
const colors = ['#c870d7', '#98b0f9', '#a5dff9', '#a9a9a9'];

const passAt1 = {
  "Robo-Instruct": [68.75, 66.25, 68.75, 65],
  "Evol-Instruct": [57.5, 57.5, 65, 60],
  "Self-Instruct": [55, 55, 62.5, 57.5],
  "Base Model": [40, 42.5, 55, 51.5]
};

const data = categories.map((category, idx) => ({
  x: models,
  y: passAt1[category],
  type: 'bar',
  name: category,
  marker: {
    color: colors[idx],
    line: { color: '#333', width: 1.2 }
  },
  text: passAt1[category],
  textposition: 'auto',
  opacity: 0.85
}));

const layout = {
  barmode: 'group',
  xaxis: { title: 'Models' },
  yaxis: { title: 'Pass@1 (%)', range: [0, 80] },
  legend: { orientation: 'h', y: 1.1 }
};

Plotly.newPlot('bar-chart', data, layout);
