const DATASET_URL = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv";
const TARGET = "Profit";
const els = Object.fromEntries(["status","bestK","bestRmse","bestR2","rowCount","targetBadge","resultsBody"].map(id => [id, document.getElementById(id)]));
let rmseChart;
let r2Chart;

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  const split = line => line.split(/,(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)/).map(v => v.trim().replace(/^"|"$/g, ""));
  const headers = split(lines[0]);
  return lines.slice(1).map(line => Object.fromEntries(split(line).map((value, i) => [headers[i], value])));
}

function numericColumns(rows) {
  return Object.keys(rows[0]).filter(key => rows.every(row => row[key] !== "" && Number.isFinite(Number(row[key]))));
}

function isIndexColumn(key) {
  return !key || /^unnamed|^index$/i.test(key);
}

function encodeRows(rows, target) {
  const numeric = numericColumns(rows).filter(key => key === target || !isIndexColumn(key));
  const categorical = Object.keys(rows[0]).filter(key => key !== target && !numeric.includes(key) && !isIndexColumn(key));
  const categories = Object.fromEntries(categorical.map(key => [key, [...new Set(rows.map(row => row[key]))].sort().slice(1)]));
  return rows.map(row => {
    const encoded = Object.fromEntries(numeric.map(key => [key, Number(row[key])]));
    categorical.forEach(key => categories[key].forEach(value => { encoded[`${key}_${value}`] = row[key] === value ? 1 : 0; }));
    return encoded;
  });
}

function correlation(rows, feature, target) {
  const x = rows.map(r => Number(r[feature]));
  const y = rows.map(r => Number(r[target]));
  const mx = x.reduce((a,b) => a+b, 0) / x.length;
  const my = y.reduce((a,b) => a+b, 0) / y.length;
  const num = x.reduce((s,v,i) => s + (v-mx)*(y[i]-my), 0);
  const den = Math.sqrt(x.reduce((s,v) => s+(v-mx)**2,0) * y.reduce((s,v) => s+(v-my)**2,0));
  return den ? num / den : 0;
}

function solve(matrix, vector) {
  const n = vector.length;
  const a = matrix.map((row, i) => [...row, vector[i]]);
  for (let i = 0; i < n; i++) {
    let pivot = i;
    for (let j = i + 1; j < n; j++) if (Math.abs(a[j][i]) > Math.abs(a[pivot][i])) pivot = j;
    [a[i], a[pivot]] = [a[pivot], a[i]];
    if (Math.abs(a[i][i]) < 1e-10) throw new Error("特徵矩陣無法求解，請移除重複或常數欄位。");
    const divisor = a[i][i];
    for (let k = i; k <= n; k++) a[i][k] /= divisor;
    for (let j = 0; j < n; j++) if (j !== i) {
      const factor = a[j][i];
      for (let k = i; k <= n; k++) a[j][k] -= factor * a[i][k];
    }
  }
  return a.map(row => row[n]);
}

function fitPredict(train, test, features, target) {
  const X = train.map(r => [1, ...features.map(f => Number(r[f]))]);
  const y = train.map(r => Number(r[target]));
  const p = X[0].length;
  const xtx = Array.from({length:p}, (_,i) => Array.from({length:p}, (_,j) => X.reduce((s,row) => s + row[i]*row[j], 0)));
  const xty = Array.from({length:p}, (_,i) => X.reduce((s,row,k) => s + row[i]*y[k], 0));
  const beta = solve(xtx, xty);
  return test.map(r => beta[0] + features.reduce((s,f,i) => s + beta[i+1]*Number(r[f]), 0));
}

function seededSplit(rows) {
  const keyed = rows.map((row, i) => ({row, key: Math.sin((i + 1) * 999) * 10000 % 1})).sort((a,b) => a.key-b.key).map(v => v.row);
  const cut = Math.max(2, Math.floor(keyed.length * .8));
  return [keyed.slice(0, cut), keyed.slice(cut)];
}

function evaluate(actual, predicted) {
  const mean = actual.reduce((a,b) => a+b,0) / actual.length;
  const sse = actual.reduce((s,v,i) => s + (v-predicted[i])**2, 0);
  const sst = actual.reduce((s,v) => s + (v-mean)**2, 0);
  return {rmse: Math.sqrt(sse/actual.length), r2: 1-sse/sst};
}

function analyze(rows, preferredTarget) {
  const numbers = numericColumns(rows);
  const target = numbers.includes(preferredTarget) ? preferredTarget : numbers[numbers.length - 1];
  const encodedRows = encodeRows(rows, target);
  const features = Object.keys(encodedRows[0]).filter(c => c !== target);
  if (features.length < 1 || !target) throw new Error("Dataset 至少需要一個數值特徵及一個數值目標欄位。");
  const ranked = features.sort((a,b) => Math.abs(correlation(encodedRows,b,target))-Math.abs(correlation(encodedRows,a,target))).slice(0,5);
  const [train,test] = seededSplit(encodedRows);
  const actual = test.map(r => Number(r[target]));
  const results = ranked.map((_, i) => {
    const selected = ranked.slice(0,i+1);
    const predicted = fitPredict(train,test,selected,target);
    return {k:i+1, features:selected, ...evaluate(actual,predicted)};
  });
  return {target, results};
}

function render(rows, analysis) {
  const best = analysis.results.reduce((a,b) => b.rmse < a.rmse ? b : a);
  els.bestK.textContent = best.k;
  els.bestRmse.textContent = best.rmse.toLocaleString(undefined,{maximumFractionDigits:2});
  els.bestR2.textContent = best.r2.toFixed(4);
  els.rowCount.textContent = rows.length;
  els.targetBadge.textContent = `Target: ${analysis.target}`;
  els.resultsBody.innerHTML = analysis.results.map(r => `<tr class="${r.k===best.k?'best':''}"><td>${r.k}</td><td class="features">[${r.features.join(", ")}]</td><td class="${r.k===best.k?'good':''}">${r.rmse.toFixed(6)}</td><td>${r.r2.toFixed(6)}</td></tr>`).join("");
  const labels = analysis.results.map(r => r.k);
  const options = title => ({responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{title:{display:true,text:"Number of Features"}},y:{title:{display:true,text:title},grid:{color:"#ebe7de"}}}});
  rmseChart?.destroy(); r2Chart?.destroy();
  rmseChart = new Chart(document.getElementById("rmseChart"),{type:"line",data:{labels,datasets:[{data:analysis.results.map(r=>r.rmse),borderColor:"#1f6f54",backgroundColor:"#1f6f54",pointRadius:5,tension:.15}]},options:options("RMSE")});
  r2Chart = new Chart(document.getElementById("r2Chart"),{type:"line",data:{labels,datasets:[{data:analysis.results.map(r=>r.r2),borderColor:"#d79a3b",backgroundColor:"#d79a3b",pointRadius:5,tension:.15}]},options:options("R-squared")});
}

async function run() {
  els.status.textContent = "分析中…";
  try {
    const response = await fetch(DATASET_URL);
    if (!response.ok) throw new Error(`Dataset 載入失敗 (${response.status})`);
    const text = await response.text();
    const rows = parseCsv(text);
    const output = analyze(rows, TARGET);
    render(rows, output);
    els.status.textContent = "分析完成";
    els.status.classList.remove("error");
  } catch (error) {
    els.status.textContent = error.message;
    els.status.classList.add("error");
  }
}

run();
