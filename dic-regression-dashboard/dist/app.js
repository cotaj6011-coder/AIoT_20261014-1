const DATASET_URL = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv";
const TARGET = "Profit";
const els = Object.fromEntries(["status","bestK","bestRmse","bestR2","rowCount","targetBadge","resultsBody"].map(id => [id, document.getElementById(id)]));
let rmseChart;
let r2Chart;
let modelRmseChart;
let modelR2Chart;
let actualPredictedChart;
let residualChart;
let dashboardData;
let activeDashboardModel;

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

function renderExperimentResults(comparison, importance) {
  const best = comparison.reduce((a, b) => b.cvRmse < a.cvRmse ? b : a);
  const baseline = comparison.find(row => row.method === "Baseline");
  const bestTest = comparison.reduce((a, b) => b.testRmse < a.testRmse ? b : a);
  const improvement = ((baseline.cvRmse - best.cvRmse) / baseline.cvRmse) * 100;

  document.getElementById("bestModelBadge").textContent = `Best CV RMSE · ${best.method}`;
  document.getElementById("modelComparisonBody").innerHTML = comparison.map(row => `
    <tr class="${row.method === best.method ? "best" : ""}">
      <td><strong>${row.method}</strong></td>
      <td class="features">${row.features}</td>
      <td>${row.count}</td>
      <td class="${row.method === best.method ? "good" : ""}">${row.cvRmse.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</td>
      <td>${row.testRmse.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</td>
      <td>${row.testR2.toFixed(4)}</td>
    </tr>`).join("");

  const chartOptions = title => ({responsive:true,maintainAspectRatio:false,plugins:{legend:{position:"top"}},scales:{x:{grid:{display:false}},y:{title:{display:true,text:title},grid:{color:"#ebe7de"}}}});
  const drawComparisonCharts = rows => {
    const labels = rows.map(row => row.method);
    const bestColors = rows.map(row => row.method === best.method ? "#1f6f54" : "#9fb8ad");
    modelRmseChart?.destroy(); modelR2Chart?.destroy();
    modelRmseChart = new Chart(document.getElementById("modelRmseChart"), {
      type:"bar", data:{labels,datasets:[
        {label:"CV RMSE",data:rows.map(row=>row.cvRmse),backgroundColor:bestColors},
        {label:"Test RMSE",data:rows.map(row=>row.testRmse),backgroundColor:"#d79a3b"}
      ]}, options:chartOptions("RMSE")
    });
    modelR2Chart = new Chart(document.getElementById("modelR2Chart"), {
      type:"bar", data:{labels,datasets:[{label:"Test R²",data:rows.map(row=>row.testR2),backgroundColor:bestColors}]},
      options:{...chartOptions("R-squared"),scales:{x:{grid:{display:false}},y:{min:0,max:1,title:{display:true,text:"R-squared"},grid:{color:"#ebe7de"}}}}
    });
  };
  drawComparisonCharts(comparison);
  const filters = document.getElementById("modelFilters");
  filters.innerHTML = comparison.map(row => `<label class="filter-chip"><input type="checkbox" value="${row.method}" checked> ${row.method}</label>`).join("");
  filters.addEventListener("change", () => {
    const selected = [...filters.querySelectorAll("input:checked")].map(input => input.value);
    drawComparisonCharts(comparison.filter(row => selected.includes(row.method)));
  });

  document.getElementById("analysisConclusion").textContent =
    `${best.method} 的 CV RMSE 最低（${best.cvRmse.toLocaleString(undefined,{maximumFractionDigits:2})}），選出 ${best.features}，相較 Baseline 改善 ${improvement.toFixed(2)}%。` +
    ` ${bestTest.method} 的 Test RMSE 最低（${bestTest.testRmse.toLocaleString(undefined,{maximumFractionDigits:2})}），Test R² 為 ${bestTest.testR2.toFixed(4)}。` +
    ` 五種選擇方法皆排除 State，顯示本次資料切分下地區資訊未帶來額外預測效益。`;

  const grouped = importance.reduce((groups, row) => {
    (groups[row.method] ||= []).push(row);
    return groups;
  }, {});
  document.getElementById("importanceGrid").innerHTML = comparison.map(model => {
    const rows = grouped[model.method] || [];
    return `<article class="panel method-card"><h3>${model.method}</h3><div class="method-meta">${model.count} feature groups</div>${rows.map(row => `
      <div class="importance-row"><span>${row.feature}</span><div class="importance-track"><div class="importance-fill" style="width:${(row.importance*100).toFixed(2)}%"></div></div><span class="importance-value">${(row.importance*100).toFixed(1)}%</span></div>`).join("")}</article>`;
  }).join("");
}

function formatMoney(value) {
  return Number(value).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
}

function predictWithModel(model, values) {
  return model.intercept + model.coefficients.reduce((total, item) => {
    let value = Number(values[item.feature] ?? 0);
    if (item.feature.startsWith("State_")) value = values.State === item.feature.replace("State_", "") ? 1 : 0;
    return total + item.coefficient * value;
  }, 0);
}

function updateLivePrediction() {
  if (!activeDashboardModel) return;
  const values = {
    "R&D Spend": Number(document.getElementById("predictRD").value),
    "Administration": Number(document.getElementById("predictAdmin").value),
    "Marketing Spend": Number(document.getElementById("predictMarketing").value),
    State: document.getElementById("predictState").value
  };
  document.getElementById("livePrediction").textContent = `$${formatMoney(predictWithModel(activeDashboardModel, values))}`;
}

function zoomOptions() {
  return {pan:{enabled:true,mode:"xy"},zoom:{wheel:{enabled:true},pinch:{enabled:true},mode:"xy"}};
}

function renderActiveModel(model) {
  activeDashboardModel = model;
  document.getElementById("activeMethod").textContent = model.method;
  document.getElementById("activeCvRmse").textContent = formatMoney(model.cvRmse);
  document.getElementById("activeTestRmse").textContent = formatMoney(model.testRmse);
  document.getElementById("activeTestR2").textContent = model.testR2.toFixed(4);
  const terms = model.coefficients.map(item => `${item.coefficient >= 0 ? "+" : "−"} ${Math.abs(item.coefficient).toFixed(4)} × ${item.feature}`);
  document.getElementById("regressionFormula").textContent = `Profit = ${model.intercept.toFixed(2)} ${terms.join(" ")}`;
  document.getElementById("coefficientBody").innerHTML = [
    `<tr><td>Intercept</td><td>${model.intercept.toFixed(4)}</td></tr>`,
    ...model.coefficients.map(item => `<tr><td>${item.feature}</td><td>${item.coefficient.toFixed(4)}</td></tr>`)
  ].join("");
  document.getElementById("assumptionGrid").innerHTML = model.assumptions.map(item => `<article class="assumption ${item.passed ? "pass" : "fail"}"><strong>${item.name} · ${item.passed ? "通過" : "注意"}</strong><small>${item.display}</small></article>`).join("");

  const points = model.testRows.map(row => ({x:row.actual,y:row.predicted,index:row.index}));
  const residualPoints = model.testRows.map(row => ({x:row.predicted,y:row.residual,index:row.index}));
  const values = points.flatMap(point => [point.x,point.y]);
  const min = Math.min(...values), max = Math.max(...values);
  actualPredictedChart?.destroy(); residualChart?.destroy();
  actualPredictedChart = new Chart(document.getElementById("actualPredictedChart"), {
    type:"scatter", data:{datasets:[
      {label:"Test samples",data:points,backgroundColor:"#1f6f54",pointRadius:6},
      {label:"Perfect prediction",data:[{x:min,y:min},{x:max,y:max}],type:"line",borderColor:"#d79a3b",borderDash:[6,5],pointRadius:0}
    ]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{zoom:zoomOptions(),tooltip:{callbacks:{label:ctx=>ctx.raw.index === undefined ? "Perfect prediction" : `Startup #${ctx.raw.index}: actual ${formatMoney(ctx.raw.x)}, predicted ${formatMoney(ctx.raw.y)}`}}},scales:{x:{title:{display:true,text:"Actual Profit"}},y:{title:{display:true,text:"Predicted Profit"}}}}
  });
  residualChart = new Chart(document.getElementById("residualChart"), {
    type:"scatter", data:{datasets:[
      {label:"Residual",data:residualPoints,backgroundColor:"#a84f40",pointRadius:6},
      {label:"Zero line",data:[{x:Math.min(...residualPoints.map(p=>p.x)),y:0},{x:Math.max(...residualPoints.map(p=>p.x)),y:0}],type:"line",borderColor:"#64716b",pointRadius:0}
    ]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{zoom:zoomOptions(),tooltip:{callbacks:{label:ctx=>ctx.raw.index === undefined ? "Residual = 0" : `Startup #${ctx.raw.index}: residual ${formatMoney(ctx.raw.y)}`}}},scales:{x:{title:{display:true,text:"Predicted Profit"}},y:{title:{display:true,text:"Residual"}}}}
  });
  document.querySelectorAll("#featureChecks input").forEach(input => { input.checked = model.selectedFeatures.includes(input.value); });
  updateLivePrediction();
}

function designRow(row, features, fitIntercept) {
  const values = fitIntercept ? [1] : [];
  const names = fitIntercept ? ["Intercept"] : [];
  features.forEach(feature => {
    if (feature === "State") {
      ["Florida","New York"].forEach(state => { values.push(row.State === state ? 1 : 0); names.push(`State_${state}`); });
    } else { values.push(Number(row[feature])); names.push(feature); }
  });
  return {values,names};
}

function fitMatrix(rows, features, fitIntercept) {
  const built = rows.map(row => designRow(row, features, fitIntercept));
  const X = built.map(item => item.values), y = rows.map(row => Number(row.Profit));
  const p = X[0].length;
  const xtx = Array.from({length:p},(_,i)=>Array.from({length:p},(_,j)=>X.reduce((sum,row)=>sum+row[i]*row[j],0)));
  const xty = Array.from({length:p},(_,i)=>X.reduce((sum,row,k)=>sum+row[i]*y[k],0));
  return {beta:solve(xtx,xty),names:built[0].names,X};
}

function predictMatrix(rows, model, features, fitIntercept) {
  return rows.map(row => designRow(row,features,fitIntercept).values.reduce((sum,value,i)=>sum+value*model.beta[i],0));
}

function corrArrays(a,b) {
  const ma=a.reduce((s,v)=>s+v,0)/a.length, mb=b.reduce((s,v)=>s+v,0)/b.length;
  const numerator=a.reduce((s,v,i)=>s+(v-ma)*(b[i]-mb),0);
  const denominator=Math.sqrt(a.reduce((s,v)=>s+(v-ma)**2,0)*b.reduce((s,v)=>s+(v-mb)**2,0));
  return denominator ? numerator/denominator : 0;
}

function customAssumptions(predicted,residuals,trainMatrix) {
  const dw=residuals.slice(1).reduce((s,v,i)=>s+(v-residuals[i])**2,0)/residuals.reduce((s,v)=>s+v*v,0);
  const mean=residuals.reduce((s,v)=>s+v,0)/residuals.length;
  const variance=residuals.reduce((s,v)=>s+(v-mean)**2,0)/residuals.length;
  const skew=residuals.reduce((s,v)=>s+(v-mean)**3,0)/residuals.length/Math.pow(variance,1.5);
  const kurt=residuals.reduce((s,v)=>s+(v-mean)**4,0)/residuals.length/(variance**2);
  const jb=residuals.length/6*(skew**2+(kurt-3)**2/4), normalP=Math.exp(-jb/2);
  const linear=Math.abs(corrArrays(predicted,residuals)), scale=Math.abs(corrArrays(predicted,residuals.map(Math.abs)));
  return [
    {name:"線性關係",display:`|corr(ŷ, residual)| = ${linear.toFixed(3)}`,passed:linear<.2},
    {name:"誤差獨立性",display:`Durbin-Watson = ${dw.toFixed(3)}`,passed:dw>=1.5&&dw<=2.5},
    {name:"同質變異性",display:`|corr(ŷ, |residual|)| = ${scale.toFixed(3)}`,passed:scale<.3},
    {name:"殘差常態性",display:`Jarque-Bera p = ${normalP.toFixed(3)}`,passed:normalP>.05},
    {name:"無嚴重共線性",display:`Design rank = ${trainMatrix[0].length}`,passed:true}
  ];
}

function trainCustomModel() {
  const status=document.getElementById("trainingStatus");
  try {
    status.textContent="Training Set 重新訓練中…";
    const features=[...document.querySelectorAll("#featureChecks input:checked")].map(input=>input.value);
    if(!features.length) throw new Error("請至少選擇一個特徵。");
    const fitIntercept=document.getElementById("fitIntercept").checked;
    const folds=Number(document.getElementById("cvFolds").value);
    const trainRows=dashboardData.metadata.trainIndices.map(index=>dashboardData.dataset[index]);
    const testRows=dashboardData.metadata.testIndices.map(index=>dashboardData.dataset[index]);
    const foldPredictions=[];
    for(let fold=0;fold<folds;fold++){
      const validation=trainRows.filter((_,i)=>i%folds===fold), training=trainRows.filter((_,i)=>i%folds!==fold);
      const foldModel=fitMatrix(training,features,fitIntercept);
      const predicted=predictMatrix(validation,foldModel,features,fitIntercept);
      validation.forEach((row,i)=>foldPredictions.push({actual:Number(row.Profit),predicted:predicted[i]}));
    }
    const cvRmse=Math.sqrt(foldPredictions.reduce((s,row)=>s+(row.actual-row.predicted)**2,0)/foldPredictions.length);
    const fitted=fitMatrix(trainRows,features,fitIntercept), predicted=predictMatrix(testRows,fitted,features,fitIntercept);
    const actual=testRows.map(row=>Number(row.Profit)), metrics=evaluate(actual,predicted), residuals=actual.map((v,i)=>v-predicted[i]);
    const interceptIndex=fitted.names[0]==="Intercept"?0:-1;
    const custom={method:"Custom",selectedFeatures:features,cvRmse,testRmse:metrics.rmse,testR2:metrics.r2,intercept:interceptIndex===0?fitted.beta[0]:0,coefficients:fitted.names.map((name,i)=>({feature:name,coefficient:fitted.beta[i]})).filter(item=>item.feature!=="Intercept"),testRows:testRows.map((row,i)=>({index:dashboardData.metadata.testIndices[i],actual:actual[i],predicted:predicted[i],residual:residuals[i]})),assumptions:customAssumptions(predicted,residuals,fitted.X)};
    dashboardData.models=dashboardData.models.filter(model=>model.method!=="Custom"); dashboardData.models.push(custom);
    const selector=document.getElementById("activeModel");
    if(![...selector.options].some(option=>option.value==="Custom")) selector.add(new Option("Custom · 自訂模型","Custom"));
    selector.value="Custom"; renderActiveModel(custom); status.textContent=`完成：僅使用 Training Set 進行 ${folds}-Fold CV，Test Set 最後評估一次。`;
  } catch(error){status.textContent=error.message; status.classList.add("error");}
}

function initializeInteractivePlatform(details,dataset) {
  dashboardData={...details,dataset};
  const selector=document.getElementById("activeModel");
  selector.innerHTML=details.models.map(model=>`<option value="${model.method}">${model.method}</option>`).join("");
  const features=["R&D Spend","Administration","Marketing Spend","State"];
  document.getElementById("featureChecks").innerHTML=features.map(feature=>`<label class="check-chip"><input type="checkbox" value="${feature}" checked> ${feature}</label>`).join("");
  selector.addEventListener("change",()=>renderActiveModel(dashboardData.models.find(model=>model.method===selector.value)));
  document.getElementById("trainCustom").addEventListener("click",trainCustomModel);
  ["predictRD","predictAdmin","predictMarketing","predictState"].forEach(id=>document.getElementById(id).addEventListener("input",updateLivePrediction));
  document.getElementById("resetZoom").addEventListener("click",()=>{actualPredictedChart?.resetZoom();residualChart?.resetZoom();});
  renderActiveModel(details.models[0]);
}

async function loadExperimentResults() {
  const [comparisonResponse, importanceResponse, detailsResponse, datasetResponse] = await Promise.all([
    fetch("data/model_comparison.csv"),
    fetch("data/feature_importance.csv"),
    fetch("data/model_details.json"),
    fetch(DATASET_URL)
  ]);
  if (!comparisonResponse.ok || !importanceResponse.ok || !detailsResponse.ok || !datasetResponse.ok) throw new Error("實驗結果載入失敗");
  const comparison = parseCsv(await comparisonResponse.text()).map(row => ({
    method: row.Method,
    features: row["Selected Features"],
    count: Number(row["Feature Count"]),
    cvRmse: Number(row["CV RMSE"]),
    testRmse: Number(row["Test RMSE"]),
    testR2: Number(row["Test R2"])
  }));
  const importance = parseCsv(await importanceResponse.text()).map(row => ({
    method: row.Method,
    feature: row.Feature,
    importance: Number(row.Importance)
  }));
  renderExperimentResults(comparison, importance);
  initializeInteractivePlatform(await detailsResponse.json(), parseCsv(await datasetResponse.text()));
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
loadExperimentResults().catch(error => {
  document.getElementById("analysisConclusion").textContent = error.message;
});
