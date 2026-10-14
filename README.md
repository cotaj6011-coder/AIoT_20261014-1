# 50 Startups Multiple Linear Regression

## Demo

**網站：** [https://cotaj6011-coder.github.io/AIoT_20261014-1/](https://cotaj6011-coder.github.io/AIoT_20261014-1/)

![50 Startups 網站截圖](dic-regression-dashboard-preview.png)

## 專案說明

使用 50 Startups Dataset 建立多元線性迴歸模型，根據以下欄位預測公司利潤：

- R&D Spend
- Administration
- Marketing Spend
- State

`State` 使用 One-Hot Encoding，並刪除第一個類別作為基準，避免 dummy variable trap。

## CRISP-DM 流程

1. Business Understanding：預測新創公司的 Profit。
2. Data Understanding：檢查資料欄位、統計摘要及遺漏值。
3. Data Preparation：類別編碼並切分訓練集與測試集。
4. Modeling：使用 scikit-learn `LinearRegression`。
5. Evaluation：以 MAE、RMSE 與 R² 評估模型。
6. Deployment：透過 GitHub Pages 呈現分析圖表與特徵選擇結果。

## 執行方式

安裝套件：

```bash
pip install -r requirement.txt
```

執行模型：

```bash
python startup_regression.py
```

## 專案檔案

```text
50_Startups.csv
startup_regression.py
requirement.txt
dic-regression-dashboard/
└── dist/
    ├── index.html
    └── app.js
```
