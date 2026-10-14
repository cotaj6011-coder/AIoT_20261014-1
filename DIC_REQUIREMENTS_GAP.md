# DIC 作業內容

## 題目

50 Startups 多元線性迴歸：依據 R&D Spend、Administration、Marketing Spend 與 State 預測 Profit。

## CRISP-DM

1. Business Understanding：預測新創公司的利潤。
2. Data Understanding：檢查 50 筆資料、欄位及遺漏值。
3. Data Preparation：State 採 one-hot encoding，資料分為訓練集與測試集。
4. Modeling：使用 scikit-learn `LinearRegression`。
5. Evaluation：比較 RMSE、R2 與不同特徵數量的模型結果。
6. Deployment：以網頁呈現圖表、指標與特徵選擇結果。
