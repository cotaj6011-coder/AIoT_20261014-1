# DIC 作業需求差距清單

## 核對範圍

- 本地成果：`startup_regression.py`、`50_Startups.csv`
- 參考成果：[AIoT-DIC2-MultipleLR](https://huanchen1107.github.io/AIoT-DIC2-MultipleLR/)
- 圖片：`a99f95a388c7fbb071767020ef99203d.png` 未出現在目前 workspace，尚無法核對圖片內容。

## 已完成

- Problem 1：50 Startups 多元線性迴歸。
- 使用 scikit-learn 建立資料前處理與 `LinearRegression` pipeline。
- 程式結構對應 CRISP-DM 六階段。
- 包含 train/test split、MAE、RMSE、R2、係數及範例預測。
- 已提供並成功讀取 `50_Startups.csv`。
- 已建立套件清單 `requirement.txt`。

## 尚缺必要內容

| 需求 | 目前狀態 | 尚缺內容 |
|---|---|---|
| 完成兩個 problems | 已完成 | Problem 1 為 50 Startups，Problem 2 為 Advertising Sales。 |
| Problem 2 直接套用 skill 或 workflow | 已完成 | 已以相同迴歸分析 workflow 完成 Advertising Sales。 |
| 兩題擇一上傳 | 未完成 | 需要在兩題完成後選定正式上傳題目，整理可部署成果。 |
| Result 網頁 | 已完成（本地） | 已建立互動式靜態網頁，公開部署尚未完成。 |

## 加分項目

| 配分 | 需求 | 目前狀態 | 尚缺內容 |
|---:|---|---|---|
| 1 分 | 下拉視窗呈現兩種 problem | 已完成 | 可切換 50 Startups 與 Advertising Sales。 |
| 2 分 | 可任意輸入 Dataset | 已完成 | 支援 Dataset URL 與本機 CSV 上傳，會重新計算並顯示結果。 |

Dataset URL 範例：

```text
https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv
```

## 與參考 Result 的差距

- 參考網站有互動式圖表、模型指標、係數表、殘差表與即時預測；本地成果只輸出到終端機。
- 參考網站的按鈕用於切換特徵選擇演算法，不是「兩種 problem」的下拉選單，因此仍不符合 1 分加分項目。
- 參考網站只連結固定的 50 Startups Dataset，沒有讓使用者輸入任意 Dataset URL 或上傳 CSV，因此仍不符合 2 分加分項目。
- 本地程式的欄位名稱固定為 `Profit`，部署版若要支援任意資料集，還需要讓使用者指定目標欄位與特徵欄位。

## 待完成清單

- [x] 建立 Problem 2，套用指定 skill 或 workflow。
- [ ] 決定兩題中要上傳的主題。
- [x] 建立可部署的網頁介面。
- [x] 加入兩種 problem 的下拉選單。
- [x] 加入 Dataset URL／CSV 上傳功能。
- [x] 自動辨識數值目標欄位與特徵欄位。
- [x] 在網頁顯示模型指標與預測結果。
- [ ] 部署並驗證公開 Result URL。
- [ ] 補上 `a99f95a388c7fbb071767020ef99203d.png` 後再次核對。
