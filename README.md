
# 台灣房市先行指標儀表板（自動更新）

這個專案會**每週自動抓取**官方/開放資料，彙整成 `docs/data.json`，並用靜態網站（GitHub Pages）呈現。
不需 Google Sheet、不需手動貼數字。

## 內容與指標
- 建物買賣移轉棟數（全國＋縣市）
- 五大行庫新承作房貸年利率（CBC）
- 聯徵中心個人房屋貸款件數（JCIC）
- 全體金融機構建築貸款餘額（CBC）
- 營造工程物價指數（主計總處）
- CPI 房租年增率（主計總處）
- 建造執照／使用執照（國土管理署，開工/完工代理）
- 建材營造類股指數（月均，TWSE）
- 住宅價格指數（季，內政部/信義房屋）

> 預設覆蓋至少 10 年的每月資料（季資料會對齊季底月份）。

## 一鍵部署步驟（GitHub Pages）
1. 在 GitHub 建立一個新 Repository，名稱建議：`tw-housing-dashboard`。
2. 將此專案內容上傳（或直接上傳我提供的 ZIP 解壓後的檔案）。
3. 進入 **Settings → Pages**：
   - Source 選擇 `Deploy from a branch`
   - Branch 選擇 `main`，資料夾選 `/**docs**`，按 `Save`
   - GitHub 會發給你一個公開網址（如：`https://<你的帳號>.github.io/tw-housing-dashboard/`）
4. 進入 **Actions** 分頁，啟用 workflows。之後系統會：
   - 每週自動執行 `scripts/fetch_and_build.py`，更新 `docs/data.json`
   - 你的網站會自動載入最新數據

> 想手動立即更新：在 Actions 裡面選擇 `Update dashboard data (weekly)`，按 **Run workflow**。

## 本地開發
- 直接用 VS Code 的 Live Server 或任何靜態伺服器開 `docs/index.html` 即可。

## 常見問題
- 某些政府單位的 XLS/XLSX 檔格式會變動，已加入防呆。若抓取失敗，對應圖表會暫時無資料，待下次成功抓取會自動恢復。
- 若你想加入更多指標，只要在 `scripts/fetch_and_build.py` 裡新增抓取函式並輸出到 `docs/data.json`，前端就能讀取。

## 授權
- 原始碼 MIT；數據版權屬各資料提供單位。

