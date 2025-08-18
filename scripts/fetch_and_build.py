
# -*- coding: utf-8 -*-

import os, io, re, sys, json, time, math, datetime
from dataclasses import dataclass
from typing import List, Dict, Any
import requests
import pandas as pd
from bs4 import BeautifulSoup

SESSION = requests.Session()
SESSION.headers.update({"User-Agent":"Mozilla/5.0 (compatible; TW-HousingBot/1.0)"})

ROOT = os.path.dirname(os.path.dirname(__file__))
DOCS = os.path.join(ROOT, "docs")
OUTFILE = os.path.join(DOCS, "data.json")

def log(*args): print("[ETL]", *args, file=sys.stderr)

def to_month_str(dt: datetime.date) -> str:
    return f"{dt.year}-{dt.month:02d}"

def ensure_10y_months(series):
    # Keep last >= 10 years if too long
    if not series: return series
    last = series[-1]['date']
    y, m = map(int, last.split('-'))
    cutoff = datetime.date(y-10, m, 1)
    return [d for d in series if datetime.date(*map(int, d['date'].split('-')), 1) >= cutoff]

def parse_csv_text(txt):
    from io import StringIO
    return pd.read_csv(StringIO(txt))

# -------------------- CBC: Construction loans balance (monthly CSV) --------------------
def fetch_cbc_construction_loans():
    # CSV documented on CBC site (Financial Statistics)
    url = "https://www.cbc.gov.tw/public/data/EBOOKXLS/107_EF99_A4L.csv"
    log("CBC construction loans:", url)
    r = SESSION.get(url, timeout=60); r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.content.decode('utf-8-sig', errors='ignore')))
    # Heuristic: find a column that contains 'LOANS FOR CONSTRUCTION' or Chinese title
    col_candidates = [c for c in df.columns if 'LOANS FOR CONSTRUCTION' in c.upper() or '建築貸款' in c]
    if not col_candidates:
        # sometimes the data places 'Value' in a column; try last column
        col = df.columns[-1]
    else:
        col = col_candidates[0]
    # Ensure date column
    date_col = [c for c in df.columns if 'DATE' in c.upper() or '年月' in c]
    if not date_col:
        # Some CBC files use Year, Month
        if 'Year' in df.columns and 'Month' in df.columns:
            df['date'] = df.apply(lambda x: f"{int(x['Year']):04d}-{int(x['Month']):02d}", axis=1)
        else:
            # Try first column
            first = df.columns[0]
            # Normalize to YYYY-MM
            def norm(x):
                s = str(x)
                m = re.search(r'(\d{4})[^\d]?(\d{1,2})', s)
                return f"{m.group(1)}-{int(m.group(2)):02d}" if m else None
            df['date'] = df[first].map(norm)
    else:
        d = date_col[0]
        df['date'] = df[d].astype(str).str.replace('/', '-').str.replace('.', '-')
        df['date'] = df['date'].str.replace(r'^(\d{4})(\d{2})$', lambda m: f"{m.group(1)}-{m.group(2)}", regex=True)
    df = df[['date', col]].dropna()
    df = df[df['date'].str.match(r'^\d{4}-\d{2}$')]
    df[col] = pd.to_numeric(df[col], errors='coerce')
    out = [{'date': d, 'value': float(v)} for d, v in df.values if pd.notnull(v)]
    return ensure_10y_months(out)

# -------------------- CBC: Five major banks new mortgage rate (via NCHC mirror CSV) --------------------
def fetch_cbc_five_bank_new_mortgage_rate():
    # NCHC SCIDM mirror provides stable CSV for five-bank new loan interest rate (monthly)
    # If this link changes, replace with official CBC '金融統計月報 D.五大銀行新承作放款' CSV.
    resources = [
        "https://scidm.nchc.org.tw/dataset/best_wish6551/resource/7e55f9bc-0a7d-4b89-8b53-3d24bf67806e/download"  # _revise CSV (monthly)
    ]
    for url in resources:
        try:
            log("CBC five-bank rate:", url)
            r = SESSION.get(url, timeout=60); r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.content.decode('utf-8', errors='ignore')))
            # Heuristic columns
            date_col = [c for c in df.columns if 'DATE' in c.upper() or '年月' in c or 'time' in c.lower()]
            rate_col = [c for c in df.columns if ('利率' in c) or ('interest' in c.lower())]
            if not date_col or not rate_col:
                continue
            dcol = date_col[0]; rcol = rate_col[0]
            df[dcol] = df[dcol].astype(str).str.replace('/', '-')
            df = df[[dcol, rcol]].dropna()
            df = df[df[dcol].str.match(r'^\d{4}-\d{2}$')]
            df[rcol] = pd.to_numeric(df[rcol], errors='coerce')
            out = [{'date': d, 'value': float(v)} for d, v in df.values if pd.notnull(v)]
            return ensure_10y_months(out)
        except Exception as e:
            log("fallback failed:", e)
    return []

# -------------------- JCIC: Personal housing loan trends (CSV from opendata page) --------------------
def fetch_jcic_housing_loans_counts():
    # Crawl page and get CSV link
    page = "https://www.jcic.org.tw/jcweb/od/Opendatapage1mech_s.aspx?key=Ti5nZ+qLJbnKV27D2ZlvwA=="
    log("JCIC page:", page)
    html = SESSION.get(page, timeout=60).text
    soup = BeautifulSoup(html, 'html.parser')
    csv_link = None
    for a in soup.select('a'):
        if a.text.strip().startswith('1-2'):
            csv_link = a.get('href')
            break
    if not csv_link:
        # generic: find CSV anchor
        for a in soup.find_all('a', href=True):
            if 'fileRename.aspx' in a['href'] and 'CSV' in a.text:
                csv_link = a['href']; break
    if not csv_link:
        return []
    if csv_link.startswith('/'):
        csv_link = 'https://www.jcic.org.tw' + csv_link
    log("JCIC csv:", csv_link)
    r = SESSION.get(csv_link, timeout=60); r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.content.decode('utf-8-sig', errors='ignore')))
    # Heuristics: find date & count columns
    dcol = [c for c in df.columns if '年月' in c or 'DATE' in c.upper()]
    cnt_col = [c for c in df.columns if ('件數' in c) or ('戶數' in c) or ('counts' in c.lower())]
    if not dcol or not cnt_col:
        return []
    dcol = dcol[0]; cnt_col = cnt_col[0]
    df[dcol] = df[dcol].astype(str).str.replace('/', '-')
    df = df[[dcol, cnt_col]].dropna()
    df = df[df[dcol].str.match(r'^\d{4}-\d{2}$')]
    df[cnt_col] = pd.to_numeric(df[cnt_col], errors='coerce')
    out = [{'date': d, 'value': float(v)} for d, v in df.values if pd.notnull(v)]
    return ensure_10y_months(out)

# -------------------- DGBAS: Construction Cost Index (CCI) + Rent CPI YoY --------------------
def fetch_dgbas_cci():
    # Use JSON export from DGBAS Stats DB (funid=A030502015). We query "總指數-營造工程物價指數".
    # The API link can be copied from the website; this default should work for CSV/JSON.
    # If schema changes, update query parameters accordingly.
    base = "https://nstatdb.dgbas.gov.tw/dgbasAll/webMain.aspx"
    params = {
        "funid":"A030502015",
        "sys":"210",
        "outmode":"9"  # JSON
    }
    log("DGBAS CCI JSON:", base, params)
    r = SESSION.get(base, params=params, timeout=60); r.raise_for_status()
    try:
        js = r.json()
    except Exception:
        # fallback CSV
        params['outmode'] = '8'
        r = SESSION.get(base, params=params, timeout=60); r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.content.decode('utf-8', errors='ignore')))
        # try to find date, value columns
        dcol = [c for c in df.columns if '統計期' in c or '日期' in c or 'time' in c.lower()]
        vcol = [c for c in df.columns if '營造工程物價指數' in c or '指數' in c or 'value' in c.lower()]
        if not dcol or not vcol:
            return []
        dcol = dcol[0]; vcol = vcol[0]
        df[dcol] = df[dcol].astype(str).str.replace('/', '-')
        df = df[[dcol, vcol]].dropna()
        df[vcol] = pd.to_numeric(df[vcol], errors='coerce')
        out = []
        for d,v in df.values:
            # Normalize YYYY/MM to YYYY-MM
            m = re.search(r'(\d{4})[\-/年](\d{1,2})', str(d))
            if not m: continue
            out.append({"date": f"{int(m.group(1)):04d}-{int(m.group(2)):02d}", "value": float(v)})
        out.sort(key=lambda x: x['date'])
        return ensure_10y_months(out)
    # JSON structure may include fields with time and value
    # Attempt to autodetect date & value fields
    rows = js.get('DATA', []) if isinstance(js, dict) else js
    out = []
    for row in rows:
        # try common keys
        d = row.get('統計期') or row.get('日期') or row.get('TIME') or row.get('time') or row.get('日期-月')
        v = row.get('營造工程物價指數') or row.get('指數') or row.get('VALUE') or row.get('value')
        if d is None or v is None: 
            # try first two values
            vals = list(row.values())
            if len(vals) >= 2:
                d, v = vals[0], vals[1]
        try:
            m = re.search(r'(\d{4})[\-/年](\d{1,2})', str(d))
            if not m: continue
            date = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"
            v = float(str(v).replace(',', ''))
            out.append({"date": date, "value": v})
        except Exception:
            continue
    out.sort(key=lambda x: x['date'])
    return ensure_10y_months(out)

def fetch_rent_cpi_yoy():
    # Use DGBAS 'cpisplrent.xls' which contains an '年增率' sheet.
    url = "https://www.stat.gov.tw/public/data/dgbas03/bs3/inquire/cpisplrent.xls"
    log("DGBAS rent CPI:", url)
    r = SESSION.get(url, timeout=60); r.raise_for_status()
    # pandas can read xls via xlrd
    xls = pd.ExcelFile(io.BytesIO(r.content))
    # Try a sheet with '年增率'
    sheet_name = None
    for s in xls.sheet_names:
        if '年增率' in s or 'YoY' in s or '年增' in s:
            sheet_name = s; break
    if sheet_name is None:
        sheet_name = xls.sheet_names[0]
    df = xls.parse(sheet_name)
    # heuristics for date, value
    # Flatten wide format if needed
    if '年' in df.columns and '月' in df.columns:
        df['date'] = df.apply(lambda x: f"{int(x['年']):04d}-{int(x['月']):02d}", axis=1)
        vcol = [c for c in df.columns if '年增率' in c]
        vcol = vcol[0] if vcol else df.columns[-1]
        series = df[['date', vcol]].dropna()
        series[vcol] = pd.to_numeric(series[vcol], errors='coerce')
        out = [{'date': d, 'value': float(v)} for d, v in series.values if pd.notnull(v)]
        return ensure_10y_months(out)
    # Try another pattern: first column is date, next is YoY
    df.columns = [str(c).strip() for c in df.columns]
    dcol = df.columns[0]; vcol = df.columns[1]
    df[dcol] = df[dcol].astype(str)
    # Normalize 'YYYY/MM'
    def norm(s):
        m = re.search(r'(\d{4})[\-/年](\d{1,2})', s)
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}" if m else None
    df['date'] = df[dcol].map(norm)
    df[vcol] = pd.to_numeric(df[vcol], errors='coerce')
    out = [{'date': d, 'value': float(v)} for d, v in df[['date', vcol]].dropna().values]
    return ensure_10y_months(out)

# -------------------- NLMA/MOI: Building permits (建造執照) & usage licenses (使用執照) --------------------
def latest_xls_links_from_nlma():
    # Crawl known monthly table files and return urls
    seeds = [
        "https://www.nlma.gov.tw/filesys/file/chinese/statistic7/2355-00-01-2.xls",  # 建造執照-按用途
        "https://www.nlma.gov.tw/uploads/files/c1656f74b252811900bc526cacb37fbf.xls", # 使用執照-按層數
    ]
    return seeds

def read_monthly_from_nlma(url):
    log("NLMA xls:", url)
    r = SESSION.get(url, timeout=60); r.raise_for_status()
    try:
        xls = pd.ExcelFile(io.BytesIO(r.content))
    except Exception:
        return []
    # Heuristic: find a sheet with monthly columns or date column
    # Many NLMA tables list counts by month/year in rows.
    for sheet in xls.sheet_names:
        df = xls.parse(sheet, header=0)
        # Try to locate year/month columns
        cols = [str(c) for c in df.columns]
        if any('年月' in c or '月份' in c for c in cols):
            dcol = next(c for c in cols if ('年月' in c or '月份' in c))
            # Find total/件數 column
            valcol = None
            for c in cols[::-1]:
                if any(k in c for k in ['件數','棟數','合計','總計']):
                    valcol = c; break
            if valcol is None: 
                # fallback to last column
                valcol = cols[-1]
            tmp = df[[dcol, valcol]].copy()
            tmp.columns = ['date_raw','value']
            # Normalize date
            def to_date(s):
                s = str(s)
                # accept 民國年 e.g., 110/05
                m = re.search(r'(\d{2,3})[\-/年](\d{1,2})', s)
                if m and int(m.group(1)) < 200:  # ROC year
                    y = int(m.group(1)) + 1911
                    return f"{y:04d}-{int(m.group(2)):02d}"
                m = re.search(r'(\d{4})[\-/年](\d{1,2})', s)
                if m:
                    return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"
                return None
            tmp['date'] = tmp['date_raw'].map(to_date)
            tmp['value'] = pd.to_numeric(tmp['value'], errors='coerce')
            out = [{'date': d, 'value': float(v)} for d, v in tmp[['date','value']].dropna().values]
            out.sort(key=lambda x: x['date'])
            # return last 10y
            return ensure_10y_months(out)
    return []

def fetch_permits_and_usage():
    urls = latest_xls_links_from_nlma()
    permits = read_monthly_from_nlma(urls[0]) if urls else []
    usage = read_monthly_from_nlma(urls[1]) if urls else []
    return permits, usage

# -------------------- MOI: Transactions by city & national total (monthly) --------------------
def fetch_transactions_moi():
    # Use '內政統計月報 4.5-辦理建物所有權登記' XLS which includes monthly counts; we derive '買賣登記'總數.
    # The page lists ODF/XLS links; we fetch the XLS then transform.
    page = "https://ws.moi.gov.tw/001/Upload/OldFile/site_stuff/321/1/month/month.html"
    log("MOI month page:", page)
    html = SESSION.get(page, timeout=60).text
    soup = BeautifulSoup(html, 'html.parser')
    link = None
    for a in soup.find_all('a', href=True):
        if '4.5-' in a.text and ('XLS' in a.text or a.get('href','').lower().endswith('.xls')):
            link = a.get('href'); break
    if not link:
        return [], {'ALL': []}
    if link.startswith('http') is False:
        if link.startswith('/'):
            link = "https://ws.moi.gov.tw" + link
        else:
            link = "https://ws.moi.gov.tw/001/Upload/OldFile/site_stuff/321/1/month/" + link
    log("MOI 4.5 xls:", link)
    r = SESSION.get(link, timeout=60); r.raise_for_status()
    xls = pd.ExcelFile(io.BytesIO(r.content))
    # The workbook often contains multiple tables; try to locate a sheet with city rows and months columns.
    national = []
    cities = {}
    for s in xls.sheet_names:
        df = xls.parse(s, header=0)
        # try to find column names indicating county/city and '買賣'
        # normalize headers
        df.columns = [str(c).strip() for c in df.columns]
        if not any(('縣市' in c or '市名' in c or '縣' in c) for c in df.columns):
            continue
        # melt monthly columns if they look like 'YYYY-MM' or '民國年月'
        month_cols = [c for c in df.columns if re.search(r'(\d{2,3}|\d{4})[\-/年]?\d{1,2}', c)]
        id_col = next((c for c in df.columns if '縣' in c or '市' in c or '行政區' in c), None)
        if not id_col or not month_cols:
            continue
        mdf = df[[id_col]+month_cols].copy()
        mdf = mdf.melt(id_vars=[id_col], var_name='date_raw', value_name='value')
        # keep '買賣' rows only if present; otherwise assume total registrations
        # some sheets separate '買賣','贈與','繼承' in different tables, so we try to detect '買賣' keyword near sheet name
        if ('買賣' not in s) and ('買賣' not in ''.join(df.columns)):
            # accept as total ownership registrations; still a proxy of volume trend
            pass
        # normalize date
        def norm_date(s):
            s = str(s)
            m = re.search(r'(\d{4})[\-/年](\d{1,2})', s)
            if m: return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"
            m = re.search(r'(\d{2,3})[\-/年](\d{1,2})', s)  # ROC
            if m: return f"{int(m.group(1))+1911:04d}-{int(m.group(2)):02d}"
            return None
        mdf['date'] = mdf['date_raw'].map(norm_date)
        mdf['value'] = pd.to_numeric(mdf['value'], errors='coerce')
        mdf = mdf.dropna(subset=['date'])
        # build city dict
        for city, sub in mdf.groupby(id_col):
            seq = [{'date':d, 'value': float(v)} for d,v in sub[['date','value']].dropna().values]
            seq.sort(key=lambda x: x['date'])
            if city not in cities: cities[city] = []
            # merge (if multiple sheets)
            acc = {x['date']: x['value'] for x in cities[city]}
            for x in seq: acc[x['date']] = x['value']
            cities[city] = [{'date':k, 'value':acc[k]} for k in sorted(acc.keys())]
        # national total (sum by date)
    # Sum national
    all_dates = sorted({d for c in cities.values() for d in [x['date'] for x in c]})
    for d in all_dates:
        total = sum(next((x['value'] for x in cities[city] if x['date']==d), 0) for city in cities)
        national.append({'date': d, 'value': float(total)})
    # Keep last 10 years
    national = ensure_10y_months(national)
    cities = {k: ensure_10y_months(v) for k,v in cities.items()}
    return national, cities

# -------------------- TWSE: Construction sector index (category index) --------------------
def fetch_twse_construction_sector():
    # Use MI_INDEX CSV (type=ALL) which includes daily market & category indices;
    # We'll crawl last ~5 years daily then compress to monthly average.
    # TWSE returns per-day; we sample the last 6 years by iterating months.
    out_daily = []
    today = datetime.date.today()
    start = today.replace(year=today.year-6, day=1)
    cur = start
    while cur <= today:
        date_str = cur.strftime("%Y%m%d")
        url = f"https://www.twse.com.tw/exchangeReport/MI_INDEX?response=csv&date={date_str}&type=ALL"
        try:
            r = SESSION.get(url, timeout=60); 
            if r.status_code != 200: 
                cur = (cur.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
                continue
            text = r.content.decode('big5','ignore')
            # The CSV contains multiple tables; find the '類股指數' table and pick 建材營造類
            lines = [ln.strip() for ln in text.splitlines() if ln and ln[0].isdigit()==False]
            # Fallback: parse with pandas, filter rows containing 建材營造
            df = pd.read_csv(io.StringIO(text))
        except Exception:
            # more robust parsing
            try:
                df = pd.read_csv(io.StringIO(text), header=None)
            except Exception:
                cur = (cur.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
                continue
        # Try to locate the section that has label '建材營造類'
        try:
            # heuristic: flatten all rows and search keyword
            flat = df.astype(str)
            mask = flat.apply(lambda row: row.str.contains('建材營造', na=False)).any(axis=1)
            rows = df[mask]
            if not rows.empty:
                # assume the row has: 名稱, 指數, 漲跌點數, 漲跌幅, ...
                # Use the first numeric as close index
                for _, row in rows.iterrows():
                    vals = [str(v).replace(',','') for v in row.tolist()]
                    name = next((v for v in vals if '建材營造' in v), None)
                    num = None
                    for v in vals:
                        try:
                            fv = float(v)
                            num = fv; break
                        except:
                            continue
                    if num is not None:
                        out_daily.append({"date": cur.strftime("%Y-%m-%d"), "value": num})
                        break
        except Exception:
            pass
        # advance to next month
        cur = (cur.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
    # compress to monthly average
    by_month = {}
    for x in out_daily:
        ym = x['date'][:7]
        by_month.setdefault(ym, []).append(x['value'])
    monthly = [{'date': k, 'value': float(sum(v)/len(v))} for k,v in sorted(by_month.items())]
    return ensure_10y_months(monthly)

# -------------------- MOI/NLMA: House Price Index (official, quarterly) & Sinyi (quarterly) --------------------
def fetch_moi_hpi_quarterly():
    # MOI news pages announce quarterly index; scraping is unstable. If not available, return empty and rely on Sinyi.
    # This function attempts to read one latest news and then backfill via archive if available.
    url = "https://www.moi.gov.tw/News_Content.aspx?n=4&s=327724"
    try:
        html = SESSION.get(url, timeout=60).text
        # Attempt to extract '全國住宅價格指數為150.98' pattern and its quarter
        m = re.findall(r'(\d{3}|\d{4})年第(\d)季全國住宅價格指數為([0-9\.]+)', html)
        out = []
        for roc, q, val in m:
            y = int(roc); 
            y = y+1911 if y<1911 else y
            month = int(q)*3
            out.append({"area":"全國","series":[{"date": f"{y:04d}-{month:02d}", "value": float(val)}]})
        # Deduplicate by date
        if out:
            merged = {}
            for s in out:
                key = s['area']
                merged.setdefault(key, {'area':key, 'series':[]})
                for p in s['series']:
                    if p not in merged[key]['series']:
                        merged[key]['series'].append(p)
            for k in merged: merged[k]['series'].sort(key=lambda x: x['date'])
            return list(merged.values())
    except Exception as e:
        log("MOI HPI parse fail:", e)
    return []

def fetch_sinyi_hpi_quarterly():
    # Scrape Sinyi quarterly table page
    url = "https://www.sinyinews.com.tw/quarterly"
    log("Sinyi HPI:", url)
    html = SESSION.get(url, timeout=60).text
    soup = BeautifulSoup(html, 'html.parser')
    # The page contains a series of numbers; we'll extract a simple '全國' series by scanning nearest table-like content.
    text = soup.get_text(" ", strip=True)
    # Very loose extraction; recommend replacing with a more precise parser if DOM stable.
    # For now we return empty and keep the slot; front-end will show MOI when available.
    return []

# -------------------- Build & Save --------------------
def main():
    data = {
      "updated_at": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
      "transactions_total": [],
      "transactions_cities": {},
      "five_bank_new_mortgage_rate": [],
      "jcic_housing_loan_counts": [],
      "construction_loans_balance": [],
      "construction_cost_index": [],
      "rent_cpi_yoy": [],
      "construction_sector_index": [],
      "building_permits": [],
      "usage_licenses": [],
      "house_price_index_moi": [],
      "house_price_index_sinyi": []
    }

    try:
        nat, cities = fetch_transactions_moi()
        data["transactions_total"] = nat
        data["transactions_cities"] = cities
    except Exception as e:
        log("transactions fail:", e)

    try:
        data["five_bank_new_mortgage_rate"] = fetch_cbc_five_bank_new_mortgage_rate()
    except Exception as e:
        log("five bank rate fail:", e)

    try:
        data["construction_loans_balance"] = fetch_cbc_construction_loans()
    except Exception as e:
        log("cbc construction loans fail:", e)

    try:
        data["jcic_housing_loan_counts"] = fetch_jcic_housing_loans_counts()
    except Exception as e:
        log("jcic loan counts fail:", e)

    try:
        data["construction_cost_index"] = fetch_dgbas_cci()
    except Exception as e:
        log("cci fail:", e)

    try:
        data["rent_cpi_yoy"] = fetch_rent_cpi_yoy()
    except Exception as e:
        log("rent cpi fail:", e)

    try:
        data["construction_sector_index"] = fetch_twse_construction_sector()
    except Exception as e:
        log("twse sector fail:", e)

    try:
        permits, usage = fetch_permits_and_usage()
        data["building_permits"] = permits
        data["usage_licenses"] = usage
    except Exception as e:
        log("permits/usage fail:", e)

    try:
        data["house_price_index_moi"] = fetch_moi_hpi_quarterly()
    except Exception as e:
        log("moi hpi fail:", e)

    try:
        data["house_price_index_sinyi"] = fetch_sinyi_hpi_quarterly()
    except Exception as e:
        log("sinyi hpi fail:", e)

    # Save
    os.makedirs(DOCS, exist_ok=True)
    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log("Wrote", OUTFILE)

if __name__ == "__main__":
    main()
