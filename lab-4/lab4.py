import os
import re
import time
import json
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup


CAFES = [
    {
        "name": "SHOco.",
        "url": "https://glovoapp.com/uk/ua/lviv/stores/shoco1-lvi",
    },
    {
        "name": "KREDENS CAFE",
        "url": "https://glovoapp.com/uk/ua/lviv/stores/kredens-cafe-lvi",
    },
    {
        "name": "CACAO",
        "url": "https://glovoapp.com/uk/ua/lviv/stores/cacao-lvi",
    },
    {
        "name": "Cinnamon Bakery",
        "url": "https://glovoapp.com/uk/ua/lviv/stores/cinnamon-bakery-lvi",
    },
    {
        "name": "Tempo del Caffe",
        "url": "https://glovoapp.com/uk/ua/lviv/stores/tempo-del-caffe-lvi",
    },
]

HEADLESS = False
SCROLL_PAUSE = 1.5
SCROLL_STEPS = 12
PAGE_LOAD_WAIT = 6

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CATEGORY_KEYWORDS = {
    "Кава": ["кава", "еспресо", "еспрессо", "капучино", "капучіно", "лате", "латте",
             "американо", "флет", "раф", "мокачино", "макіато", "ристрето",
             "колд брю", "cappuccino", "latte", "espresso", "americano", "coffee",
             "флэт", "doppio", "допіо", "кортадо"],
    "Чай": ["чай", "матча", "матalmа", "tea", "пуер", "улун", "ройбуш", "каркаде"],
    "Десерт": ["чізкейк", "торт", "тістечко", "круасан", "круассан", "еклер",
               "тірамісу", "брауні", "мафін", "маффін", "наполеон", "медовик",
               "штрудель", "печиво", "десерт", "панкейк", "панкейки", "вафлі",
               "морозиво", "пиріг", "макарон", "профітролі", "cheesecake",
               "тарт", "фондан", "крембрюле", "шарлотка"],
    "Сніданок": ["сніданок", "омлет", "сирники", "сирник", "каша", "боул",
                 "гранола", "тост", "яйця", "яєчня", "млинці", "панкейк",
                 "breakfast", "бенедикт", "скрембл", "шакшука", "поридж"],
    "Напій": ["сік", "лимонад", "смузі", "какао", "коктейль", "вода", "морс",
              "глінтвейн", "пунш", "фреш", "айс", "кола", "напій", "milkshake",
              "молочний", "збитень", "узвар", "компот"],
}

STOP_WORDS = {
    "з", "та", "і", "на", "по", "у", "в", "до", "зі", "із", "для", "без",
    "грн", "uah", "мл", "ml", "г", "гр", "шт", "the", "and", "with",
    "класичний", "класичне", "домашній", "великий", "малий",
    "style", "class", "href", "src", "div", "span",
}


def make_driver():
    options = Options()
    if HEADLESS:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1280,2000")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--lang=uk-UA")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    driver = webdriver.Chrome(options=options)
    return driver


def scroll_page(driver):
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(SCROLL_STEPS):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


def fetch_html(driver, url):
    driver.get(url)
    time.sleep(PAGE_LOAD_WAIT)
    scroll_page(driver)
    return driver.page_source


def classify(dish):
    text = dish.lower()
    for category, words in CATEGORY_KEYWORDS.items():
        for w in words:
            if w in text:
                return category
    return "Інше"


def clean_name(text):
    text = re.sub(r"(style|class|href|src|width|height)\s*=\s*\"[^\"]*\"", " ", text, flags=re.I)
    text = re.sub(r"[{<>}]", " ", text)
    text = re.sub(r"\b\d{1,4}\s*(?:мл|ml|г|гр|шт)\b", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[•\-\*\d\.\)]+\s*", "", text)
    return text


def extract_items(html, cafe_name):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    rows = []
    seen = set()
    price_pattern = re.compile(r"(\d{1,4})[.,]?(\d{0,2})\s*(?:грн|₴|UAH)", re.IGNORECASE)

    candidates = soup.find_all(
        ["div", "li", "article", "section"],
        attrs={"class": re.compile(r"(product|item|card|tile|dish|menu)", re.I)},
    )
    if not candidates:
        candidates = soup.find_all(["li", "article", "div"])

    for node in candidates:
        text = node.get_text(separator=" ", strip=True)
        if not text or len(text) > 200:
            continue
        match = price_pattern.search(text)
        if not match:
            continue
        price = float(match.group(1) + ("." + match.group(2) if match.group(2) else ""))
        if price < 10 or price > 2000:
            continue
        name = price_pattern.sub("", text).strip()
        name = clean_name(name)
        if len(name) < 3 or len(name) > 80:
            continue
        key = (cafe_name, name.lower(), price)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "cafe": cafe_name,
            "category": classify(name),
            "dish": name,
            "price": price,
        })
    return rows


def collect_dataframe():
    driver = make_driver()
    all_rows = []
    try:
        for cafe in CAFES:
            if cafe["url"].startswith("ВСТАВ"):
                print("Пропущено (не задано URL):", cafe["name"])
                continue
            print("Парсинг:", cafe["name"], "->", cafe["url"])
            html = fetch_html(driver, cafe["url"])
            items = extract_items(html, cafe["name"])
            print("   Знайдено позицій:", len(items))
            all_rows.extend(items)
            with open(os.path.join(RESULTS_DIR, f"raw_{cafe['name']}.html"),
                      "w", encoding="utf-8") as f:
                f.write(html)
    finally:
        driver.quit()
    df = pd.DataFrame(all_rows)
    return df


def olap_cube(df):
    cube = df.groupby(["cafe", "category"]).agg(
        positions=("price", "count"),
        avg_price=("price", "mean"),
        min_price=("price", "min"),
        max_price=("price", "max"),
    ).round(2)
    return cube


def slice_by_category(df, category):
    return df[df["category"] == category].copy()


def dice_cafes_category(df, cafes, categories):
    return df[df["cafe"].isin(cafes) & df["category"].isin(categories)].copy()


def pivot_avg_price(df):
    return pd.pivot_table(
        df, values="price", index="cafe", columns="category", aggfunc="mean"
    ).round(2)


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^а-яіїєґ'a-z\s]", " ", text)
    tokens = [t.strip("'") for t in text.split()]
    return [t for t in tokens if len(t) > 2 and t not in STOP_WORDS]


def text_mining(df):
    corpus = " ".join(df["dish"].tolist())
    return Counter(tokenize(corpus))


def plot_positions_per_cafe(df):
    counts = df.groupby("cafe")["dish"].count().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(counts.index, counts.values, color="#4C72B0")
    plt.ylabel("Кількість позицій")
    plt.title("Обсяг меню кав'ярень м. Львів")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_positions.png"), dpi=150)
    plt.close()


def plot_avg_price_by_category(df):
    table = pivot_avg_price(df)
    table.plot(kind="bar", figsize=(11, 6))
    plt.ylabel("Середня ціна, грн")
    plt.title("Середня ціна за категоріями у кав'ярнях м. Львів")
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Категорія", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_avg_price.png"), dpi=150)
    plt.close()


def plot_category_share(df):
    shares = df.groupby("category")["dish"].count().sort_values(ascending=False)
    plt.figure(figsize=(7, 7))
    plt.pie(shares.values, labels=shares.index, autopct="%1.1f%%",
            startangle=90, colors=plt.cm.Set3.colors)
    plt.title("Структура меню за категоріями (узагальнено)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_category_share.png"), dpi=150)
    plt.close()


def plot_coffee_price_compare(df):
    coffee = slice_by_category(df, "Кава")
    if coffee.empty:
        return
    table = coffee.groupby("cafe")["price"].mean().sort_values()
    plt.figure(figsize=(10, 5))
    plt.barh(table.index, table.values, color="#C44E52")
    plt.xlabel("Середня ціна кави, грн")
    plt.title("Порівняння середньої ціни кави між кав'ярнями")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_coffee_price.png"), dpi=150)
    plt.close()


def plot_heatmap(df):
    table = pivot_avg_price(df)
    fig, ax = plt.subplots(figsize=(10, 6))
    data = table.fillna(0).values
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_xticklabels(table.columns, rotation=30, ha="right")
    ax.set_yticklabels(table.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] > 0:
                ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center",
                        color="black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Середня ціна, грн")
    ax.set_title("OLAP-теплокарта: кав'ярня × категорія")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_heatmap.png"), dpi=150)
    plt.close()


def plot_wordcloud(freq):
    if not freq:
        return
    wc = WordCloud(width=1000, height=600, background_color="white",
                   colormap="viridis").generate_from_frequencies(freq)
    plt.figure(figsize=(11, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Хмара слів меню кав'ярень м. Львів")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_wordcloud.png"), dpi=150)
    plt.close()


def plot_top_words(freq, top_n=15):
    common = freq.most_common(top_n)
    if not common:
        return
    words = [w for w, _ in common][::-1]
    values = [c for _, c in common][::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(words, values, color="#55A868")
    plt.xlabel("Частота")
    plt.title(f"Топ-{top_n} слів у назвах страв")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_top_words.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    df = collect_dataframe()

    if df.empty:
        print("Дані не зібрано. Перевірте URL у списку CAFES та доступність сайту.")
        raise SystemExit

    print("=" * 60)
    print("ЗІБРАНІ ДАНІ МЕНЮ КАВ'ЯРЕНЬ М. ЛЬВІВ")
    print("=" * 60)
    print(df.to_string(index=False))
    print()

    print("Кількість кав'ярень:", df["cafe"].nunique())
    print("Загальна кількість позицій меню:", len(df))
    print("Категорії:", ", ".join(sorted(df["category"].unique())))
    print()

    cube = olap_cube(df)
    print("=" * 60)
    print("OLAP-КУБ: агрегація за вимірами (кав'ярня x категорія)")
    print("=" * 60)
    print(cube.to_string())
    print()

    pivot = pivot_avg_price(df)
    print("=" * 60)
    print("ЗВЕДЕНА ТАБЛИЦЯ: середня ціна (грн)")
    print("=" * 60)
    print(pivot.to_string())
    print()

    coffee = slice_by_category(df, "Кава")
    if not coffee.empty:
        print("=" * 60)
        print("SLICE: зріз за категорією 'Кава'")
        print("=" * 60)
        print("Середня ціна кави по місту:", round(coffee["price"].mean(), 2), "грн")
        print("Найдешевша кава:", coffee.loc[coffee["price"].idxmin(), "dish"],
              "-", coffee["price"].min(), "грн (",
              coffee.loc[coffee["price"].idxmin(), "cafe"], ")")
        print("Найдорожча кава:", coffee.loc[coffee["price"].idxmax(), "dish"],
              "-", coffee["price"].max(), "грн (",
              coffee.loc[coffee["price"].idxmax(), "cafe"], ")")
        print()

    freq = text_mining(df)
    print("=" * 60)
    print("TEXT MINING: топ-15 слів у назвах страв")
    print("=" * 60)
    for word, count in freq.most_common(15):
        print(f"{word:25s} {count}")
    print()

    print("=" * 60)
    print("УЗАГАЛЬНЕНІ СТАТИСТИЧНІ ХАРАКТЕРИСТИКИ ЦІН")
    print("=" * 60)
    print("Математичне сподівання ціни:", round(df["price"].mean(), 2), "грн")
    print("Дисперсія ціни:", round(df["price"].var(), 2))
    print("СКВ ціни:", round(df["price"].std(), 2), "грн")
    print("Медіана ціни:", round(df["price"].median(), 2), "грн")
    print("Мінімальна ціна:", df["price"].min(), "грн")
    print("Максимальна ціна:", df["price"].max(), "грн")
    print()

    plot_positions_per_cafe(df)
    plot_avg_price_by_category(df)
    plot_category_share(df)
    plot_coffee_price_compare(df)
    plot_heatmap(df)
    plot_wordcloud(freq)
    plot_top_words(freq)

    df.to_csv(os.path.join(RESULTS_DIR, "menu_dataset.csv"),
              index=False, encoding="utf-8-sig")
    cube.to_csv(os.path.join(RESULTS_DIR, "olap_cube.csv"), encoding="utf-8-sig")

    print("Графіки та таблиці збережено у:", RESULTS_DIR)
