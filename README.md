# Data Science Labs

Laboratory works for the university course **"Data Science Technologies"** (Igor Sikorsky Kyiv Polytechnic Institute), written as standalone Python scripts.

The eight labs cover the core of a practical data science workflow: generating and analysing synthetic data, real-world exchange-rate data, cleaning anomalous measurements, least-squares regression and forecasting, recursive filtering, multi-criteria decision making, web scraping and OLAP analysis, clustering and computer vision, neural-network forecasting, business analytics and credit scoring.

> **Note:** code is in Python, while comments, console output and chart labels are in Ukrainian.

## Contents

| Lab | Topic | Main tools |
|-----|-------|------------|
| [Lab 1](lab-1/Lab_work_1.py) | Statistical modelling of data with noise and anomalies | NumPy, pandas, Matplotlib |
| [Lab 2](lab-2/Lab_work_2.py) | Anomaly cleaning, least-squares regression, forecasting and α-β(-γ) filters | NumPy, pandas, Matplotlib |
| [Lab 3](lab-3/lab3.py) | Multi-criteria decision making: choosing the best bank credit offer | NumPy, pandas, Matplotlib |
| [Lab 4](lab-4/lab4.py) | Web scraping of café menus, OLAP cube and text mining | Selenium, BeautifulSoup, pandas, WordCloud |
| [Lab 5](lab-5/lab-5/lab5.py) | Machine learning: K-means clustering, image segmentation, object counting | scikit-learn, OpenCV |
| [Lab 6](lab-6/lab6.py) | Time-series forecasting with a neural network | scikit-learn (`MLPRegressor`) |
| [Lab 7](lab-7/lab7.py) | Sales performance analytics, pivot tables and profit forecast | pandas, NumPy, Matplotlib |
| [Lab 8](lab-8/lab8.py) | Credit scoring, discriminant analysis and fraud detection | scikit-learn (LDA), pandas |

## Repository structure

```
data-science-labs/
├── lab-1/   Lab_work_1.py, Oschadbank (USD).xls
├── lab-2/   Lab_work_2.py, Oschadbank (USD).xls, fig_*.png
├── lab-3/   lab3.py, credit_offers.xlsx, results.xlsx, fig*.png
├── lab-4/   lab4.py, results/ (menu_dataset.csv, olap_cube.csv, raw_*.html, fig_*.png)
├── lab-5/   lab-5/ (lab5.py, Oschadbank (USD).xls, images/, figures/),
│            report (.pdf, .docx), lab-5.rar
├── lab-6/   lab6.py, metrics_table.csv, summary.json, figs/
├── lab-7/   lab7.py, Data_Set_9.xlsx, region_dynamics_output.xlsx, figures/
└── lab-8/   lab8.py, sample_data.xlsx, data_description.xlsx,
             scoring_results.xlsx, metrics.csv, fig_*.png
```

Each folder is self-contained: the script sits next to the data it reads, and the generated charts and result tables are committed as well, so the results can be viewed on GitHub without running anything.

## Labs

### Lab 1. Statistical modelling of data with noise and anomalies

Builds synthetic measurement series and compares them with real data.

- Generates 10,000-point samples from a **quadratic** and a **linear** trend with additive **exponential** and **uniform** noise.
- Injects **anomalous measurements** (10% of points) drawn from a normal distribution.
- Computes the statistical characteristics of each sample: mean, variance, standard deviation and the R² of a least-squares quadratic trend.
- Loads the 2022 USD exchange rates of Oschadbank (buy, sell and the National Bank of Ukraine rate) from `Oschadbank (USD).xls` and applies the same analysis to the real data.

### Lab 2. Anomaly cleaning, regression, forecasting and recursive filtering

Continues Lab 1 with a full processing pipeline.

- Generates a quadratic trend with exponential noise and anomalies, and a linear trend with uniform noise and anomalies.
- **Detects and cleans anomalies** with a sliding window: a point that deviates from the local median by more than n·σ is replaced by the median.
- **Chooses the polynomial degree** automatically by comparing R² for degrees 1 to 4.
- Fits the trend with the **least squares method** (implemented with matrix algebra) and **extrapolates** it 50% beyond the observation interval.
- Smooths the data with recursive **α-β** and **α-β-γ filters** and compares them by R² and RMSE.
- Runs the whole pipeline on the three Oschadbank exchange-rate series and saves the charts as `fig_*.png`.

### Lab 3. Multi-criteria choice of a bank credit offer

Ranks credit offers from 12 Ukrainian banks (`credit_offers.xlsx`) by eight criteria: interest rate, down payment, term, maximum amount, fee, overpayment, processing time and bank rating.

- Normalizes the decision matrix, taking into account whether each criterion should be maximized or minimized.
- Computes the integral efficiency score with two scalar convolution methods: **multiplicative** and **additive** (weighted sum).
- Ranks the alternatives, names the best offer and compares both rankings with **Spearman's rank correlation**.
- Plots the scores, a comparison of both methods, a heatmap of the normalized matrix and a radar chart of the top three offers. The ranking is saved to `results.xlsx`.

### Lab 4. Web scraping, OLAP analysis and text mining of café menus

Collects and analyses the menus of five cafés in Lviv from the Glovo delivery service.

- **Scrapes** the store pages with Selenium (scrolling to load the full menu) and parses the dishes and prices with BeautifulSoup and regular expressions. The raw HTML of each page is saved to `results/`.
- Classifies the dishes into categories (coffee, tea, desserts, breakfasts, drinks, other) by keywords.
- Builds an **OLAP cube** (café × category) with the number of items and min/avg/max price, plus **slice**, **dice** and **pivot** operations.
- Performs simple **text mining** on dish names: tokenization, stop-word removal and word frequencies.
- Reports price statistics and plots the menu size per café, average prices by category, category shares, a coffee price comparison, an OLAP heatmap, a word cloud and the top words.

### Lab 5. Machine learning: clustering and computer vision

Solves three tasks from the lab assignment (option 9).

- **Clustering of numeric data:** clusters the Oschadbank exchange-rate data with **K-means**, choosing the number of clusters (2 to 7) by the elbow method and the silhouette score, and visualises the clusters in 3D and over time.
- **Colour-based image segmentation:** clusters the pixels of `images/color_scene.png` in RGB space with K-means, shows the segmented image for k = 4 and k = 5 and a mask for each colour cluster.
- **Object counting:** counts the coins in `images/coins.png` with OpenCV: Otsu thresholding, morphological opening, distance transform and the **watershed** algorithm.

### Lab 6. Time-series forecasting with a neural network

Forecasts a noisy periodic signal with a multilayer perceptron.

- Generates a 5,000-point **sinusoid** with normal noise (σ = 15) and **40% anomalous measurements**.
- Cleans the anomalies with a rolling median and a robust MAD-based n-sigma rule.
- Turns the series into sliding windows of 100 points and trains an `MLPRegressor` (64, 32) on the first 80% of the data.
- Evaluates **one-step-ahead** and **recursive multi-step** forecasts on the test part with R², RMSE and MAE.
- Compares nine network architectures, from one hidden layer of 4 neurons to (128, 64), by accuracy, number of parameters and training time.

On the saved run the main network reached **R² ≈ 0.909**, and the best architecture, (128, 64), reached **R² ≈ 0.911** (see `metrics_table.csv` and `summary.json`).

### Lab 7. Sales performance analytics

Analyses a small sales funnel dataset (`Data_Set_9.xlsx`, from *Python for Data Analysis* by Wes McKinney): accounts, sales reps, managers, products, quantities, prices and deal status.

- Runs a preliminary analysis: data types, missing values, duplicates and descriptive statistics.
- Computes **sales** and **profit** (with a margin per product) and the overall profitability.
- Aggregates the results by manager (region): sales, profit, number of deals, **win rate** and margin.
- Builds a **pivot table** of sales by manager and deal status, and a summary by product.
- Fits a linear **least-squares model** of profit by sales rep and makes a three-step forecast.
- Saves six charts to `figures/` and the regional summary to `region_dynamics_output.xlsx`.

### Lab 8. Credit scoring and fraud detection

Builds a scoring model on 500 consumer-loan applications (`sample_data.xlsx`, field descriptions in `data_description.xlsx`).

- Selects the relevant fields, cleans the data (age from birth date, median and mode imputation, clipping) and engineers features such as disposable income, expense ratio, payment burden and loan-to-income ratio.
- Computes an expert **scoring card** (`Scor`) and makes a credit decision by a percentile threshold.
- Trains a **Linear Discriminant Analysis** classifier to predict whether a loan will be repaid and evaluates it with accuracy, precision, recall, F1, ROC-AUC and a confusion matrix.
- Flags **suspicious applications** with eight rule-based fraud indicators (implausible age, income outliers, expenses above income, seniority greater than age and so on).
- Saves the per-application results to `scoring_results.xlsx` and the metrics to `metrics.csv`.

On the saved run the LDA model reached accuracy 0.74 and F1 0.85, but ROC-AUC is only 0.54, so it separates the classes only slightly better than chance and leans towards predicting repayment.

## Getting started

1. Clone the repository:

   ```bash
   git clone https://github.com/lilvload/data-science-labs.git
   cd data-science-labs
   ```

2. Create a virtual environment and install the dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate

   pip install numpy pandas matplotlib scikit-learn openpyxl xlrd opencv-python selenium beautifulsoup4 wordcloud
   ```

   `xlrd` is needed to read the old `.xls` files, and `openpyxl` reads and writes `.xlsx`.

3. Run a script from its own folder, because the scripts load data and save results by relative path:

   ```bash
   cd lab-2
   python Lab_work_2.py
   ```


### Notes on running

- **Lab 1** shows its charts in interactive windows; the other labs save them to image files.
- **Lab 4** opens a real Chrome window through Selenium (set `HEADLESS = True` in the script to hide it), so it needs Google Chrome and an internet connection. The scraping depends on Glovo's current page layout, so a new run may collect a different menu than the saved `results/`.
- **Labs 1 and 4** use random sampling or live data without a fixed seed, so re-running them gives slightly different numbers. Labs 2, 5, 6 and 8 fix the random seed.

## Technologies

Python · NumPy · pandas · Matplotlib · scikit-learn · OpenCV · Selenium · BeautifulSoup · WordCloud
