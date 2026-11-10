import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 11


def file_parsing(file_name):
    d = pd.read_excel(file_name)
    print('---------------- ВХІДНІ ДАНІ ----------------')
    print(d.to_string())
    print('Джерело даних: William McKinney, Python for Data Analysis')
    return d


def preliminary_analysis(df):
    print('\n---------------- ПОПЕРЕДНІЙ АНАЛІЗ ----------------')
    print('Розмір таблиці (рядки, стовпці):', df.shape)
    print('\nСтовпці та типи даних:')
    print(df.dtypes.to_string())
    print('\nКількість пропущених значень:')
    print(df.isnull().sum().to_string())
    print('\nКількість дублікатів рядків:', df.duplicated().sum())
    print('\nУнікальні менеджери:', list(df['Manager'].unique()))
    print('Унікальні представники:', list(df['Rep'].unique()))
    print('Унікальні продукти:', list(df['Product'].unique()))
    print('Унікальні статуси угод:', list(df['Status'].unique()))
    print('\nОписова статистика числових полів:')
    print(df[['Quantity', 'Price']].describe().to_string())
    return df


def add_efficiency_metrics(df):
    df = df.copy()
    df['Sales'] = df['Quantity'] * df['Price']
    margin = {'CPU': 0.25, 'Software': 0.55, 'Maintenance': 0.40, 'Monitor': 0.30}
    df['Profit'] = df.apply(lambda r: r['Sales'] * margin.get(r['Product'], 0.20), axis=1)
    print('\n---------------- ПОКАЗНИКИ ЕФЕКТИВНОСТІ ----------------')
    print(df[['Name', 'Manager', 'Product', 'Quantity', 'Price', 'Sales', 'Profit', 'Status']].to_string())
    print('\nЗагальний обсяг продажів:', round(df['Sales'].sum(), 2))
    print('Загальний прибуток:', round(df['Profit'].sum(), 2))
    print('Середня рентабельність:', round(df['Profit'].sum() / df['Sales'].sum() * 100, 2), '%')
    return df


def region_dynamics(df):
    by_region = df.groupby('Manager').agg(
        Sales=('Sales', 'sum'),
        Profit=('Profit', 'sum'),
        Deals=('Status', 'count'),
        Won=('Status', lambda s: (s == 'won').sum())
    ).reset_index()
    by_region['WinRate'] = (by_region['Won'] / by_region['Deals'] * 100).round(2)
    by_region['Margin'] = (by_region['Profit'] / by_region['Sales'] * 100).round(2)
    print('\n---------------- ДИНАМІКА ЗА РЕГІОНАМИ (МЕНЕДЖЕРАМИ) ----------------')
    print(by_region.to_string())
    return by_region


def status_pivot(df):
    piv = pd.pivot_table(df, values='Sales', index='Manager', columns='Status',
                         aggfunc='sum', fill_value=0, margins=True, margins_name='Разом')
    print('\n---------------- ЗВЕДЕНА ТАБЛИЦЯ ОБСЯГУ ПРОДАЖІВ ЗА СТАТУСОМ ----------------')
    print(piv.round(0).to_string())
    return piv


def product_summary(df):
    prod = df.groupby('Product').agg(
        Quantity=('Quantity', 'sum'),
        Sales=('Sales', 'sum'),
        Profit=('Profit', 'sum')
    ).reset_index().sort_values('Sales', ascending=False)
    print('\n---------------- АНАЛІЗ ЗА ПРОДУКТАМИ ----------------')
    print(prod.to_string())
    return prod


def MNK(x, y):
    n = len(x)
    F = np.ones((n, 2))
    F[:, 1] = x
    Yin = y.reshape(-1, 1)
    FT = F.T
    C = np.linalg.inv(FT.dot(F)).dot(FT).dot(Yin)
    Yout = F.dot(C)
    return C, Yout.flatten()


def determination(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0


def regression_and_forecast(by_region, df):
    rep_profit = df.groupby('Rep')['Profit'].sum().sort_values(ascending=False)
    x = np.arange(len(rep_profit))
    y = rep_profit.values.astype(float)
    C, y_hat = MNK(x, y)
    r2 = determination(y, y_hat)
    print('\n---------------- МАТЕМАТИЧНА МОДЕЛЬ (МНК) ----------------')
    print('Лінійна модель прибутку за представниками: y = a0 + a1*t')
    print('Коефіцієнт a0 =', round(C[0, 0], 2))
    print('Коефіцієнт a1 =', round(C[1, 0], 2))
    print('Коефіцієнт детермінації R2 =', round(r2, 4))

    x_future = np.arange(len(rep_profit), len(rep_profit) + 3)
    y_future = C[0, 0] + C[1, 0] * x_future
    print('\n---------------- ПРОГНОЗ ПРИБУТКУ ----------------')
    forecast = pd.DataFrame({
        'Крок': x_future + 1,
        'ПрогнозованийПрибуток': np.round(y_future, 2)
    })
    print(forecast.to_string(index=False))
    return rep_profit, x, y, y_hat, x_future, y_future, r2, C


def fmt_k(x, _):
    return f'{x/1000:.0f}k'


def plot_sales_by_region(by_region):
    plt.clf()
    fig, ax = plt.subplots()
    ax.bar(by_region['Manager'], by_region['Sales'], color='#4C72B0', alpha=0.85)
    ax.set_ylabel('Обсяг продажів, грош. од.')
    ax.set_title('Обсяг продажів за регіонами (менеджерами)')
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_k))
    for i, v in enumerate(by_region['Sales']):
        ax.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('figures/fig1_sales_region.png', dpi=130)
    plt.close()


def plot_profit_by_region(by_region):
    plt.clf()
    fig, ax = plt.subplots()
    ax.bar(by_region['Manager'], by_region['Profit'], color='#55A868', alpha=0.85)
    ax.set_ylabel('Прибуток, грош. од.')
    ax.set_title('Прибуток за регіонами (менеджерами)')
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_k))
    for i, v in enumerate(by_region['Profit']):
        ax.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('figures/fig2_profit_region.png', dpi=130)
    plt.close()


def plot_status(df):
    plt.clf()
    order = ['won', 'pending', 'presented', 'declined']
    counts = df['Status'].value_counts().reindex(order).fillna(0)
    colors = ['#55A868', '#DD8452', '#4C72B0', '#C44E52']
    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values, color=colors, alpha=0.85)
    ax.set_ylabel('Кількість угод')
    ax.set_title('Розподіл угод за статусом')
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(int(v)), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('figures/fig3_status.png', dpi=130)
    plt.close()


def plot_products(prod):
    plt.clf()
    fig, ax = plt.subplots()
    ax.barh(prod['Product'], prod['Sales'], color='#8172B3', alpha=0.85)
    ax.set_xlabel('Обсяг продажів, грош. од.')
    ax.set_title('Обсяг продажів за продуктами')
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    plt.tight_layout()
    plt.savefig('figures/fig4_products.png', dpi=130)
    plt.close()


def plot_pivot_heat(df):
    plt.clf()
    piv = pd.pivot_table(df, values='Sales', index='Manager', columns='Status',
                         aggfunc='sum', fill_value=0)
    fig, ax = plt.subplots()
    im = ax.imshow(piv.values, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            ax.text(j, i, f'{piv.values[i, j]:,.0f}', ha='center', va='center', fontsize=9)
    ax.set_title('Теплова карта: обсяг продажів (менеджер × статус)')
    fig.colorbar(im, ax=ax, format=FuncFormatter(fmt_k))
    ax.grid(False)
    plt.tight_layout()
    plt.savefig('figures/fig5_pivot_heat.png', dpi=130)
    plt.close()


def plot_regression(rep_profit, x, y, y_hat, x_future, y_future, r2):
    plt.clf()
    fig, ax = plt.subplots()
    labels = list(rep_profit.index)
    ax.scatter(x, y, color='#4C72B0', s=70, zorder=3, label='Факт')
    ax.plot(x, y_hat, color='#DD8452', linewidth=2, label='МНК (тренд)')
    ax.plot(x_future, y_future, 'o--', color='#C44E52', linewidth=2, label='Прогноз')
    all_x = list(x) + list(x_future)
    all_lab = labels + [f'+{i+1}' for i in range(len(x_future))]
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_lab, rotation=30, ha='right')
    ax.set_ylabel('Прибуток, грош. од.')
    ax.set_title(f'Регресія та прогноз прибутку (R2 = {r2:.3f})')
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax.legend()
    plt.tight_layout()
    plt.savefig('figures/fig6_regression.png', dpi=130)
    plt.close()


if __name__ == '__main__':
    FILE = 'Data_Set_9.xlsx'

    df = file_parsing(FILE)
    df = preliminary_analysis(df)
    df = add_efficiency_metrics(df)
    by_region = region_dynamics(df)
    piv = status_pivot(df)
    prod = product_summary(df)
    rep_profit, x, y, y_hat, x_future, y_future, r2, C = regression_and_forecast(by_region, df)

    plot_sales_by_region(by_region)
    plot_profit_by_region(by_region)
    plot_status(df)
    plot_products(prod)
    plot_pivot_heat(df)
    plot_regression(rep_profit, x, y, y_hat, x_future, y_future, r2)

    by_region.to_excel('region_dynamics_output.xlsx', index=False)
    print('\nФайл результату збережено: region_dynamics_output.xlsx')
    print('Графіки збережено у каталозі figures/')
