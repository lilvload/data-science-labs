import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

FILE_NAME = "credit_offers.xlsx"


def file_parsing(File_name):
    d = pd.read_excel(File_name)
    names = d.iloc[:, 0].to_numpy()
    X = d.iloc[:, 1:].to_numpy(dtype=float)
    crit = list(d.columns[1:])
    print('Вхідна матриця критеріїв (decision matrix):')
    print(d.to_string(index=False))
    return names, X, crit


def normalize(X, directions):
    m, n = X.shape
    R = np.zeros((m, n))
    for j in range(n):
        col = X[:, j]
        if directions[j] > 0:                # максимізація
            R[:, j] = col / col.max()
        else:                                # мінімізація
            R[:, j] = col.min() / col
    return R


def multiplicative_criterion(R, w):
    return np.prod(R ** w, axis=1)


def additive_criterion(R, w):
    return R.dot(w)


def make_ranking(names, score):
    order = np.argsort(-score)
    rank = np.empty(len(score), dtype=int)
    rank[order] = np.arange(1, len(score) + 1)
    return order, rank


def plot_scores(names, score, title, fname):
    idx = np.argsort(score)
    colors = ['#1f77b4'] * len(score)
    colors[idx[-1]] = '#2ca02c'
    plt.figure(figsize=(9, 5))
    plt.barh(np.array(names)[idx], np.array(score)[idx], color=np.array(colors)[idx])
    plt.xlabel('Iнтегральний критерiй ефективностi')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=130)
    plt.close()


def plot_compare(names, sc_mult, sc_add, fname):
    a = (sc_mult - sc_mult.min()) / (sc_mult.max() - sc_mult.min())
    b = (sc_add - sc_add.min()) / (sc_add.max() - sc_add.min())
    x = np.arange(len(names))
    plt.figure(figsize=(11, 5))
    plt.bar(x - 0.2, a, 0.4, label='Мультиплiкативний', color='#1f77b4')
    plt.bar(x + 0.2, b, 0.4, label='Адитивний', color='#ff7f0e')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Нормована iнтегральна оцiнка')
    plt.title('Порiвняння методiв скалярної згортки')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=130)
    plt.close()


def plot_heatmap(names, R, crit, fname):
    plt.figure(figsize=(11, 6))
    im = plt.imshow(R, aspect='auto', cmap='RdYlGn')
    plt.colorbar(im, label='Нормоване значення (1 - найкраще)')
    plt.xticks(range(len(crit)), crit, rotation=45, ha='right')
    plt.yticks(range(len(names)), names)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            plt.text(j, i, f'{R[i, j]:.2f}', ha='center', va='center', fontsize=7)
    plt.title('Нормована матриця критерiїв')
    plt.tight_layout()
    plt.savefig(fname, dpi=130)
    plt.close()


def plot_radar(names, R, crit, order, fname):
    top = order[:3]
    angles = np.linspace(0, 2 * np.pi, len(crit), endpoint=False).tolist()
    angles += angles[:1]
    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    palette = ['#2ca02c', '#1f77b4', '#ff7f0e']
    for k, i in enumerate(top):
        vals = R[i].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=palette[k], linewidth=2, label=names[i])
        ax.fill(angles, vals, color=palette[k], alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(crit, fontsize=8)
    ax.set_title('Профiлi трьох найкращих альтернатив', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.10))
    plt.tight_layout()
    plt.savefig(fname, dpi=130)
    plt.close()


if __name__ == '__main__':
    names, X, crit = file_parsing(FILE_NAME)

    directions = np.array([-1, -1, +1, +1, -1, -1, -1, +1])

    w = np.array([0.22, 0.14, 0.08, 0.06, 0.10, 0.20, 0.08, 0.12])

    print('\nСума вагових коефiцiєнтiв =', round(w.sum(), 5))
    print('Напрям оптимiзацiї (+1 макс / -1 мiн):', dict(zip(crit, directions)))

    R = normalize(X, directions)

    score_mult = multiplicative_criterion(R, w)
    score_add = additive_criterion(R, w)

    order_m, rank_m = make_ranking(names, score_mult)
    order_a, rank_a = make_ranking(names, score_add)

    res = pd.DataFrame({
        'Альтернатива': names,
        'I_мульт': np.round(score_mult, 4),
        'Ранг_мульт': rank_m,
        'I_адит': np.round(score_add, 4),
        'Ранг_адит': rank_a,
    }).sort_values('Ранг_мульт').reset_index(drop=True)

    print('\n================ ЗВЕДЕНА ТАБЛИЦЯ РЕЗУЛЬТАТIВ ================')
    print(res.to_string(index=False))

    best = names[order_m[0]]
    print('\nНАЙКРАЩА кредитна пропозицiя (мультиплiкативний критерiй):', best)
    print('Збiг рангу лiдера за двома методами:',
          'ТАК' if rank_a[order_m[0]] == 1 else 'НI')
    d2 = np.sum((rank_m - rank_a) ** 2)
    nN = len(names)
    spearman = 1 - 6 * d2 / (nN * (nN ** 2 - 1))
    print('Коефiцiєнт рангової кореляцiї Спiрмена (мульт vs адит):',
          round(spearman, 4))

    plot_scores(names, score_mult,
                'Iнтегральний критерiй (мультиплiкативна згортка)',
                'fig1_scores_mult.png')
    plot_compare(names, score_mult, score_add, 'fig2_compare.png')
    plot_heatmap(names, R, crit, 'fig3_heatmap.png')
    plot_radar(names, R, crit, order_m, 'fig4_radar.png')

    res.to_excel('results.xlsx', index=False)
    print('\nГрафiки та results.xlsx збережено.')
