import numpy as np
import pandas as pd
import math as mt
import matplotlib.pyplot as plt

def file_parsing(URL, File_name, Data_name):
    d = pd.read_excel(File_name)
    values = d[Data_name].values
    S_real = np.zeros(len(values))
    for i in range(len(values)):
        S_real[i] = values[i]
    print('Джерело даних:', URL)
    print('Показник:', Data_name, '| Обсяг вибірки:', len(S_real))
    return S_real

def random_uniform(low, high, iter):
    S = np.random.uniform(low, high, iter)
    mS = np.mean(S); dS = np.var(S); scvS = mt.sqrt(dS)
    print('----- статистичні характеристики РІВНОМІРНОГО закону розподілу ВВ -----')
    print('математичне сподівання ВВ =', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ =', scvS)
    print('-----------------------------------------------------------------------')
    return S

def random_exp(alfa, iter):
    S = np.random.exponential(alfa, iter)
    S = S - np.mean(S)
    mS = np.mean(S); dS = np.var(S); scvS = mt.sqrt(dS)
    print('--- статистичні характеристики ЕКСПОНЕНЦІЙНОГО закону розподілу ВВ ---')
    print('математичне сподівання ВВ =', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ =', scvS)
    print('-----------------------------------------------------------------------')
    return S

def Model_quad(n):
    S0 = np.zeros(n)
    for i in range(n):
        S0[i] = 0.00005 * i * i
    return S0

def Model_lin(n):
    S0 = np.zeros(n)
    for i in range(n):
        S0[i] = 0.005 * i
    return S0

def Model_additive(S0, S_noise, n):
    SV = np.zeros(n)
    for i in range(n):
        SV[i] = S0[i] + S_noise[i]
    return SV

def Model_with_AV(SV, nAV, Q_AV, dsig, n):
    SV_AV = np.copy(SV)
    AV_indices = np.random.randint(0, n, nAV)
    AV_values = np.random.normal(0, Q_AV * dsig, nAV)
    for k in range(nAV):
        idx = AV_indices[k]
        SV_AV[idx] = SV[idx] + AV_values[k]
    print('Кількість аномальних вимірів (АВ):', nAV)
    return SV_AV, AV_indices

def MNK_poly(S0, degree=2):
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, degree + 1))
    for i in range(iter):
        Yin[i, 0] = float(S0[i])
        for p in range(1, degree + 1):
            F[i, p] = float(i) ** p
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    C = FFTI.dot(FT).dot(Yin)
    Yout = F.dot(C)
    return Yout, C

def MNK_extrapol(S0, degree, koef):
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, degree + 1))
    for i in range(iter):
        Yin[i, 0] = float(S0[i])
        for p in range(1, degree + 1):
            F[i, p] = float(i) ** p
    C = np.linalg.inv(F.T.dot(F)).dot(F.T).dot(Yin)
    iter_extra = int(iter * (1 + koef))
    F_extra = np.ones((iter_extra, degree + 1))
    for i in range(iter_extra):
        for p in range(1, degree + 1):
            F_extra[i, p] = float(i) ** p
    Yout_extra = F_extra.dot(C)
    return Yout_extra, iter, iter_extra

def clean_anomalies(S, window=15, n_sigma=2.0):
    S_clean = np.copy(S)
    n = len(S)
    n_detected = 0
    half = window // 2
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        local = np.concatenate([S[a:i], S[i+1:b]])
        if len(local) < 2:
            continue
        med = np.median(local)
        sigma = np.std(local)
        if sigma > 0 and abs(S[i] - med) > n_sigma * sigma:
            S_clean[i] = med
            n_detected += 1
    print('Виявлено та очищено аномальних вимірів:', n_detected)
    return S_clean

def quality_metrics(S_real, S_model, text=''):
    S_real = np.asarray(S_real).ravel()
    S_model = np.asarray(S_model).ravel()
    n = len(S_real)
    ss_res = np.sum((S_real - S_model) ** 2)
    ss_tot = np.sum((S_real - np.mean(S_real)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    RMSE = mt.sqrt(ss_res / n)
    print('---------', text, '---------')
    print('R^2 (коеф. детермінації) =', R2)
    print('RMSE =', RMSE)
    print('-----------------------------------------')
    return R2, RMSE

def optimize_degree(S, max_degree=4):
    best_deg, best_R2 = 1, -1e9
    print('=== Оптимізація степеня поліноміальної моделі ===')
    for deg in range(1, max_degree + 1):
        Yout, _ = MNK_poly(S, deg)
        R2, RMSE = quality_metrics(S, Yout, f'поліном степеня {deg}')
        if R2 > best_R2 + 0.005:
            best_R2, best_deg = R2, deg
    print('>> Оптимальний степінь полінома:', best_deg, '| R^2 =', best_R2)
    return best_deg

def Stat_characteristics(SL, Text):
    Yout, _ = MNK_poly(SL, 2)
    SL0 = SL - Yout.ravel()
    mS = np.mean(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)
    print('------------', Text, '------------')
    print('математичне сподівання випадкової складової =', mS)
    print('дисперсія =', dS)
    print('СКВ =', scvS)
    print('-----------------------------------------------------')
    return

def ABF(S):
    n = len(S)
    Yout = np.zeros(n)
    T0 = 1.0
    Xsp = S[0]
    Xspeed = (S[1] - S[0]) / T0
    Xextra = Xsp + Xspeed * T0
    Yout[0] = Xsp
    for i in range(1, n):
        alfa = (2 * (2 * i - 1)) / (i * (i + 1))
        beta = 6 / (i * (i + 1))
        Xsp = Xextra + alfa * (S[i] - Xextra)
        Xspeed = Xspeed + (beta / T0) * (S[i] - Xextra)
        Xextra = Xsp + Xspeed * T0
        Yout[i] = Xsp
    return Yout

def ABGF(S):
    n = len(S)
    Yout = np.zeros(n)
    T0 = 1.0
    Xsp = S[0]
    Xspeed = (S[1] - S[0]) / T0
    Xaccel = 0.0
    Xextra = Xsp + Xspeed * T0 + 0.5 * Xaccel * T0 * T0
    Yout[0] = Xsp
    for i in range(1, n):
        alfa = (3 * (3 * i * i - 3 * i + 2)) / (i * (i + 1) * (i + 2))
        beta = (18 * (2 * i - 1)) / (i * (i + 1) * (i + 2))
        gamma = 60 / (i * (i + 1) * (i + 2))
        err = S[i] - Xextra
        Xsp = Xextra + alfa * err
        Xspeed = Xspeed + (beta / T0) * err
        Xaccel = Xaccel + (gamma / (T0 * T0)) * err
        Xextra = Xsp + Xspeed * T0 + 0.5 * Xaccel * T0 * T0
        Yout[i] = Xsp
    return Yout

def Plot_two(S0, SV, text, fname, label0='модель/тренд', label1='вибірка'):
    plt.figure(figsize=(8, 5))
    plt.plot(SV, label=label1, linewidth=0.8)
    plt.plot(S0, label=label0, linewidth=2, color='orange')
    plt.ylabel(text); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(fname, dpi=120); plt.close()

def Plot_hist(S, text, fname):
    plt.figure(figsize=(7, 4.5))
    plt.hist(S, bins=25, facecolor='blue', alpha=0.5)
    plt.title(text); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(fname, dpi=120); plt.close()


if __name__ == '__main__':
    np.random.seed(9)

    n = 1000
    Q_AV = 5
    nAVv = 10
    nAV = int((n * nAVv) / 100)
    dsig = 5
    alfa_exp = 5
    koef_extrapol = 0.5

    print('############ ГРУПА ВИМОГ 1 ############\n')

    S0_quad = Model_quad(n)
    SN_exp = random_exp(alfa_exp, n)
    SV_quad = Model_additive(S0_quad, SN_exp, n)
    SV_quad_AV, idx_quad = Model_with_AV(SV_quad, nAV, Q_AV, dsig, n)
    Plot_two(S0_quad, SV_quad_AV, 'Квадратична модель + експон. шум + АВ',
             'fig_input_quad.png', 'квадратичний тренд', 'вибірка з АВ')
    Stat_characteristics(SV_quad_AV, 'Вхідна вибірка (квадр. + експон. + АВ)')

    S0_lin = Model_lin(n)
    SN_uni = random_uniform(-8, 8, n)
    SV_lin = Model_additive(S0_lin, SN_uni, n)
    SV_lin_AV, idx_lin = Model_with_AV(SV_lin, nAV, Q_AV, dsig, n)
    Plot_two(S0_lin, SV_lin_AV, 'Лінійна модель + рівномірний шум + АВ',
             'fig_input_lin.png', 'лінійний тренд', 'вибірка з АВ')

    print('\n--- Очищення вхідних даних від аномальних вимірів ---')
    SV_quad_clean = clean_anomalies(SV_quad_AV, window=15, n_sigma=2.0)
    Plot_two(SV_quad_AV, SV_quad_clean, 'Очищення від АВ (квадратична модель)',
             'fig_clean_quad.png', 'очищена вибірка', 'вибірка з АВ')

    print('\n--- Визначення показників якості та оптимізація моделі ---')
    best_deg = optimize_degree(SV_quad_clean, max_degree=4)

    print('\n--- Статистичне навчання поліноміальної моделі (МНК) ---')
    Yout_mnk, C = MNK_poly(SV_quad_clean, best_deg)
    quality_metrics(SV_quad_clean, Yout_mnk, 'МНК поліном (навчена модель)')
    Plot_two(Yout_mnk.ravel(), SV_quad_clean, 'МНК поліноміальна регресія',
             'fig_mnk.png', 'МНК-апроксимація', 'очищена вибірка')

    print('\n--- Прогнозування (екстраполяція) на 0.5 інтервалу ---')
    Yextra, n_obs, n_full = MNK_extrapol(SV_quad_clean, best_deg, koef_extrapol)
    print(f'Інтервал спостереження: {n_obs} | Прогноз до: {n_full} (+{n_full-n_obs})')
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(n_obs), SV_quad_clean, label='очищена вибірка', linewidth=0.8)
    plt.plot(np.arange(n_full), Yextra.ravel(), label='МНК-прогноз', linewidth=2, color='red')
    plt.axvline(n_obs, color='green', linestyle='--', label='межа спостереження')
    plt.ylabel('Прогнозування квадратичного процесу')
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig('fig_extrapol.png', dpi=120); plt.close()

    print('\n############ ГРУПА ВИМОГ 2 ############\n')
    print('--- Рекурентне згладжування alfa-beta фільтром ---')
    Y_abf = ABF(SV_quad_clean)
    quality_metrics(SV_quad_clean, Y_abf, 'alfa-beta фільтр')
    Plot_two(Y_abf, SV_quad_clean, 'alfa-beta фільтр',
             'fig_abf.png', 'ABF-згладжування', 'очищена вибірка')

    print('--- Рекурентне згладжування alfa-beta-gamma фільтром ---')
    Y_abgf = ABGF(SV_quad_clean)
    quality_metrics(SV_quad_clean, Y_abgf, 'alfa-beta-gamma фільтр')
    Plot_two(Y_abgf, SV_quad_clean, 'alfa-beta-gamma фільтр',
             'fig_abgf.png', 'ABGF-згладжування', 'очищена вибірка')

    print('\n############ РЕАЛЬНІ ДАНІ (3 показники) ############\n')
    URL = 'https://www.oschadbank.ua/rates-archive'
    FILE = 'Oschadbank (USD).xls'
    for indicator in ['Купівля', 'Продаж', 'КурсНбу']:
        print(f'\n===== Показник: {indicator} =====')
        S_real = file_parsing(URL, FILE, indicator)
        Stat_characteristics(S_real, f'Реальні дані ({indicator})')
        S_real_clean = clean_anomalies(S_real, window=11, n_sigma=2.5)
        deg_r = optimize_degree(S_real_clean, max_degree=3)
        Yout_r, _ = MNK_poly(S_real_clean, deg_r)
        quality_metrics(S_real_clean, Yout_r, f'МНК ({indicator})')
        Yextra_r, no, nf = MNK_extrapol(S_real_clean, deg_r, 0.5)
        Y_abf_r = ABF(S_real_clean)
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(no), S_real_clean, label='реальні дані', linewidth=1)
        plt.plot(np.arange(nf), Yextra_r.ravel(), label='МНК-прогноз', linewidth=2, color='red')
        plt.plot(np.arange(no), Y_abf_r, label='alfa-beta фільтр', linewidth=1.2, color='green')
        plt.axvline(no, color='gray', linestyle='--')
        plt.ylabel(f'USD: {indicator} (Ощадбанк 2022)')
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(f'fig_real_{indicator}.png', dpi=120); plt.close()

    print('\n############ ОБРОБКУ ЗАВЕРШЕНО ############')
